"""The 9 graph nodes + their conditional-edge routers (ARCHITECTURE_SPEC §3.12 / §B.5).

Every node is a plain ``(GraphState) -> dict`` function returning a *partial*
state update (never the full state); every node appends its own name to
``events`` (the append-only reducer turns that into a derived execution
trace). Errors are never raised out of a node -- a hard failure is written
into ``state["error"]``/``state["status"]`` and routed to the terminal
``error`` node as a graph edge.

Phase 4 scope (the full agentic layer): ``plan`` runs the REAL LLM planner
(``plan_request`` → classify + fill, with the last rejection's ``plan_feedback`` threaded
back on a re-plan); ``review_intent`` / ``review_output`` run the REAL LLM reviewers through
``get_adapter()`` (provider-selected; ``LLM_PROVIDER`` unset ⇒ the zero-network
``StubAdapter``, so the whole offline suite stays network-free). ``execute`` does REAL dispatch
by ``query_class`` (``_dispatch_execute`` → the six live deterministic classes incl.
``single_value``) and stashes a bounded ``fetched_records`` index for the Output Reviewer's
record-grounded citation re-verify. Offline test sentinels
(``_force_error/_empty/_too_large/_canned/_force_plan/_force_reject/_force_revise``)
short-circuit before any network / LLM call so the structural graph tests stay deterministic.
The LLM decides WHAT to compute; the deterministic tools compute it — the model never emits a
number (a structured-output model has no count field; the builder inserts every number).

Escalation budget (§B.5): shared and <=1 across the three re-plan triggers
(Plan Checker reject, Intent Reviewer revise, execute zero-results). Because
LangGraph conditional-edge routers are pure functions of state (they cannot
themselves write a state update), the increment lives in ``plan``: the first
call (``iter_count == 0``) never increments; any re-entry (``iter_count >
0``, i.e. a back-edge from check/review_intent/execute) bumps
``escalation_count`` once. All three routers then just read the same shared
counter -- which is exactly what "shared, <=1 across three triggers" means.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging

from app import config
from app.api.schemas import Citation
from app.cache import RESPONSE_CACHE, plan_cache_key
from app.ctgov.aggregate import _nct_id
from app.ctgov.client import CTGovClient
from app.ctgov.params import build_search_params
from app.ctgov.tools import (
    DATE_PROJECTION,
    NETWORK_FIELDS,
    aggregate_by,
    aggregate_by_counts,
    build_network,
    compare,
    count_trials,
    is_count_aggregatable,
    study_duration_histogram,
    timeseries,
)
from app.graph import guards
from app.graph.clarify import detect_dangling_reference
from app.graph.state import GraphState
from app.llm.adapter import get_adapter
from app.llm.planner import plan_request
from app.llm.reviewers import IntentVerdict, review_output_llm, should_skip_intent_review
from app.llm.reviewers import review_intent as review_intent_llm
from app.logging_setup import log_event
from app.plan.checker import check_plan
from app.plan.models import CheckResult, Plan
from app.viz.review import (
    computed_numbers,
    deterministic_precheck,
    note_number_safe,
    record_grounded_reverify,
)
from app.viz.spec import build_clarification_envelope, build_envelope

logger = logging.getLogger(__name__)

# Over-budget threshold (§B.7): a match set above this is refused, not paged.
# Sourced from config so an operator can retune the DoS bound with one env var
# (the config default is 20_000 — behaviour-preserving).
_TOO_LARGE_THRESHOLD = config.TOO_LARGE_THRESHOLD

# plan.entities dimension -> ClinicalTrials.gov query-area code (§A(d)). Only the
# dimensions actually present on the plan become query.<area> selectors.
_ENTITY_TO_AREA: dict[str, str] = {
    "condition": "cond",
    "drug": "intr",
    "sponsor": "spons",
    "country": "locn",
    "term": "term",
}


def _has_force_sentinel(merged_inputs: dict | None) -> bool:
    """Is any offline test sentinel (``_force_*``) active? The response cache is
    bypassed when so — the structural/offline suite must stay deterministic and
    cache-free (a forced/canned plan must never be served from, or stored into,
    the shared cache and leak across tests)."""
    return any(str(k).startswith("_force") for k in (merged_inputs or {}))


def _adapter():
    # The one provider-agnostic seam every LLM node calls through (C-99). Provider + per-node
    # model come from env (LLM_PROVIDER / LLM_MODEL_PLANNER / LLM_MODEL_REVIEWER); the adapter
    # picks the planner model for propose() and the cheaper reviewer model for verify(). Unset
    # LLM_PROVIDER ⇒ StubAdapter (zero network) so the offline suite never makes a call.
    return get_adapter()


def _plan_query(plan: Plan) -> dict[str, str]:
    """Derive the ``{area_code: value}`` search selectors from ``plan.entities``."""
    return {
        area: plan.entities[dim]
        for dim, area in _ENTITY_TO_AREA.items()
        if plan.entities.get(dim)
    }


def _plan_filters(plan: Plan) -> dict:
    """Derive the validated filter dict: the plan's own filters, plus the
    interventional-only toggle when set (CC-5/E-38)."""
    filters = dict(plan.filters or {})
    if plan.interventional_only:
        filters["interventional_only"] = True
    return filters


# --- canned Phase-0 execute payload -----------------------------------------

# A fixed distribution-by-phase bucket set (ARCHITECTURE_SPEC §3.6 stands in
# for the real AggregationCore until Phase 1). Shaped exactly like the real
# tool's output (CC-3 dual counts, CC-9 inline citations) so the viz-builder
# and Output Reviewer exercise the real downstream contract even though the
# numbers are fixed.
_CANNED_PHASE_BUCKETS: list[dict] = [
    {
        "value": "PHASE1",
        "label": "Phase 1",
        "count_trials": 32,
        "count_mentions": 34,
        "source_ids": ["NCT00000001"],
        "citations": [
            {
                "nct_id": "NCT00000001",
                "field_path": "protocolSection.designModule.phases",
                "value": ["PHASE1"],
                "excerpt": "PHASE1",
            }
        ],
        "contributing_count": 32,
    },
    {
        "value": "PHASE2",
        "label": "Phase 2",
        "count_trials": 54,
        "count_mentions": 58,
        "source_ids": ["NCT00000002"],
        "citations": [
            {
                "nct_id": "NCT00000002",
                "field_path": "protocolSection.designModule.phases",
                "value": ["PHASE2"],
                "excerpt": "PHASE2",
            }
        ],
        "contributing_count": 54,
    },
    {
        "value": "NA",
        "label": "NA (not applicable)",
        "count_trials": 40,
        "count_mentions": 40,
        "source_ids": ["NCT00000003"],
        "citations": [
            {
                "nct_id": "NCT00000003",
                "field_path": "protocolSection.designModule.phases",
                "value": ["NA"],
                "excerpt": "NA",
            }
        ],
        "contributing_count": 40,
    },
]


# --- nodes -------------------------------------------------------------------


def merge_inputs(state: GraphState) -> dict:
    """Raw structured-field normalization only (§B.5) -- the CC-1 dimension
    precedence (which dimension the free-text query names) is the Planner's
    job, since deciding that requires the NL parse.

    Updates (never replaces) any pre-existing ``merged_inputs`` so that
    ``build.initial_state``'s test-only ``overrides`` (e.g. ``_force_error``)
    survive this node rather than being clobbered.

    Unset optional request fields arrive here as explicit ``None`` values
    (``VisualizeRequest.model_dump()`` always includes every optional field);
    those are dropped rather than copied through, so ``merged_inputs`` only
    carries fields the caller actually specified -- otherwise a present-but-
    ``None`` key would shadow a downstream ``dict.get(key, default)`` fallback
    (``dict.get`` only applies its default when the key is *absent*).
    """
    merged = dict(state.get("merged_inputs") or {})
    merged.update({k: v for k, v in (state.get("raw_fields") or {}).items() if v is not None})
    merged["query"] = state["question"]
    return {
        "merged_inputs": merged,
        "escalation_count": 0,
        "iter_count": 0,
        "scratch": {},
        # Phase-5 guard state (reset per request; ``deadline_at`` is seeded by the
        # entry point in ``initial_state``, so it is NOT reset here — that would
        # clobber the wall-clock budget the first node inherits).
        "tool_call_count": 0,
        "seen_signatures": [],
        "events": ["merge_inputs"],
    }


def plan(state: GraphState) -> dict:
    """The ReAct planner (§3.2): ``plan_request`` classifies the NL query into one of the six
    query classes and fills that recipe's slots, emitting a typed Plan through the adapter (C-99).
    The reason→act→observe loop is the bounded escalation back-edge — a re-entry threads the last
    rejection's ``plan_feedback`` in so the re-plan is a real correction.

    Also owns the shared escalation-budget increment: a re-entry into this node
    (``iter_count > 0``, i.e. a back-edge from check/review_intent/execute) bumps
    ``escalation_count`` once.
    """
    # --- Phase-5 runtime-harness guards, BEFORE an LLM call is spent (§4/§B.4) ---
    # Wall-clock deadline / iteration cap / node-visit backstop. A trip
    # short-circuits to a REDACTED error (``route_after_plan`` -> ``error``),
    # never a hang. Under v1's single-shot planner the iteration/step caps are
    # headroom (this node runs <=2x); ``tests/test_guards.py`` proves they fire.
    tripped = guards.check_pre_plan_guards(state)
    if tripped is not None:
        logger.warning("plan: runtime guard tripped (%s)", tripped)
        return {"error": guards.guard_error(tripped), "status": "error", "events": ["plan"]}

    # ``_force_plan`` (initial_state overrides only, never the wire) injects a
    # hardcoded Plan so the deterministic engine for every query_class can be driven
    # end-to-end through the FULL graph before the Phase-4 LLM planner exists — the
    # Phase-2 breadth gate. Same test-injection pattern as _force_error/_force_canned.
    forced_plan = (state.get("merged_inputs") or {}).get("_force_plan")
    if forced_plan is not None:
        result_plan = forced_plan
    else:
        # The reason -> act -> observe loop (§3.12): on a back-edge re-entry the last
        # rejection's machine reason (``plan_feedback``) is threaded into the planner so the
        # bounded re-plan is a real correction, not a blind retry.
        result_plan = plan_request(
            _adapter(), state["merged_inputs"], feedback=state.get("plan_feedback")
        )

    # --- stall detector (SET-based signature, G-41g/ENG-57) ------------------
    # A repeated plan signature aborts as a no-progress stall / A->B->A oscillation
    # — but ONLY BEYOND the sanctioned single escalation (``iter_count >= 2``, i.e.
    # a 3rd+ plan entry). Under v1 the escalation budget (<=1) already bounds the
    # loop, and the ONE legal re-plan legitimately reproduces the same plan when the
    # planner can't improve it — that path must still proceed to settle a clean
    # ``empty`` or ship best-effort (§B.5), so aborting there would be wrong. Hence
    # this is a forward-compat backstop for a future multi-iteration planner
    # (unreachable under v1's <=1 escalation); ``tests/test_guards.py`` proves it
    # fires when the pathological state is injected. The signature is always
    # recorded so the SET is complete the moment a real multi-iteration loop exists.
    seen = list(state.get("seen_signatures") or [])
    if state.get("iter_count", 0) >= 2 and guards.is_stalled(result_plan, seen):
        logger.warning("plan: stalled — re-plan reproduced an already-seen plan signature")
        return {
            "error": guards.guard_error(guards.STALLED_NO_PROGRESS),
            "status": "error",
            "events": ["plan"],
        }
    seen.append(guards.plan_signature(result_plan))

    iter_count = state.get("iter_count", 0)
    escalation_count = state.get("escalation_count", 0)
    if iter_count > 0:
        escalation_count += 1
    update: dict = {
        "plan": result_plan,
        "iter_count": iter_count + 1,
        "escalation_count": escalation_count,
        "seen_signatures": seen,
        "plan_feedback": None,  # consumed — clear it so it can't leak into a later iteration
        "events": ["plan"],
    }
    # --- dangling-reference → clarification (E-13/P5-INPUT) -------------------
    # If the NL query made a demonstrative reference ("this drug") to a dimension it
    # never resolved, ask instead of guessing: short-circuit to build_spec (via
    # route_after_plan), skipping check/execute — there is nothing to compute yet.
    clarification_q = detect_dangling_reference(state.get("merged_inputs"), result_plan)
    if clarification_q:
        update["clarification"] = clarification_q
        update["status"] = "empty"
    return update


def check(state: GraphState) -> dict:
    """The Plan Checker (§3.3) -- mechanical validation, code only.

    Supports a ``_force_reject`` test-injection sentinel (initial_state overrides
    only, never the wire) read off ``merged_inputs``, mirroring the ``execute``
    sentinel pattern, so the checker-reject escalation edge (``route_after_check``)
    stays exercisable offline+deterministically without a real Plan that
    mechanically fails. On a reject the machine reason is written to
    ``plan_feedback`` for the bounded re-plan.
    """
    if (state.get("merged_inputs") or {}).get("_force_reject"):
        validation = CheckResult(ok=False, reason="injected_reject")
    else:
        validation = check_plan(state["plan"])
    update: dict = {"validation": validation, "events": ["check"]}
    if not validation.ok:
        # Feed the precise machine reason back into the bounded re-plan (§3.12).
        update["plan_feedback"] = f"plan rejected by the checker: {validation.reason}"
    return update


def review_intent(state: GraphState) -> dict:
    """The Intent Reviewer (§3.4) -- semantic judgment on a mechanically-valid
    Plan. Stashes the verdict in ``scratch`` (last-write-wins) so
    ``route_after_intent`` can read a single current decision, in addition to
    the append-only ``verifications`` audit trail.

    Supports a ``_force_revise`` test-injection sentinel (initial_state overrides
    only, never the wire) read off ``merged_inputs``, mirroring the ``execute``
    sentinel pattern, so the intent-revise escalation edge (``route_after_intent``)
    stays exercisable offline+deterministically without a real reviewer that
    semantically disagrees. Skippable on an all-structured plan
    (``should_skip_intent_review``); on ``revise`` the reason is written to
    ``plan_feedback`` for the bounded re-plan.
    """
    merged_inputs = state.get("merged_inputs") or {}
    plan = state["plan"]
    if merged_inputs.get("_force_revise"):
        verdict = IntentVerdict(decision="revise", reason="injected_revise")
    elif should_skip_intent_review(merged_inputs, plan):
        # Skippable (§3.4): an all-structured-field plan had no NL parse to misread, so
        # there is nothing to catch -- approve without spending an LLM call.
        verdict = IntentVerdict(decision="approve", reason="skipped: all-structured plan")
    else:
        verdict = review_intent_llm(_adapter(), state["question"], plan)
    scratch = dict(state.get("scratch") or {})
    scratch["intent_verdict"] = verdict.model_dump()
    update: dict = {
        "verifications": [verdict.model_dump()],
        "scratch": scratch,
        "events": ["review_intent"],
    }
    if verdict.decision == "revise":
        target = f" (field: {verdict.field})" if verdict.field else ""
        update["plan_feedback"] = (
            f"intent review asked to revise{target}: {verdict.reason}"
        )
    return update


def execute(state: GraphState) -> dict:
    """The REAL executor (§3.6/§3.12) -- count-then-aggregate against the live
    ClinicalTrials.gov v2 API via the deterministic tool layer.

    The sentinels read off ``merged_inputs`` short-circuit BEFORE any network
    call, so the graph's offline routing tests stay deterministic:
    ``_force_error`` (a hard upstream failure), ``_force_empty`` (a
    zero-results aggregation), ``_force_too_large`` (an over-budget match set),
    and ``_force_canned`` (the fixed ``_CANNED_PHASE_BUCKETS`` structural path
    the doctor / structural tests exercise offline). None are real request
    fields (``VisualizeRequest`` forbids unknown fields) -- they only ever
    arrive via ``build.initial_state``'s test-only ``overrides``.

    Default = real dispatch: derive ``query``/``filters`` from the validated
    Plan, take the exact ``countTotal`` oracle, refuse over-budget (§B.7),
    settle zero-results as ``empty``, else page + aggregate. Any exception is
    caught and returned as a REDACTED error (the real cause -- which can embed
    the upstream URL/params -- is logged server-side only, never on the wire;
    LESSON B4). ``count_total`` + ``bucket_mode`` are written into state for the
    Output Reviewer's reconciliation pre-check.
    """
    merged_inputs = state.get("merged_inputs") or {}
    plan: Plan = state["plan"]
    retrieved_at = dt.datetime.now(dt.UTC).isoformat()
    condition = merged_inputs.get("condition", "pancreatic cancer")
    canned_provenance = {
        "endpoint": "/api/v2/studies",
        "params": {
            "query.cond": condition,
            "countTotal": True,
            "fields": "NCTId|Phase",
        },
    }

    if merged_inputs.get("_force_error"):
        return {
            "error": {"code": "upstream_error", "message": "injected for test"},
            "status": "error",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }

    if merged_inputs.get("_force_empty"):
        return {
            "tool_results": [{"tool": "aggregate_by", "buckets": []}],
            "status": "empty",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            # Mirror the real empty path so the offline budget tests exercise empty-trigger
            # feedback threading (test fidelity — the live _status_result sets the same).
            "plan_feedback": (
                "the previous plan matched zero trials; broaden the entities/filters or "
                "reconsider the query class."
            ),
            "events": ["execute"],
        }

    if merged_inputs.get("_force_too_large"):
        return {
            "tool_results": [{"tool": "count_trials", "total_count": 142_411}],
            "status": "too_large",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }

    if merged_inputs.get("_force_canned"):
        # Offline structural path: the fixed bucket set the doctor / structural
        # tests reconcile against (Σ = 32+54+40 = 126, combine mode).
        return {
            "tool_results": [
                {"tool": "aggregate_by", "buckets": copy.deepcopy(_CANNED_PHASE_BUCKETS)}
            ],
            "status": "ok",
            "count_total": 126,
            "bucket_mode": "combine",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }

    # --- Phase-5 guards immediately before the real (networked) dispatch -----
    # The wall-clock deadline and the tool-call budget are checked here, after the
    # offline sentinels (which never touch the network). NOTE on scope: the deadline
    # is checked at this node's ENTRY, not inside the paging loop — so a single slow
    # ``execute`` is bounded by the PAGE BUDGET (config.PAGE_BUDGET_PAGES pages) plus
    # the per-call timeout (config.PER_CALL_TIMEOUT_SECONDS), which together make it
    # finite; the wall-clock deadline additionally catches a request that is slow
    # ACROSS nodes. Threading the deadline into ``iter_studies`` per-page is a
    # documented future tightening, not needed for a finite bound. A trip → a
    # redacted error routed to the error node. ``deadline_at`` is None on the
    # structural offline path, so this is a no-op there.
    if guards.over_deadline(state.get("deadline_at")):
        logger.warning("execute: wall-clock deadline exceeded before dispatch")
        return {
            "error": guards.guard_error(guards.DEADLINE_EXCEEDED),
            "status": "error",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }
    tool_call_count = state.get("tool_call_count", 0) + 1
    over_budget = guards.check_tool_budget(tool_call_count)
    if over_budget is not None:
        logger.warning("execute: tool-call budget exceeded (%s)", over_budget)
        return {
            "error": guards.guard_error(over_budget),
            "status": "error",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }

    # --- response cache (§3.10 · keyed on the normalized plan, non-authoritative) ---
    # A HIT replays the prior fully-computed envelope and short-circuits the rest of
    # the pipeline (build_spec + review_output) — ``route_after_execute`` sees a
    # populated ``spec`` and routes straight to ``respond``. Bypassed entirely when
    # a test sentinel is active (deterministic offline suite) or the cache is off.
    # The cache never overrides a live count; it only hands back a code-computed
    # envelope a prior miss produced for this exact plan.
    if config.CACHE_ENABLED and not _has_force_sentinel(merged_inputs):
        cached = RESPONSE_CACHE.get(plan_cache_key(plan))
        if cached is not None:
            log_event(logger, "execute", query_class=plan.query_class, cache="hit")
            return {
                "spec": cached,
                "status": cached.status,
                "cache_hit": True,
                "retrieved_at": retrieved_at,
                "events": ["execute"],
            }

    # --- real dispatch (by query_class) --------------------------------------
    try:
        update = dict(_dispatch_execute(plan, retrieved_at))
    except Exception:  # noqa: BLE001 -- redact upstream detail, log server-side (LESSON B4)
        logger.exception("execute: real dispatch failed")
        return {
            "error": {
                "code": "upstream_error",
                "message": "failed to retrieve or aggregate trial data",
            },
            "status": "error",
            "retrieved_at": retrieved_at,
            "query_provenance": canned_provenance,
            "events": ["execute"],
        }
    update["tool_call_count"] = tool_call_count  # accumulate the fan-out across re-executes
    # Structured event — DECIDED shape only (query_class + the computed total), NEVER
    # the raw query or a free-text arg value (§A(i)/SEC-47). count_total is a computed number.
    log_event(
        logger, "execute", query_class=plan.query_class, cache="miss",
        status=update.get("status"), count_total=update.get("count_total"),
    )
    return update


# --- per-class execute dispatch (§3.6 runner; one core, five classes) --------


def _provenance(search_params: dict, projection: str) -> dict:
    """Reproducibility stamp: endpoint + the effective, validated wire params."""
    return {
        "endpoint": "/api/v2/studies",
        "params": {**search_params, "countTotal": "true", "pageSize": 1000, "fields": projection},
    }


def _status_result(status: str, tool_results: list, retrieved_at: str, provenance: dict) -> dict:
    """A terminal non-ok execute update (too_large / empty) — shared shape.

    ``empty`` (zero results) carries a ``plan_feedback`` hint so the bounded re-plan
    (``route_after_execute`` empty ∧ esc<1 → plan) is a real reason -> act -> observe step:
    the planner sees "the last plan matched zero trials" and can broaden or reclassify."""
    update: dict = {
        "tool_results": tool_results,
        "status": status,
        "retrieved_at": retrieved_at,
        "query_provenance": provenance,
        "events": ["execute"],
    }
    if status == "empty":
        update["plan_feedback"] = (
            "the previous plan matched zero trials; broaden the entities/filters or "
            "reconsider the query class."
        )
    return update


def _series_query(entities: dict) -> dict:
    """The ``{area: value}`` selectors for one compare arm's entities."""
    return {area: entities[dim] for dim, area in _ENTITY_TO_AREA.items() if entities.get(dim)}


def _dispatch_execute(plan: Plan, retrieved_at: str) -> dict:
    """Route the validated plan to its class runner — the single dispatch point
    that makes breadth 'one core, five classes' (CC-11) literal."""
    query = _plan_query(plan)
    filters = _plan_filters(plan)
    if plan.query_class == "single_value":
        return _execute_single_value(plan, query, filters, retrieved_at)
    if plan.query_class == "compare":
        return _execute_compare(plan, retrieved_at)
    if plan.query_class == "network":
        return _execute_network(plan, query, filters, retrieved_at)
    return _execute_single(plan, query, filters, retrieved_at)


def _execute_single(plan: Plan, query: dict, filters: dict, retrieved_at: str) -> dict:
    """distribution / geographic / timeseries / histogram — one population, one
    ``countTotal`` oracle + budget gate, then the class's tool. The reconciliation
    anchor (distinct-nctId == countTotal) holds for all four (combine or explode)."""
    search_params = build_search_params(query, filters)
    if plan.query_class == "timeseries":
        projection = f"NCTId|{DATE_PROJECTION.get(plan.date_field or '', plan.date_field)}"
    elif plan.query_class == "geographic":
        projection = "NCTId|LocationCountry"
    elif plan.field == "study_duration":
        projection = "NCTId|StartDate|CompletionDate"
    else:
        projection = "NCTId|Phase"
    provenance = _provenance(search_params, projection)

    total = count_trials(query, filters)  # exact oracle + budget gate
    if total > _TOO_LARGE_THRESHOLD:
        # Over budget, but a bounded-token CATEGORICAL distribution (status / sponsorClass /
        # interventionType) can be computed EXACTLY via one count query per token — no paging, no
        # biased prefix — so it charts instead of refusing (scales to any size). phase (composites)
        # and country (unbounded) are NOT count-aggregatable → they still refuse (§B.7).
        if plan.query_class == "distribution" and is_count_aggregatable(plan.field):
            result = aggregate_by_counts(query, filters, plan.field)
            return {
                "tool_results": [result],
                "status": "ok",
                "count_total": total,
                "bucket_mode": result["mode"],
                "retrieved_at": retrieved_at,
                "query_provenance": provenance,
                "fetched_records": result.get("record_index"),
                "events": ["execute"],
            }
        return _status_result(
            "too_large", [{"tool": "count_trials", "total_count": total}], retrieved_at, provenance
        )
    if total == 0:
        return _status_result(
            "empty", [{"tool": "aggregate_by", "buckets": []}], retrieved_at, provenance
        )

    if plan.query_class == "timeseries":
        result = timeseries(query, filters, plan.date_field, plan.grain or "year")
    elif plan.query_class == "geographic":
        result = aggregate_by(query, filters, "country")
    elif plan.field == "study_duration":
        result = study_duration_histogram(query, filters)
    else:
        result = aggregate_by(query, filters, plan.field)

    update: dict = {
        "tool_results": [result],
        "status": "ok",
        "count_total": total,
        "bucket_mode": result["mode"],
        "retrieved_at": retrieved_at,
        "query_provenance": provenance,
        "fetched_records": result.get("record_index"),
        "events": ["execute"],
    }
    # Genuine truncation (distinct < total) discloses a partial AND fails
    # reconciliation downstream; a trailing token on an exactly-full final page does
    # not (every distinct trial was still fetched — K3 boundary).
    if result.get("truncated") and result.get("distinct_trials", 0) < total:
        update["partial"] = {"truncated": True, "of_total": total}
    return update


def _execute_compare(plan: Plan, retrieved_at: str) -> dict:
    """compare — ≥2 independently-filtered arms (G-24). Each arm is budget-gated and
    self-reconciled by its own ``aggregate_by``; the union spans two populations so
    ``count_total`` is None and the Output Reviewer reconciliation is waived
    (``reconcile=False``), while the excerpt/cited checks still run."""
    # Cap the number of arms (E-21): keep the first MAX_COMPARE_ENTITIES, disclose the
    # dropped ones in meta.notes — a grouped bar with too many series is unreadable, and
    # silent truncation would misstate the comparison.
    all_arms = list(plan.series or [])
    dropped_labels: list[str] = []
    if len(all_arms) > config.MAX_COMPARE_ENTITIES:
        dropped_labels = [a.label for a in all_arms[config.MAX_COMPARE_ENTITIES:]]
        all_arms = all_arms[: config.MAX_COMPARE_ENTITIES]

    series_list: list[dict] = []
    for arm in all_arms:
        arm_query = _series_query(arm.entities)
        arm_filters = {**(plan.filters or {}), **(arm.filters or {})}
        series_list.append({"label": arm.label, "query": arm_query, "filters": arm_filters})

    provenance = _provenance(
        build_search_params(series_list[0]["query"], series_list[0]["filters"])
        if series_list
        else {},
        f"NCTId|{plan.field}",
    )
    if not series_list:  # defense-in-depth: the checker requires >=2 arms, never assume it ran
        return _status_result("empty", [{"tool": "compare", "buckets": []}], retrieved_at, provenance)

    for arm in series_list:
        # Per-arm exact countTotal: the budget gate AND the honest % denominator the
        # compare tool uses (F3 — the arm total is threaded in, not discarded).
        n = count_trials(arm["query"], arm["filters"])
        if n > _TOO_LARGE_THRESHOLD:
            return _status_result(
                "too_large", [{"tool": "count_trials", "total_count": n}], retrieved_at, provenance
            )
        arm["count_total"] = n
    if all(arm["count_total"] == 0 for arm in series_list):
        return _status_result("empty", [{"tool": "compare", "buckets": []}], retrieved_at, provenance)

    result = compare(series_list, plan.field)
    if dropped_labels:
        note = (
            f"Showing the first {config.MAX_COMPARE_ENTITIES} of {len(plan.series)} requested "
            f"series; dropped for legibility: {', '.join(dropped_labels)} (E-21 cap)."
        )
        result = {**result, "notes": [*(result.get("notes") or []), note]}
    return {
        "tool_results": [result],
        "status": "ok",
        "count_total": None,  # multi-population — reconcile=False in review_output
        "bucket_mode": "compare",
        "retrieved_at": retrieved_at,
        "query_provenance": provenance,
        "fetched_records": result.get("record_index"),
        "events": ["execute"],
    }


def _execute_network(plan: Plan, query: dict, filters: dict, retrieved_at: str) -> dict:
    """network — one population, budget-gated, then the graph builder. A network is
    reconciliation-exempt (its ``data`` is a NetworkData, not a row list, so the
    Output Reviewer's list-only reconciliation never fires); ``count_total`` is
    stamped for provenance only."""
    search_params = build_search_params(query, filters)
    provenance = _provenance(search_params, NETWORK_FIELDS)
    total = count_trials(query, filters)
    if total > _TOO_LARGE_THRESHOLD:
        return _status_result(
            "too_large", [{"tool": "count_trials", "total_count": total}], retrieved_at, provenance
        )
    if total == 0:
        return _status_result(
            "empty",
            [{"tool": "build_network",
              "graph": {"nodes": [], "edges": [], "notes": ["No trials matched this query."]}}],
            retrieved_at,
            provenance,
        )

    result = build_network(query, plan.network.kind, filters)
    graph = result.get("graph") or {}

    # Degeneracy fallback (G-41e): a ≤1-node OR edges==0 graph is not a graph. Fall
    # back to the cited BAR of individual drug frequencies the network layer computed
    # (graph["fallback"]) — the "knows when NOT to graph" path. If even that is empty
    # (a population with no DRUG interventions at all), settle as a normal empty.
    if graph.get("degenerate"):
        fallback = graph.get("fallback") or {}
        fb_buckets = fallback.get("buckets") or []
        if not fb_buckets:
            return _status_result(
                "empty",
                [{"tool": "build_network",
                  "graph": {"nodes": [], "edges": [],
                            "notes": [f"{total:,} trial(s) matched but none study a drug "
                                      "intervention, so there is nothing to graph."]}}],
                retrieved_at,
                provenance,
            )
        # Notes = ONLY the fallback's own bar-appropriate disclosures (drug-node
        # formation + top-N truncation), NEVER the graph's edge/cap/prune notes which
        # describe a graph this bar does not render (L3-1). Plus the count-basis gap:
        # some matched trials have no drug node and are excluded from the bar total.
        drug_trials = int(fallback.get("distinct_trials", 0))
        gap_notes: list[str] = []
        if total and drug_trials < total:
            gap_notes.append(
                f"{drug_trials:,} of {total:,} matched trials study ≥1 drug intervention; "
                f"the remaining {total - drug_trials:,} have none and are not shown here."
            )
        fallback_result = {
            "tool": "build_network_fallback",
            "mode": "explode",
            "network_fallback": True,  # marker: build_spec renders a BAR, review waives reconcile
            "distinct_trials": drug_trials,
            "truncated": bool(result.get("truncated")),
            "buckets": fb_buckets,
            "notes": [*(fallback.get("notes") or []), *gap_notes, fallback.get("note", "")],
        }
        return {
            "tool_results": [fallback_result],
            "status": "ok",
            "count_total": None,  # a derived drug-frequency bar, not countTotal-reconciled
            "bucket_mode": "network_fallback",
            "retrieved_at": retrieved_at,
            "query_provenance": provenance,
            "fetched_records": result.get("record_index"),
            "events": ["execute"],
        }

    return {
        "tool_results": [result],
        "status": "ok",
        "count_total": total,
        "bucket_mode": "network",
        "retrieved_at": retrieved_at,
        "query_provenance": provenance,
        "fetched_records": result.get("record_index"),
        "events": ["execute"],
    }


_NCTID_PATH = "protocolSection.identificationModule.nctId"


def _execute_single_value(plan: Plan, query: dict, filters: dict, retrieved_at: str) -> dict:
    """single_value (CC-7) -- one exact ``count_trials`` over the plan's scope, rendered as a
    scalar stat card (``kind:"visualization"``) or a yes/no (``kind:"answer"``). The number is the
    API's exact ``countTotal`` (code-inserted, never LLM-authored); the citation is an honest
    "this trial is in the counted set" reference (excerpt = the nctId at the identification path,
    which round-trips via ``is_substring_at``). One extra cheap page (``fields=NCTId``) fetches a
    bounded citation sample only when there is something to cite."""
    search_params = build_search_params(query, filters)
    provenance = _provenance(search_params, "NCTId")
    total = count_trials(query, filters)  # exact oracle + budget gate
    if total > _TOO_LARGE_THRESHOLD:
        return _status_result(
            "too_large", [{"tool": "count_trials", "total_count": total}], retrieved_at, provenance
        )

    citations: list[Citation] = []
    record_index: dict[str, dict] = {}
    if total > 0:
        records, _ = CTGovClient().iter_studies(search_params, fields="NCTId", max_pages=1)
        for record in sorted(records, key=lambda r: _nct_id(r) or "")[:20]:
            nct = _nct_id(record)
            if not isinstance(nct, str) or not nct:
                continue
            record_index[nct] = record
            citations.append(
                Citation(nct_id=nct, field_path=_NCTID_PATH, value=nct, excerpt=nct)
            )

    kind = "answer" if plan.answer_kind == "answer" else "visualization"
    result = {
        "tool": "count_trials",
        "total_count": total,
        "kind": kind,
        "citations": [c.model_dump() for c in citations],
        "record_index": record_index,
    }
    return {
        "tool_results": [result],
        "status": "ok",
        "count_total": total,
        "bucket_mode": "single_value",
        "retrieved_at": retrieved_at,
        "query_provenance": provenance,
        "fetched_records": record_index,
        "events": ["execute"],
    }


def build_spec(state: GraphState) -> dict:
    """The viz-spec builder (§3.7) -- reads whatever ``status`` execute set
    (default ``"ok"`` for a from-scratch invocation).

    Also surfaces the planner's interpretation notes — the CC-1 dimension-override echo above all
    (field wins, echo the override — CC-1 / G-18 / §B.5) — onto ``meta.notes``. The echo can be the
    code-templated ``"Override: …"`` (fires when the LLM emitted a value the typed field then
    overrode) OR the LLM's own CC-1 note (when the model applied the field itself), so we thread
    ALL planner notes rather than guess a prefix — but each is run through the SAME §1 number
    post-check (``note_number_safe``) so a note can't smuggle a fabricated count, and the internal
    offline-stub note is excluded. meta.notes is untrusted display text by contract (§A(c))."""
    # A dangling-reference clarification (E-13) short-circuits here: emit the code-owned
    # question envelope, no viz/data (the plan node set state.clarification and routed
    # straight to build_spec, skipping check/execute).
    clarification_q = state.get("clarification")
    if clarification_q:
        spec = build_clarification_envelope(
            question=clarification_q,
            plan=state.get("plan"),
            retrieved_at=state.get("retrieved_at"),
            query_provenance=state.get("query_provenance"),
        )
        return {"spec": spec, "status": "empty", "events": ["build_spec"]}

    plan = state["plan"]
    spec = build_envelope(
        plan=plan,
        tool_results=state.get("tool_results", []),
        status=state.get("status", "ok"),
        question=state["question"],
        retrieved_at=state.get("retrieved_at"),
        query_provenance=state.get("query_provenance"),
        partial=state.get("partial"),
    )
    if spec.meta is not None:
        allowed = computed_numbers(spec)
        threaded = [
            str(n)
            for n in (getattr(plan, "notes", None) or [])
            if str(n).strip()
            and not str(n).startswith("Offline stub plan")
            and note_number_safe(str(n), allowed)
        ]
        if threaded:
            updated_meta = spec.meta.model_copy(update={"notes": [*spec.meta.notes, *threaded]})
            spec = spec.model_copy(update={"meta": updated_meta})
    return {"spec": spec, "events": ["build_spec"]}


# Generic wire message for a deterministic-precheck hard fail (the machine code
# carries the specifics; the message never leaks internals, API-22).
_PRECHECK_FAIL_MESSAGE = "output failed deterministic provenance/reconciliation checks"


def _latest_aggregate(tool_results: list | None) -> dict | None:
    """The most recent tool-result dict carrying a ``distinct_trials`` anchor
    (``aggregate_by`` / ``timeseries`` / ``study_duration_histogram`` / network),
    or ``None`` — the reconciliation anchor the Output Reviewer reads."""
    for result in reversed(tool_results or []):
        if isinstance(result, dict) and "distinct_trials" in result:
            return result
    return None


def review_output(state: GraphState) -> dict:
    """The Output Reviewer (§3.8). Runs the deterministic pre-checks FIRST
    (excerpt-substring, ``countTotal`` reconciliation, partial-iff-truncated,
    cited-or-derived), then the LLM half.

    On a deterministic hard fail the spec is replaced with a REDACTED error
    envelope and ``status`` flipped to ``"error"`` (``route_after_output``
    always -> ``respond``, which emits it). Otherwise a within-tolerance
    reconciliation drift is disclosed on ``meta.notes``, and the LLM half runs
    (through the adapter, C-99) exactly as before -- a ``flag`` verdict appends
    a caveat without ever rebuilding the already-computed spec."""
    spec = state["spec"]

    # A clarification (E-13) carries no computed data — nothing to reconcile or
    # fact-check, and no LLM call to spend; pass it straight through.
    if spec is not None and spec.kind == "clarification":
        return {"events": ["review_output"]}

    aggregate = _latest_aggregate(state.get("tool_results"))
    count_total = state.get("count_total")
    mode = state.get("bucket_mode")
    distinct_trials = aggregate.get("distinct_trials") if aggregate else None
    if distinct_trials is None:
        distinct_trials = count_total
    truncated = bool(state.get("partial"))

    # compare spans TWO populations (two countTotals) -- no single oracle to
    # reconcile the union against, so the single-oracle count check is waived
    # (each arm self-reconciled in-tool); the excerpt/cited checks still run. The
    # network degeneracy fallback (a derived drug-frequency bar of the DRUG-bearing
    # subset, not the whole countTotal population) is likewise count-reconciliation
    # exempt -- its excerpt/cited checks still run (teeth kept).
    plan = state.get("plan")
    reconcile = not (
        (plan is not None and plan.query_class == "compare")
        or mode == "network_fallback"
    )

    pc = deterministic_precheck(
        spec,
        count_total=count_total,
        mode=mode,
        distinct_trials=distinct_trials,
        truncated=truncated,
        reconcile=reconcile,
    )
    # Phase-4 hardening (§3.8): re-verify each excerpt against the ACTUAL fetched record (an
    # independent ground truth), now that the LLM is in the loop -- catches a fabricated citation
    # even when excerpt == value, and gives is_substring_at a runtime caller (LESSON M3).
    rg = record_grounded_reverify(spec, state.get("fetched_records"))
    failed = pc if pc.hard_fail else (rg if rg.hard_fail else None)

    if failed is not None:
        error_spec = build_envelope(
            plan=state.get("plan"),
            tool_results=state.get("tool_results", []),
            status="error",
            question=state["question"],
            retrieved_at=state.get("retrieved_at"),
            query_provenance=state.get("query_provenance"),
            error={"code": failed.reason or "internal", "message": _PRECHECK_FAIL_MESSAGE},
        )
        return {
            "verifications": [{"decision": "reject", "reason": failed.reason}],
            "spec": error_spec,
            "status": "error",
            "events": ["review_output"],
        }

    if pc.disclosure:
        updated_meta = spec.meta.model_copy(update={"notes": [*spec.meta.notes, pc.disclosure]})
        spec = spec.model_copy(update={"meta": updated_meta})

    # Best-effort caveat (P4-ROUTING / §B.5): if the Intent Reviewer's last verdict was still
    # ``revise`` yet we reached execute, the escalation budget was exhausted and we shipped the
    # legal plan best-effort -- disclose that honestly on meta.notes (the Checker already proved
    # the plan is legal; only aptness was unconfirmed).
    intent_verdict = (state.get("scratch") or {}).get("intent_verdict") or {}
    if intent_verdict.get("decision") == "revise":
        caveat = (
            "Intent review could not fully confirm the plan matched the question; shipped "
            "best-effort after the re-plan budget was exhausted."
        )
        updated_meta = spec.meta.model_copy(update={"notes": [*spec.meta.notes, caveat]})
        spec = spec.model_copy(update={"meta": updated_meta})

    verdict = review_output_llm(_adapter(), state["question"], spec)
    if verdict.decision == "flag":
        # §1: an LLM-authored meta.notes entry (the flag reason) must pass a deterministic
        # digit post-check — a fabricated count (a digit-run absent from the computed data) is
        # withheld for a fixed code-owned caveat, so the model cannot smuggle a number onto the
        # wire via its flag prose. A digit-free reason ships as-is.
        raw = (verdict.reason or "").strip()
        if raw and note_number_safe(raw, computed_numbers(spec)):
            caveat = raw
        else:
            if raw:
                logger.info("output-reviewer flag reason withheld (non-data number in note)")
            caveat = (
                "Output Reviewer flagged this result for interpretation; the computed values "
                "and citations are unchanged."
            )
        updated_meta = spec.meta.model_copy(update={"notes": [*spec.meta.notes, caveat]})
        spec = spec.model_copy(update={"meta": updated_meta})
    return {
        "verifications": [verdict.model_dump()],
        "spec": spec,
        "events": ["review_output"],
    }


def respond(state: GraphState) -> dict:
    """Terminal passthrough (§3.12) -- the final spec is already in state;
    this only pins ``status`` if nothing upstream set one, and populates the
    response cache with a freshly-computed public result.

    Cache store policy (§3.10 · SEC-48): store ONLY a real, freshly-built envelope
    (not a cache-hit replay), only for a public computed status (ok / empty /
    too_large — never an error, and never a sentinel-driven offline result), keyed
    on the normalized plan. A cache-store failure never breaks the response."""
    status = state.get("status", "ok")
    spec = state.get("spec")
    plan = state.get("plan")
    if (
        config.CACHE_ENABLED
        and not state.get("cache_hit")
        and spec is not None
        and spec.kind != "clarification"  # a clarification is cheap + not a computed result
        and plan is not None
        and status in ("ok", "empty", "too_large")
        and not _has_force_sentinel(state.get("merged_inputs"))
    ):
        try:
            RESPONSE_CACHE.set(plan_cache_key(plan), spec)
        except Exception:  # noqa: BLE001 -- a cache write must never break the response
            logger.exception("respond: response-cache store failed (ignored)")
    return {"status": status, "events": ["respond"]}


def error(state: GraphState) -> dict:
    """Terminal error envelope (§3.12) -- builds the error ``VisualizeResponse``
    through the same viz-builder every other status uses, then funnels into
    the single terminal ``respond`` node (wired in ``app.graph.build``)."""
    err = state.get("error") or {"code": "internal", "message": "unspecified"}
    spec = build_envelope(
        plan=state.get("plan"),
        tool_results=state.get("tool_results", []),
        status="error",
        question=state["question"],
        retrieved_at=state.get("retrieved_at"),
        query_provenance=state.get("query_provenance"),
        error=err,
    )
    return {"spec": spec, "status": "error", "events": ["error"]}


# --- conditional-edge routers ------------------------------------------------


def route_after_merge(state: GraphState) -> str:
    """Defensive only: ``VisualizeRequest`` already guarantees a non-empty
    ``query`` before the graph ever runs, so ``merge_inputs`` cannot actually
    produce an invalid ``merged_inputs`` in this build. Kept as a real check
    (not a stub) so the edge is honest about what it guards."""
    if not (state.get("merged_inputs") or {}).get("query"):
        return "error"
    return "plan"


def route_after_plan(state: GraphState) -> str:
    """plan -> check normally; a tripped runtime-harness guard (deadline /
    iteration cap / node-visit backstop / stall) short-circuits to the ``error``
    node instead (the ``plan`` node set ``status:"error"`` when it tripped).

    On a normal or escalation re-entry ``status`` is ``"ok"``/``"empty"`` (never
    ``"error"`` — a hard execute error routes straight to ``error``, and a
    checker/intent rejection routes to ``plan`` without touching ``status``), so
    this returns ``"check"`` unless THIS ``plan`` invocation tripped a guard."""
    if state.get("status") == "error":
        return "error"
    if state.get("clarification"):
        return "build_spec"  # a dangling-reference clarification skips check/execute (E-13)
    return "check"


def route_after_check(state: GraphState) -> str:
    """ok -> review_intent; reject -> one bounded re-plan (esc < 1), else
    -> error (escalation budget exhausted)."""
    validation = state["validation"]
    if validation.ok:
        return "review_intent"
    if state.get("escalation_count", 0) < 1:
        return "plan"
    return "error"


def route_after_intent(state: GraphState) -> str:
    """approve -> execute; revise ∧ esc<1 -> one bounded re-plan; revise ∧ esc≥1 ->
    **execute (best-effort)**, NOT error (P4-ROUTING, reconciled to ARCHITECTURE_SPEC §B.5).

    The Intent Reviewer is advisory — the Plan Checker has already proven the plan is
    mechanically LEGAL — so an exhausted re-plan budget ships the legal plan best-effort with a
    disclosed ``meta.notes`` caveat (added in ``review_output``), rather than refusing to answer
    because an advisory reviewer disagreed twice. (checker-reject with an exhausted budget still
    routes to ``error`` — an *illegal* plan cannot ship.)"""
    verdict = (state.get("scratch") or {}).get("intent_verdict") or {}
    if verdict.get("decision", "approve") == "approve":
        return "execute"
    if state.get("escalation_count", 0) < 1:
        return "plan"
    return "execute"


def route_after_execute(state: GraphState) -> str:
    """cache hit (execute populated ``spec``) -> respond (skips build_spec +
    review_output — the envelope is already final); done -> build_spec;
    too_large -> build_spec (over-budget is an always-refuse, not
    escalation-eligible, §B.7); empty (zero-results) -> one bounded re-plan
    (esc < 1), else settle into build_spec(empty); a hard error -> the dedicated
    error node (never build_spec)."""
    status = state.get("status", "ok")
    if state.get("cache_hit"):
        return "respond"
    if status == "error":
        return "error"
    if status == "too_large":
        return "build_spec"
    if status == "empty":
        if state.get("escalation_count", 0) < 1:
            return "plan"
        return "build_spec"
    return "build_spec"


def route_after_output(state: GraphState) -> str:
    """review_output never loops (§3.12) -- always terminal."""
    return "respond"
