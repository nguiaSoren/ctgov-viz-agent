"""The LangGraph state schema (ARCHITECTURE_SPEC §3.12 / §B.5).

A single ``TypedDict`` shared by every node. Three fields are **append-only**
(``Annotated[list, operator.add]``): ``tool_results``, ``verifications``, and
``events``. Every node that contributes to one of these returns a small list
and LangGraph concatenates it onto the running total -- this is what lets
``events`` double as a derived execution trace (a node's name is appended the
moment it runs) without any node needing to know what ran before it. Every
other field is last-write-wins: a node returning a partial update simply
overwrites that key in the merged state.

Writer -> reader matrix, restated field-by-field. This is the matrix for the code as
built, verified against ``app.graph.nodes`` — where it differs from ARCHITECTURE_SPEC
§B.5's sketch (which predates the cache short-circuit, the clarification path and the
Phase-4 reviewers) the code is what is written here:

* ``question`` / ``raw_fields`` -- seeded once by ``build.initial_state`` from the
  validated ``VisualizeRequest``; no node rewrites either. ``raw_fields`` is read only by
  ``merge_inputs`` (which folds it into ``merged_inputs``); ``question`` is read by
  ``review_intent``, ``build_spec``, ``review_output`` and ``error``.
* ``merged_inputs`` -- written by ``merge_inputs`` (update-not-replace, so
  ``initial_state``'s test-only ``overrides`` survive); read by ``plan`` (the planner's
  input), ``check``/``review_intent``/``execute`` (their ``_force_*`` sentinels),
  ``review_intent`` again (``should_skip_intent_review``), ``respond`` (cache-store
  policy) and ``route_after_merge``.
* ``plan`` -- written by ``plan``; read by ``check``, ``review_intent``, ``execute``,
  ``build_spec``, ``review_output`` (the compare reconciliation waiver), ``respond``
  (the cache key) and ``error``.
* ``escalation_count`` -- zeroed by ``merge_inputs``, incremented by ``plan`` on a
  back-edge re-entry; read by ``route_after_check`` / ``route_after_intent`` /
  ``route_after_execute`` (the shared, <=1 escalation budget across all three re-plan
  triggers).
* ``validation`` -- written by ``check``; read by ``route_after_check`` only.
* ``tool_results`` (append-only) -- written by ``execute``; read by ``build_spec``,
  ``review_output`` (the reconciliation anchor) and ``error``.
* ``iter_count`` -- bumped by ``plan`` on every entry; read by ``plan`` itself (the
  escalation-increment condition and the stall gate) and by the iteration cap in
  ``guards.check_pre_plan_guards``. It counts plan ENTRIES; there is no loop inside the
  node (the planner makes one call — see ``nodes.plan``).
* ``scratch`` -- reset by ``merge_inputs``; ``review_intent`` stashes its verdict there
  (last-write-wins) for ``route_after_intent`` and ``review_output`` to read.
* ``partial`` -- written ONLY by ``execute`` (``nodes._execute_single``), and only on a
  genuine truncation (paged distinct < ``countTotal``); read by ``build_spec`` and by
  ``review_output``'s partial-iff-truncated check. No other node writes it.
* ``spec`` -- written by ``build_spec``, by ``error``, by ``execute`` on a response-cache
  hit (the replayed envelope — the mechanism the whole cache short-circuit rests on) and
  by ``review_output`` (twice: the redacted envelope on a deterministic hard fail, and the
  meta.notes-appended copy on the normal path); read by ``review_output``/``respond``.
* ``verifications`` (append-only) -- written by ``review_intent`` and
  ``review_output``; read by ``respond``.
* ``error`` -- written by ``plan`` (a tripped guard) and ``execute`` (a guard trip or a
  redacted upstream failure); read by the terminal ``error`` node, which turns it into
  the error envelope. ``review_output``'s hard fail builds its envelope directly and does
  not go through this field.
* ``status`` -- written by ``plan`` (a tripped guard, or a clarification),
  ``execute``, ``build_spec``, ``review_output`` (a deterministic hard fail),
  ``respond`` and ``error``. Effectively: any node that can decide the outcome.
* ``count_total`` / ``bucket_mode`` -- written by ``execute`` (the API's exact
  ``countTotal`` oracle + the aggregation ``combine``/``explode`` mode); read by
  ``review_output`` for the deterministic reconciliation pre-check.
* ``retrieved_at`` / ``query_provenance`` -- written by ``execute`` (only; ``retrieved_at``
  is stamped on every ``execute`` return, including the cache hit); read by ``build_spec``,
  ``review_output`` (the hard-fail envelope) and ``error``.
* ``fetched_records`` -- the bounded ``{nct_id: record}`` index ``execute`` stashes
  from the records it actually paged (Phase 4, built by ``tools._bounded_record_index``);
  read by ``review_output`` for the record-grounded citation re-verify (each citation's
  ``matched_value`` re-checked as a real substring at its ``field_path`` in the actual
  fetched record via ``is_substring_at`` -- giving the load-bearing primitive a RUNTIME
  caller, LESSON M3). ``None`` when the path didn't page (too_large / offline sentinels).
* ``plan_feedback`` -- the machine reason from the last rejected attempt, written by
  ``check`` (reject), ``review_intent`` (revise) and ``execute`` (zero results); read by
  ``plan`` on a re-entry (and cleared to ``None`` there once consumed) so the bounded
  re-plan is a real ``reason -> act -> observe`` step, not a blind retry.
* ``deadline_at`` -- the ABSOLUTE ``time.monotonic()`` wall-clock deadline
  (Phase 5, SEC-36): seeded by the entry point (60s sync / 90s SSE, §B.4). Read at
  exactly TWO points -- the top of ``plan`` (via ``guards.check_pre_plan_guards``) and
  the top of ``execute``, before the network -- so an over-deadline run short-circuits
  to ``error`` instead of hanging. It is NOT checked before the ``review_intent`` /
  ``review_output`` LLM calls, and no explicit client timeout is set on the provider
  SDKs, so a slow reviewer call is bounded only by the provider SDK's own default
  timeout. A ``review_intent`` overrun is caught at the next guarded node (``execute``);
  a ``review_output`` overrun is not caught at all -- nothing guarded runs after it.
  ``None`` when unset (structural offline tests).
* ``tool_call_count`` -- ``execute`` ENTRIES this request (Phase 5, ENG-27/§B.4),
  capped at ``MAX_TOOL_CALLS`` = 12. A forward-compat backstop: v1 enters ``execute`` at
  most twice (the zero-results re-plan is the only way back), so it cannot fire in normal
  operation, but a future multi-tool planner is bounded by construction. A response-cache
  hit is budget-CHECKED but records no spend (it makes no upstream call).
* ``seen_signatures`` -- the SET (as a list) of canonical plan signatures already produced
  (Phase 5, ENG-57/G-41g; the signature is ``cache.plan_cache_key``, so "the same plan"
  means one thing to both the cache and the stall detector). ``plan`` appends on every
  entry, but only CONSULTS it from the third entry onward (``iter_count >= 2``) — which
  v1's ``<=1`` escalation budget never reaches, so the stall abort is unreachable today
  and the list is pure forward-compat bookkeeping. The one sanctioned re-plan is allowed
  to repeat a signature: it must still be able to settle a clean ``empty``.
* ``clarification`` -- a code-templated disambiguation question, written by ``plan`` when
  the query makes a demonstrative reference it never resolved (Phase 5, E-13); read by
  ``route_after_plan`` (short-circuit to ``build_spec``, skipping check/execute) and by
  ``build_spec`` to emit a ``kind:"clarification"`` envelope. ``None`` in the normal path.
* ``cache_hit`` -- seeded ``False`` by ``build.initial_state`` and set ``True`` by
  ``execute`` on a response-cache hit; read by ``route_after_execute`` (straight to
  ``respond``) and ``respond`` (never re-store a replay).
* ``events`` (append-only) -- written by every node; a derived stream via
  node transitions, read by tests/observability (also the ``MAX_GRAPH_STEPS``
  node-visit backstop reads ``len(events)``).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.api.schemas import VisualizeResponse
from app.plan.models import CheckResult, Plan


class GraphState(TypedDict, total=False):
    question: str
    raw_fields: dict
    merged_inputs: dict
    plan: Plan | None
    escalation_count: int
    validation: CheckResult | None
    tool_results: Annotated[list, operator.add]
    iter_count: int
    scratch: dict
    partial: dict | None
    spec: VisualizeResponse | None
    verifications: Annotated[list, operator.add]
    error: dict | None
    status: str
    count_total: int | None
    bucket_mode: str | None
    retrieved_at: str | None
    query_provenance: dict | None
    fetched_records: dict[str, dict] | None  # {nct_id: record}, bounded (tools._bounded_record_index)
    plan_feedback: str | None
    deadline_at: float | None  # absolute time.monotonic() wall-clock deadline (Phase 5, SEC-36)
    tool_call_count: int  # total tool fan-out, capped at MAX_TOOL_CALLS (Phase 5, ENG-27)
    seen_signatures: list  # SET of plan signatures already produced — stall detector (Phase 5, ENG-57)
    clarification: str | None  # code-templated disambiguation question (Phase 5, E-13)
    cache_hit: bool  # True when `execute` served the envelope from the response cache (Phase 5, C-74)
    events: Annotated[list, operator.add]
