"""Human-readable per-stage pipeline trace (dev/observability affordance).

OFF by default (``config.PIPELINE_TRACE`` / env ``PIPELINE_TRACE`` unset). When
enabled, :func:`stage` prints each graph node's salient result to stdout as the
request runs, so a live ``POST /visualize`` shows the whole pipeline in the
server console:

    planning → validating → plan_approved → fetching → aggregating →
    building_spec → verifying → done

This is a DEBUG affordance, deliberately distinct from the structured audit log
(``app.logging_setup.log_event``): that log keeps the raw query and the provider
key out *by construction* (a closed field allowlist). This trace intentionally
shows the query, entities, and every intermediate decision, because an operator
opted in by setting the flag — but the one real secret, the provider key, is
still run through ``logging_setup._scrub`` so even this verbose path cannot leak
it. The trace never changes the graph's behaviour or its returned envelope; it
only observes the node updates the run already produces.
"""

from __future__ import annotations

import sys
from typing import Any

from app import config
from app.logging_setup import _scrub

# Graph node → the label to show for it. ``plan``..``respond`` are the eight
# published SSE stages (``app.main.SSE_STATUS_ENUM``); ``merge_inputs`` is the
# input-normalization pre-stage that precedes them and ``error`` is the terminal
# failure node. ``execute`` fuses fetching + aggregating (one deterministic runner).
_STAGE_LABEL: dict[str, str] = {
    "merge_inputs": "merge · normalize inputs",
    "plan": "planning",
    "check": "validating",
    "review_intent": "plan_approved · intent review",
    "execute": "fetching + aggregating",
    "build_spec": "building_spec",
    "review_output": "verifying · output review",
    "respond": "done",
    "error": "error",
}

_RULE = "━" * 74


def enabled() -> bool:
    """Is the pipeline trace turned on for this process?"""
    return config.PIPELINE_TRACE


def _p(line: str = "") -> None:
    """Print one scrubbed line to stdout, flushed so it appears mid-request."""
    print(_scrub(line), file=sys.stdout, flush=True)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a pydantic model (attribute) or a dict (item)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enum_value(value: Any) -> Any:
    """A ``(str, Enum)`` member (e.g. ``ChartType.BAR``) → its wire value ("bar")."""
    return getattr(value, "value", value)


# --- per-node formatters -----------------------------------------------------


def _fmt_merge(u: Any) -> None:
    merged = _get(u, "merged_inputs") or {}
    shown = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    _p(f"   merged_inputs : {shown}")


def _fmt_plan(u: Any) -> None:
    plan = _get(u, "plan")
    iter_count = _get(u, "iter_count")
    escalation = _get(u, "escalation_count")
    replan = "   ⟲ RE-PLAN (escalation back-edge fired)" if (iter_count or 0) > 1 else ""
    _p(f"   query_class   : {_get(plan, 'query_class')}{replan}")
    _p(f"   entities      : {_get(plan, 'entities')}")
    _p(f"   filters       : {_get(plan, 'filters')}")
    _p(
        f"   field         : {_get(plan, 'field')}"
        f"    chart: {_enum_value(_get(plan, 'chart_type'))}"
        f"    interventional_only: {_get(plan, 'interventional_only')}"
    )
    date_field, grain = _get(plan, "date_field"), _get(plan, "grain")
    if date_field or grain:
        _p(f"   date_field    : {date_field}    grain: {grain}")
    notes = _get(plan, "notes") or []
    if notes:
        _p(f"   planner notes : {list(notes)}")
    _p(f"   budget        : iter_count={iter_count}  escalation_count={escalation}")


def _fmt_check(u: Any) -> None:
    validation = _get(u, "validation")
    ok = _get(validation, "ok")
    reason = _get(validation, "reason")
    _p(f"   validation    : {'PASS ✓' if ok else 'REJECT ✗'}" + (f"    reason: {reason}" if reason else ""))
    feedback = _get(u, "plan_feedback")
    if feedback:
        _p(f"   → re-plan feedback: {feedback}")


def _fmt_intent(u: Any) -> None:
    verifications = _get(u, "verifications") or []
    verdict = verifications[-1] if verifications else {}
    decision = _get(verdict, "decision")
    _p(f"   intent verdict: {'APPROVE ✓' if decision == 'approve' else 'REVISE ↻'}")
    field = _get(verdict, "field")
    if field:
        _p(f"   offending field: {field}")
    reason = _get(verdict, "reason")
    if reason:
        _p(f"   reason        : {reason}")
    if _get(u, "plan_feedback"):
        _p("   → threaded back to the planner for one bounded re-plan")


def _fmt_execute(u: Any) -> None:
    status = _get(u, "status")
    count_total = _get(u, "count_total")
    mode = _get(u, "bucket_mode")
    _p(f"   status        : {status}    countTotal (exact oracle): {count_total}    bucket_mode: {mode}")
    provenance = _get(u, "query_provenance") or {}
    params = provenance.get("params") if isinstance(provenance, dict) else None
    if params:
        _p(f"   wire params   : {params}")
    tool_results = _get(u, "tool_results") or []
    result = tool_results[0] if tool_results and isinstance(tool_results[0], dict) else None
    if result is None:
        return
    distinct = result.get("distinct_trials")
    _p(f"   tool={result.get('tool')}    distinct_trials={distinct}    truncated={result.get('truncated')}")
    buckets = result.get("buckets") or []
    if buckets:
        _p(f"   aggregated buckets ({len(buckets)}):")
        running = 0
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            count = bucket.get("count_trials")
            running += count or 0
            _p(f"       {str(bucket.get('value')):<16} {count}")
        if count_total is not None:
            verdict = "reconciles ✓" if running == count_total else "MISMATCH ✗"
            _p(f"       Σ = {running}  vs countTotal {count_total}  → {verdict}")
    elif result.get("total_count") is not None:
        _p(f"   total_count   : {result.get('total_count')}")


def _fmt_build(u: Any) -> None:
    spec = _get(u, "spec")
    kind = _get(spec, "kind")
    viz = _get(spec, "visualization")
    if viz is not None:
        data = _get(viz, "data") or []
        _p(f"   built {kind}: chart={_enum_value(_get(viz, 'type'))}  title={_get(viz, 'title')!r}  rows={len(data)}")
        return
    question, answer = _get(spec, "question"), _get(spec, "answer")
    if question:
        _p(f"   built clarification: {question!r}")
    elif answer:
        _p(f"   built answer: {answer!r}")
    else:
        _p(f"   built {kind}")


def _fmt_output(u: Any) -> None:
    if _get(u, "status") == "error":
        verifications = _get(u, "verifications") or []
        reason = _get(verifications[-1], "reason") if verifications else None
        _p(f"   deterministic precheck: HARD-FAIL ✗  reason={reason}  → error envelope")
        return
    _p(
        "   deterministic prechecks: PASS ✓ "
        "(matched-value quotes · Σ==countTotal · partial-iff-truncated · cited-or-derived · record-grounded)"
    )
    verifications = _get(u, "verifications") or []
    verdict = verifications[-1] if verifications else {}
    decision = _get(verdict, "decision")
    reason = _get(verdict, "reason")
    _p(f"   LLM output verdict: {'APPROVE ✓' if decision == 'approve' else 'FLAG ⚑'}" + (f"  reason: {reason}" if reason else ""))


def _fmt_respond(u: Any) -> None:
    _p(f"   final status  : {_get(u, 'status')}")


def _fmt_error(u: Any) -> None:
    err = _get(u, "error") or {}
    _p(f"   error         : code={_get(err, 'code')}  message={_get(err, 'message')}")


_FORMATTERS = {
    "merge_inputs": _fmt_merge,
    "plan": _fmt_plan,
    "check": _fmt_check,
    "review_intent": _fmt_intent,
    "execute": _fmt_execute,
    "build_spec": _fmt_build,
    "review_output": _fmt_output,
    "respond": _fmt_respond,
    "error": _fmt_error,
}


# --- public API (called from app.graph.build.run_sync) -----------------------


def banner(query: str) -> None:
    """Print the trace header at the start of a traced request."""
    if not enabled():
        return
    _p()
    _p("┏" + _RULE)
    _p(f"┃ PIPELINE TRACE   query: {query!r}")
    _p("┗" + _RULE)


def stage(step: int, node: str, update: Any) -> None:
    """Print one node's stage header + its salient result (no-op if disabled)."""
    if not enabled():
        return
    label = _STAGE_LABEL.get(node, node)
    _p()
    _p(f"── [{step}] {node} → {label} ".ljust(76, "─"))
    formatter = _FORMATTERS.get(node)
    if formatter is not None:
        formatter(update)


def footer(spec: Any) -> None:
    """Print the terminal summary (final envelope) at the end of a traced request."""
    if not enabled():
        return
    viz = _get(spec, "visualization")
    meta = _get(spec, "meta")
    citations = _get(spec, "citations") or {}
    _p()
    _p("┏" + _RULE)
    line = f"┃ DONE  status={_get(spec, 'status')}  kind={_get(spec, 'kind')}"
    if viz is not None:
        line += f"  chart={_enum_value(_get(viz, 'type'))}"
    _p(line)
    count_basis = _get(meta, "count_basis") if meta is not None else None
    if count_basis is not None:
        _p(f"┃ count_basis: {count_basis}")
    _p(f"┃ citations index: {len(citations)} trials")
    notes = _get(meta, "notes") if meta is not None else None
    if notes:
        _p(f"┃ meta.notes: {list(notes)}")
    _p("┗" + _RULE)
    _p()
