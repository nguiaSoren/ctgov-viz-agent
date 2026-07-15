"""Regression tests for the D3 hardening fixes (graph wiring + escalation
coverage).

Three fixes under test:

1. ``build_spec`` now reads ``state["partial"]`` and forwards it into
   ``build_envelope(partial=...)`` -- previously unwired, so a genuine
   truncation would silently vanish from ``meta.partial``.
2. ``execute`` now hands out a fresh ``copy.deepcopy`` of the canned bucket
   list per call instead of the shared module-global list object -- a
   cross-request aliasing hazard.
3. ``check``/``review_intent`` now support ``_force_reject``/``_force_revise``
   test-injection sentinels (mirroring the existing ``execute`` sentinels),
   making the checker-reject and intent-revise escalation edges
   (``route_after_check``/``route_after_intent``) exercisable at the graph
   level.
"""

from __future__ import annotations

from app.api.schemas import VisualizeRequest
from app.graph.build import build_graph, initial_state, run_sync
from app.graph.nodes import _CANNED_PHASE_BUCKETS, build_spec
from app.llm.adapter import get_adapter
from app.llm.planner import plan_request

# --- FIX 1: partial reader wiring --------------------------------------------


def test_build_spec_forwards_partial_into_envelope() -> None:
    """A state carrying a genuine ``partial`` truncation must surface it in
    the built spec's ``meta.partial`` -- proving ``build_spec`` actually reads
    ``state["partial"]`` instead of silently dropping it."""
    merged_inputs = {"condition": "pancreatic cancer", "query": "phase distribution"}
    plan = plan_request(get_adapter(), merged_inputs)
    state = {
        "plan": plan,
        "tool_results": [{"tool": "aggregate_by", "buckets": _CANNED_PHASE_BUCKETS}],
        "status": "ok",
        "question": "phase distribution",
        "retrieved_at": None,
        "query_provenance": None,
        "partial": {"truncated": True, "of_total": 500},
    }

    result = build_spec(state)
    spec = result["spec"]

    assert spec.meta.partial is not None
    assert spec.meta.partial.truncated is True
    assert spec.meta.partial.of_total == 500


def test_build_spec_leaves_partial_null_when_absent() -> None:
    """No ``partial`` in state (the common case) must still produce
    ``meta.partial:null`` -- the fix must not force a ``Partial`` into every
    response."""
    merged_inputs = {"condition": "pancreatic cancer", "query": "phase distribution"}
    plan = plan_request(get_adapter(), merged_inputs)
    state = {
        "plan": plan,
        "tool_results": [{"tool": "aggregate_by", "buckets": _CANNED_PHASE_BUCKETS}],
        "status": "ok",
        "question": "phase distribution",
        "retrieved_at": None,
        "query_provenance": None,
        "partial": None,
    }

    result = build_spec(state)
    assert result["spec"].meta.partial is None


# --- FIX 3: checker-reject escalation edge ------------------------------------


def test_checker_reject_triggers_one_bounded_replan_then_errors() -> None:
    response = run_sync(
        VisualizeRequest(query="x", condition="y"), overrides={"_force_reject": True}
    )

    assert response.status == "error"
    assert response.error is not None


def test_checker_reject_events_show_one_replan_then_error() -> None:
    graph = build_graph()
    request = VisualizeRequest(query="x", condition="y")
    final_state = graph.invoke(initial_state(request, overrides={"_force_reject": True}))

    events = final_state["events"]
    # Exactly one bounded re-plan: "plan" runs twice, never a third time.
    assert events.count("plan") == 2
    assert events[-2:] == ["error", "respond"]
    assert "review_intent" not in events  # a rejected plan never reaches intent review


# --- P4-ROUTING: intent-revise escalation is best-effort, NOT a hard stop -----
# Reconciled to ARCHITECTURE_SPEC §B.5 in Phase 4 (was a hard stop to `error` in the
# Phase-0 skeleton): the Checker already proved the plan LEGAL, so a persistently-revised
# plan with the re-plan budget exhausted ships best-effort to `execute` + a disclosed
# `meta.notes` caveat, rather than refusing over an advisory reviewer's disagreement.
# `_force_canned` keeps `execute` offline+deterministic so this stays a structural test.


def test_intent_revise_exhausted_budget_ships_best_effort_not_error() -> None:
    response = run_sync(
        VisualizeRequest(query="x", condition="y"),
        overrides={"_force_revise": True, "_force_canned": True},
    )

    assert response.status == "ok"  # best-effort, not error
    assert response.visualization is not None
    # The best-effort routing is disclosed honestly on meta.notes.
    assert any("best-effort" in note.lower() for note in response.meta.notes)


def test_intent_revise_events_show_one_replan_then_best_effort_execute() -> None:
    graph = build_graph()
    request = VisualizeRequest(query="x", condition="y")
    final_state = graph.invoke(
        initial_state(request, overrides={"_force_revise": True, "_force_canned": True})
    )

    events = final_state["events"]
    # Exactly one bounded re-plan: "plan" runs twice, never a third time.
    assert events.count("plan") == 2
    # Then best-effort execute → build_spec → review_output → respond (NOT error).
    assert "execute" in events
    assert "error" not in events
    assert events[-1] == "respond"


# --- FIX 2: execute's canned bucket list is no longer shared across requests --


def test_execute_returns_independent_bucket_objects_per_run() -> None:
    graph = build_graph()
    request = VisualizeRequest(query="x", condition="y")

    # ``_force_canned`` exercises the deepcopy-isolated ``_CANNED_PHASE_BUCKETS``
    # path offline (the default path is now a live API call).
    state_a = graph.invoke(initial_state(request, {"_force_canned": True}))
    state_b = graph.invoke(initial_state(request, {"_force_canned": True}))

    buckets_a = state_a["tool_results"][-1]["buckets"]
    buckets_b = state_b["tool_results"][-1]["buckets"]

    assert buckets_a is not buckets_b
    for bucket_a, bucket_b in zip(buckets_a, buckets_b, strict=True):
        assert bucket_a is not bucket_b

    # Mutating a returned bucket must never leak back into the module global.
    buckets_a[0]["count_trials"] = 999999
    assert _CANNED_PHASE_BUCKETS[0]["count_trials"] != 999999
