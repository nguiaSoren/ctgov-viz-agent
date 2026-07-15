"""Real tests for the Phase-0 LangGraph skeleton (ARCHITECTURE_SPEC §3.12/§B.5).

Four guarantees:

1. ``build_graph`` compiles without error (no checkpointer -- ENG-47) and is
   cached (built once).
2. The happy path flows end-to-end to a schema-valid ``ok`` visualization,
   and the ``events`` trace visits every node in the exact §B.5 order.
3. The injected hard-error sentinel routes through the dedicated ``error``
   node to a schema-valid error envelope (never a half-built viz).
4. The injected over-budget / zero-results sentinels exercise the
   ``too_large`` refuse contract and the bounded zero-results escalation
   (one re-plan, then settle into an empty envelope) respectively.
"""

from __future__ import annotations

from app.api.schemas import ChartType, VisualizeRequest
from app.graph.build import build_graph, initial_state, run_sync

HAPPY_PATH_EVENTS = [
    "merge_inputs",
    "plan",
    "check",
    "review_intent",
    "execute",
    "build_spec",
    "review_output",
    "respond",
]


def _happy_request() -> VisualizeRequest:
    return VisualizeRequest(
        query="Phase distribution of pancreatic cancer trials",
        condition="pancreatic cancer",
        interventional_only=True,
    )


# --- graph compiles ----------------------------------------------------------


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_build_graph_is_cached() -> None:
    """A second call returns the same compiled graph object, not a rebuild."""
    assert build_graph() is build_graph()


# --- happy path ----------------------------------------------------------------


def test_happy_path_returns_ok_visualization() -> None:
    # ``execute``'s default path is now a LIVE API call; the offline structural
    # ``_force_canned`` sentinel exercises the same ok-envelope shape without a
    # network dependency (the live path is covered by the doctor's c6 + the X-2 gate).
    response = run_sync(_happy_request(), overrides={"_force_canned": True})

    assert response.status == "ok"
    assert response.kind == "visualization"
    assert response.visualization is not None
    assert response.visualization.type == ChartType.BAR
    assert len(response.visualization.data) >= 1
    assert response.meta.source == "clinicaltrials.gov"


def test_happy_path_events_trace_visits_every_node_in_order() -> None:
    graph = build_graph()
    final_state = graph.invoke(initial_state(_happy_request(), {"_force_canned": True}))
    assert final_state["events"] == HAPPY_PATH_EVENTS


# --- injected error ------------------------------------------------------------


def test_injected_error_returns_error_status() -> None:
    response = run_sync(VisualizeRequest(query="x"), overrides={"_force_error": True})

    assert response.status == "error"
    assert response.error is not None
    assert response.error.code == "upstream_error"
    assert response.visualization is None


def test_injected_error_events_include_error_node() -> None:
    graph = build_graph()
    request = VisualizeRequest(query="x")
    final_state = graph.invoke(initial_state(request, overrides={"_force_error": True}))

    assert "error" in final_state["events"]
    assert final_state["events"][-1] == "respond"
    # A hard error short-circuits straight to the error node -- it never
    # reaches build_spec/review_output.
    assert "build_spec" not in final_state["events"]


# --- injected too_large ----------------------------------------------------------


def test_injected_too_large_returns_answer_envelope() -> None:
    response = run_sync(VisualizeRequest(query="y"), overrides={"_force_too_large": True})

    assert response.status == "too_large"
    assert response.kind == "answer"
    assert response.answer is not None
    assert response.visualization is None
    assert response.vega_lite is None
    assert response.meta.partial is None
    assert response.meta.count_basis.trials == 142_411


# --- injected empty (zero-results) — exercises the bounded escalation edge ------


def test_injected_empty_triggers_one_bounded_replan_then_settles() -> None:
    """Zero-results is an escalation-eligible trigger (§B.5): the first pass
    bounces back to ``plan`` once (the shared esc<=1 budget), and the second
    pass's zero-results settles into a schema-valid empty envelope."""
    graph = build_graph()
    request = VisualizeRequest(query="z")
    final_state = graph.invoke(initial_state(request, overrides={"_force_empty": True}))

    spec = final_state["spec"]
    assert spec.status == "empty"
    assert spec.kind == "visualization"
    assert spec.visualization is not None
    assert spec.visualization.data == []

    # The escalation actually fired once: plan/check/review_intent/execute
    # each ran twice, and the shared budget prevented a third attempt.
    assert final_state["events"].count("plan") == 2
    assert final_state["events"].count("execute") == 2
    assert final_state["escalation_count"] == 1
