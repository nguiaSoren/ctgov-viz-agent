"""Phase-5 runtime-harness guard tests (ARCHITECTURE_SPEC §4 · §B.4 · §B.7).

Proves every execution-engine backstop FIRES on a pathological state — the
"tested to abort a pathological loop" half of the P5-GUARDS decision. Under v1's
single-shot planner + shared ``escalation <= 1`` budget these caps cannot trip in
normal operation (see the graph/hardening suites, which show the happy paths
untouched), so each is exercised here by INJECTING the pathological state
directly into the pure guard functions and into the ``plan``/``execute`` nodes.
All offline (``_force_plan`` short-circuits the LLM; no network).
"""

from __future__ import annotations

import time

import pytest

from app import config
from app.api.schemas import VisualizeRequest
from app.graph import guards
from app.graph.build import build_graph, initial_state
from app.graph.nodes import execute, plan
from app.plan.models import Plan


def _plan() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        filters={},
        field="phase",
        chart_type="bar",
    )


# --- pure guard functions ----------------------------------------------------


def test_over_deadline_past_present_and_unset():
    now = time.monotonic()
    assert guards.over_deadline(now - 1.0, now=now) is True
    assert guards.over_deadline(now + 1.0, now=now) is False
    assert guards.over_deadline(None, now=now) is False  # unset → never trips (offline path)


def test_plan_signature_is_stable_and_discriminating():
    p1 = _plan()
    p2 = _plan()
    assert guards.plan_signature(p1) == guards.plan_signature(p2)  # same plan → same sig
    p3 = p1.model_copy(update={"field": "overallStatus"})
    assert guards.plan_signature(p3) != guards.plan_signature(p1)  # different field → different sig


def test_is_stalled_predicate():
    p = _plan()
    sig = guards.plan_signature(p)
    assert guards.is_stalled(p, [sig]) is True
    assert guards.is_stalled(p, []) is False
    assert guards.is_stalled(p, None) is False


def test_check_pre_plan_guards_each_trip():
    now_past = time.monotonic() - 5.0
    assert guards.check_pre_plan_guards({"deadline_at": now_past}) == guards.DEADLINE_EXCEEDED
    assert (
        guards.check_pre_plan_guards({"iter_count": config.MAX_REACT_ITERATIONS})
        == guards.MAX_ITERATIONS_EXCEEDED
    )
    assert (
        guards.check_pre_plan_guards({"events": ["x"] * config.MAX_GRAPH_STEPS})
        == guards.MAX_STEPS_EXCEEDED
    )
    assert guards.check_pre_plan_guards({"iter_count": 0, "events": [], "deadline_at": None}) is None


def test_check_tool_budget():
    assert guards.check_tool_budget(config.MAX_TOOL_CALLS) is None  # at the cap is OK
    assert guards.check_tool_budget(config.MAX_TOOL_CALLS + 1) == guards.MAX_TOOL_CALLS_EXCEEDED


def test_guard_error_is_redacted():
    err = guards.guard_error(guards.DEADLINE_EXCEEDED)
    assert err["code"] == guards.DEADLINE_EXCEEDED
    assert err["message"] == guards.GUARD_MESSAGE  # fixed generic message, no internals


# --- the plan node aborts on each pre-plan / stall guard ----------------------


def _plan_state(**overrides) -> dict:
    state = {
        "merged_inputs": {"_force_plan": _plan()},
        "iter_count": 0,
        "escalation_count": 0,
        "seen_signatures": [],
        "deadline_at": None,
        "events": [],
    }
    state.update(overrides)
    return state


def test_plan_node_aborts_on_deadline():
    out = plan(_plan_state(deadline_at=time.monotonic() - 1.0))
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.DEADLINE_EXCEEDED
    assert out["events"] == ["plan"]  # the node returned an error update and appended itself
    assert "plan" not in out  # short-circuited: no plan was produced (no LLM/forced read)


def test_plan_node_aborts_on_iteration_cap():
    out = plan(_plan_state(iter_count=config.MAX_REACT_ITERATIONS))
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.MAX_ITERATIONS_EXCEEDED


def test_plan_node_aborts_on_step_backstop():
    out = plan(_plan_state(events=["x"] * config.MAX_GRAPH_STEPS))
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.MAX_STEPS_EXCEEDED


def test_plan_node_stall_fires_only_beyond_escalation():
    sig = guards.plan_signature(_plan())
    # iter_count >= 2 (a 3rd+ plan entry, unreachable under v1's <=1 escalation) AND
    # the signature already seen → abort as stalled.
    out = plan(_plan_state(iter_count=2, seen_signatures=[sig]))
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.STALLED_NO_PROGRESS


def test_plan_node_stall_does_not_fire_on_the_sanctioned_single_replan():
    # iter_count == 1 (the ONE legal escalation re-plan) with the SAME signature must
    # NOT abort — it proceeds so the loop can settle a clean empty / ship best-effort.
    sig = guards.plan_signature(_plan())
    out = plan(_plan_state(iter_count=1, seen_signatures=[sig]))
    assert out.get("status") != "error"
    assert out["plan"] is not None  # a real plan was produced, no abort


# --- the execute node aborts on deadline / tool budget -----------------------


def test_execute_node_aborts_on_deadline_before_any_network():
    state = {
        "merged_inputs": {},
        "plan": _plan(),
        "deadline_at": time.monotonic() - 1.0,
        "tool_call_count": 0,
    }
    out = execute(state)  # deadline trips BEFORE _dispatch_execute → no network call
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.DEADLINE_EXCEEDED


def test_execute_node_aborts_on_tool_budget_before_any_network():
    state = {
        "merged_inputs": {},
        "plan": _plan(),
        "deadline_at": None,
        "tool_call_count": config.MAX_TOOL_CALLS,  # +1 in execute → over budget
    }
    out = execute(state)
    assert out["status"] == "error"
    assert out["error"]["code"] == guards.MAX_TOOL_CALLS_EXCEEDED


# --- end-to-end: a blown deadline routes cleanly to an error envelope ---------


def test_end_to_end_deadline_routes_to_error_envelope():
    req = VisualizeRequest(query="phase distribution of pancreatic cancer trials")
    graph = build_graph()
    # A negative budget puts deadline_at in the past; _force_plan keeps it offline.
    state = initial_state(req, {"_force_plan": _plan()}, deadline_seconds=-0.001)
    final = graph.invoke(state)
    spec = final["spec"]
    assert spec.status == "error"
    assert spec.error is not None
    assert spec.error.code == guards.DEADLINE_EXCEEDED
    assert spec.visualization is None  # never a half-viz
    assert "plan" in final["events"] and "error" in final["events"]


@pytest.mark.parametrize("code", [
    guards.DEADLINE_EXCEEDED,
    guards.MAX_ITERATIONS_EXCEEDED,
    guards.MAX_STEPS_EXCEEDED,
    guards.MAX_TOOL_CALLS_EXCEEDED,
    guards.STALLED_NO_PROGRESS,
])
def test_all_guard_codes_are_distinct_nonempty(code):
    assert isinstance(code, str) and code
