"""Phase-5 response-cache INTEGRATION into the graph (§3.10 · C-73/C-74 · SEC-48).

The pure cache is covered by ``tests/test_cache.py``; this proves the WIRING:
``execute`` serves a hit (no network), ``respond`` stores a public result, and
neither errors nor sentinel-driven offline runs pollute the cache. All offline
(a cache hit short-circuits before ``execute`` touches the network; the end-to-end
test uses the zero-network StubAdapter and a pre-seeded cache).
"""

from __future__ import annotations

from app.api.schemas import VisualizeRequest, VisualizeResponse
from app.cache import RESPONSE_CACHE, plan_cache_key
from app.graph.build import run_sync
from app.graph.nodes import execute, respond
from app.llm.adapter import get_adapter
from app.llm.planner import plan_request
from app.plan.models import Plan


def _plan() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        filters={},
        field="phase",
        chart_type="bar",
    )


def _envelope() -> VisualizeResponse:
    return VisualizeResponse(status="ok", kind="visualization", meta={})


def test_execute_serves_from_cache_without_network():
    p = _plan()
    env = _envelope()
    RESPONSE_CACHE.set(plan_cache_key(p), env)
    # No _force sentinel + a seeded cache → execute returns the cached envelope BEFORE
    # any network dispatch (a miss here would try to page the live API).
    out = execute({"merged_inputs": {}, "plan": p, "deadline_at": None, "tool_call_count": 0})
    assert out["cache_hit"] is True
    assert out["spec"].model_dump() == env.model_dump()  # value-equal deep copy (not a shared alias)
    assert out["status"] == "ok"


def test_execute_bypasses_cache_when_sentinel_active():
    p = _plan()
    RESPONSE_CACHE.set(plan_cache_key(p), _envelope())
    # A sentinel is active → the cache is bypassed even though an entry exists
    # (the canned structural path must stay deterministic + cache-free).
    out = execute({"merged_inputs": {"_force_canned": True}, "plan": p, "deadline_at": None})
    assert not out.get("cache_hit")
    assert out["status"] == "ok"
    assert out["tool_results"][0]["tool"] == "aggregate_by"  # the canned path, not the cache


def test_respond_stores_public_result():
    p = _plan()
    env = _envelope()
    out_state = {"status": "ok", "spec": env, "plan": p, "merged_inputs": {}, "cache_hit": False}
    respond(out_state)
    assert RESPONSE_CACHE.get(plan_cache_key(p)).model_dump() == env.model_dump()  # stored (deep copy)


def test_respond_does_not_store_errors():
    p = _plan()
    env = VisualizeResponse(status="error", kind="answer", meta={}, error={"code": "x", "message": "y"})
    respond({"status": "error", "spec": env, "plan": p, "merged_inputs": {}, "cache_hit": False})
    assert RESPONSE_CACHE.get(plan_cache_key(p)) is None  # errors are never cached


def test_respond_does_not_restore_a_cache_hit_replay():
    p = _plan()
    env = _envelope()
    # cache_hit=True → this envelope came FROM the cache; don't re-store it.
    respond({"status": "ok", "spec": env, "plan": p, "merged_inputs": {}, "cache_hit": True})
    assert len(RESPONSE_CACHE) == 0


def test_respond_bypasses_store_on_sentinel():
    p = _plan()
    respond({
        "status": "ok", "spec": _envelope(), "plan": p,
        "merged_inputs": {"_force_canned": True}, "cache_hit": False,
    })
    assert RESPONSE_CACHE.get(plan_cache_key(p)) is None  # sentinel results never stored


def test_end_to_end_stub_run_is_served_from_cache_offline():
    # Compute the plan the StubAdapter produces for this request, seed the cache,
    # then run the full graph — execute must serve the seeded envelope with NO
    # network (the StubAdapter is offline; the cache short-circuits before dispatch).
    req = VisualizeRequest(query="phase distribution of pancreatic cancer", condition="pancreatic cancer")
    merged = {"query": req.query, "condition": "pancreatic cancer"}
    plan = plan_request(get_adapter(), merged)
    env = _envelope()
    RESPONSE_CACHE.set(plan_cache_key(plan), env)
    resp = run_sync(req)
    # served from cache (deep copy), deterministic, zero network calls
    assert resp.model_dump() == env.model_dump()
