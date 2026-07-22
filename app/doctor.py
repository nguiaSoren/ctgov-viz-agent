"""`ct-doctor` — the end-to-end self-check (run: `uv run ct-doctor` or `python -m app.doctor`).

Asserts, in order, that:
  1. the app package imports (every module),
  2. the LangGraph pipeline compiles,
  3. the Pydantic envelope validates every golden fixture,
  4. a dummy request traverses ALL graph nodes to a schema-valid `ok` envelope
     (offline structural path via the `_force_canned` sentinel),
  5. the FastAPI surface answers /healthz and routes /visualize end-to-end (in-process, no server),
  6. the real X-2 request reconciles Σ-buckets to the live API's `countTotal`
     (SKIP-as-pass when the network is unreachable, so the doctor stays green offline).

Checks 1-4 never touch the network. Check 5 runs in-process (TestClient, no server) but its
POST goes down the real `execute` path, so online it DOES make live API calls — which is why
check 6 clears the response cache first, or it would replay check 5's envelope instead of
fetching (see `_c6_live_reconciliation`).

Exits 0 if every check passes, 1 otherwise. It started as the "does the skeleton hold
together?" gate from BUILD_PLAN Phase 0 and grew a live reconciliation check with the
Phase-1 API wiring — if it fails, whatever is built on top is built on sand.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# All node names the happy path must traverse (BUILD_PLAN Phase-0 graph acceptance).
_EXPECTED_NODES = [
    "merge_inputs", "plan", "check", "review_intent",
    "execute", "build_spec", "review_output", "respond",
]

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# ANSI colors as constants (backslash escapes inside f-string braces are illegal < py3.12,
# and our floor is 3.11 — keep the escapes out of every f-string expression).
_GREEN, _RED, _RESET = "\033[32m", "\033[31m", "\033[0m"


def _check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  {_GREEN}PASS{_RESET}  {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:  # noqa: BLE001 — the doctor reports every failure, never raises
        print(f"  {_RED}FAIL{_RESET}  {label} — {type(exc).__name__}: {exc}")
        return False


def _c1_imports() -> str:
    import app.api.schemas  # noqa: F401
    import app.ctgov.aggregate  # noqa: F401
    import app.ctgov.citations
    import app.ctgov.client
    import app.ctgov.dates  # noqa: F401
    import app.ctgov.enums
    import app.ctgov.tools
    import app.graph.build  # noqa: F401
    import app.graph.nodes
    import app.graph.state
    import app.llm.adapter  # noqa: F401
    import app.llm.planner
    import app.llm.reviewers
    import app.main  # noqa: F401
    import app.plan.checker  # noqa: F401
    import app.plan.models
    import app.plan.recipes
    import app.viz.spec  # noqa: F401
    return "all app modules import"


def _c2_graph_compiles() -> str:
    from app.graph.build import build_graph
    build_graph()
    return "LangGraph pipeline compiles (stateless, no checkpointer)"


def _c3_goldens_validate() -> str:
    from app.api.schemas import VisualizeResponse
    files = sorted(glob.glob(str(_FIXTURES / "golden_*.json")))
    if not files:
        raise AssertionError(f"no golden fixtures found under {_FIXTURES}")
    for f in files:
        resp = VisualizeResponse.model_validate(json.loads(Path(f).read_text()))
        assert resp.meta.source == "clinicaltrials.gov", f"{Path(f).name}: meta.source wrong"
    return f"{len(files)} goldens validate against VisualizeResponse"


def _c4_dummy_traverses_all_nodes() -> str:
    from app.api.schemas import VisualizeRequest, VisualizeResponse
    from app.graph.build import build_graph, initial_state
    req = VisualizeRequest(
        query="Phase distribution of interventional pancreatic cancer trials",
        condition="pancreatic cancer",
        interventional_only=True,
    )
    # ``_force_canned`` = the offline structural path (``execute``'s default is a
    # live API call now; the live reconciliation is c6's job).
    final = build_graph().invoke(initial_state(req, {"_force_canned": True}))
    spec = final["spec"]
    assert isinstance(spec, VisualizeResponse), "spec is not a VisualizeResponse"
    assert spec.status == "ok" and spec.visualization is not None, "dummy run not ok/visualization"
    events = final.get("events", [])
    missing = [n for n in _EXPECTED_NODES if n not in events]
    assert not missing, f"nodes never visited: {missing} (events={events})"
    return f"traversed {len(_EXPECTED_NODES)} nodes → status={spec.status}, type={spec.visualization.type.value}"


def _c5_api_end_to_end() -> str:
    from fastapi.testclient import TestClient  # local import: only the doctor needs the test dep

    from app.api.schemas import VisualizeResponse
    from app.main import _selfcheck_payload, app
    client = TestClient(app)
    h = client.get("/healthz")
    assert h.status_code == 200 and h.json() == {"status": "ok"}, f"/healthz -> {h.status_code} {h.text}"
    # The HTTP surface can't carry the offline ``_force_canned`` sentinel, so this
    # asserts the surface ROUTES end-to-end to a schema-valid envelope (offline: a
    # valid redacted error; online: ok). The real live path is c6.
    v = client.post("/visualize", json=_selfcheck_payload())
    assert v.status_code == 200, f"/visualize -> {v.status_code} {v.text}"
    body = v.json()
    resp = VisualizeResponse.model_validate(body)  # a schema-valid envelope, always
    assert resp.status in {"ok", "empty", "too_large", "error"}, f"bad status {resp.status}"
    assert resp.meta.source == "clinicaltrials.gov", "meta.source drifted"
    return f"/healthz 200, /visualize routes end-to-end (status={resp.status})"


def _c6_live_reconciliation() -> str:
    """Run the real X-2 request live and assert Σ-buckets reconciles to the API's
    ``countTotal``.

    Check 5 POSTs the SAME request through the app, so on a successful online run it
    has already populated the response cache for this exact plan — without the
    ``RESPONSE_CACHE.clear()`` below, this check would replay check 5's envelope and
    the "live" in its name would be a lie. Clearing forces a real fetch + aggregate,
    and the ``countTotal`` cross-check at the end is a second, independent live call.

    Offline (unreachable) or a transient upstream failure (rate-limit / 5xx) →
    SKIP-as-pass: the pipeline returns a clean redacted ``upstream_error`` envelope,
    which is not a logic failure (the deterministic reconciliation is unit-covered).
    A DIFFERENT error code (e.g. ``reconciliation_failed``) or a bad ``ok`` still
    FAILS — that would be a real regression."""
    import httpx

    from app.api.schemas import VisualizeRequest
    from app.cache import RESPONSE_CACHE
    from app.ctgov.tools import count_trials
    from app.graph.build import run_sync

    try:
        httpx.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"pageSize": 1, "countTotal": "true"},
            timeout=8.0,
        )
    except httpx.HTTPError:
        return "SKIP (offline) — live ClinicalTrials.gov API unreachable"

    RESPONSE_CACHE.clear()  # evict check 5's entry so this really is a live fetch, not a replay
    resp = run_sync(
        VisualizeRequest(
            query="Phase distribution of interventional pancreatic cancer trials",
            condition="pancreatic cancer",
            interventional_only=True,
        )
    )
    if resp.status == "error":
        code = resp.error.code if resp.error else "unknown"
        assert code == "upstream_error", f"live X-2 errored with non-transient code {code!r}"
        return "SKIP (transient upstream error) — live API rate-limited/unavailable"

    assert resp.status == "ok" and resp.visualization is not None, f"live X-2 not ok: {resp.status}"
    summed = sum(d.count_trials for d in resp.visualization.data)
    # Independent countTotal cross-check (the precheck already reconciled internally
    # to reach status "ok"); tolerate a transient failure of this extra live call.
    try:
        total = count_trials({"cond": "pancreatic cancer"}, {"interventional_only": True})
    except Exception:  # noqa: BLE001 — transient; status "ok" already proves reconciliation
        return f"live X-2 reconciles (status=ok, Σbuckets={summed}); independent countTotal check skipped"
    drift = abs(summed - total)
    assert drift <= 20, f"Σbuckets {summed} vs countTotal {total} drift {drift} > 20"
    return f"live X-2 reconciles: Σbuckets={summed} ≈ countTotal={total} (drift {drift})"


def main() -> int:
    print("ct-doctor — self-check (1-4 offline, 5 in-process, 6 live)\n")
    checks = [
        ("app package imports", _c1_imports),
        ("graph compiles", _c2_graph_compiles),
        ("goldens validate", _c3_goldens_validate),
        ("dummy request traverses all nodes", _c4_dummy_traverses_all_nodes),
        ("FastAPI surface (in-process)", _c5_api_end_to_end),
        ("live X-2 reconciliation", _c6_live_reconciliation),
    ]
    results = [_check(label, fn) for label, fn in checks]
    ok = all(results)
    verdict = f"{_GREEN}OK{_RESET}" if ok else f"{_RED}FAILED{_RESET}"
    print(f"\n{verdict}: {sum(results)}/{len(results)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
