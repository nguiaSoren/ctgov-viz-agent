"""Phase-5 live red-team battery (P5-HARDEN) — a small FIXED regression set that
exercises the REAL end-to-end path (the "wired isn't run" gate for the hardening
trunk): the response cache, the runtime guards, the Essie neutralization, the
too_large refuse, and the clarification outcome. Uses the deterministic execute
layer live (the StubAdapter plans a distribution-by-phase, so no provider key is
needed); every number is the real API's.

Run deliberately (spends real API calls): ``./.venv/bin/python scripts/run_phase5_battery.py``
"""

from __future__ import annotations

import sys
import time

from app.api.schemas import VisualizeRequest
from app.cache import RESPONSE_CACHE
from app.graph import guards
from app.graph.build import build_graph, initial_state, run_sync

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name} — {detail}")


def main() -> int:
    print("Phase-5 live red-team battery\n" + "=" * 60)

    # 1. HAPPY PATH — real execute + reconciliation (interventional pancreatic).
    RESPONSE_CACHE.clear()
    req = VisualizeRequest(
        query="Phase distribution of interventional pancreatic cancer trials",
        condition="pancreatic cancer",
        interventional_only=True,
    )
    t0 = time.monotonic()
    r1 = run_sync(req)
    dt1 = time.monotonic() - t0
    total = r1.meta.count_basis.trials if r1.meta and r1.meta.count_basis else None
    sigma = sum(d.count_trials for d in (r1.visualization.data or [])) if r1.visualization else None
    check("happy: real cited distribution", r1.status == "ok" and r1.kind == "visualization",
          f"status={r1.status} kind={r1.kind} total={total} Σbars={sigma}")
    check("happy: Σbars reconciles to countTotal", sigma == total and total is not None,
          f"Σ={sigma} == countTotal={total}")

    # 2. CACHE — the identical request is served from cache (much faster, same total).
    t0 = time.monotonic()
    r2 = run_sync(req)
    dt2 = time.monotonic() - t0
    total2 = r2.meta.count_basis.trials if r2.meta and r2.meta.count_basis else None
    check("cache: repeat served from cache", total2 == total and dt2 < dt1,
          f"first={dt1:.2f}s cached={dt2:.3f}s total={total2}")

    # 3. ESSIE INJECTION — an operator in the condition value is neutralized to a
    #    literal, so it does NOT blow up into the OR-union population.
    RESPONSE_CACHE.clear()
    inj = run_sync(VisualizeRequest(query="phase distribution", condition="cancer OR diabetes"))
    inj_total = inj.meta.count_basis.trials if inj.meta and inj.meta.count_basis else None
    # cancer OR diabetes as an operator union is ~145k (too_large); neutralized to the
    # literal string it matches a small/zero population → NOT a too_large union.
    check("essie: operator neutralized (not a union)", inj.status != "too_large" and (inj_total or 0) < 20000,
          f"status={inj.status} total={inj_total} (raw-union would be ~145k)")

    # 4. TOO_LARGE — an unscoped 'cancer by phase' (>20k, phase not count-aggregatable) refuses.
    RESPONSE_CACHE.clear()
    tl = run_sync(VisualizeRequest(query="phase distribution of all cancer trials", condition="cancer"))
    tl_total = tl.meta.count_basis.trials if tl.meta and tl.meta.count_basis else None
    check("too_large: refuses the chart, exact total via answer",
          tl.status == "too_large" and tl.kind == "answer" and tl.visualization is None
          and tl.vega_lite is None and (tl.meta.partial is None) and (tl_total or 0) > 20000,
          f"status={tl.status} total={tl_total} viz={tl.visualization} partial={tl.meta.partial}")

    # 5. CLARIFICATION — a demonstrative referent with no antecedent asks (offline path).
    RESPONSE_CACHE.clear()
    cl = run_sync(VisualizeRequest(query="How many trials are there for this drug?"))
    check("clarification: dangling reference asks, never guesses",
          cl.kind == "clarification" and cl.status == "empty" and bool(cl.question)
          and cl.visualization is None,
          f"kind={cl.kind} question={cl.question!r}")

    # 6. GUARD — a blown wall-clock deadline routes to a redacted error (inject a past deadline).
    RESPONSE_CACHE.clear()
    graph = build_graph()
    st = initial_state(VisualizeRequest(query="phase distribution", condition="pancreatic cancer"),
                       deadline_seconds=-0.001)
    guarded = graph.invoke(st)["spec"]
    check("guard: blown deadline → redacted error, no half-viz",
          guarded.status == "error" and guarded.error is not None
          and guarded.error.code == guards.DEADLINE_EXCEEDED and guarded.visualization is None,
          f"status={guarded.status} code={guarded.error.code if guarded.error else None}")

    print("=" * 60)
    n_fail = sum(1 for verdict, _, _ in _results if verdict == FAIL)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
