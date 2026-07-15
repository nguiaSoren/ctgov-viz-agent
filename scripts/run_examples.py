"""Regenerate the Phase-2 example runs (X-1..X-5 + X-TL) from LIVE API calls.

Drives each query CLASS end-to-end through the FULL graph (merge -> plan -> check
-> review_intent -> execute -> build_spec -> review_output -> respond) with a
HARDCODED Plan injected via the ``_force_plan`` sentinel -- the Phase-2 breadth
gate. The LLM planner is Phase 4; Phase 2 proves the deterministic engine covers
all five classes off one core, reconciled + cited, WITHOUT the LLM.

Run:  ./.venv/bin/python scripts/run_examples.py
Writes: examples/run_{x1_timeseries,x3_compare,x4_geographic,x5_network,
        xtl_too_large}.json (X-2 has its own scripts/run_gate.py).
Prints: per-example status + the reconciliation/structure proof.

Every example is a sub-20k population that CHARTS, except X-TL ("overall cancer
trials by phase") which deliberately exceeds the budget -> a ``too_large`` refuse
(the showpiece: the system knows when NOT to chart).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.api.schemas import ChartType, VisualizeRequest
from app.ctgov.client import UpstreamError
from app.graph.build import run_sync
from app.plan.models import NetworkSpec, Plan, Series

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "examples"


def _plan_x1() -> Plan:
    return Plan(
        query_class="timeseries",
        entities={"condition": "melanoma"},
        filters={"start_year": 2015},
        date_field="startDate",
        grain="year",
        chart_type=ChartType.TIME_SERIES,
    )


def _plan_x3() -> Plan:
    return Plan(
        query_class="compare",
        field="overallStatus",
        chart_type=ChartType.GROUPED_BAR,
        series=[
            Series(label="Pembrolizumab", entities={"drug": "pembrolizumab"}),
            Series(label="Nivolumab", entities={"drug": "nivolumab"}),
        ],
    )


def _plan_x4() -> Plan:
    return Plan(
        query_class="geographic",
        entities={"condition": "diabetes"},
        filters={"overallStatus": "RECRUITING"},
        field="country",
        chart_type=ChartType.BAR,
    )


def _plan_x5() -> Plan:
    return Plan(
        query_class="network",
        entities={"condition": "melanoma"},
        chart_type=ChartType.NETWORK_GRAPH,
        network=NetworkSpec(kind="drug_drug"),
    )


def _plan_x5b() -> Plan:
    # Phase-3: the sponsor<->drug network (same population as X-5) — the generality
    # proof that the ONE graph builder serves a different entity graph, not just one.
    return Plan(
        query_class="network",
        entities={"condition": "melanoma"},
        chart_type=ChartType.NETWORK_GRAPH,
        network=NetworkSpec(kind="sponsor_drug"),
    )


def _plan_x5c() -> Plan:
    # Phase-3: the degeneracy showpiece. Progeria's drug trials have no repeated
    # co-occurrence, so drug<->drug is degenerate (edges==0 under the k=2 default) ->
    # the system REFUSES the hairball and falls back to a cited individual-drug bar
    # ("knows when NOT to graph", sibling of X-TL's too_large refuse).
    return Plan(
        query_class="network",
        entities={"condition": "progeria"},
        chart_type=ChartType.NETWORK_GRAPH,
        network=NetworkSpec(kind="drug_drug"),
    )


def _plan_x6() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        field="study_duration",
        chart_type=ChartType.HISTOGRAM,
        interventional_only=True,
    )


def _plan_xtl() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "cancer"},
        field="phase",
        chart_type=ChartType.BAR,
    )


_EXAMPLES = [
    ("x1_timeseries", "Melanoma trials started per year since 2015", _plan_x1),
    ("x3_compare", "Pembrolizumab vs nivolumab trials by status", _plan_x3),
    ("x4_geographic", "Recruiting diabetes trials by country", _plan_x4),
    ("x5_network", "Drugs studied together in melanoma trials", _plan_x5),
    ("x5b_sponsor_drug", "Sponsor-drug network for melanoma trials", _plan_x5b),
    ("x5c_degenerate_fallback", "Drugs studied together in progeria trials", _plan_x5c),
    ("x6_histogram_duration", "Study-duration distribution of interventional pancreatic cancer trials", _plan_x6),
    ("xtl_too_large", "Overall distribution of cancer trials by phase", _plan_xtl),
]


def _run_one(name: str, query: str, plan: Plan) -> int:
    request = VisualizeRequest(query=query)
    try:
        spec = run_sync(request, overrides={"_force_plan": plan})
    except UpstreamError as exc:
        print(f"  [{name}] TRANSIENT upstream ({exc.code}) -- left untouched, retry later")
        return 2
    body = spec.model_dump()

    print(f"\n=== {name}: {query} ===")
    print(f"  status={body['status']} kind={body['kind']}")
    if body["status"] == "error":
        print(f"  ERROR: {body.get('error')}")
        return 1

    if body["status"] == "too_large":
        total = body["meta"]["count_basis"]["trials"]
        ok = body["visualization"] is None and body["vega_lite"] is None
        print(f"  too_large: exact total={total:,}  answer={body['answer'][:80]!r}")
        print(f"  visualization/vega_lite both null: {ok}")
        _save(name, body)
        return 0 if ok else 1

    viz = body["visualization"]
    print(f"  chart={viz['type']}  citations_index={len(body['citations'])}")
    if viz["type"] == "network_graph":
        nodes, edges = viz["data"]["nodes"], viz["data"]["edges"]
        cited = all(len(e["citations"]) == 2 for e in edges)
        print(f"  nodes={len(nodes)} edges={len(edges)}  every-edge-2-citations={cited}")
        print(f"  distinct_trials(basis)={body['meta']['count_basis']['trials']}")
        print(f"  vega_lite is null (networks never Vega): {body['vega_lite'] is None}")
    else:
        data = viz["data"]
        basis = body["meta"]["count_basis"]
        print(f"  buckets={len(data)}  count_basis={basis}")
        sample = [(d["value"], d["count_trials"]) for d in data[:8]]
        print(f"  data(head)={sample}")
        if "series" in (data[0] if data else {}):
            print("  series present (grouped bar):", sorted({d.get("series") for d in data}))
        if any(d.get("planned") for d in data):
            print("  planned bucket:", [d["value"] for d in data if d.get("planned")])
    if body["meta"]["notes"]:
        print(f"  notes: {body['meta']['notes'][:2]}")
    _save(name, body)
    return 0


def _save(name: str, body: dict) -> None:
    _OUT.mkdir(exist_ok=True)
    path = _OUT / f"run_{name}.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(_ROOT)}")


def main() -> int:
    # Optional CLI filter: `run_examples.py x5 x5b x5c` regenerates only the named
    # examples (matched on the leading token before the first "_"), so a re-run after
    # a change that only touched some classes doesn't re-page the untouched ones — the
    # right response to a per-IP rate limit is FEWER calls, not parallelism (which
    # shares the same quota and trips 429 harder, LESSON L5).
    wanted = set(sys.argv[1:])
    rc = 0
    for name, query, plan_fn in _EXAMPLES:
        if wanted and name.split("_", 1)[0] not in wanted:
            continue
        result = _run_one(name, query, plan_fn())
        if result == 1:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
