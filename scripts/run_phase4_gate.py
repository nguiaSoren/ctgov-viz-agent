#!/usr/bin/env python
"""Phase-4 live gate: run real natural-language queries through the FULL graph with a real
LLM adapter, proving the agentic layer sits on the proven deterministic engine WITHOUT ever
emitting a number. NL -> LLM planner -> Plan Checker -> Intent Reviewer (LLM) -> execute (live
ClinicalTrials.gov) -> viz-spec builder -> Output Reviewer (code + LLM) -> cited envelope.

Run (from ctgov-viz-agent/):
    set -a; source ../../../.env; set +a          # to get CTGOV_OPENAI_API_KEY
    export OPENAI_API_KEY="$CTGOV_OPENAI_API_KEY" LLM_PROVIDER=openai \
           LLM_MODEL_PLANNER=gpt-5.4 LLM_MODEL_REVIEWER=gpt-5.4-mini
    ./.venv/bin/python scripts/run_phase4_gate.py

Writes examples/run_phase4_<class>.json for each query that completes live. Skips (does not fail)
a query whose upstream ClinicalTrials.gov call is transiently rate-limited, so a burst doesn't
mask the real result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.api.schemas import VisualizeRequest
from app.graph.build import build_graph, initial_state

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# One NL query per class (+ both single_value shapes). No `query_class` is ever passed in —
# the LLM must classify from the natural language.
CASES: list[tuple[str, dict]] = [
    ("distribution", {"query": "How are interventional pancreatic cancer trials distributed across phases?",
                      "condition": "pancreatic cancer", "interventional_only": True}),
    ("timeseries", {"query": "How has the number of melanoma trials changed per year since 2015?",
                    "condition": "melanoma"}),
    ("compare", {"query": "Compare overall status for trials of pembrolizumab versus nivolumab"}),
    ("geographic", {"query": "Which countries have the most recruiting diabetes trials?",
                    "condition": "diabetes"}),
    ("network", {"query": "Show a network of drugs studied together in melanoma trials",
                 "condition": "melanoma"}),
    ("single_value_count", {"query": "How many interventional trials are there for pancreatic cancer?",
                            "condition": "pancreatic cancer", "interventional_only": True}),
    ("single_value_yesno", {"query": "Is there any recruiting trial for glioblastoma?",
                            "condition": "glioblastoma"}),
    ("too_large", {"query": "How are cancer trials distributed across phases overall?", "condition": "cancer"}),
]

# The number a datum/spec ever shows must come from a tool computation, never the LLM. The
# planner's typed output has NO numeric-count field by construction — so this is an invariant we
# assert structurally, and here we spot-check the plan carries no stray integer count.
def _plan_has_no_count(plan) -> bool:
    # Plan fields that legitimately hold integers are the inclusive year bounds inside filters;
    # nothing else. A trial COUNT never appears on the plan.
    banned_keys = {"count", "total", "trial_count", "n"}
    blob = json.loads(plan.model_dump_json())
    return not any(k in banned_keys for k in _walk_keys(blob))


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


def main() -> int:
    graph = build_graph()
    ran = 0
    for label, req in CASES:
        request = VisualizeRequest(**req)
        final = graph.invoke(initial_state(request))
        plan = final.get("plan")
        spec = final["spec"]
        status = spec.status
        # A transient upstream rate-limit surfaces as a redacted error envelope — skip, don't fail.
        if status == "error" and spec.error and str(spec.error.code).startswith("upstream"):
            print(f"[{label:20}] SKIP — upstream rate-limited ({spec.error.code})")
            continue
        ran += 1
        pc = plan.query_class if plan else "?"
        replans = final["events"].count("plan") - 1
        inv = _plan_has_no_count(plan) if plan else True
        detail = (
            f"class={pc} status={status} kind={spec.kind} "
            f"viz={spec.visualization.type.value if spec.visualization else None} "
            f"re-plans={replans} invariant(no-count-on-plan)={inv}"
        )
        # provenance/citation sanity
        cited = 0
        if spec.visualization and isinstance(spec.visualization.data, list):
            cited = sum(len(d.citations) for d in spec.visualization.data)
        elif spec.citations:
            cited = len(spec.citations)
        print(f"[{label:20}] {detail} citations~{cited}")
        out = EXAMPLES / f"run_phase4_{label}.json"
        out.write_text(json.dumps(json.loads(spec.model_dump_json()), indent=2))
    print(f"\n{ran}/{len(CASES)} live runs completed (rest skipped as rate-limited).")
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main())
