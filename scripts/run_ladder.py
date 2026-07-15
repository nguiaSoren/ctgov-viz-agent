"""Phase-6 example ladder — the SHIPPED example runs (P6-SRC / P6-LADDER).

Drives 13 real natural-language queries through the FULL agentic graph with the
REAL LLM planner (NL -> LLM planner -> Plan Checker -> Intent Reviewer -> execute
[live ClinicalTrials.gov] -> viz-spec builder -> Output Reviewer -> cited envelope).
No ``query_class`` is ever passed in: the LLM classifies every query from the
natural language. The ladder runs simple -> highly complex, one capability per rung,
and the ACTUAL JSON of each rung is saved to ``examples/run_NN_<slug>.json``
(A-57/V-9: script-regenerated, never hand-edited).

Why serial (not fanned out): the public CT.gov API rate-limits under burst
(~3 req/s, LESSON H2/L5); the right response is FEWER calls, not parallelism that
shares the same quota and trips 429 harder. A transiently rate-limited rung is
SKIPPED (its JSON left untouched), never overwritten with an error.

Run (from ctgov-viz-agent/):
    set -a; source ../../../.env; set +a          # to get CTGOV_OPENAI_API_KEY
    export OPENAI_API_KEY="$CTGOV_OPENAI_API_KEY" LLM_PROVIDER=openai \
           LLM_MODEL_PLANNER=gpt-5.4 LLM_MODEL_REVIEWER=gpt-5.4-mini
    ./.venv/bin/python scripts/run_ladder.py               # all 15 rungs
    ./.venv/bin/python scripts/run_ladder.py 02 12         # only rungs 02 and 12
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.api.schemas import VisualizeRequest, VisualizeResponse
from app.graph.build import run_sync

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "examples"

# Each rung: (NN, slug, one-line "what this proves", request-kwargs). The request
# carries ONLY the natural-language query + the OPTIONAL structured fields a real
# user would supply (condition / drug_name / interventional_only / start_year) —
# never a query_class, never a plan. The LLM does the classification + filter
# extraction; the deterministic core does every number.
LADDER: list[tuple[str, str, str, dict]] = [
    # --- the simple end + the chart-type coverage (one of each renderable mark) ---
    ("01", "single_value_yesno",
     "trivial scalar — 'is a chart even needed?' (assignment step 3 / CC-7): answers in prose, no viz",
     {"query": "Is there any recruiting trial for glioblastoma?", "condition": "glioblastoma"}),
    ("02", "distribution_phase",
     "BAR + the killer gate (X-2): phase distribution, Σ distinct-nctId == countTotal, explicit 63%-class NA bucket",
     {"query": "How are interventional pancreatic cancer trials distributed across phases?",
      "condition": "pancreatic cancer", "interventional_only": True}),
    ("03", "timeseries_year",
     "TIME_SERIES: per-year bins, fill-0 gap years, genuine future startDate -> a flagged 'planned' bucket",
     {"query": "How has the number of melanoma trials changed per year since 2015?",
      "condition": "melanoma", "start_year": 2015}),
    ("04", "histogram_duration",
     "HISTOGRAM: study duration (completionDate - startDate) binned into ranges over the interventional pancreatic set (a continuous-magnitude mark, not a categorical bar)",
     {"query": "Show the distribution of study durations for interventional pancreatic cancer trials.",
      "condition": "pancreatic cancer", "interventional_only": True}),
    ("05", "compare_grouped_bar",
     "GROUPED_BAR: two drugs side-by-side by status; synonym recall (Keytruda == pembrolizumab); category union",
     {"query": "Compare the overall status of pembrolizumab versus nivolumab trials"}),
    ("06", "geographic_ranked_bar",
     "RANKED BAR: geographic (not choropleth), per-trial country dedup (THE trap), top-N + 'Other', free-text country",
     {"query": "Which countries have the most recruiting diabetes trials?", "condition": "diabetes"}),
    ("07", "network_drug_drug",
     "NETWORK_GRAPH (the richest viz): drug<->drug co-occurrence; every edge weight traces to 2 cited nctIds; placebo-free, synonym-merged",
     {"query": "Show a network of drugs studied together in melanoma trials", "condition": "melanoma"}),
    # --- the judgment cases: knowing when NOT to answer, and how to answer honestly ---
    ("08", "too_large_refuse",
     "knows when NOT to chart: 121k cancer trials by phase exceeds the paging budget -> refuse + exact total, no biased prefix",
     {"query": "How are cancer trials distributed across phases overall?", "condition": "cancer"}),
    ("09", "exact_at_scale_status",
     "the counterpoint to rung 08: the SAME 121k population charts EXACTLY by status via per-token count queries (no paging, no bias)",
     {"query": "How are cancer trials distributed across overall recruitment status?", "condition": "cancer"}),
    ("10", "network_degenerate_fallback",
     "knows when NOT to graph: progeria's 10 drug trials have no repeated co-occurrence -> refuse the hairball, fall back to a cited bar",
     {"query": "Show a network of drugs studied together in progeria trials", "condition": "progeria"}),
    ("11", "clarification",
     "asks rather than guesses: a demonstrative referent with no antecedent -> a first-class clarification, nothing fabricated",
     {"query": "How many trials are there for this drug?"}),
    ("12", "cc1_field_vs_query_conflict",
     "input precedence (CC-1): the query says 'Keytruda' but the structured drug_name says 'nivolumab' -> field wins, echoed in meta.notes",
     {"query": "How many trials are there for Keytruda?", "drug_name": "nivolumab"}),
    # --- the boss tier: composed / adversarial queries the whole architecture is built to survive ---
    ("13", "boss_stacked_filters",
     "BOSS #1 (compound plan): ONE sentence -> condition + FOUR stacked filters (recruiting + industry + interventional + since-2020) that collapse 3950 -> 229 and still RECONCILE on the reduced set (filters bite; the AC3 dropped-filter regression, proven fixed)",
     {"query": "How are recruiting, industry-sponsored interventional pancreatic cancer trials "
               "that started in 2020 or later distributed across phases?",
      "condition": "pancreatic cancer", "interventional_only": True, "start_year": 2020}),
    ("14", "boss_injection_neutralized",
     "BOSS #2 (security): a structured field value carrying an Essie operator (raw = a 121k+ union / DoS-amplification) is neutralized live to an inert literal -> the real, small population",
     {"query": "How are the matching trials distributed across phases?",
      "condition": "cancer OR diabetes"}),
    ("15", "boss_compare_filtered_arms",
     "BOSS #3 (the hardest plan): a COMPARE where EACH arm carries its OWN 4-filter stack. The planner must decompose one sentence into 2 series x 4 filters + a sponsor-type aggregation; each arm collapses (pembro 2903->123, nivo 2011->40) and self-reconciles -> a grouped bar with a visibly rich, non-trivial answer.",
     {"query": "Compare how recruiting, interventional Phase 3 pembrolizumab trials versus "
               "nivolumab trials, started in 2018 or later, break down by lead sponsor type."}),
]


def _reconcile(spec: VisualizeResponse) -> str:
    """One-line reconciliation / behavior proof for the rung's console summary."""
    m = spec.meta
    basis = m.count_basis.trials if m and m.count_basis else None
    if spec.status == "too_large":
        return f"REFUSE  exact total={basis:,}  viz=null vega=null partial={m.partial}"
    if spec.kind == "clarification":
        return f"CLARIFY question={spec.question!r}"
    if spec.kind == "answer":
        return f"ANSWER  {spec.answer!r}  basis={basis}"
    viz = spec.visualization
    if viz is None:
        return f"status={spec.status} (no viz)"
    if viz.type.value == "network_graph":
        nodes, edges = viz.data.nodes, viz.data.edges
        every2 = all(len(e.citations) == 2 for e in edges) if edges else "n/a(fallback?)"
        return f"NETWORK nodes={len(nodes)} edges={len(edges)} every-edge-2-cites={every2} basis={basis}"
    rows = viz.data
    sigma = sum(d.count_trials for d in rows)
    # combine reconciles Σ==countTotal; explode is Σ>=distinct by design.
    ok = "Σ==countTotal" if sigma == basis else f"Σ={sigma} vs countTotal={basis} (explode: Σ>=distinct OK)"
    return f"{viz.type.value:12} buckets={len(rows)} {ok}"


def _params(spec: VisualizeResponse) -> str:
    """The effective wire params — proves which filters actually reached the API
    (the BOSS-#1 'filters bite' and BOSS-#2 'neutralized value' evidence)."""
    prov = (spec.meta.query_provenance if spec.meta else {}) or {}
    p = prov.get("params", {})
    keep = {k: v for k, v in p.items() if k.startswith("query.") or k.startswith("filter.")}
    return json.dumps(keep, ensure_ascii=False)


def _run_one(nn: str, slug: str, proves: str, req_kwargs: dict) -> int:
    request = VisualizeRequest(**req_kwargs)
    try:
        spec = run_sync(request)
    except Exception as exc:  # transport blip surfaces as a raised error here
        print(f"[{nn}] {slug}: EXCEPTION {type(exc).__name__}: {exc} — left untouched")
        return 2
    # A transient upstream rate-limit surfaces as a redacted error envelope — skip, don't overwrite.
    if spec.status == "error" and spec.error and str(spec.error.code).startswith("upstream"):
        print(f"[{nn}] {slug}: SKIP — upstream rate-limited ({spec.error.code}) — JSON left untouched")
        return 2

    body = json.loads(spec.model_dump_json())
    path = _OUT / f"run_{nn}_{slug}.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[{nn}] {slug}")
    print(f"     proves: {proves}")
    print(f"     query : {req_kwargs['query']!r}")
    print(f"     status={spec.status} kind={spec.kind}")
    print(f"     proof : {_reconcile(spec)}")
    print(f"     params: {_params(spec)}")
    if spec.meta and spec.meta.notes:
        for note in spec.meta.notes[:3]:
            print(f"     note  : {note}")
    print(f"     wrote {path.relative_to(_ROOT)}")
    return 0


def main() -> int:
    wanted = set(sys.argv[1:])
    rc = 0
    for nn, slug, proves, req_kwargs in LADDER:
        if wanted and nn not in wanted:
            continue
        result = _run_one(nn, slug, proves, req_kwargs)
        if result == 1:
            rc = 1
        time.sleep(0.5)  # be polite to the shared per-IP quota
    return rc


if __name__ == "__main__":
    sys.exit(main())
