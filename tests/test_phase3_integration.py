"""Phase-3 integration tests (offline) — the orchestrator-wired seam.

Covers the two Wave-B integrations that live only at the graph/viz boundary:

1. **Composite-citation teeth** — the Output-Reviewer precheck must reject a
   fabricated ``matched_tokens`` member (not just a fabricated primary excerpt),
   so a composite bucket can't smuggle an unverifiable token past the gate.
2. **Degeneracy → BAR fallback** — a ``network_fallback`` tool-result (emitted by
   ``_execute_network`` when ``build_graph`` reports a degenerate graph) must
   render as a schema-valid BAR (not a network_graph), with a bar-appropriate
   title + the "too sparse to graph" disclosure, while a normal network result
   still renders as a network.
"""

from __future__ import annotations

from app.api.schemas import (
    ChartType,
    Citation,
    Datum,
    EncodingChannel,
    Meta,
    Visualization,
    VisualizeResponse,
)
from app.plan.models import NetworkSpec, Plan
from app.viz.review import deterministic_precheck
from app.viz.spec import build_envelope

# --- 1. composite-citation teeth (matched_tokens verified) ------------------


def _spec_with_citation(citation: Citation) -> VisualizeResponse:
    datum = Datum(value="PHASE1|PHASE2", label="Phase 1/2", count_trials=3, citations=[citation])
    viz = Visualization(
        type=ChartType.BAR,
        title="t",
        encoding={"x": EncodingChannel(field="value"), "y": EncodingChannel(field="count_trials")},
        data=[datum],
    )
    return VisualizeResponse(
        status="ok", kind="visualization", visualization=viz, citations={}, meta=Meta()
    )


def test_valid_composite_matched_tokens_pass() -> None:
    cit = Citation(
        nct_id="NCT00000001",
        field_path="protocolSection.designModule.phases",
        value=["PHASE1", "PHASE2"],
        matched_value="PHASE1",
        matched_tokens=["PHASE1", "PHASE2"],
    )
    pc = deterministic_precheck(
        _spec_with_citation(cit), count_total=3, mode="combine", distinct_trials=3, truncated=False
    )
    assert not pc.hard_fail


def test_fabricated_excerpt_token_hard_fails() -> None:
    # The PRIMARY excerpt is valid, but a fabricated member token ("PHASE9") is not
    # an element of value → the composite must NOT sneak it past the gate.
    cit = Citation(
        nct_id="NCT00000001",
        field_path="protocolSection.designModule.phases",
        value=["PHASE1", "PHASE2"],
        matched_value="PHASE1",
        matched_tokens=["PHASE1", "PHASE9"],
    )
    pc = deterministic_precheck(
        _spec_with_citation(cit), count_total=3, mode="combine", distinct_trials=3, truncated=False
    )
    assert pc.hard_fail
    assert pc.reason == "citation_invalid"


# --- 2. degeneracy → BAR fallback -------------------------------------------


def _network_plan() -> Plan:
    return Plan(
        query_class="network",
        entities={"condition": "progeria"},
        chart_type=ChartType.NETWORK_GRAPH,
        network=NetworkSpec(kind="drug_drug"),
    )


def _fallback_tool_result() -> dict:
    return {
        "tool": "build_network_fallback",
        "mode": "explode",
        "network_fallback": True,
        "distinct_trials": 9,
        "truncated": False,
        "buckets": [
            {
                "value": "Lonafarnib",
                "label": "Lonafarnib",
                "count_trials": 4,
                "count_mentions": 4,
                "source_ids": ["NCT00000001"],
                "citations": [
                    Citation(
                        nct_id="NCT00000001",
                        field_path="protocolSection.armsInterventionsModule.interventions[].name",
                        value="Lonafarnib",
                        matched_value="Lonafarnib",
                    )
                ],
                "citations_truncated": False,
                "contributing_count": 4,
            }
        ],
        "notes": ["Degenerate-network fallback: individual drug frequencies."],
    }


def test_degenerate_network_renders_bar() -> None:
    spec = build_envelope(
        plan=_network_plan(),
        tool_results=[_fallback_tool_result()],
        status="ok",
        question="Drugs studied together in progeria trials",
    )
    assert spec.status == "ok"
    assert spec.kind == "visualization"
    # rendered as a BAR (row data), NOT a network_graph — the schema validator would
    # reject a NETWORK_GRAPH type carrying a row list, so the chart_type override is
    # load-bearing here.
    assert spec.visualization is not None
    assert spec.visualization.type is ChartType.BAR
    assert isinstance(spec.visualization.data, list)
    assert spec.visualization.title.startswith("Most-studied drugs in progeria")
    assert spec.vega_lite is not None  # a bar gets a Vega-Lite convenience block
    assert any("too sparse" in n for n in spec.meta.notes)
    # the fallback bar is still cited
    assert spec.visualization.data[0].citations


def test_degenerate_bar_passes_precheck_reconcile_waived() -> None:
    spec = build_envelope(
        plan=_network_plan(), tool_results=[_fallback_tool_result()], status="ok"
    )
    # reconcile is waived for the fallback (count_total None), but the excerpt +
    # cited-or-derived checks still run and must pass.
    pc = deterministic_precheck(
        spec, count_total=None, mode="explode", distinct_trials=9, truncated=False, reconcile=False
    )
    assert not pc.hard_fail
