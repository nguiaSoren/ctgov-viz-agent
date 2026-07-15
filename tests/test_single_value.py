"""Tests for the single_value / no-viz (scalar) path (Phase 4, CC-7).

Offline + network-free. Covers the three files Agent-D owns:

* ``app.plan.recipes`` — the ``single_value`` recipe row (count_trials → stat card).
* ``app.plan.checker`` — a ``single_value`` plan (no aggregation field, no
  series/network; even unscoped) passes ``check_plan``.
* ``app.viz.spec`` — ``build_envelope`` renders the ``count_trials`` scalar
  tool-result as either a SINGLE_VALUE stat card (kind:visualization) or a
  code-templated yes/no (kind:answer). The NUMBER is the exact ``total_count``
  inserted by CODE (G-30/CC-16), never LLM-authored.
"""

from __future__ import annotations

from app.api.schemas import ChartType, VisualizeResponse
from app.plan.checker import check_plan
from app.plan.models import Plan
from app.plan.recipes import RECIPES
from app.viz.spec import build_envelope

# --- fixtures ---------------------------------------------------------------


def _single_value_plan(**overrides) -> Plan:
    base = dict(
        query_class="single_value",
        entities={"condition": "melanoma"},
        chart_type=ChartType.SINGLE_VALUE,
        alternates=[ChartType.TABLE],
        filters={},
    )
    base.update(overrides)
    return Plan(**base)


def _sv_tool_result(total: int, kind: str) -> dict:
    """The exact shape the trunk executor puts into ``tool_results`` for a
    single_value plan: a scalar ``count_trials`` with a code-inserted total, the
    kind marker, and a membership-proof citation sample (the nctId is in the
    counted set — a legitimate 'this trial was counted' provenance)."""
    return {
        "tool": "count_trials",
        "total_count": total,
        "kind": kind,
        "citations": [
            {
                "nct_id": "NCT00000001",
                "field_path": "protocolSection.identificationModule.nctId",
                "value": "NCT00000001",
                "excerpt": "NCT00000001",
            }
        ],
    }


# --- recipe registry --------------------------------------------------------


def test_single_value_recipe_registered() -> None:
    assert len(RECIPES) == 6
    recipe = RECIPES["single_value"]
    assert recipe.allowed_tools == ["count_trials"]
    assert recipe.chart_type is ChartType.SINGLE_VALUE
    assert ChartType.TABLE in recipe.alternates
    assert recipe.degeneracy_fallback is None


# --- checker ----------------------------------------------------------------


def test_scoped_single_value_plan_passes_checker() -> None:
    result = check_plan(_single_value_plan())
    assert result.ok is True
    assert result.reason is None
    assert result.normalized_plan is not None


def test_unscoped_single_value_plan_passes_checker() -> None:
    # No entity, no filter — the too_large gate refuses a huge population at execute
    # time (G-39); the checker lets an unscoped scalar count through.
    result = check_plan(_single_value_plan(entities={}, filters={}))
    assert result.ok is True


def test_single_value_with_table_alternate_passes_checker() -> None:
    # TABLE is an allowed alternate mark for the recipe.
    result = check_plan(_single_value_plan(chart_type=ChartType.TABLE))
    assert result.ok is True


# --- viz: the stat-card (kind:visualization) path ---------------------------


def test_build_envelope_single_value_visualization() -> None:
    n = 142
    spec = build_envelope(
        plan=_single_value_plan(),
        tool_results=[_sv_tool_result(n, kind="visualization")],
        status="ok",
        question="how many melanoma trials are there?",
    )
    # schema-valid envelope (constructing VisualizeResponse ran its validators)
    assert isinstance(spec, VisualizeResponse)
    VisualizeResponse.model_validate(spec.model_dump())

    assert spec.status == "ok"
    assert spec.kind == "visualization"
    assert spec.answer is None
    assert spec.error is None
    assert spec.visualization is not None
    assert spec.visualization.type is ChartType.SINGLE_VALUE
    # the NUMBER is the exact total_count, inserted by code
    assert spec.visualization.data[0].count_trials == n
    assert spec.visualization.data[0].value == str(n)
    assert spec.meta.count_basis is not None
    assert spec.meta.count_basis.trials == n
    # a stat card has no natural Vega-Lite mark
    assert spec.vega_lite is None
    # membership-proof citation surfaced into the top-level dedup index
    assert "NCT00000001" in spec.citations


def test_single_value_datum_citations_are_membership_proof() -> None:
    spec = build_envelope(
        plan=_single_value_plan(),
        tool_results=[_sv_tool_result(5, kind="visualization")],
        status="ok",
    )
    datum = spec.visualization.data[0]
    assert len(datum.citations) == 1
    assert datum.citations[0].nct_id == "NCT00000001"
    assert datum.citations[0].excerpt == "NCT00000001"
    # the sample (1) is honestly smaller than the counted set (5)
    assert datum.contributing_count == 5
    assert datum.citations_truncated is True


# --- viz: the yes/no (kind:answer) path -------------------------------------


def test_build_envelope_single_value_answer() -> None:
    n = 9
    spec = build_envelope(
        plan=_single_value_plan(),
        tool_results=[_sv_tool_result(n, kind="answer")],
        status="ok",
        question="are there any melanoma trials?",
    )
    assert isinstance(spec, VisualizeResponse)
    VisualizeResponse.model_validate(spec.model_dump())

    assert spec.kind == "answer"
    assert spec.visualization is None
    assert spec.vega_lite is None
    # the answer is code-templated and carries the code-inserted number
    assert spec.answer is not None
    assert str(n) in spec.answer
    assert spec.meta.count_basis is not None
    assert spec.meta.count_basis.trials == n


def test_single_value_answer_zero_is_no_match() -> None:
    spec = build_envelope(
        plan=_single_value_plan(),
        tool_results=[_sv_tool_result(0, kind="answer")],
        status="ok",
    )
    assert spec.kind == "answer"
    assert spec.visualization is None
    assert spec.answer is not None
    assert "No trials match" in spec.answer
