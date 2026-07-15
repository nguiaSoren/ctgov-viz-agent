"""Offline self-verification for the Phase-4 planner (classify->fill, CC-1 precedence).

Everything here runs through :class:`~app.llm.adapter.StubAdapter` — zero network, zero
provider key. The guarantees under test:

1. ``PlannerOutput.to_plan`` maps the closed planner keys onto the EXACT internal ``Plan``
   dict keys the Plan Checker validates against (``KNOWN_FILTER_KEYS`` / the entity dimensions).
2. ``plan_request(StubAdapter(), ...)`` returns a ``Plan`` that passes ``check_plan(...).ok``.
3. CC-1 field precedence: a typed structured field lands on its dimension and, on a query/field
   disagreement, an override note is echoed.
4. ``feedback=...`` threads through the prompt without crashing (a no-op for the stub).
"""

from __future__ import annotations

from app.api.schemas import ChartType
from app.llm.adapter import StubAdapter
from app.llm.planner import (
    PlannerEntities,
    PlannerFilters,
    PlannerNetwork,
    PlannerOutput,
    PlannerSeries,
    _apply_field_precedence,
    plan_request,
)
from app.plan.checker import KNOWN_FILTER_KEYS, check_plan
from app.plan.models import Plan

# --- to_plan key mapping ---------------------------------------------------


def test_to_plan_filter_keys_are_checker_spellings() -> None:
    """Every snake_case planner filter re-spells to the exact KNOWN_FILTER_KEYS name."""
    out = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        filters=PlannerFilters(
            phase=["PHASE1", "PHASE2"],
            overall_status="RECRUITING",
            study_type="INTERVENTIONAL",
            sponsor_class="INDUSTRY",
            intervention_type="DRUG",
            start_year=2015,
            end_year=2020,
            country="France",
        ),
        field="phase",
        chart_type=ChartType.BAR,
        alternates=[ChartType.HISTOGRAM, ChartType.TABLE],
    )
    plan = out.to_plan()

    assert set(plan.filters).issubset(KNOWN_FILTER_KEYS)
    assert set(plan.filters) == {
        "phase",
        "overallStatus",
        "studyType",
        "sponsorClass",
        "interventionType",
        "start_year",
        "end_year",
        "country",
    }
    assert plan.filters["phase"] == ["PHASE1", "PHASE2"]
    assert plan.filters["overallStatus"] == "RECRUITING"
    # A fully-filled distribution plan is still mechanically legal.
    assert check_plan(plan).ok is True


def test_to_plan_entities_use_dimension_keys() -> None:
    out = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="glioma", drug="temozolomide"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    plan = out.to_plan()
    assert plan.entities == {"condition": "glioma", "drug": "temozolomide"}
    assert check_plan(plan).ok is True


def test_to_plan_unset_filters_are_absent_not_null() -> None:
    """Only SET planner fields appear in Plan.filters — no ``None`` sentinels leak through."""
    out = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        filters=PlannerFilters(overall_status="COMPLETED"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    plan = out.to_plan()
    assert plan.filters == {"overallStatus": "COMPLETED"}


def test_to_plan_compare_series_and_network_lower_correctly() -> None:
    """series arms and the network block lower to their internal counterparts."""
    compare = PlannerOutput(
        query_class="compare",
        field="phase",
        chart_type=ChartType.GROUPED_BAR,
        series=[
            PlannerSeries(label="Pembro", entities=PlannerEntities(drug="pembrolizumab")),
            PlannerSeries(
                label="Nivo",
                entities=PlannerEntities(drug="nivolumab"),
                filters=PlannerFilters(overall_status="RECRUITING"),
            ),
        ],
    ).to_plan()
    assert compare.series is not None and len(compare.series) == 2
    assert compare.series[0].entities == {"drug": "pembrolizumab"}
    assert compare.series[1].filters == {"overallStatus": "RECRUITING"}
    assert check_plan(compare).ok is True

    network = PlannerOutput(
        query_class="network",
        chart_type=ChartType.NETWORK_GRAPH,
        network=PlannerNetwork(kind="sponsor_drug", entity_a="melanoma"),
    ).to_plan()
    assert network.network is not None
    assert network.network.kind == "sponsor_drug"
    assert network.network.entity_a == "melanoma"
    assert check_plan(network).ok is True


def test_to_plan_interventional_only_rides_on_plan_not_filters() -> None:
    plan = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
        interventional_only=True,
    ).to_plan()
    assert plan.interventional_only is True
    assert "interventional_only" not in plan.filters


# --- plan_request returns a checker-legal Plan -----------------------------


def test_plan_request_returns_checker_legal_plan() -> None:
    plan = plan_request(
        StubAdapter(),
        {"query": "How are melanoma trials distributed across phases?", "condition": "melanoma"},
    )
    assert isinstance(plan, Plan)
    assert plan.query_class == "distribution"
    assert plan.field == "phase"
    assert plan.chart_type == ChartType.BAR
    assert plan.entities.get("condition") == "melanoma"
    assert check_plan(plan).ok is True


def test_plan_request_defaults_condition_when_absent() -> None:
    plan = plan_request(StubAdapter(), {"query": "distribution by phase"})
    assert plan.entities.get("condition") == "pancreatic cancer"
    assert plan.interventional_only is False
    assert check_plan(plan).ok is True


# --- CC-1 field precedence -------------------------------------------------


def test_plan_request_drug_name_lands_in_entities() -> None:
    """A structured ``drug_name`` field is placed on the drug dimension (CC-1 gap-fill)."""
    plan = plan_request(
        StubAdapter(),
        {"query": "trials for this drug", "drug_name": "aspirin"},
    )
    assert plan.entities.get("drug") == "aspirin"
    assert check_plan(plan).ok is True


def test_cc1_field_wins_and_echoes_override_note() -> None:
    """When the planner (from the query) and a typed field disagree, the field wins + a note."""
    proposed = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(drug="pembrolizumab"),  # as if inferred from the query
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(proposed, {"drug_name": "nivolumab"})
    assert proposed.entities.drug == "nivolumab"
    assert any("Override" in note and "nivolumab" in note for note in proposed.notes)

    plan = proposed.to_plan()
    assert plan.entities.get("drug") == "nivolumab"
    assert any("Override" in note for note in plan.notes)


def test_cc1_gap_fill_emits_no_override_note() -> None:
    """Filling an unset dimension from a typed field is NOT a disagreement — no note."""
    proposed = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(proposed, {"drug_name": "aspirin"})
    assert proposed.entities.drug == "aspirin"
    assert not any("Override" in note for note in proposed.notes)


def test_cc1_year_bounds_and_interventional_only_apply() -> None:
    proposed = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(
        proposed, {"start_year": 2010, "end_year": 2020, "interventional_only": True}
    )
    plan = proposed.to_plan()
    assert plan.filters["start_year"] == 2010
    assert plan.filters["end_year"] == 2020
    assert plan.interventional_only is True
    assert check_plan(plan).ok is True


def test_cc1_human_phase_string_is_normalized_and_applied() -> None:
    """A human phase string ("Phase 2") is now NORMALIZED to wire tokens and applied
    authoritatively (CC-1 / E-16 — P5-INPUT made trial_phase a validated, closed-vocab
    field); the resulting plan stays checker-legal. A combined form ("Phase 1/2")
    becomes the composite token list; an already-tokenized value is applied unchanged."""
    human = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(human, {"trial_phase": "Phase 2"})
    assert human.filters.phase == ["PHASE2"]  # normalized human string -> applied token
    assert check_plan(human.to_plan()).ok is True

    combo = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(combo, {"trial_phase": "Phase 1/2"})
    assert combo.filters.phase == ["PHASE1", "PHASE2"]  # composite (CC-15) from "1/2"
    assert check_plan(combo.to_plan()).ok is True

    tokenized = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(tokenized, {"trial_phase": "PHASE2"})
    assert tokenized.filters.phase == ["PHASE2"]
    plan = tokenized.to_plan()
    assert plan.filters["phase"] == ["PHASE2"]
    assert check_plan(plan).ok is True


def test_cc1_study_type_hint_is_tokenized_when_legal() -> None:
    proposed = PlannerOutput(
        query_class="distribution",
        entities=PlannerEntities(condition="melanoma"),
        field="phase",
        chart_type=ChartType.BAR,
    )
    _apply_field_precedence(proposed, {"study_type": "interventional"})
    plan = proposed.to_plan()
    assert plan.filters["studyType"] == "INTERVENTIONAL"
    assert check_plan(plan).ok is True


# --- feedback re-plan does not crash ---------------------------------------


def test_plan_request_with_feedback_still_returns_valid_plan() -> None:
    plan = plan_request(
        StubAdapter(),
        {"query": "melanoma trials by phase", "condition": "melanoma"},
        feedback="unknown_field:'foo'",
    )
    assert isinstance(plan, Plan)
    assert check_plan(plan).ok is True
