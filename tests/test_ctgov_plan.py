"""Tests for the ctgov tool surface + plan recipes/checker (Phase 0 scaffold).

Recipes and the Plan Checker are REAL in Phase 0; the 7 tool bodies are stubs that
raise NotImplementedError until Phase 1 wires the live API. The date/citation helpers
are pure functions and are real now.
"""

import inspect

import pytest
from pydantic import ValidationError

from app.api.schemas import ChartType
from app.ctgov import citations, dates, tools
from app.ctgov.enums import (
    DATE_FIELDS,
    OVERALL_STATUS_TOKENS,
    PHASE_TOKENS,
    QUERY_AREAS,
)
from app.plan.checker import check_plan
from app.plan.models import Plan
from app.plan.recipes import RECIPES, Recipe, get_recipe

# --- recipe registry (real data, one row per query class) ---

def test_distribution_recipe_is_a_recipe():
    assert isinstance(RECIPES["distribution"], Recipe)
    assert isinstance(get_recipe("distribution"), Recipe)


def test_all_query_classes_registered():
    # Six as of Phase 4: the five chart classes + single_value (the no-viz / scalar path).
    assert set(RECIPES) == {
        "distribution", "timeseries", "compare", "geographic", "network", "single_value"
    }


def test_every_recipe_chart_type_and_tools_are_legal():
    for name, recipe in RECIPES.items():
        assert recipe.chart_type in ChartType, name
        for alt in recipe.alternates:
            assert alt in ChartType, (name, alt)
        # every tool a recipe is allowed to call must be a real registered tool
        assert set(recipe.allowed_tools) <= tools.TOOL_NAMES, (name, recipe.allowed_tools)


# --- Plan Checker (mechanical anti-hallucination gate) ---

def _good_distribution_plan() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        field="phase",
        chart_type=ChartType.BAR,
        alternates=[ChartType.HISTOGRAM],
        filters={},
    )


def test_checker_passes_a_valid_plan():
    result = check_plan(_good_distribution_plan())
    assert result.ok is True
    assert result.normalized_plan is not None


def test_checker_rejects_invented_phase_token():
    plan = _good_distribution_plan()
    plan.filters = {"phase": ["PHASE9"]}  # not a real token
    result = check_plan(plan)
    assert result.ok is False
    assert result.reason  # a machine reason, not silence


def test_checker_rejects_wrong_chart_for_recipe():
    plan = _good_distribution_plan()
    plan.chart_type = ChartType.NETWORK_GRAPH  # network chart on a distribution plan
    result = check_plan(plan)
    assert result.ok is False
    assert result.reason


def test_bad_date_field_is_rejected_at_construction():
    # DateField is a Literal, so an invented date field can't even be constructed —
    # the anti-hallucination gate starts at the type boundary.
    with pytest.raises(ValidationError):
        Plan(
            query_class="timeseries",
            field=None,
            date_field="madeUpDate",
            chart_type=ChartType.TIME_SERIES,
            filters={},
        )


# --- the 7 tools are importable signatures with stub bodies ---

def test_all_tools_registered():
    # Phase 2 adds study_duration_histogram (the duration recipe) alongside the
    # original 7 named tools.
    assert set(tools.TOOL_REGISTRY) == {
        "count_trials", "aggregate_by", "timeseries", "compare", "build_network",
        "study_duration_histogram", "get_trial", "resolve_entity",
    }
    assert tools.TOOL_NAMES == frozenset(tools.TOOL_REGISTRY)


# Implemented through Phase 2 (their live paths are exercised in the live gate
# tests); calling them here with dummy args would issue a real network request.
_IMPLEMENTED = {
    "count_trials", "aggregate_by", "timeseries", "compare",
    "build_network", "study_duration_histogram",
}
# Deliberately still stubbed (Phase 2/3 stretch — drill-down + display canon).
_STILL_STUBBED = {"get_trial", "resolve_entity"}


def test_stretch_tool_bodies_still_refuse():
    # The two not-yet-built tools must refuse, never silently return — the implemented
    # tools are excluded (dummy args would hit the network). ``get_trial`` now also
    # refuses a MALFORMED nctId early via the path-injection guard (ValueError), so a
    # stubbed tool may raise either NotImplementedError (valid args, fetch unbuilt) or
    # ValueError (the guard fired on the dummy "x") — both are a refusal.
    assert _IMPLEMENTED | _STILL_STUBBED == set(tools.TOOL_REGISTRY)
    for name in _STILL_STUBBED:
        fn = tools.TOOL_REGISTRY[name]
        sig = inspect.signature(fn)
        kwargs = {}
        for p in sig.parameters.values():
            if p.default is not inspect.Parameter.empty:
                continue
            kwargs[p.name] = {} if "dict" in str(p.annotation) else "x"
        with pytest.raises((NotImplementedError, ValueError)):
            fn(**kwargs)


def test_get_trial_rejects_malformed_nctid_before_any_fetch():
    # The nctId path-injection guard (G-8/R-20) fires BEFORE the stubbed fetch: a
    # malformed id is a ValueError; a well-formed id reaches the NotImplementedError.
    with pytest.raises(ValueError):
        tools.get_trial("NCT123")  # too short
    with pytest.raises(ValueError):
        tools.get_trial("'; DROP TABLE studies;--")  # injection payload
    with pytest.raises(NotImplementedError):
        tools.get_trial("NCT01234567")  # well-formed → guard passes, fetch still a stub


def test_implemented_tools_are_no_longer_stubs():
    # Every implemented tool must not be the NotImplementedError stub any longer.
    import dis

    for name in _IMPLEMENTED:
        source_consts = tools.TOOL_REGISTRY[name].__code__.co_consts
        assert not any(
            isinstance(c, str) and c.startswith("Phase 1: ") for c in source_consts
        ), f"{name} still raises the stub"
        assert dis.Bytecode(tools.TOOL_REGISTRY[name])  # importable/executable code object


# --- enum token sets are grounded in the API Reality Brief ---

def test_enum_token_sets_present():
    assert "NA" in PHASE_TOKENS and "EARLY_PHASE1" in PHASE_TOKENS
    assert len(OVERALL_STATUS_TOKENS) == 14
    assert QUERY_AREAS == frozenset({"term", "cond", "intr", "spons", "locn"})
    assert "studyFirstPostDate" in DATE_FIELDS


# --- pure date + citation helpers (real now) ---

def test_date_parsing_and_future_flag():
    assert dates.parse_ct_date("2015") == (2015, None)
    assert dates.parse_ct_date("2015-05") == (2015, 5)
    assert dates.parse_ct_date("2030-12")[0] == 2030
    assert dates.is_future(2030, 2026) is True
    assert dates.is_future(2015, 2026) is False


def test_excerpt_extraction_and_roundtrip():
    record = {"protocolSection": {"designModule": {"phases": ["PHASE1", "PHASE2"]}}}
    excerpt = citations.extract_excerpt(record, "protocolSection.designModule.phases")
    # excerpt is string-extracted from the record, never authored
    assert "PHASE1" in excerpt
    assert citations.is_substring_at(record, "protocolSection.designModule.phases", "PHASE1") is True
