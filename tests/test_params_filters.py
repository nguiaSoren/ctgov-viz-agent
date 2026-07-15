"""``build_search_params`` composes phase / sponsorClass / interventionType filters into the
Essie ``filter.advanced`` expression (offline). Regression for the gap the Phase-4 hard battery
exposed: these keys were accepted by the Checker + emitted by the LLM planner but SILENTLY DROPPED
at the query layer, so a compound query ran on a broader population than requested. Essie syntax
live-verified (interventional pancreatic 3950 → +phase[1,2] 2601 → +industry compound 928)."""

from __future__ import annotations

import pytest

from app.ctgov.params import build_search_params


def _advanced(filters: dict) -> str:
    return build_search_params({"cond": "pancreatic cancer"}, filters).get("filter.advanced", "")


def test_phase_list_becomes_an_or_group() -> None:
    adv = _advanced({"phase": ["PHASE1", "PHASE2"]})
    assert "AREA[Phase](PHASE1 OR PHASE2)" in adv


def test_single_phase_is_a_group_of_one() -> None:
    assert "AREA[Phase](PHASE3)" in _advanced({"phase": ["PHASE3"]})
    assert "AREA[Phase](PHASE3)" in _advanced({"phase": "PHASE3"})  # scalar accepted too


def test_sponsor_class_and_intervention_type_clauses() -> None:
    assert "AREA[LeadSponsorClass]COVERAGE[FullMatch]INDUSTRY" in _advanced({"sponsorClass": "INDUSTRY"})
    assert "AREA[InterventionType]COVERAGE[FullMatch]DRUG" in _advanced({"interventionType": "DRUG"})


def test_compound_filters_all_compose_and_dont_drop() -> None:
    """The exact hard-battery case: interventional + phase[1,2] + industry all appear in ONE
    AND-joined expression (none silently dropped)."""
    adv = _advanced({"studyType": "INTERVENTIONAL", "phase": ["PHASE1", "PHASE2"],
                     "sponsorClass": "INDUSTRY"})
    assert "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL" in adv
    assert "AREA[Phase](PHASE1 OR PHASE2)" in adv
    assert "AREA[LeadSponsorClass]COVERAGE[FullMatch]INDUSTRY" in adv
    assert adv.count(" AND ") == 2  # three clauses, two joins


def test_invalid_tokens_fail_loud_at_the_boundary() -> None:
    """The SSRF/Essie boundary re-validates every token — an invented one raises, never reaches
    the wire (the anti-hallucination gate, even though the Checker also validates upstream)."""
    with pytest.raises(ValueError):
        _advanced({"phase": ["PHASE9"]})
    with pytest.raises(ValueError):
        _advanced({"sponsorClass": "MEGACORP"})
    with pytest.raises(ValueError):
        _advanced({"interventionType": "TELEPATHY"})
