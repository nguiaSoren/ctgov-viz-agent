"""Regression tests for the D1 security/data-helper hardening pass.

Covers, one test class per fix:

* FIX 1 — ``CTGovClient`` SSRF base-pin bypass (userinfo trick, suffix trick,
  non-https) is closed; the legitimate base URL still works.
* FIX 2 — ``check_plan`` never raises on an unhashable/malformed filter value.
* FIX 3 — ``check_plan`` rejects an unknown filter key (anti-hallucination gate).
* FIX 4 — ``is_substring_at`` verifies a genuinely-present value at a non-first
  list index (multi-value trial), not just element 0.
* FIX 5 — ``dates`` helpers reject a bad month / empty / ``None`` input with a
  clear ``ValueError`` instead of crashing or silently mis-parsing.
"""

from __future__ import annotations

import pytest

from app.api.schemas import ChartType
from app.ctgov import citations, dates
from app.ctgov.client import CTGovClient
from app.plan.checker import check_plan
from app.plan.models import Plan

# --- FIX 1: SSRF base-pin bypass -------------------------------------------


class TestCTGovClientSSRFGuard:
    def test_userinfo_trick_is_rejected(self):
        """``https://clinicaltrials.gov@evil.com/...`` — the real host is
        evil.com; clinicaltrials.gov is just basic-auth userinfo."""
        with pytest.raises(ValueError):
            CTGovClient(base_url="https://clinicaltrials.gov@evil.com/api/v2")

    def test_suffix_trick_is_rejected(self):
        """``https://clinicaltrials.gov.evil.com/...`` — a subdomain of
        evil.com, not the pinned host."""
        with pytest.raises(ValueError):
            CTGovClient(base_url="https://clinicaltrials.gov.evil.com/api/v2")

    def test_non_https_is_rejected(self):
        with pytest.raises(ValueError):
            CTGovClient(base_url="http://clinicaltrials.gov/api/v2")

    def test_legit_base_url_is_accepted(self):
        client = CTGovClient(base_url="https://clinicaltrials.gov/api/v2")
        assert client.base_url == "https://clinicaltrials.gov/api/v2"


# --- FIX 2 / FIX 3: checker hardening ---------------------------------------


def _good_distribution_plan(filters: dict | None = None) -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        field="phase",
        chart_type=ChartType.BAR,
        alternates=[ChartType.HISTOGRAM],
        filters=filters if filters is not None else {},
    )


class TestCheckPlanFilterHardening:
    def test_unhashable_filter_value_does_not_raise(self):
        """FIX 2: a dict filter value used to raise TypeError: unhashable
        type: 'dict'. A checker must never raise — it returns ok=False."""
        plan = _good_distribution_plan(filters={"phase": {"x": 1}})
        result = check_plan(plan)  # must not raise
        assert result.ok is False
        assert result.reason

    def test_unknown_filter_key_is_rejected(self):
        """FIX 3: a typo'd/invented filter key must fail, not pass silently —
        the checker is the anti-hallucination gate."""
        plan = _good_distribution_plan(filters={"phaze": "PHASE1"})
        result = check_plan(plan)
        assert result.ok is False
        assert "phaze" in (result.reason or "")

    def test_valid_distribution_plan_with_empty_filters_still_passes(self):
        result = check_plan(_good_distribution_plan(filters={}))
        assert result.ok is True
        assert result.normalized_plan is not None


# --- FIX 4: citation round-trip on non-first list element -------------------


class TestIsSubstringAtMultiValue:
    _RECORD = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Drug A"},
                    {"type": "BIOLOGICAL", "name": "Vaccine B"},
                ]
            },
        }
    }
    _PATH = "protocolSection.armsInterventionsModule.interventions[].type"

    def test_second_element_value_verifies_true(self):
        assert citations.is_substring_at(self._RECORD, self._PATH, "BIOLOGICAL") is True

    def test_first_element_value_still_verifies_true(self):
        assert citations.is_substring_at(self._RECORD, self._PATH, "DRUG") is True

    def test_fabricated_value_verifies_false(self):
        assert citations.is_substring_at(self._RECORD, self._PATH, "FABRICATED") is False


# --- FIX 5: date guards ------------------------------------------------------


class TestDateGuards:
    def test_month_13_raises(self):
        with pytest.raises(ValueError):
            dates.parse_ct_date("2015-13")

    def test_month_0_raises(self):
        with pytest.raises(ValueError):
            dates.parse_ct_date("2015-00")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            dates.parse_ct_date("")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            dates.parse_ct_date(None)  # type: ignore[arg-type]

    def test_precision_of_rejects_empty_and_none(self):
        with pytest.raises(ValueError):
            dates.precision_of("")
        with pytest.raises(ValueError):
            dates.precision_of(None)  # type: ignore[arg-type]

    def test_valid_dates_still_parse_exactly_as_before(self):
        assert dates.parse_ct_date("2015") == (2015, None)
        assert dates.parse_ct_date("2015-05") == (2015, 5)
        assert dates.parse_ct_date("2015-05-10") == (2015, 5)
        assert dates.parse_ct_date("2030-12") == (2030, 12)

    def test_precision_of_still_correct_for_valid_dates(self):
        assert dates.precision_of("2015") == "year"
        assert dates.precision_of("2015-05") == "month"
        assert dates.precision_of("2015-05-10") == "day"
