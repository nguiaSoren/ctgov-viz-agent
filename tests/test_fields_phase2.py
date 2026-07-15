"""Phase 2 explode fields -- offline units for ``app.ctgov.fields`` (no network).

Covers the two Phase-2 additions (``country`` + ``interventionType`` explode
specs) and the shared ``count_desc_sort_key``:

* ``country_key_fn`` -- DISTINCT-country dedupe within a trial (CC-13: US, US, UK
  -> one distinct-trial count each), no-location -> ``UNKNOWN``, raw free-string
  value (no ISO), routed through ``AggregationCore.page_and_group`` in explode
  mode (the real consumption path, ``_FakeClient`` stands in for the network).
* ``intervention_type_key_fn`` -- distinct-type dedupe, human labels, unknown
  token -> itself, no-interventions -> ``UNKNOWN``.
* ``count_desc_sort_key`` -- ``-count_trials`` then value, with sentinels
  (``UNKNOWN``/``NA``/``MISSING``/``Other``) pinned LAST even with a huge count.
* TOTAL discipline (K1/B2): ``protocolSection: null`` / a bare string / a
  present-but-``None`` / a non-list / non-dict elements must all degrade, never
  raise -- and one bad record must not sink a whole page_and_group batch.
* The ``phase`` spec is left behavior-identical (its ``sort_key`` reproduces the
  hardcoded ``phase_sort_key(bucket["value"])``; ``top_n`` stays ``None``).
"""

from __future__ import annotations

from app import config
from app.ctgov.aggregate import AggregationCore
from app.ctgov.fields import (
    FIELD_SPEC,
    count_desc_sort_key,
    country_key_fn,
    intervention_type_key_fn,
    phase_sort_key,
)

# --- synthetic ClinicalTrials.gov-shaped records ----------------------------


def _country_record(nct_id: str, countries: list[str] | None) -> dict:
    """A record whose locations carry ``countries`` (``None`` -> no locations key)."""
    module: dict = {}
    if countries is not None:
        module["locations"] = [{"country": c} for c in countries]
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "contactsLocationsModule": module,
        }
    }


def _intervention_record(nct_id: str, types: list[str] | None) -> dict:
    """A record whose interventions carry ``types`` (``None`` -> no interventions key)."""
    module: dict = {}
    if types is not None:
        module["interventions"] = [{"type": t, "name": f"drug-{t}"} for t in types]
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "armsInterventionsModule": module,
        }
    }


class _FakeClient:
    """A stand-in for ``CTGovClient`` returning canned records, no network."""

    def __init__(self, records: list[dict], *, truncated: bool = False) -> None:
        self._records = records
        self._truncated = truncated
        self.calls: list[dict] = []

    def iter_studies(self, search_params, *, fields, page_size=1000, max_pages=20):
        self.calls.append({"fields": fields, "max_pages": max_pages})
        return list(self._records), self._truncated


def _group_country(records: list[dict]) -> AggregationCore:
    return AggregationCore(_FakeClient(records))


# --- country: dedupe-per-trial (CC-13) --------------------------------------


class TestCountryExplode:
    def test_us_twice_plus_uk_dedupes_within_trial(self):
        # One trial listing US, US, UK -> US=1 & UK=1 distinct-trial counts, and
        # US counts ONE mention despite being listed twice (per-trial dedupe).
        records = [_country_record("NCT01", ["United States", "United States", "United Kingdom"])]
        result = _group_country(records).page_and_group(
            {"query.cond": "x"},
            fields="NCTId|LocationCountry",
            key_fn=country_key_fn,
            mode="explode",
        )
        by_value = {b.value: b for b in result.buckets}

        assert set(by_value) == {"United States", "United Kingdom"}
        assert by_value["United States"].count_trials == 1
        assert by_value["United States"].count_mentions == 1  # US listed 2x -> 1 mention (CC-13)
        assert by_value["United Kingdom"].count_trials == 1
        # value == label == the raw free string (no ISO canon).
        assert by_value["United States"].label == "United States"
        assert result.distinct_trials == 1

    def test_country_aggregates_distinct_trials_across_records(self):
        records = [
            _country_record("NCT01", ["United States", "United Kingdom"]),
            _country_record("NCT02", ["United States"]),
        ]
        result = _group_country(records).page_and_group(
            {}, fields="NCTId|LocationCountry", key_fn=country_key_fn, mode="explode"
        )
        by_value = {b.value: b for b in result.buckets}
        assert by_value["United States"].count_trials == 2  # NCT01 + NCT02
        assert by_value["United Kingdom"].count_trials == 1
        assert result.distinct_trials == 2

    def test_no_location_maps_to_unknown(self):
        records = [
            _country_record("NCT01", None),  # no locations key at all
            _country_record("NCT02", []),  # empty locations list
        ]
        result = _group_country(records).page_and_group(
            {}, fields="NCTId|LocationCountry", key_fn=country_key_fn, mode="explode"
        )
        by_value = {b.value: b for b in result.buckets}
        assert set(by_value) == {"UNKNOWN"}
        assert by_value["UNKNOWN"].label == "Unknown"
        assert by_value["UNKNOWN"].count_trials == 2

    def test_free_string_value_preserved_verbatim(self):
        # Free string with punctuation + unicode round-trips exactly (no canon).
        assert country_key_fn(_country_record("NCT01", ["Turkey (Türkiye)"])) == [
            ("Turkey (Türkiye)", "Turkey (Türkiye)")
        ]
        assert country_key_fn(_country_record("NCT01", ["South Korea"])) == [
            ("South Korea", "South Korea")
        ]

    def test_location_missing_or_empty_country_degrades(self):
        # A location dict with no / empty / non-string country contributes nothing;
        # a mixed list keeps only the usable country.
        rec = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT01"},
                "contactsLocationsModule": {
                    "locations": [
                        None,
                        {"country": None},
                        {"country": ""},
                        {"no_country": 1},
                        {"country": 123},
                        {"country": "Japan"},
                    ]
                },
            }
        }
        assert country_key_fn(rec) == [("Japan", "Japan")]


# --- interventionType: distinct-type dedupe + labels ------------------------


class TestInterventionTypeExplode:
    def test_explode_dedupes_types_and_labels(self):
        records = [
            _intervention_record("NCT01", ["DRUG", "DRUG", "DEVICE"]),  # DRUG twice
            _intervention_record("NCT02", ["DRUG"]),
            _intervention_record("NCT03", None),  # no interventions -> UNKNOWN
        ]
        result = AggregationCore(_FakeClient(records)).page_and_group(
            {}, fields="NCTId|InterventionType", key_fn=intervention_type_key_fn, mode="explode"
        )
        by_value = {b.value: b for b in result.buckets}

        assert set(by_value) == {"DRUG", "DEVICE", "UNKNOWN"}
        assert by_value["DRUG"].count_trials == 2  # NCT01 + NCT02
        assert by_value["DRUG"].count_mentions == 2  # NCT01's double-DRUG -> 1 mention
        assert by_value["DRUG"].label == "Drug"
        assert by_value["DEVICE"].count_trials == 1
        assert by_value["DEVICE"].label == "Device"
        assert by_value["UNKNOWN"].count_trials == 1
        assert by_value["UNKNOWN"].label == "Unknown"
        assert result.distinct_trials == 3

    def test_human_labels_and_unknown_token_fallback(self):
        assert intervention_type_key_fn(_intervention_record("NCT01", ["DRUG"])) == [("DRUG", "Drug")]
        assert intervention_type_key_fn(_intervention_record("NCT01", ["DIAGNOSTIC_TEST"])) == [
            ("DIAGNOSTIC_TEST", "Diagnostic Test")
        ]
        # An unknown/garbage token maps to itself (label == value), never raises.
        assert intervention_type_key_fn(_intervention_record("NCT01", ["MADE_UP"])) == [
            ("MADE_UP", "MADE_UP")
        ]

    def test_no_interventions_maps_to_unknown(self):
        assert intervention_type_key_fn(_intervention_record("NCT01", None)) == [("UNKNOWN", "Unknown")]
        assert intervention_type_key_fn(_intervention_record("NCT01", [])) == [("UNKNOWN", "Unknown")]


# --- count_desc_sort_key: sentinels pinned last -----------------------------


class TestCountDescSortKey:
    def test_pins_unknown_and_other_last_despite_huge_count(self):
        buckets = [
            {"value": "United States", "count_trials": 50},
            {"value": "UNKNOWN", "count_trials": 100_000},  # huge, still pinned last
            {"value": "United Kingdom", "count_trials": 30},
            {"value": "Other", "count_trials": 999_999},  # huge, still pinned last
            {"value": "France", "count_trials": 30},
        ]
        ordered = [b["value"] for b in sorted(buckets, key=count_desc_sort_key)]
        # Real categories by count desc (30-count tie broken by value: France < UK),
        # then BOTH sentinels regardless of their (huge) counts.
        assert ordered == ["United States", "France", "United Kingdom", "Other", "UNKNOWN"]

    def test_all_four_sentinels_pinned_after_real_categories(self):
        buckets = [
            {"value": "DRUG", "count_trials": 10},
            {"value": "NA", "count_trials": 5_000},
            {"value": "MISSING", "count_trials": 5_000},
            {"value": "UNKNOWN", "count_trials": 5_000},
            {"value": "Other", "count_trials": 5_000},
            {"value": "DEVICE", "count_trials": 8},
        ]
        ordered = [b["value"] for b in sorted(buckets, key=count_desc_sort_key)]
        assert ordered[:2] == ["DRUG", "DEVICE"]  # real categories first, by count desc
        assert set(ordered[2:]) == {"NA", "MISSING", "UNKNOWN", "Other"}
        # Equal-count sentinels order deterministically by value.
        assert ordered[2:] == ["MISSING", "NA", "Other", "UNKNOWN"]

    def test_total_on_malformed_bucket_dicts(self):
        # A missing / non-int count and a missing / non-str value never raise.
        assert count_desc_sort_key({}) == (0, 0, "")
        assert count_desc_sort_key({"value": None, "count_trials": None}) == (0, 0, "None")
        assert count_desc_sort_key({"value": 5, "count_trials": "abc"}) == (0, 0, "5")
        # A numeric-string count is coerced; a sentinel value is still pinned.
        assert count_desc_sort_key({"value": "UNKNOWN", "count_trials": "12"}) == (1, -12, "UNKNOWN")


# --- TOTAL discipline: never raise on malformed / live-shaped garbage --------

_MALFORMED_RECORDS = [
    {},  # empty
    {"protocolSection": None},  # protocolSection is present-but-None (the .get trap)
    {"protocolSection": "not-a-dict"},  # a bare string is ONE token, not iterated
    {"protocolSection": 123},
    {"protocolSection": []},
]


class TestTotality:
    def test_country_key_fn_total_on_malformed(self):
        # Every malformed shape degrades to the Unknown bucket, never raises.
        country_specific = _MALFORMED_RECORDS + [
            {"protocolSection": {"contactsLocationsModule": None}},
            {"protocolSection": {"contactsLocationsModule": "x"}},
            {"protocolSection": {"contactsLocationsModule": {"locations": None}}},  # present-but-None
            {"protocolSection": {"contactsLocationsModule": {"locations": "United States"}}},  # non-list
            {"protocolSection": {"contactsLocationsModule": {"locations": [None, 5, "x", {}]}}},
        ]
        for record in country_specific:
            assert country_key_fn(record) == [("UNKNOWN", "Unknown")]

    def test_intervention_type_key_fn_total_on_malformed(self):
        intervention_specific = _MALFORMED_RECORDS + [
            {"protocolSection": {"armsInterventionsModule": None}},
            {"protocolSection": {"armsInterventionsModule": "x"}},
            {"protocolSection": {"armsInterventionsModule": {"interventions": None}}},  # present-but-None
            {"protocolSection": {"armsInterventionsModule": {"interventions": "DRUG"}}},  # non-list
            {"protocolSection": {"armsInterventionsModule": {"interventions": [None, 5, {"type": None}, {}]}}},
        ]
        for record in intervention_specific:
            assert intervention_type_key_fn(record) == [("UNKNOWN", "Unknown")]

    def test_one_bad_record_does_not_sink_the_batch(self):
        # A malformed record mid-batch must not raise inside page_and_group; the
        # good records still bucket, and an id-less malformed record can't be
        # counted as a distinct trial (no nctId to reconcile against, K3).
        records = [
            _country_record("NCT01", ["United States"]),
            {"protocolSection": None},  # malformed + id-less -> Unknown mention only
            _country_record("NCT02", None),  # id-bearing but location-less -> Unknown trial
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT03"},
                    "contactsLocationsModule": {"locations": "bad-non-list"},
                }
            },
        ]
        result = _group_country(records).page_and_group(
            {}, fields="NCTId|LocationCountry", key_fn=country_key_fn, mode="explode"
        )
        by_value = {b.value: b for b in result.buckets}
        assert by_value["United States"].count_trials == 1
        # NCT02 + NCT03 are id-bearing Unknowns; the id-less record adds no trial.
        assert by_value["UNKNOWN"].count_trials == 2
        assert result.distinct_trials == 3  # NCT01, NCT02, NCT03


# --- phase spec left behavior-identical + new specs registered --------------


class TestFieldSpecRegistrations:
    def test_phase_spec_behavior_preserved(self):
        phase = FIELD_SPEC["phase"]
        assert phase.mode == "combine"
        assert phase.top_n is None
        assert phase.fields_projection == "NCTId|Phase|BriefTitle"
        # sort_key reproduces tools.py's hardcoded phase_sort_key(bucket["value"]).
        for value in ("MISSING", "NA", "PHASE1", "PHASE1|PHASE2", "PHASE4"):
            assert phase.sort_key({"value": value}) == phase_sort_key(value)

    def test_country_spec_registered(self):
        country = FIELD_SPEC["country"]
        assert country.mode == "explode"
        assert country.top_n == config.TOP_N_CATEGORIES  # 50 (P5-TOPN reconciled 15 → spec's 50)
        assert country.field_path == "protocolSection.contactsLocationsModule.locations[].country"
        assert country.fields_projection == "NCTId|LocationCountry|BriefTitle"
        assert country.sort_key is count_desc_sort_key
        assert country.key_fn is country_key_fn

    def test_intervention_type_spec_registered(self):
        itype = FIELD_SPEC["interventionType"]
        assert itype.mode == "explode"
        assert itype.top_n is None
        assert itype.field_path == "protocolSection.armsInterventionsModule.interventions[].type"
        assert itype.fields_projection == "NCTId|InterventionType|BriefTitle"
        assert itype.sort_key is count_desc_sort_key
        assert itype.key_fn is intervention_type_key_fn
