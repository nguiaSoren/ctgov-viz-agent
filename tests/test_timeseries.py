"""Offline units for the time-series layer (W1c).

Two surfaces, no network:

* ``year_key_fn`` driven through the real ``AggregationCore.page_and_group``
  (combine) over synthetic ClinicalTrials.gov-shaped records — the exact path the
  ``timeseries`` tool uses — then handed to ``finalize_timeseries``. Proves the
  observable contract: a gap year fills with a 0-count / no-citation bucket, a
  future year is flagged ``planned`` (not clamped, not dropped), a missing-date
  record lands in a kept ``MISSING`` bucket, and ``Σ count_trials`` over ALL
  emitted datums reconciles to the record count.
* ``year_key_fn`` alone as a TOTAL key function: mixed precision all normalizes to
  the year, and every malformed/absent-date shape maps to ``MISSING`` without
  raising.
"""

from __future__ import annotations

from app.ctgov.aggregate import AggregationCore
from app.ctgov.citations import build_bucket_citations, is_substring_at
from app.ctgov.timeseries import finalize_timeseries, year_key_fn

_START_PATH = "protocolSection.statusModule.startDateStruct.date"
_CURRENT_YEAR = 2026


# --- synthetic record + fake client -----------------------------------------


def _record(nct_id: str, date: str | None) -> dict:
    """A minimal record: nctId + optional startDateStruct.date (``None`` -> absent)."""
    status: dict = {}
    if date is not None:
        status["startDateStruct"] = {"date": date}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "statusModule": status,
        }
    }


class _FakeClient:
    """Stand-in for ``CTGovClient`` returning canned records (no network)."""

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def iter_studies(self, search_params, *, fields, page_size=1000, max_pages=20):
        return list(self._records), False


def _buckets_from(records: list[dict]) -> list[dict]:
    """Run ``year_key_fn`` through the core, then build the citation-carrying bucket
    dicts exactly as ``tools.timeseries`` would before calling ``finalize``."""
    core = AggregationCore(_FakeClient(records))
    group = core.page_and_group(
        {"query.cond": "x"},
        fields="NCTId|StartDate",
        key_fn=year_key_fn(_START_PATH),
        mode="combine",
    )
    buckets: list[dict] = []
    for bucket in group.buckets:
        citations, contributing_count, truncated = build_bucket_citations(
            bucket.records, _START_PATH, k=20
        )
        buckets.append(
            {
                "value": bucket.value,
                "label": bucket.label,
                "count_trials": bucket.count_trials,
                "count_mentions": bucket.count_mentions,
                "source_ids": [c.nct_id for c in citations],
                "citations": citations,
                "citations_truncated": truncated,
                "contributing_count": contributing_count,
            }
        )
    return buckets


def _by_period(datums: list[dict]) -> dict[str | None, dict]:
    return {d.get("period"): d for d in datums}


# --- finalize: gap / planned / missing / degrade ----------------------------


class TestFinalizeTimeseries:
    def test_gap_year_filled_with_zero_and_no_citations(self):
        records = [
            _record("NCT01", "2019-04-01"),
            _record("NCT02", "2019-06-01"),
            _record("NCT03", "2021-02"),  # gap at 2020
        ]
        datums, notes, degrade = finalize_timeseries(
            _buckets_from(records), current_year=_CURRENT_YEAR
        )
        by_period = _by_period(datums)

        assert set(by_period) == {"2019", "2020", "2021"}
        gap = by_period["2020"]
        assert gap["count_trials"] == 0
        assert gap["count_mentions"] == 0
        assert gap["citations"] == []
        assert gap["source_ids"] == []
        assert gap["contributing_count"] == 0
        assert gap["period"] == "2020"
        # Two non-zero real years -> not degraded.
        assert degrade is False
        assert any("Gap year" in n for n in notes)

    def test_future_year_flagged_planned_not_clamped_not_gapfilled(self):
        records = [
            _record("NCT01", "2020-01-01"),
            _record("NCT02", "2020-05-01"),
            _record("NCT03", "2030-12-18"),  # genuine future
        ]
        datums, notes, _ = finalize_timeseries(
            _buckets_from(records), current_year=_CURRENT_YEAR
        )
        by_period = _by_period(datums)

        planned = by_period["2030"]
        assert planned["planned"] is True
        assert planned["label"] == "2030 (planned)"
        assert planned["period"] == "2030"  # NOT clamped to 2026
        assert planned["value"] == "2030"
        assert planned["count_trials"] == 1
        # Future year is NOT dragged behind a decade of gap-fill (no 2021..2029).
        assert set(by_period) == {"2020", "2030"}
        assert any("planned" in n for n in notes)

    def test_missing_date_record_kept_and_reconciles(self):
        records = [
            _record("NCT01", "2019-04-01"),
            _record("NCT02", "2020-04-01"),
            _record("NCT03", None),  # no start date -> MISSING
        ]
        buckets = _buckets_from(records)
        datums, notes, _ = finalize_timeseries(buckets, current_year=_CURRENT_YEAR)

        missing = [d for d in datums if d["value"] == "MISSING"]
        assert len(missing) == 1
        assert missing[0]["period"] is None
        assert missing[0]["count_trials"] == 1
        # MISSING datum is last.
        assert datums[-1]["value"] == "MISSING"
        # Reconciliation: Σ over ALL datums (real + gap + missing) == record count.
        assert sum(d["count_trials"] for d in datums) == len(records) == 3
        assert any("Missing" in n for n in notes)

    def test_single_real_year_degrades(self):
        records = [_record("NCT01", "2019-04-01"), _record("NCT02", "2019-08-01")]
        datums, _, degrade = finalize_timeseries(
            _buckets_from(records), current_year=_CURRENT_YEAR
        )
        assert degrade is True  # one non-zero real year -> not a trend
        assert {d["period"] for d in datums} == {"2019"}

    def test_full_reconciliation_with_gap_planned_and_missing(self):
        records = [
            _record("NCT01", "2019-04-01"),
            _record("NCT02", "2021-02"),  # gap at 2020
            _record("NCT03", "2030-01-01"),  # planned
            _record("NCT04", None),  # missing
        ]
        datums, _, _ = finalize_timeseries(
            _buckets_from(records), current_year=_CURRENT_YEAR
        )
        # gap year contributes 0; every real trial is represented exactly once.
        assert sum(d["count_trials"] for d in datums) == 4
        # sorted ascending by year, MISSING last (period None).
        periods = [d["period"] for d in datums]
        assert periods == ["2019", "2020", "2021", "2030", None]

    def test_planned_bucket_keeps_its_citations(self):
        records = [_record("NCT01", "2020-01-01"), _record("NCT02", "2030-01-01")]
        datums, _, _ = finalize_timeseries(
            _buckets_from(records), current_year=_CURRENT_YEAR
        )
        planned = _by_period(datums)["2030"]
        assert len(planned["citations"]) == 1
        citation = planned["citations"][0]
        assert citation.field_path == _START_PATH
        # Excerpt is string-extracted, round-trips against a reconstructed record.
        record = {
            "protocolSection": {
                "identificationModule": {"nctId": citation.nct_id},
                "statusModule": {"startDateStruct": {"date": citation.value}},
            }
        }
        assert is_substring_at(record, _START_PATH, citation.matched_value)


# --- year_key_fn: normalization + TOTAL-ness --------------------------------


class TestYearKeyFn:
    def test_mixed_precision_all_normalize_to_year(self):
        key = year_key_fn(_START_PATH)
        assert key(_record("NCT01", "2021-02")) == [("2021", "2021")]
        assert key(_record("NCT02", "1998")) == [("1998", "1998")]
        assert key(_record("NCT03", "2019-04-12")) == [("2019", "2019")]

    def test_missing_and_malformed_map_to_missing_without_raising(self):
        key = year_key_fn(_START_PATH)
        missing = [("MISSING", "Missing (no start date)")]

        # Absent struct, present-but-None struct, empty/garbage/non-string dates,
        # and a non-dict protocolSection — every one is MISSING, none raises.
        assert key(_record("NCT01", None)) == missing
        assert key(
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT02"},
                    "statusModule": {"startDateStruct": None},
                }
            }
        ) == missing
        assert key(_record("NCT03", "")) == missing
        assert key(_record("NCT04", "   ")) == missing
        assert key(_record("NCT05", "not-a-date")) == missing
        assert key(_record("NCT06", "2020-13")) == missing  # month out of range
        assert key({"protocolSection": None}) == missing
        assert key({}) == missing
        # A bare-string statusModule must not crash the descent.
        assert key({"protocolSection": {"statusModule": "oops"}}) == missing
