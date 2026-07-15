"""Offline units for the duration histogram layer (W1c).

Pure/offline over synthetic ClinicalTrials.gov-shaped records (no network). Proves
the observable contract: durations land in the right half-open month bins, empty
bins are kept, a trial missing completion (or with a negative duration) goes to a
kept ``Unknown (undated)`` bucket, the sum reconciles to the record count, bucket
citations point at the START date field and round-trip, and a malformed record
never raises (TOTAL).
"""

from __future__ import annotations

from app.ctgov.citations import is_substring_at
from app.ctgov.histogram import bin_durations

_START_PATH = "protocolSection.statusModule.startDateStruct.date"
_END_PATH = "protocolSection.statusModule.completionDateStruct.date"


def _record(nct_id: str, start: str | None, end: str | None) -> dict:
    """Minimal record: nctId + optional start/completion date structs."""
    status: dict = {}
    if start is not None:
        status["startDateStruct"] = {"date": start}
    if end is not None:
        status["completionDateStruct"] = {"date": end}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "statusModule": status,
        }
    }


def _by_label(datums: list[dict]) -> dict[str, dict]:
    return {d["label"]: d for d in datums}


class TestBinDurations:
    def test_binning_across_all_six_bins(self):
        records = [
            _record("NCT01", "2020-01", "2020-04"),  # 3 mo  -> 0–6 mo
            _record("NCT02", "2020-01", "2020-09"),  # 8 mo  -> 6–12 mo
            _record("NCT03", "2020-01", "2021-06"),  # 17 mo -> 1–2 yr
            _record("NCT04", "2020-01", "2023-01"),  # 36 mo -> 2–4 yr
            _record("NCT05", "2020-01", "2026-01"),  # 72 mo -> 4–10 yr
            _record("NCT06", "2010-01", "2025-01"),  # 180 mo -> 10+ yr
        ]
        datums, notes = bin_durations(records, start_path=_START_PATH, end_path=_END_PATH)
        by_label = _by_label(datums)

        for label in ("0–6 mo", "6–12 mo", "1–2 yr", "2–4 yr", "4–10 yr", "10+ yr"):
            assert by_label[label]["count_trials"] == 1, label
            assert by_label[label]["count_mentions"] == 1

        # bin edges are carried in months (float), open upper edge on the last bin.
        assert by_label["0–6 mo"]["bin_start"] == 0.0
        assert by_label["0–6 mo"]["bin_end"] == 6.0
        assert by_label["10+ yr"]["bin_start"] == 120.0
        assert by_label["10+ yr"]["bin_end"] is None

        # All six real trials binned, none undated; Σ == record count.
        assert "Unknown (undated)" not in by_label
        assert sum(d["count_trials"] for d in datums) == len(records) == 6
        assert any("R-16" in n for n in notes)

    def test_empty_bins_are_kept_with_no_citations(self):
        # A single 3-month trial: the other five bins must still appear, empty.
        datums, _ = bin_durations([_record("NCT01", "2020-01", "2020-04")])
        by_label = _by_label(datums)
        assert len(by_label) == 6  # all six default bins present, no undated
        empty = by_label["10+ yr"]
        assert empty["count_trials"] == 0
        assert empty["citations"] == []
        assert empty["source_ids"] == []
        assert empty["contributing_count"] == 0

    def test_missing_completion_goes_to_undated_and_reconciles(self):
        records = [
            _record("NCT01", "2020-01", "2020-06"),  # 5 mo -> 0–6 mo
            _record("NCT02", "2020-01", None),  # no completion -> undated
        ]
        datums, notes = bin_durations(records, start_path=_START_PATH, end_path=_END_PATH)
        by_label = _by_label(datums)

        undated = by_label["Unknown (undated)"]
        assert undated["value"] == "UNDATED"
        assert undated["count_trials"] == 1
        assert undated["bin_start"] is None and undated["bin_end"] is None
        # undated bucket is appended last.
        assert datums[-1]["label"] == "Unknown (undated)"
        # Reconciliation: bins + undated == record count.
        assert sum(d["count_trials"] for d in datums) == len(records) == 2
        assert any("undated" in n for n in notes)

    def test_negative_duration_goes_to_undated(self):
        # Completion strictly before start -> impossible duration -> undated.
        records = [_record("NCT01", "2022-01", "2020-01")]
        datums, _ = bin_durations(records)
        by_label = _by_label(datums)
        assert by_label["Unknown (undated)"]["count_trials"] == 1
        assert all(d["count_trials"] == 0 for d in datums if d["value"] != "UNDATED")

    def test_mixed_precision_month_defaults_to_january(self):
        # year-only start ("2020" -> Jan) to "2020-07" is a 6-month span -> 6–12 mo.
        records = [_record("NCT01", "2020", "2020-07")]
        datums, _ = bin_durations(records)
        by_label = _by_label(datums)
        assert by_label["6–12 mo"]["count_trials"] == 1
        assert by_label["0–6 mo"]["count_trials"] == 0

    def test_boundary_duration_is_half_open(self):
        # exactly 6 months belongs to [6,12), not [0,6).
        records = [_record("NCT01", "2020-01", "2020-07")]  # 6 mo
        by_label = _by_label(bin_durations(records)[0])
        assert by_label["6–12 mo"]["count_trials"] == 1
        assert by_label["0–6 mo"]["count_trials"] == 0

    def test_citations_cite_start_path_and_round_trip(self):
        records = [_record("NCT01", "2019-04-12", "2020-01-01")]
        datums, _ = bin_durations(records, start_path=_START_PATH, end_path=_END_PATH)
        bucket = _by_label(datums)["6–12 mo"]  # 9 months
        assert bucket["count_trials"] == 1
        citation = bucket["citations"][0]
        assert citation.field_path == _START_PATH
        assert citation.nct_id == "NCT01"
        assert is_substring_at(_record("NCT01", citation.value, None), _START_PATH, citation.excerpt)

    def test_distinct_nctid_dedup(self):
        # A duplicate page row (same nctId, same dates) must not double-count (K3).
        records = [
            _record("NCT01", "2020-01", "2020-04"),
            _record("NCT01", "2020-01", "2020-04"),
        ]
        datums, _ = bin_durations(records)
        by_label = _by_label(datums)
        assert by_label["0–6 mo"]["count_trials"] == 1
        assert by_label["0–6 mo"]["contributing_count"] == 1

    def test_total_on_malformed_records(self):
        records = [
            _record("NCT01", "2020-01", "2020-06"),  # good -> 0–6 mo
            {  # nctId present, statusModule a bare string -> dates unresolvable -> undated
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT02"},
                    "statusModule": "oops",
                }
            },
            {  # present-but-None startDateStruct -> undated
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT03"},
                    "statusModule": {"startDateStruct": None, "completionDateStruct": {"date": "2021"}},
                }
            },
            {"protocolSection": None},  # no nctId -> not counted at all
            "not-a-record",  # not even a dict -> not counted
        ]
        datums, _ = bin_durations(records)  # must not raise
        by_label = _by_label(datums)
        assert by_label["0–6 mo"]["count_trials"] == 1
        # NCT02 + NCT03 are undatable; the id-less / non-dict records are dropped.
        assert by_label["Unknown (undated)"]["count_trials"] == 2

    def test_custom_bins_generate_labels(self):
        records = [_record("NCT01", "2020-01", "2020-03")]  # 2 mo
        datums, _ = bin_durations(records, bins=[(0, 12), (12, None)])
        by_label = _by_label(datums)
        assert by_label["0–12 mo"]["count_trials"] == 1
        assert by_label["12+ mo"]["count_trials"] == 0
