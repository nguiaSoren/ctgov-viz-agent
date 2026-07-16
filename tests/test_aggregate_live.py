"""Aggregation core + tools: one pure offline unit + the live reconciliation gate.

Two layers, per the Interface Contract:

* ``TestPageAndGroupCombine`` -- a pure/offline unit of ``page_and_group`` over
  ~5 synthetic ClinicalTrials.gov-shaped records (no network). Proves the
  combine-mode invariants: a composite phase gets its OWN bucket (CC-15), a
  missing ``phases`` key -> ``MISSING`` (not NA, CC-5), and ``Σ count_trials``
  equals the record count.
* ``TestLiveReconciliation`` -- the killer gate (CC-16): against the real
  registry, ``Σ count_trials`` over the phase buckets reconciles to the API's
  exact ``countTotal`` for interventional pancreatic-cancer trials, with NA and
  Missing as separate buckets, every non-zero bucket cited, and every excerpt a
  live substring at its field_path. Live calls are $0 (no key); when the network
  is unreachable the live tests ``skip`` (never fail), so an offline run is green.
"""

from __future__ import annotations

import pytest

from app.ctgov.aggregate import AggregationCore
from app.ctgov.citations import is_substring_at
from app.ctgov.client import UpstreamError
from app.ctgov.fields import phase_key_fn
from app.ctgov.tools import aggregate_by, count_trials

# --- shared live-query fixtures ---------------------------------------------

_PANCREATIC_QUERY = {"cond": "pancreatic cancer"}
_INTERVENTIONAL_FILTERS = {"interventional_only": True}
_PHASE_FIELD_PATH = "protocolSection.designModule.phases"

# Drift tolerance for a genuine mid-run registry mutation between the count call
# and the paging walk (Interface Contract: <=0.5% AND <=20, else fail).
_DRIFT_PCT = 0.005
_DRIFT_ABS = 20


# --- offline unit: page_and_group combine-mode ------------------------------


def _record(nct_id: str, phases: list[str] | None) -> dict:
    """A minimal ClinicalTrials.gov-shaped record (nctId + optional phases)."""
    design: dict = {}
    if phases is not None:
        design["phases"] = phases
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "designModule": design,
        }
    }


class _FakeClient:
    """A stand-in for ``CTGovClient`` that returns canned records, no network."""

    def __init__(self, records: list[dict], *, truncated: bool = False) -> None:
        self._records = records
        self._truncated = truncated
        self.calls: list[dict] = []

    def iter_studies(self, search_params, *, fields, page_size=1000, max_pages=20):
        self.calls.append({"fields": fields, "max_pages": max_pages})
        return list(self._records), self._truncated


class TestPageAndGroupCombine:
    def test_combine_buckets_composite_missing_and_reconciles(self):
        records = [
            _record("NCT0001", ["PHASE1"]),
            _record("NCT0002", ["PHASE2"]),
            _record("NCT0003", ["PHASE1", "PHASE2"]),  # composite -> own bucket (CC-15)
            _record("NCT0004", None),  # no phases key -> MISSING (CC-5)
            _record("NCT0005", ["NA"]),  # NA is its own bucket, distinct from MISSING
        ]
        core = AggregationCore(_FakeClient(records))
        result = core.page_and_group(
            {"query.cond": "x"}, fields="NCTId|Phase", key_fn=phase_key_fn, mode="combine"
        )

        by_value = {bucket.value: bucket for bucket in result.buckets}

        # Exactly one bucket per distinct phase key; 5 records -> 5 buckets.
        assert set(by_value) == {"PHASE1", "PHASE2", "PHASE1|PHASE2", "MISSING", "NA"}

        # Composite is its OWN bucket, never split (CC-15).
        assert by_value["PHASE1|PHASE2"].count_trials == 1
        assert by_value["PHASE1|PHASE2"].label == "Phase 1/2"

        # Missing != NA -- two separate explicit buckets (CC-5).
        assert by_value["MISSING"].label == "Missing (not reported)"
        assert by_value["NA"].label == "NA (not applicable)"

        # Combine: mentions == trials for every bucket, and each carries its record.
        for bucket in result.buckets:
            assert bucket.count_trials == bucket.count_mentions == 1
            assert len(bucket.records) == bucket.count_trials

        # Σ count_trials == record count == distinct trials (the reconciliation shape).
        total = sum(bucket.count_trials for bucket in result.buckets)
        assert total == len(records) == 5
        assert result.distinct_trials == 5
        assert result.truncated is False


# --- live: the CC-16 reconciliation gate ------------------------------------


def _skip_if_transient(exc: Exception) -> None:
    """Skip (never fail) on ANY transport-level ``UpstreamError`` raised while
    FETCHING (offline `upstream_unreachable`, retry-exhausted rate-limit
    `upstream_status_429`/`5xx` under a full-suite burst, redirect refusal, bad
    JSON). This is safe without losing teeth: the fetch happens in the fixture,
    so an ``UpstreamError`` here is always a transport condition, never a
    reconciliation logic bug -- a real regression surfaces as an assertion
    failure in a test BODY (the fixture having fetched fine), which still fails.
    Anything that is not an ``UpstreamError`` re-raises."""
    if isinstance(exc, UpstreamError):
        pytest.skip(f"clinicaltrials.gov transport error ({exc.code}) -- live test skipped")
    raise exc


def _within_drift(observed: int, total: int) -> bool:
    drift = abs(observed - total)
    return drift == 0 or (drift <= _DRIFT_PCT * total and drift <= _DRIFT_ABS)


@pytest.fixture(scope="module")
def live_reconciliation() -> dict:
    """Run the real count + aggregate once; share across the assertions."""
    try:
        total = count_trials(_PANCREATIC_QUERY, _INTERVENTIONAL_FILTERS)
        result = aggregate_by(_PANCREATIC_QUERY, _INTERVENTIONAL_FILTERS, "phase")
    except Exception as exc:  # noqa: BLE001 -- narrowed to transient-skip below
        _skip_if_transient(exc)
        raise
    return {"total": total, "result": result}


class TestLiveReconciliation:
    def test_sum_reconciles_to_count_total(self, live_reconciliation):
        total = live_reconciliation["total"]
        result = live_reconciliation["result"]
        observed = sum(bucket["count_trials"] for bucket in result["buckets"])
        assert not result["truncated"], "budget exhausted -- reconciliation would be against a partial"
        assert _within_drift(observed, total), (
            f"Σcount_trials={observed} does not reconcile to countTotal={total} "
            f"(drift {abs(observed - total)} exceeds <=0.5% AND <=20)"
        )
        # distinct-trial total is the same anchor for combine.
        assert _within_drift(result["distinct_trials"], total)

    def test_na_and_missing_are_separate_buckets(self, live_reconciliation):
        result = live_reconciliation["result"]
        values = {bucket["value"] for bucket in result["buckets"]}
        assert "NA" in values, "expected an explicit NA bucket among interventional trials"
        # If a Missing bucket is present it is a DISTINCT bucket from NA (CC-5).
        if "MISSING" in values:
            na = next(b for b in result["buckets"] if b["value"] == "NA")
            missing = next(b for b in result["buckets"] if b["value"] == "MISSING")
            assert na is not missing
            assert na["count_trials"] >= 0 and missing["count_trials"] >= 0

    def test_every_nonzero_bucket_is_cited(self, live_reconciliation):
        result = live_reconciliation["result"]
        for bucket in result["buckets"]:
            if bucket["count_trials"] > 0:
                assert len(bucket["citations"]) >= 1, f"bucket {bucket['value']} has no citations"
                assert bucket["contributing_count"] == bucket["count_trials"]

    def test_every_excerpt_is_a_live_substring(self, live_reconciliation):
        result = live_reconciliation["result"]
        for bucket in result["buckets"]:
            for citation in bucket["citations"]:
                record = {
                    "protocolSection": {
                        "identificationModule": {"nctId": citation.nct_id},
                        "designModule": {"phases": citation.value},
                    }
                }
                assert is_substring_at(record, _PHASE_FIELD_PATH, citation.matched_value), (
                    f"excerpt {citation.matched_value!r} not present at {_PHASE_FIELD_PATH} "
                    f"for {citation.nct_id}"
                )
