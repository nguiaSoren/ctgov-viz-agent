"""Regression tests for the Phase-1 adversarial-review findings (2026-07-15).

Each test pins a defect the 6-lens adversarial pass found, so the fix can't
silently regress. All offline (fake client / direct calls) — no network.
Grouped by finding code (K1..K5 in tasks/LESSONS.md).
"""

from __future__ import annotations

import httpx
import pytest

from app.ctgov.aggregate import AggregationCore
from app.ctgov.citations import is_substring_at
from app.ctgov.client import CTGovClient, UpstreamError
from app.ctgov.fields import phase_key_fn
from app.viz.review import _excerpt_in_value, deterministic_precheck
from tests.test_review import _datum, _ok_spec  # reuse the spec builders

_PHASES = "protocolSection.designModule.phases"


def _rec(nct_id: str | None, phases: object) -> dict:
    ident = {"nctId": nct_id} if nct_id is not None else {}
    design = {"phases": phases} if phases is not None else {}
    return {"protocolSection": {"identificationModule": ident, "designModule": design}}


class _FakeClient:
    def __init__(self, records: list[dict], *, truncated: bool = False) -> None:
        self._records, self._truncated = records, truncated

    def iter_studies(self, sp, *, fields, page_size=1000, max_pages=20):
        return list(self._records), self._truncated


# --- K3: count_trials is a DISTINCT-nctId count (dedup) -----------------------


def test_combine_dedups_duplicate_and_idless_records() -> None:
    """A duplicate page-row / id-less record must NOT inflate the bar past the
    distinct-trial total it reconciles against (the headline reconciliation MED)."""
    core = AggregationCore(
        _FakeClient([_rec("NCT1", ["PHASE1"]), _rec("NCT1", ["PHASE1"]), _rec(None, ["PHASE2"])])
    )
    gr = core.page_and_group({}, fields="x", key_fn=phase_key_fn, mode="combine")
    total = sum(b.count_trials for b in gr.buckets)
    assert total == gr.distinct_trials == 1  # was 3 (raw record sum) before the fix
    p1 = next(b for b in gr.buckets if b.value == "PHASE1")
    assert p1.count_trials == p1.count_mentions == 1
    assert len(p1.records) == 1


def test_explode_counts_distinct_trials_and_per_occurrence_mentions() -> None:
    """The Phase-2 explode branch, now exercised: count_trials = distinct trials
    carrying a value; count_mentions = per (trial, value) occurrence; dedup by nctId."""

    def multi_key(record):
        vals = record["protocolSection"]["designModule"]["phases"]
        return [(v, v) for v in vals]

    core = AggregationCore(
        _FakeClient([_rec("NCT1", ["A", "B"]), _rec("NCT1", ["A", "B"]), _rec("NCT2", ["A"])])
    )
    gr = core.page_and_group({}, fields="x", key_fn=multi_key, mode="explode")
    by = {b.value: b for b in gr.buckets}
    assert by["A"].count_trials == 2 and by["B"].count_trials == 1  # distinct trials
    assert gr.distinct_trials == 2


# --- K2: element-precise provenance matching ----------------------------------


@pytest.mark.parametrize(
    ("value", "matched_value", "expected"),
    [
        (["PHASE10"], "PHASE1", False),  # substring false positive
        (["PHASE10"], "1", False),
        (["PHASE1", "PHASE2"], "', '", False),  # list-repr punctuation
        (["PHASE1", "PHASE2"], "[", False),
        (["PHASE1", "PHASE2"], "PHASE2", True),  # genuine element
        (["PHASE3"], "", False),  # empty excerpt vs present value
        (None, "", True),  # empty excerpt vs absent value = valid absence
        ([], "", True),
    ],
)
def test_excerpt_in_value_element_precise(value, matched_value, expected) -> None:
    assert _excerpt_in_value(matched_value, value) is expected


def test_is_substring_at_element_precise() -> None:
    assert is_substring_at(_rec("NCT1", ["PHASE10"]), _PHASES, "PHASE1") is False
    assert is_substring_at(_rec("NCT1", ["PHASE1", "PHASE2"]), _PHASES, "PHASE2") is True
    assert is_substring_at(_rec("NCT1", ["PHASE1", "PHASE2"]), _PHASES, "', '") is False
    assert is_substring_at(_rec("NCT1", None), _PHASES, "") is True  # absence
    assert is_substring_at(_rec("NCT1", ["PHASE1"]), _PHASES, "") is False  # present


def test_precheck_rejects_fabricated_element_citation() -> None:
    """A citation whose excerpt is not a real element of its value hard-fails."""
    d = _datum("PHASE1", 1)
    d.citations[0].value = ["PHASE10"]  # excerpt "PHASE1" is NOT an element of ["PHASE10"]
    d.citations[0].matched_value = "PHASE1"
    pc = deterministic_precheck(
        _ok_spec([d]), count_total=1, mode="combine", distinct_trials=1, truncated=False
    )
    assert pc.hard_fail and pc.reason == "citation_invalid"


# --- K1: totality of phase_key_fn (one bad record must not sink the chart) -----


@pytest.mark.parametrize(
    ("record", "expected_value"),
    [
        ({"protocolSection": None}, "MISSING"),
        ({"protocolSection": {"designModule": None}}, "MISSING"),
        (_rec("NCT1", "PHASE1"), "PHASE1"),  # bare string -> one token, not char-sharded
        (_rec("NCT1", ["PHASE1", "PHASE1"]), "PHASE1"),  # dup token -> one bucket
        (_rec("NCT1", 123), "MISSING"),  # non-list/non-str -> MISSING
        (_rec("NCT1", []), "MISSING"),
    ],
)
def test_phase_key_fn_is_total(record, expected_value) -> None:
    keys = phase_key_fn(record)
    assert len(keys) == 1
    assert keys[0][0] == expected_value


def test_phase_key_fn_stringifies_nonstring_tokens() -> None:
    """A stray non-string token can't crash the sort/join (composite still forms)."""
    keys = phase_key_fn(_rec("NCT1", ["PHASE1", 123]))
    assert len(keys) == 1 and "|" in keys[0][0]  # a deterministic composite, no crash


# --- K5: client is total on a malformed API response --------------------------


class _StubResp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status
        self.headers = {}

    def json(self):
        return self._body


def _client_with(monkeypatch, body):
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, params=None: _StubResp(body))
    return CTGovClient()


def test_count_rejects_missing_or_nonint_total(monkeypatch) -> None:
    for body in [{"studies": []}, {"totalCount": None}, {"totalCount": "3950"}, [1, 2]]:
        with pytest.raises(UpstreamError) as ei:
            _client_with(monkeypatch, body).count({"query.cond": "x"})
        assert ei.value.code == "upstream_bad_response"


def test_iter_studies_handles_null_and_nonlist_studies(monkeypatch) -> None:
    recs, trunc = _client_with(monkeypatch, {"studies": None}).iter_studies(
        {}, fields="NCTId", max_pages=1
    )
    assert recs == [] and trunc is False  # null studies -> no records, not a crash


def test_iter_studies_empty_token_is_completion_not_cursor(monkeypatch) -> None:
    recs, trunc = _client_with(
        monkeypatch, {"studies": [_rec("NCT1", ["PHASE1"])], "nextPageToken": ""}
    ).iter_studies({}, fields="NCTId", max_pages=20)
    assert trunc is False and len(recs) == 1  # "" token = done, not an infinite/mislabeled walk


def test_iter_studies_clamps_page_size_floor_and_cap(monkeypatch) -> None:
    seen = {}

    def capture(self, url, params=None):
        seen.update(params or {})
        return _StubResp({"studies": []})

    monkeypatch.setattr(httpx.Client, "get", capture)
    CTGovClient().iter_studies({}, fields="NCTId", page_size=0, max_pages=1)
    assert seen["pageSize"] == 1  # floored at 1, never 0/negative
    CTGovClient().iter_studies({}, fields="NCTId", page_size=99999, max_pages=1)
    assert seen["pageSize"] == 1000  # capped at the hard 1000
