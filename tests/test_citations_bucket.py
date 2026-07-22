"""Tests for the per-bucket citation builder (``build_bucket_citations``, CC-9).

Pure functions — no network. Records are synthetic ClinicalTrials.gov-shaped
dicts carrying only the two paths the builder reads: the nctId (for the
deterministic sort) and the field being cited (whose value becomes
``matched_value``). They carry no ``briefTitle``, so ``Citation.excerpt`` falls
back to ``matched_value`` here — this file says nothing about the brief-title
excerpt path.
"""

from __future__ import annotations

from app.api.schemas import Citation
from app.ctgov.citations import build_bucket_citations, is_substring_at

FIELD_PATH = "protocolSection.designModule.phases"


def _record(nct_id: str, phases: list[str] | None = None) -> dict:
    """A minimal record: an nctId + a phases array (default ``["PHASE2"]``)."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "designModule": {"phases": phases if phases is not None else ["PHASE2"]},
        }
    }


def test_truncates_at_k_but_reports_exact_contributing_count() -> None:
    """25 records, k=20 → 20 citations, contributing_count==25 (exact), truncated."""
    records = [_record(f"NCT{i:08d}") for i in range(25)]

    citations, contributing_count, truncated = build_bucket_citations(
        records, FIELD_PATH, k=20
    )

    assert len(citations) == 20
    assert contributing_count == 25  # EXACT — computed before capping
    assert truncated is True
    assert all(isinstance(c, Citation) for c in citations)


def test_under_k_is_not_truncated() -> None:
    """3 records → 3 citations, contributing_count==3, not truncated."""
    records = [_record("NCT00000003"), _record("NCT00000001"), _record("NCT00000002")]

    citations, contributing_count, truncated = build_bucket_citations(
        records, FIELD_PATH, k=20
    )

    assert len(citations) == 3
    assert contributing_count == 3
    assert truncated is False


def test_every_excerpt_is_a_real_substring_at_field_path() -> None:
    """Each returned citation's ``matched_value`` — the anti-fabrication anchor the
    Output Reviewer verifies — round-trips via the existing ``is_substring_at``."""
    records = [_record(f"NCT{i:08d}", phases=["PHASE1", "PHASE2"]) for i in range(5)]

    citations, _, _ = build_bucket_citations(records, FIELD_PATH, k=20)

    assert citations  # sanity: the sample is non-empty
    for citation, record in zip(citations, records, strict=True):
        assert citation.field_path == FIELD_PATH
        assert is_substring_at(record, FIELD_PATH, citation.matched_value) is True


def test_sample_is_deterministic_first_k_by_sorted_nctid() -> None:
    """Shuffled nctIds → the sample is the lexicographically-smallest k nctIds, in order."""
    ids = [f"NCT{i:08d}" for i in range(30)]
    shuffled = [ids[7], ids[29], ids[0], ids[15], ids[3], *ids[1:3], *ids[4:7], *ids[8:15], *ids[16:29]]
    assert sorted(shuffled) == sorted(ids)  # sanity: a true permutation, no dupes/drops
    records = [_record(nct_id) for nct_id in shuffled]

    citations, contributing_count, truncated = build_bucket_citations(
        records, FIELD_PATH, k=20
    )

    sampled_ids = [c.nct_id for c in citations]
    assert sampled_ids == sorted(ids)[:20]  # the smallest 20, in ascending nctId order
    assert contributing_count == 30
    assert truncated is True


def test_missing_nctid_sorts_as_empty_string_without_crashing() -> None:
    """A record missing the nctId path sorts as "" (first) and never raises."""
    good = _record("NCT00000005")
    no_id = {"protocolSection": {"designModule": {"phases": ["PHASE2"]}}}

    citations, contributing_count, truncated = build_bucket_citations(
        [good, no_id], FIELD_PATH, k=20
    )

    assert contributing_count == 2
    assert truncated is False
    assert [c.nct_id for c in citations] == ["", "NCT00000005"]  # "" sorts first, deterministic


def test_input_records_not_mutated() -> None:
    """Pure: the builder returns a new sample and never mutates the input list/dicts."""
    records = [_record(f"NCT{i:08d}") for i in range(3)]
    before = [dict(r) for r in records]

    build_bucket_citations(records, FIELD_PATH, k=20)

    assert records == before
