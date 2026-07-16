"""Tests for composite-bucket element-precise citations (CC-15, Phase-3 Task 5).

Pure functions — no network. Records are synthetic ClinicalTrials.gov-shaped
dicts carrying only the two paths the builder reads: the nctId (for the
deterministic sort) and the ``phases`` array being cited.

A composite bucket (e.g. ``PHASE1|PHASE2``, formed from ``["PHASE1","PHASE2"]``)
must identify EVERY member token: ``excerpt`` stays the first token for display
back-compat, and ``matched_tokens`` carries every verified member literal (each a
verbatim element at ``field_path``). A single-value bucket is unchanged:
``matched_tokens is None``.
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


def test_composite_bucket_carries_every_verified_member_token() -> None:
    """PHASE1|PHASE2 composite → matched_value=="PHASE1", matched_tokens==["PHASE1","PHASE2"];
    every token round-trips element-precisely at field_path."""
    records = [
        _record(f"NCT{i:08d}", phases=["PHASE1", "PHASE2"]) for i in range(5)
    ]

    citations, contributing_count, truncated = build_bucket_citations(
        records, FIELD_PATH, k=20, member_tokens=["PHASE1", "PHASE2"]
    )

    assert contributing_count == 5
    assert truncated is False
    assert citations  # sanity: non-empty sample
    for citation, record in zip(citations, records, strict=True):
        assert isinstance(citation, Citation)
        assert citation.field_path == FIELD_PATH
        assert citation.matched_value == "PHASE1"  # first token, for display
        assert citation.matched_tokens == ["PHASE1", "PHASE2"]  # in token order
        # value stays the record's REAL resolved value, not a copy of the excerpt:
        assert citation.value == ["PHASE1", "PHASE2"]
        # each member token round-trips element-precisely at field_path:
        for token in citation.matched_tokens:
            assert is_substring_at(record, FIELD_PATH, token) is True


def test_single_value_bucket_via_member_tokens_is_back_compatible() -> None:
    """member_tokens with length 1 → single-value behavior: matched_value=="PHASE3",
    matched_tokens is None."""
    records = [_record(f"NCT{i:08d}", phases=["PHASE3"]) for i in range(3)]

    citations, _, _ = build_bucket_citations(
        records, FIELD_PATH, k=20, member_tokens=["PHASE3"]
    )

    assert citations
    for citation in citations:
        assert citation.matched_value == "PHASE3"
        assert citation.matched_tokens is None  # single-value → no composite tokens


def test_member_tokens_none_is_unchanged_single_value_behavior() -> None:
    """member_tokens=None (the default) → matched_value=="PHASE3", matched_tokens is None."""
    records = [_record(f"NCT{i:08d}", phases=["PHASE3"]) for i in range(3)]

    citations, _, _ = build_bucket_citations(records, FIELD_PATH, k=20)

    assert citations
    for citation in citations:
        assert citation.matched_value == "PHASE3"
        assert citation.matched_tokens is None


def test_composite_honesty_absent_member_token_is_excluded() -> None:
    """A record genuinely missing a member token gets only the present tokens in
    matched_tokens (honesty check) — no synthesized/unverifiable literal."""
    both = _record("NCT00000001", phases=["PHASE1", "PHASE2"])
    only_p1 = _record("NCT00000002", phases=["PHASE1"])  # PHASE2 genuinely absent

    citations, contributing_count, _ = build_bucket_citations(
        [both, only_p1], FIELD_PATH, k=20, member_tokens=["PHASE1", "PHASE2"]
    )

    assert contributing_count == 2
    by_nct = {c.nct_id: c for c in citations}
    # display excerpt is the first token on both, regardless of presence:
    assert by_nct["NCT00000001"].matched_value == "PHASE1"
    assert by_nct["NCT00000002"].matched_value == "PHASE1"
    # matched_tokens is honest: only the tokens actually present in each record:
    assert by_nct["NCT00000001"].matched_tokens == ["PHASE1", "PHASE2"]
    assert by_nct["NCT00000002"].matched_tokens == ["PHASE1"]  # PHASE2 excluded
    # every listed token still round-trips at field_path:
    for citation, record in ((by_nct["NCT00000001"], both), (by_nct["NCT00000002"], only_p1)):
        for token in citation.matched_tokens:
            assert is_substring_at(record, FIELD_PATH, token) is True


def test_composite_reports_exact_contributing_count_and_truncates_at_k() -> None:
    """25 composite records, k=20 → 20 citations, contributing_count==25 (exact),
    truncated; each citation still carries the composite matched_tokens."""
    records = [
        _record(f"NCT{i:08d}", phases=["PHASE1", "PHASE2"]) for i in range(25)
    ]

    citations, contributing_count, truncated = build_bucket_citations(
        records, FIELD_PATH, k=20, member_tokens=["PHASE1", "PHASE2"]
    )

    assert len(citations) == 20
    assert contributing_count == 25  # EXACT — computed before capping
    assert truncated is True
    assert all(c.matched_tokens == ["PHASE1", "PHASE2"] for c in citations)
