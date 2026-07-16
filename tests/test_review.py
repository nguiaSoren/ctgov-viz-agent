"""Unit tests for the Output-Reviewer deterministic pre-check (``app.viz.review``).

Pure + offline: every ``VisualizeResponse`` is constructed directly (no network,
no graph). Covers the four checks + the exemption gate:

* excerpt-substring (fabricated excerpt → hard fail),
* reconciliation (exact → ok; within-tolerance drift → ok + disclosure;
  beyond-tolerance → hard fail; explode uses ``distinct_trials``),
* partial-iff-truncated (mismatch → hard fail, both directions),
* cited-or-derived (a legitimate 0-count uncited bucket is NOT a hard fail),
* status ≠ "ok" / non-list data → exempt (no checks).
"""

from __future__ import annotations

from app.api.schemas import (
    ChartType,
    Citation,
    CountBasis,
    Datum,
    EncodingChannel,
    Meta,
    Partial,
    Visualization,
    VisualizeResponse,
)
from app.viz.review import deterministic_precheck

_FIELD_PATH = "protocolSection.designModule.phases"


def _datum(
    value: str,
    count: int,
    *,
    matched_value: str | None = None,
    cited: bool = True,
    derived: bool = False,
) -> Datum:
    citations: list[Citation] = []
    if cited:
        citations = [
            Citation(
                nct_id="NCT00000001",
                field_path=_FIELD_PATH,
                value=[value],
                matched_value=value if matched_value is None else matched_value,
            )
        ]
    return Datum(
        value=value, label=value, count_trials=count, citations=citations, derived=derived
    )


def _ok_spec(data: list[Datum], *, partial: dict | None = None) -> VisualizeResponse:
    viz = Visualization(
        type=ChartType.BAR,
        title="Phase distribution",
        encoding={
            "x": EncodingChannel(field="value"),
            "y": EncodingChannel(field="count_trials"),
        },
        data=data,
    )
    return VisualizeResponse(
        status="ok",
        kind="visualization",
        visualization=viz,
        vega_lite={},
        meta=Meta(
            count_basis=CountBasis(trials=sum(d.count_trials for d in data)),
            partial=Partial(**partial) if partial else None,
        ),
    )


def _check(spec, *, count_total, mode="combine", distinct=None, truncated=False):
    return deterministic_precheck(
        spec,
        count_total=count_total,
        mode=mode,
        distinct_trials=count_total if distinct is None else distinct,
        truncated=truncated,
    )


# --- (1) excerpt substring ----------------------------------------------------


def test_fabricated_excerpt_hard_fails() -> None:
    spec = _ok_spec([_datum("PHASE1", 50, matched_value="FABRICATED-NOT-PRESENT")])
    pc = _check(spec, count_total=50)
    assert pc.hard_fail
    assert pc.reason == "citation_invalid"


def test_real_excerpt_passes() -> None:
    spec = _ok_spec([_datum("PHASE1", 30), _datum("PHASE2", 20)])
    pc = _check(spec, count_total=50)
    assert pc.ok and not pc.hard_fail


# --- (2) reconciliation -------------------------------------------------------


def test_exact_reconciliation_ok_no_disclosure() -> None:
    spec = _ok_spec([_datum("PHASE1", 30), _datum("PHASE2", 20)])
    pc = _check(spec, count_total=50)
    assert pc.ok and not pc.hard_fail
    assert pc.disclosure is None


def test_within_tolerance_drift_ok_with_disclosure() -> None:
    # Σ = 9990 vs countTotal 10000 → drift 10 ≤ 0.5% (50) AND ≤ 20 → ok + disclosure.
    spec = _ok_spec([_datum("PHASE1", 5000), _datum("PHASE2", 4990)])
    pc = _check(spec, count_total=10000, distinct=9990)
    assert pc.ok and not pc.hard_fail
    assert pc.disclosure is not None
    assert "10,000" in pc.disclosure and "9,990" in pc.disclosure


def test_drift_beyond_pct_hard_fails() -> None:
    # Σ = 9900 vs 10000 → drift 100 > 0.5% (50) → hard fail.
    spec = _ok_spec([_datum("PHASE1", 5000), _datum("PHASE2", 4900)])
    pc = _check(spec, count_total=10000, distinct=9900)
    assert pc.hard_fail
    assert pc.reason == "reconciliation_failed"


def test_drift_within_pct_but_beyond_abs_hard_fails() -> None:
    # Σ = 99970 vs 100000 → drift 30 ≤ 0.5% (500) but > 20 abs → hard fail (AND).
    spec = _ok_spec([_datum("PHASE1", 99970)])
    pc = _check(spec, count_total=100000, distinct=99970)
    assert pc.hard_fail
    assert pc.reason == "reconciliation_failed"


def test_explode_mode_reconciles_on_distinct_trials() -> None:
    # Σ count_trials = 70 (mentions), but distinct_trials = 50 = countTotal → ok.
    spec = _ok_spec([_datum("A", 30), _datum("B", 40)])
    pc = _check(spec, count_total=50, mode="explode", distinct=50)
    assert pc.ok and not pc.hard_fail


def test_missing_count_total_hard_fails_cannot_certify() -> None:
    """An ok list-spec with NO oracle cannot be certified as reconciled — shipping
    it as ok would be a false provenance claim, so it hard-fails (K1). In Phase 1
    execute always stamps count_total for ok, so None here is a real upstream bug."""
    spec = _ok_spec([_datum("PHASE1", 30), _datum("PHASE2", 20)])
    pc = _check(spec, count_total=None)
    assert pc.hard_fail
    assert pc.reason == "reconciliation_unavailable"


# --- (3) partial iff truncated ------------------------------------------------


def test_partial_present_without_truncation_hard_fails() -> None:
    spec = _ok_spec([_datum("PHASE1", 50)], partial={"truncated": True, "of_total": 100})
    pc = _check(spec, count_total=50, truncated=False)
    assert pc.hard_fail
    assert pc.reason == "partial_inconsistent"


def test_truncation_without_partial_hard_fails() -> None:
    spec = _ok_spec([_datum("PHASE1", 50)])  # meta.partial is None
    pc = _check(spec, count_total=50, truncated=True)
    assert pc.hard_fail
    assert pc.reason == "partial_inconsistent"


def test_partial_matches_truncation_ok() -> None:
    spec = _ok_spec([_datum("PHASE1", 50)], partial={"truncated": True, "of_total": 500})
    pc = _check(spec, count_total=50, truncated=True)
    assert pc.ok and not pc.hard_fail


# --- (4) cited-or-derived -----------------------------------------------------


def test_zero_count_uncited_datum_is_not_a_hard_fail() -> None:
    # A legitimate explicit 0-fill bucket (G-35) carries no citation and must pass.
    spec = _ok_spec([_datum("PHASE1", 50), _datum("PHASE4", 0, cited=False)])
    pc = _check(spec, count_total=50)
    assert pc.ok and not pc.hard_fail


def test_nonzero_uncited_datum_hard_fails() -> None:
    spec = _ok_spec([_datum("PHASE1", 50, cited=False)])
    pc = _check(spec, count_total=50)
    assert pc.hard_fail
    assert pc.reason == "uncited_datum"


def test_derived_datum_without_citation_is_ok() -> None:
    spec = _ok_spec([_datum("RATE", 42, cited=False, derived=True)])
    pc = _check(spec, count_total=42)
    assert pc.ok and not pc.hard_fail


# --- exemption gate -----------------------------------------------------------


def test_too_large_spec_is_exempt() -> None:
    spec = VisualizeResponse(
        status="too_large",
        kind="answer",
        visualization=None,
        answer="99,999 trials match — too large to chart.",
        meta=Meta(count_basis=CountBasis(trials=99999)),
    )
    pc = deterministic_precheck(
        spec, count_total=None, mode=None, distinct_trials=None, truncated=False
    )
    assert pc.ok and not pc.hard_fail


def test_error_spec_is_exempt() -> None:
    spec = VisualizeResponse(
        status="error",
        kind="answer",
        visualization=None,
        error={"code": "upstream_error", "message": "failed"},
        meta=Meta(),
    )
    pc = deterministic_precheck(
        spec, count_total=None, mode=None, distinct_trials=None, truncated=False
    )
    assert pc.ok and not pc.hard_fail


def test_empty_spec_is_exempt() -> None:
    # An empty visualization (status "empty") carries an empty data list but is
    # EXEMPT because status != "ok" — a 0-drift reconciliation would otherwise
    # need count_total=0.
    viz = Visualization(
        type=ChartType.BAR,
        title="Phase distribution",
        encoding={"x": EncodingChannel(field="value")},
        data=[],
    )
    spec = VisualizeResponse(
        status="empty",
        kind="visualization",
        visualization=viz,
        meta=Meta(count_basis=CountBasis(trials=0)),
    )
    pc = deterministic_precheck(
        spec, count_total=None, mode="combine", distinct_trials=None, truncated=False
    )
    assert pc.ok and not pc.hard_fail
