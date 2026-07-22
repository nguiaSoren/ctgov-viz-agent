"""Unit tests for the Output-Reviewer deterministic pre-check (``app.viz.review``).

Pure + offline: every ``VisualizeResponse`` is constructed directly (no network,
no graph). Covers all FIVE checks ``deterministic_precheck`` runs, plus the
exemption gate:

* (1) citation provenance — the verified field is ``matched_value`` (and every
  ``matched_tokens`` member), NOT the field literally named ``excerpt`` (that one
  carries the trial's brief title and the reviewer never checks it): a fabricated
  ``matched_value`` → hard fail ``citation_invalid``,
* (2) reconciliation (exact → ok; within-tolerance drift → ok + disclosure;
  beyond-tolerance → hard fail ``reconciliation_failed``; explode anchors on
  ``distinct_trials``; a missing oracle → ``reconciliation_unavailable``),
* (2b) combine bar-sum consistency — Σ displayed bars must equal
  ``distinct_trials`` or it is a hard fail ``bar_sum_mismatch`` (explode is
  exempt by design: Σ bars = memberships ≥ distinct),
* (3) partial-iff-truncated (mismatch → hard fail, both directions),
* (4) cited-or-derived (a legitimate 0-count uncited bucket is NOT a hard fail),
* the exemption gate: a non-``ok`` status (too_large / error / empty) → exempt,
  no checks run. A ``network_graph`` spec (non-list ``NetworkData``) is exempt
  from the ROW checks only — its edge citations are still verified (F2), which
  is asserted below.
"""

from __future__ import annotations

from app.api.schemas import (
    ChartType,
    Citation,
    CountBasis,
    Datum,
    Edge,
    EncodingChannel,
    Meta,
    NetworkData,
    Node,
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


# --- (2b) combine bar-sum consistency -----------------------------------------
#
# The explode side of this check (Σ bars = memberships > distinct_trials is
# LEGAL, not a mismatch) is proven by ``test_explode_mode_reconciles_on_distinct_trials``
# above: 70 mentions over 50 distinct trials passes.


def test_combine_bar_sum_mismatch_hard_fails() -> None:
    """A deflated combine bar that leaves the SCALAR anchor intact still hard-fails.

    ``distinct_trials == count_total == 50``, so check (2) reconciles exactly — but the
    DISPLAYED bars sum to 45, i.e. the chart silently drops 5 trials. Check (2b) is the
    only thing standing between that and a shipped envelope (LESSON L1: reconcile the
    displayed bars, not just a scalar anchor).
    """
    spec = _ok_spec([_datum("PHASE1", 30), _datum("PHASE2", 15)])
    pc = _check(spec, count_total=50, mode="combine", distinct=50)
    assert pc.hard_fail
    assert pc.reason == "bar_sum_mismatch"


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


def _network_spec(matched_value: str) -> VisualizeResponse:
    """An ok ``network_graph`` envelope: one edge, one endpoint citation carrying
    ``matched_value`` (the value at ``field_path`` that placed the trial on the edge)."""
    viz = Visualization(
        type=ChartType.NETWORK_GRAPH,
        title="Drugs studied together",
        encoding={},
        data=NetworkData(
            nodes=[
                Node(id="drug:a", label="A", kind="drug"),
                Node(id="drug:b", label="B", kind="drug"),
            ],
            edges=[
                Edge(
                    source="drug:a",
                    target="drug:b",
                    weight=1,
                    source_ids=["NCT00000001"],
                    citations=[
                        Citation(
                            nct_id="NCT00000001",
                            field_path="protocolSection.armsInterventionsModule.interventions",
                            value=["A", "B"],
                            matched_value=matched_value,
                        )
                    ],
                )
            ],
        ),
    )
    return VisualizeResponse(
        status="ok",
        kind="visualization",
        visualization=viz,
        meta=Meta(count_basis=CountBasis(trials=1)),
    )


def test_network_spec_is_row_exempt_but_edge_citations_are_still_verified() -> None:
    """A non-list ``NetworkData`` payload skips the ROW checks (no row array, no single
    oracle) — but NOT the citation check. The old coarse list-only exemption waived
    check (1) too, so a fabricated edge excerpt shipped unverified (F2)."""
    honest = deterministic_precheck(
        _network_spec("A"), count_total=None, mode=None, distinct_trials=None, truncated=False
    )
    assert honest.ok and not honest.hard_fail  # reconciliation genuinely waived

    fabricated = deterministic_precheck(
        _network_spec("FABRICATED-NOT-AN-ENDPOINT"),
        count_total=None,
        mode=None,
        distinct_trials=None,
        truncated=False,
    )
    assert fabricated.hard_fail
    assert fabricated.reason == "citation_invalid"


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
