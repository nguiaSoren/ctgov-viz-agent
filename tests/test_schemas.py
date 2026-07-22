"""Self-verification for the API contracts (Phase 0).

Two guarantees:

1. Every hand-written golden fixture is a valid ``VisualizeResponse`` — one per
   envelope shape (distribution / timeseries / network / answer / error /
   too_large) — and each stamps ``meta.source == "clinicaltrials.gov"`` (A-33/G-2).
   The goldens are SCHEMA fixtures only, never behaviour oracles: they were
   hand-written before the engine existed and differ from real output in several
   visible ways. See ``tests/fixtures/README.md``; for "what the engine actually
   returns", use ``examples/run_*.json``.
2. ``VisualizeRequest`` enforces its documented validation rules (A-22, API-4..7,
   E-17/E-25, G-41b): non-empty query, length caps, ordered year range,
   ``interventional_only`` accepted, and ``extra="forbid"`` on unknown fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.schemas import Meta, VisualizeRequest, VisualizeResponse

FIXTURES = Path(__file__).parent / "fixtures"

GOLDENS = [
    "golden_distribution.json",
    "golden_timeseries.json",
    "golden_network.json",
    "golden_answer.json",
    "golden_error.json",
    "golden_too_large.json",
]


# --- Golden fixtures validate against the envelope ------------------------


@pytest.mark.parametrize("filename", GOLDENS)
def test_golden_validates(filename: str) -> None:
    """Each golden loads as a valid VisualizeResponse and carries the registry source."""
    payload = json.loads((FIXTURES / filename).read_text())
    resp = VisualizeResponse.model_validate(payload)
    assert resp.meta.source == "clinicaltrials.gov"


def test_all_six_goldens_present() -> None:
    """All six envelope shapes exist on disk (no silently-missing fixture)."""
    for filename in GOLDENS:
        assert (FIXTURES / filename).is_file(), f"missing fixture: {filename}"


def test_meta_source_defaults_to_registry() -> None:
    """meta.source defaults to clinicaltrials.gov even when omitted (A-33/G-2)."""
    assert Meta().source == "clinicaltrials.gov"


def test_data_union_discriminates_rows_vs_network() -> None:
    """The Visualization.data smart-union resolves list→rows and object→network."""
    dist = VisualizeResponse.model_validate(
        json.loads((FIXTURES / "golden_distribution.json").read_text())
    )
    assert isinstance(dist.visualization.data, list)  # rows

    net = VisualizeResponse.model_validate(
        json.loads((FIXTURES / "golden_network.json").read_text())
    )
    assert hasattr(net.visualization.data, "nodes")  # NetworkData
    assert hasattr(net.visualization.data, "edges")


# --- Request validation ---------------------------------------------------


def test_valid_request_minimal() -> None:
    """A minimal valid request (query only) is accepted and stripped."""
    req = VisualizeRequest(query="  trials for melanoma by phase  ")
    assert req.query == "trials for melanoma by phase"
    assert req.interventional_only is False


def test_empty_query_rejected() -> None:
    """An empty query raises (API-4)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="")


def test_whitespace_query_rejected() -> None:
    """A whitespace-only query raises (E-17)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="   \t  \n ")


def test_query_over_length_rejected() -> None:
    """A query over the 500-char cap raises (E-25/SEC-35)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="x" * 501)


def test_long_condition_rejected() -> None:
    """A structured string field over the 200-char cap raises (G-41b)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="distribution", condition="c" * 5000)


def test_interventional_only_accepted() -> None:
    """interventional_only=True is accepted (CC-5/E-38)."""
    req = VisualizeRequest(query="phase distribution", interventional_only=True)
    assert req.interventional_only is True


def test_inverted_year_range_rejected() -> None:
    """start_year > end_year raises (→ 422)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials over time", start_year=2020, end_year=2010)


def test_equal_year_range_accepted() -> None:
    """A single-year range (start == end) is valid."""
    req = VisualizeRequest(query="trials in 2020", start_year=2020, end_year=2020)
    assert req.start_year == 2020 and req.end_year == 2020


def test_unknown_field_rejected() -> None:
    """An unknown request field is rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials", nonsense_field="boom")


def test_trial_phase_accepts_human_string() -> None:
    """trial_phase accepts a human string; normalization is a downstream concern."""
    req = VisualizeRequest(query="phase 1/2 trials", trial_phase="1/2")
    assert req.trial_phase == "1/2"
