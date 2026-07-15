"""Phase-5 INPUT edge cases — request-validation behavior that works TODAY.

Complementary to ``tests/test_schemas.py`` (model-level validators) and
``tests/test_main.py`` (a few HTTP-422 cases). This file adds:

* per-field caps the existing suite does not exercise individually
  (``drug_name`` / ``sponsor`` / ``country`` at 200 chars, plus ``trial_phase`` /
  ``study_type`` at their 100-char caps),
* the ACCEPTED boundary (exactly-at-cap) for the string fields + ``query`` — the
  existing tests only assert the rejected side,
* the year ``ge/le`` fence (1900..2100) which nothing else covers, and
* an HTTP-boundary (TestClient → 422) pass for the structured fields + year
  range that ``test_main`` omits, proving FastAPI maps each to a 422.

Everything is offline: a 422 is produced by FastAPI request validation BEFORE
the endpoint body runs, so no graph / network call ever fires.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas import VisualizeRequest
from app.main import app

client = TestClient(app)

# The four structured dimension fields sharing the 200-char cap (G-41b).
_STRUCTURED_FIELDS = ["drug_name", "condition", "sponsor", "country"]


# --- a fully-populated valid request builds a valid model -------------------


def test_valid_full_request_builds_model() -> None:
    """Every documented field at a valid value constructs a model with those
    values preserved (the ``query`` stripped, defaults untouched)."""
    req = VisualizeRequest(
        query="  Phase distribution of melanoma trials over time  ",
        drug_name="pembrolizumab",
        condition="melanoma",
        sponsor="Merck",
        country="United States",
        trial_phase="1/2",
        study_type="interventional",
        start_year=2010,
        end_year=2020,
        interventional_only=True,
    )
    assert req.query == "Phase distribution of melanoma trials over time"  # stripped
    assert req.drug_name == "pembrolizumab"
    assert req.condition == "melanoma"
    assert req.sponsor == "Merck"
    assert req.country == "United States"
    assert req.trial_phase == "1/2"
    assert req.study_type == "interventional"
    assert req.start_year == 2010 and req.end_year == 2020
    assert req.interventional_only is True


# --- structured string fields: 200-char cap, both sides ---------------------


@pytest.mark.parametrize("field", _STRUCTURED_FIELDS)
def test_structured_field_over_200_rejected(field: str) -> None:
    """Each of the four structured dimension fields rejects a >200-char value
    (only ``condition`` was covered before — G-41b applies to all four)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(**{"query": "distribution", field: "x" * 201})


@pytest.mark.parametrize("field", _STRUCTURED_FIELDS)
def test_structured_field_exactly_200_accepted(field: str) -> None:
    """The accepted boundary: exactly 200 chars validates (the cap is inclusive)."""
    req = VisualizeRequest(**{"query": "distribution", field: "x" * 200})
    assert getattr(req, field) == "x" * 200


# --- query length cap: accepted boundary ------------------------------------


def test_query_exactly_500_accepted() -> None:
    """A 500-char query is accepted (test_schemas already covers 501 → reject)."""
    req = VisualizeRequest(query="q" * 500)
    assert len(req.query) == 500


# --- trial_phase / study_type 100-char caps ---------------------------------


def test_trial_phase_over_100_rejected() -> None:
    """trial_phase is free text but length-capped at 100 (G-41b)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="phase distribution", trial_phase="p" * 101)


def test_study_type_over_100_rejected() -> None:
    """study_type is a hint but length-capped at 100 (G-41b)."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="by study type", study_type="s" * 101)


# --- year ge/le fence [1900, 2100] ------------------------------------------


def test_start_year_below_min_rejected() -> None:
    """start_year < 1900 is out of the documented [1900, 2100] fence."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials over time", start_year=1899)


def test_end_year_above_max_rejected() -> None:
    """end_year > 2100 is out of the documented [1900, 2100] fence."""
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials over time", end_year=2101)


def test_start_year_at_min_boundary_accepted() -> None:
    """start_year == 1900 (the inclusive lower fence) validates."""
    req = VisualizeRequest(query="trials over time", start_year=1900)
    assert req.start_year == 1900


# --- HTTP boundary: each maps to a 422 (complements test_main) ---------------


@pytest.mark.parametrize("field", _STRUCTURED_FIELDS)
def test_http_oversized_structured_field_is_422(field: str) -> None:
    """An oversized structured field is a 422 at the FastAPI boundary (no
    network — validation short-circuits before the endpoint body)."""
    resp = client.post("/visualize", json={"query": "trials", field: "x" * 5000})
    assert resp.status_code == 422


def test_http_inverted_year_range_is_422() -> None:
    """start_year > end_year is a 422 at the transport boundary."""
    resp = client.post(
        "/visualize", json={"query": "trials over time", "start_year": 2020, "end_year": 2010}
    )
    assert resp.status_code == 422


def test_http_out_of_range_year_is_422() -> None:
    """A year outside [1900, 2100] is a 422 at the transport boundary."""
    resp = client.post("/visualize", json={"query": "trials over time", "start_year": 1899})
    assert resp.status_code == 422
