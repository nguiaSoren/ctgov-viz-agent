"""trial_phase normalization + the E-16 422 contract (P5-INPUT)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import VisualizeRequest
from app.ctgov.enums import PHASE_TOKENS
from app.ctgov.phases import InvalidTrialPhase, normalize_trial_phase


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Phase 1", ["PHASE1"]),
        ("phase 1", ["PHASE1"]),
        ("PHASE1", ["PHASE1"]),
        ("1", ["PHASE1"]),
        ("phase I", ["PHASE1"]),
        ("Phase II", ["PHASE2"]),
        ("phase iii", ["PHASE3"]),
        ("Phase 4", ["PHASE4"]),
        ("Early Phase 1", ["EARLY_PHASE1"]),
        ("early phase i", ["EARLY_PHASE1"]),
        ("Phase 0", ["EARLY_PHASE1"]),
        ("EARLY_PHASE1", ["EARLY_PHASE1"]),
        ("NA", ["NA"]),
        ("N/A", ["NA"]),
        ("not applicable", ["NA"]),
        ("1/2", ["PHASE1", "PHASE2"]),
        ("Phase 1/2", ["PHASE1", "PHASE2"]),
        ("phase 2/3", ["PHASE2", "PHASE3"]),
        ("1 and 2", ["PHASE1", "PHASE2"]),
        ("2, 3", ["PHASE2", "PHASE3"]),
        ("1/1", ["PHASE1"]),  # de-duped
    ],
)
def test_normalize_accepts_human_forms(text, expected):
    out = normalize_trial_phase(text)
    assert out == expected
    assert all(tok in PHASE_TOKENS for tok in out)  # every output is a real wire token


@pytest.mark.parametrize("bad", ["banana", "Phase 9", "phase five", "", "   ", "xyz", "Phase 1 or apple"])
def test_normalize_rejects_unknown(bad):
    with pytest.raises(InvalidTrialPhase):
        normalize_trial_phase(bad)


def test_normalize_is_total_on_nonstr():
    with pytest.raises(ValueError):
        normalize_trial_phase(None)
    with pytest.raises(ValueError):
        normalize_trial_phase(123)


def test_request_accepts_valid_trial_phase():
    req = VisualizeRequest(query="phase distribution", trial_phase="Phase 1/2")
    assert req.trial_phase == "Phase 1/2"  # original string preserved (planner re-normalizes)


def test_request_rejects_unknown_trial_phase_422():
    with pytest.raises(ValidationError) as exc:
        VisualizeRequest(query="phase distribution", trial_phase="Phase 9")
    # the 422 message enumerates the valid phases (E-16)
    assert "valid phases are" in str(exc.value)


def test_request_treats_blank_trial_phase_as_unset():
    req = VisualizeRequest(query="q", trial_phase="   ")
    assert req.trial_phase is None


def test_hyphen_and_range_forms_normalize():
    # adversarial-review regression: "Phase-1" was wrongly 422'd; "1-2" is a range.
    assert normalize_trial_phase("Phase-1") == ["PHASE1"]
    assert normalize_trial_phase("phase-2") == ["PHASE2"]
    assert normalize_trial_phase("1-2") == ["PHASE1", "PHASE2"]
    VisualizeRequest(query="q", trial_phase="Phase-1")  # accepted at the request layer (no 422)
