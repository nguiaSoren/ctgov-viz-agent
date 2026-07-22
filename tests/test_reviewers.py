"""Offline self-verification for the two LLM reviewers (ARCHITECTURE_SPEC §3.4,
§3.8; Interface Contract v4 §3).

Everything here runs through the ``StubAdapter`` — zero network, zero provider
key — so it exercises the real reviewer wiring (system prompt + message shape +
the ``verify`` seam + the canned-approve path) deterministically. The invariant
under test: the reviewers are GATES that emit ``approve``/``revise``/``flag``
verdicts only, and the skip gate is conservative.

That is a claim about the verdict SHAPE, and only that. Neither verdict model has
a numeric field, but ``reason`` is free prose and can perfectly well contain
digits — so the shape alone does not enforce "the LLM never authors a number".
The actual guard is ``note_number_safe`` (``app/viz/review.py``), applied in
``app/graph/nodes.py`` before any LLM-authored note is kept, and tested in
``tests/test_phase4_hardening.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.api.schemas import ChartType, VisualizeResponse
from app.llm.adapter import StubAdapter
from app.llm.reviewers import (
    IntentVerdict,
    OutputVerdict,
    review_intent,
    review_output_llm,
    should_skip_intent_review,
)
from app.plan.models import Plan

_GOLDEN_DISTRIBUTION = Path(__file__).parent / "fixtures" / "golden_distribution.json"


def _valid_distribution_plan() -> Plan:
    """A mechanically-valid distribution-by-phase Plan (the intent reviewer runs
    on legal plans only)."""
    return Plan(
        query_class="distribution",
        entities={"condition": "melanoma"},
        field="phase",
        chart_type=ChartType.BAR,
        alternates=[ChartType.TABLE],
    )


def _golden_spec() -> VisualizeResponse:
    return VisualizeResponse.model_validate(
        json.loads(_GOLDEN_DISTRIBUTION.read_text())
    )


# --- review_intent -----------------------------------------------------------


def test_review_intent_approves_valid_plan() -> None:
    verdict = review_intent(
        StubAdapter(),
        "How are melanoma trials distributed by phase?",
        _valid_distribution_plan(),
    )
    assert isinstance(verdict, IntentVerdict)
    assert verdict.decision == "approve"
    # A gate, not a generator: an approve carries no number-bearing payload.
    assert verdict.field is None


# --- review_output_llm -------------------------------------------------------


def test_review_output_llm_approves_golden_spec() -> None:
    verdict = review_output_llm(
        StubAdapter(),
        "How are pancreatic cancer trials distributed by phase?",
        _golden_spec(),
    )
    assert isinstance(verdict, OutputVerdict)
    assert verdict.decision == "approve"


# --- should_skip_intent_review ----------------------------------------------


def test_skip_intent_review_true_for_all_structured_plan() -> None:
    """Every dimension in the plan is backed by a typed structured field → the NL
    parse inferred no dimension to misread → skip is safe."""
    merged_inputs = {"query": "distribution by phase", "condition": "melanoma"}
    plan = _valid_distribution_plan()  # entities == {"condition": "melanoma"}
    assert should_skip_intent_review(merged_inputs, plan) is True


def test_skip_intent_review_false_for_free_text_query() -> None:
    """The condition dimension was inferred from free text (no typed `condition`
    field) → there IS an NL parse to misread → do not skip."""
    merged_inputs = {"query": "how are melanoma trials distributed by phase?"}
    plan = _valid_distribution_plan()  # entities == {"condition": "melanoma"}
    assert should_skip_intent_review(merged_inputs, plan) is False


def test_skip_intent_review_true_for_empty_query() -> None:
    """An empty/whitespace query means the request was driven entirely by typed
    fields — nothing was parsed from natural language."""
    merged_inputs = {"query": "   ", "condition": "melanoma"}
    plan = _valid_distribution_plan()
    assert should_skip_intent_review(merged_inputs, plan) is True


def test_skip_intent_review_false_for_nonempty_query_no_entities() -> None:
    """A non-empty query with no resolved dimensions still ran the NL parse to
    pick the class/field — not clearly safe to skip."""
    merged_inputs = {"query": "how many trials completed over time?"}
    plan = Plan(query_class="single_value", chart_type=ChartType.SINGLE_VALUE)
    assert should_skip_intent_review(merged_inputs, plan) is False
