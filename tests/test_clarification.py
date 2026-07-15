"""The dangling-reference → kind:"clarification" outcome (E-13 / P5-INPUT)."""

from __future__ import annotations

from app.api.schemas import VisualizeRequest
from app.graph.build import run_sync
from app.graph.clarify import detect_dangling_reference
from app.plan.models import Plan


def _plan(entities: dict | None = None) -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"} if entities is None else entities,
        filters={},
        field="phase",
        chart_type="bar",
    )


# --- the detector ------------------------------------------------------------


def test_detects_unresolved_demonstrative_drug():
    q = detect_dangling_reference({"query": "How many trials study this drug?"}, _plan())
    assert q is not None and "drug" in q.lower()


def test_no_clarification_when_drug_resolved_by_entity():
    # The planner resolved a drug entity ("that drug, pembrolizumab") → not dangling.
    plan = _plan({"condition": "melanoma", "drug": "pembrolizumab"})
    assert detect_dangling_reference({"query": "trials of that drug pembrolizumab by phase"}, plan) is None


def test_no_clarification_when_drug_resolved_by_field():
    plan = _plan()
    merged = {"query": "distribution for this drug", "drug_name": "pembrolizumab"}
    assert detect_dangling_reference(merged, plan) is None


def test_no_clarification_on_a_plain_query():
    assert detect_dangling_reference({"query": "phase distribution of pancreatic cancer"}, _plan()) is None


def test_trial_referent_always_asks():
    q = detect_dangling_reference({"query": "tell me about this trial"}, _plan())
    assert q is not None and ("NCT" in q or "trial" in q.lower())


def test_condition_and_sponsor_referents():
    assert detect_dangling_reference({"query": "trials for this condition by phase"}, _plan({})) is not None
    assert detect_dangling_reference({"query": "studies from that sponsor"}, _plan({})) is not None


def test_detector_is_total_on_missing_inputs():
    assert detect_dangling_reference(None, None) is None
    assert detect_dangling_reference({}, None) is None
    assert detect_dangling_reference({"query": ""}, _plan()) is None


# --- end-to-end (offline: a clarification short-circuits before execute) ------


def test_end_to_end_clarification_envelope_offline():
    # The StubAdapter plans a distribution over the default condition with NO drug
    # entity; "this drug" then trips the dangling-reference detector → a clarification
    # envelope, entirely offline (no execute, no network).
    resp = run_sync(VisualizeRequest(query="How many trials are there for this drug?"))
    assert resp.kind == "clarification"
    assert resp.status == "empty"
    assert resp.question and "drug" in resp.question.lower()
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.answer is None
    assert resp.error is None


def test_clarification_resolved_by_field_does_not_fire_offline_shape():
    # With drug_name supplied, "this drug" resolves → NOT a clarification. (We assert
    # only that it does not become a clarification; the real execute path is live.)
    resp_kind = detect_dangling_reference(
        {"query": "How many trials for this drug?", "drug_name": "pembrolizumab"}, _plan()
    )
    assert resp_kind is None


# --- adversarial-review regressions (clarification over/under-fire fixes) ------

def test_this_study_type_does_not_clarify():
    # "study"/"studies" dropped from the trial nouns → "this study type distribution"
    # is a legit distribution query, not a dangling trial reference.
    assert detect_dangling_reference({"query": "this study type distribution"}, _plan()) is None
    assert detect_dangling_reference({"query": "melanoma phase in this study population"}, _plan({})) is None


def test_inline_nct_id_resolves_trial_reference():
    assert detect_dangling_reference(
        {"query": "phase breakdown of this trial NCT01234567"}, _plan()
    ) is None


def test_these_those_trial_referents_now_detected():
    # symmetric with the other dimensions (was this/that only)
    assert detect_dangling_reference({"query": "tell me about those trials"}, _plan()) is not None


def test_demonstrative_extracted_as_entity_still_asks():
    # The REAL LLM planner can extract the demonstrative phrase AS the entity value
    # (query.intr="this drug"), which the offline StubAdapter never did. A bare
    # bool(entity) check treated that as resolved and never asked (a wired-isn't-run
    # gap the live ladder surfaced). The detector must see the "resolution" is itself a
    # referent and still clarify — while a genuine drug entity stays resolved (no over-fire).
    q = detect_dangling_reference(
        {"query": "How many trials are there for this drug?"}, _plan({"drug": "this drug"})
    )
    assert q is not None and "drug" in q.lower()
    assert detect_dangling_reference(
        {"query": "How many trials for this drug?"}, _plan({"drug": "pembrolizumab"})
    ) is None
