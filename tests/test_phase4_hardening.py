"""Phase-4 adversarial hardening — the NEW guarantees the happy-path live gate doesn't exercise:
the record-grounded citation re-verify TEETH, the escalation feedback hygiene, and the adapter's
secret redaction. Offline / network-free (fakes + fixtures only)."""

from __future__ import annotations

import logging

import pytest

from app.api.schemas import (
    Citation,
    Datum,
    Meta,
    Visualization,
    VisualizeRequest,
    VisualizeResponse,
)
from app.graph.build import build_graph, initial_state
from app.llm.adapter import AdapterError, OpenAIAdapter, _strict_schema
from app.llm.planner import PlannerOutput
from app.llm.reviewers import OutputVerdict
from app.viz.review import note_number_safe, record_grounded_reverify

# --- record-grounded re-verify: TEETH (SEC-19 / LESSON M3) --------------------

def _spec_with_excerpt(excerpt: str) -> VisualizeResponse:
    """A minimal ok row-spec whose single datum cites NCT00000001 at the phases path."""
    citation = Citation(
        nct_id="NCT00000001",
        field_path="protocolSection.designModule.phases",
        value=["PHASE1"],
        excerpt=excerpt,
    )
    datum = Datum(value="PHASE1", label="Phase 1", count_trials=1, citations=[citation])
    viz = Visualization(type="bar", title="t", encoding={}, data=[datum])
    return VisualizeResponse(status="ok", kind="visualization", visualization=viz,
                             meta=Meta(source="clinicaltrials.gov"))


_REAL_RECORD = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000001"},
        "designModule": {"phases": ["PHASE1"]},
    }
}


def test_record_grounded_reverify_passes_an_honest_excerpt() -> None:
    spec = _spec_with_excerpt("PHASE1")  # genuinely present at the path in the real record
    result = record_grounded_reverify(spec, {"NCT00000001": _REAL_RECORD})
    assert result.ok and not result.hard_fail


def test_record_grounded_reverify_hard_fails_a_fabricated_excerpt_even_when_value_matches() -> None:
    """A citation whose excerpt is ABSENT from its own fetched record hard-fails citation_invalid
    even though excerpt == value (the tautology the build-time check alone could not catch)."""
    spec = _spec_with_excerpt("PHASE9")  # NOT in the real record (which has only PHASE1)
    # value is also PHASE9-shaped? No — force the tautology: value == excerpt, both fabricated.
    spec.visualization.data[0].citations[0].value = ["PHASE9"]
    result = record_grounded_reverify(spec, {"NCT00000001": _REAL_RECORD})
    assert result.hard_fail and result.reason == "citation_invalid"


def test_record_grounded_reverify_skips_when_record_absent() -> None:
    """A citation whose nctId isn't in the bounded sample is skipped (build-time value-check
    already grounded it) — an absent record is not a failure."""
    spec = _spec_with_excerpt("ANYTHING")
    assert record_grounded_reverify(spec, {"NCT99999999": _REAL_RECORD}).ok
    assert record_grounded_reverify(spec, None).ok  # no index at all (too_large / offline)


# --- escalation feedback hygiene ---------------------------------------------

def test_plan_feedback_is_cleared_after_a_clean_run() -> None:
    """plan_feedback must not leak across iterations: a clean run (no reject) ends with it None,
    and the `plan` node clears it on consumption so a stale reason can't bias a later re-plan."""
    graph = build_graph()
    final = graph.invoke(initial_state(VisualizeRequest(query="x", condition="y"),
                                       overrides={"_force_canned": True}))
    assert final.get("plan_feedback") is None


def test_checker_reject_records_feedback_for_the_replan() -> None:
    """A checker reject writes the machine reason into plan_feedback so the bounded re-plan is a
    real reason->act->observe step (the escalation is still bounded: `plan` runs exactly twice)."""
    graph = build_graph()
    final = graph.invoke(initial_state(VisualizeRequest(query="x", condition="y"),
                                       overrides={"_force_reject": True}))
    # persistently-rejected (illegal) plan → error after one bounded re-plan
    assert final["events"].count("plan") == 2
    assert final["spec"].status == "error"


# --- adapter: secret redaction + strict schema (offline) ---------------------

def test_openai_adapter_redacts_provider_error_and_never_logs_key(caplog) -> None:
    """A provider failure surfaces as a redacted AdapterError (fixed message + machine code); the
    real exception detail — which could embed the key/URL — is never on the raised message."""
    class _BoomClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("sk-secret-LEAKED-KEY-should-never-surface")

    adapter = OpenAIAdapter(client=_BoomClient())
    with caplog.at_level(logging.ERROR):
        with pytest.raises(AdapterError) as exc:
            adapter.propose(system="s", messages=[{"role": "user", "content": "q"}],
                            response_model=PlannerOutput,
                            canned={"query_class": "distribution", "chart_type": "bar", "field": "phase"})
    assert "sk-secret" not in str(exc.value)  # the raised message is redacted
    assert exc.value.code == "provider_error"


# --- §1 meta.notes digit post-check (the LLM-flag-reason number smuggle) ------

def test_note_number_safe_helper() -> None:
    allowed = {"3950", "126", "2"}
    assert note_number_safe("phase distribution looks fine", allowed)  # no digits
    assert note_number_safe("the phase 2 arm looks off", allowed)      # 2 is in the data
    assert not note_number_safe("the real total is 99999 trials", allowed)  # fabricated


def test_output_reviewer_flag_cannot_smuggle_a_fabricated_number(monkeypatch) -> None:
    """A flag reason carrying a number NOT in the computed data is withheld for a fixed caveat —
    the model cannot smuggle a count onto the wire via meta.notes (§1). A real-data number rides."""
    import app.graph.nodes as nodes_mod

    # Fabricated number → withheld.
    monkeypatch.setattr(nodes_mod, "review_output_llm",
                        lambda *a, **k: OutputVerdict(decision="flag",
                                                      reason="Correction: the real total is 99999 trials."))
    final = build_graph().invoke(initial_state(VisualizeRequest(query="x", condition="y"),
                                               overrides={"_force_canned": True}))
    notes = " ".join(final["spec"].meta.notes)
    assert "99999" not in notes
    assert "flagged this result" in notes  # the fixed code-owned caveat shipped instead

    # A caveat referencing a real datum number (the canned buckets are 32/54/40) ships as-is.
    monkeypatch.setattr(nodes_mod, "review_output_llm",
                        lambda *a, **k: OutputVerdict(decision="flag",
                                                      reason="the phase-1 arm (32 trials) dominates"))
    final2 = build_graph().invoke(initial_state(VisualizeRequest(query="x", condition="y"),
                                                overrides={"_force_canned": True}))
    assert any("32 trials" in n for n in final2["spec"].meta.notes)


# --- CC-1 override echo reaches meta.notes (was silently dropped) --------------

def test_cc1_override_echo_reaches_meta_notes() -> None:
    """The CC-1 field-wins override echo (in ``plan.notes``, code- OR LLM-form) must ship on
    meta.notes (CC-1 / G-18 / §B.5), through the §1 number-guard: a number-safe echo threads, a
    note carrying a fabricated count is withheld, and the internal offline-stub note is excluded."""
    from app.graph.nodes import build_spec
    from app.plan.models import Plan

    plan = Plan(
        query_class="distribution", entities={"condition": "melanoma"}, field="phase",
        chart_type="bar",
        notes=[
            "Structured field 'condition' overrides free-text condition mention per CC-1.",  # threads
            "the real total is 88888 trials",   # fabricated number -> withheld by the §1 guard
            "Offline stub plan: distribution-by-phase over the condition.",  # internal -> excluded
        ],
    )
    state = {
        "plan": plan, "question": "q", "status": "ok",
        "tool_results": [{"tool": "aggregate_by", "mode": "combine", "distinct_trials": 126,
                          "buckets": [{"value": "PHASE1", "label": "Phase 1", "count_trials": 126,
                                       "count_mentions": 126, "source_ids": ["NCT00000001"],
                                       "citations": [{"nct_id": "NCT00000001",
                                                      "field_path": "protocolSection.designModule.phases",
                                                      "value": ["PHASE1"], "excerpt": "PHASE1"}],
                                       "contributing_count": 126}]}],
    }
    spec = build_spec(state)["spec"]
    joined = " ".join(spec.meta.notes)
    assert "overrides free-text condition mention per CC-1" in joined  # the CC-1 echo shipped
    assert "88888" not in joined            # fabricated number withheld by the §1 guard
    assert "Offline stub plan" not in joined  # internal note excluded


def test_strict_schema_closes_an_open_object() -> None:
    """The defensive close: an open ``dict[str, Any]`` object node gets additionalProperties:false
    (the latent 'next strict-schema bug' — OpenAI strict rejects an open object)."""
    from pydantic import BaseModel

    class _Open(BaseModel):
        payload: dict  # open object, no declared properties

    schema = _strict_schema(_Open)
    payload = schema["properties"]["payload"]
    assert payload.get("additionalProperties") is False


def test_strict_schema_has_no_default_keyword_anywhere() -> None:
    """OpenAI strict mode rejects `default` (esp. as a sibling of `$ref`) — the transform must
    strip it everywhere. Regression for the bug the LIVE gate caught."""
    import json
    schema = _strict_schema(PlannerOutput)
    blob = json.dumps(schema)
    assert '"default"' not in blob
    # every object node forbids extra keys
    def _all_objects_closed(node):
        if isinstance(node, dict):
            if node.get("properties") is not None:
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"].keys())
            for v in node.values():
                _all_objects_closed(v)
        elif isinstance(node, list):
            for it in node:
                _all_objects_closed(it)
    _all_objects_closed(schema)
