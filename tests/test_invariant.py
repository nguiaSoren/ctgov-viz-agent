"""The governing invariant, tested (ARCHITECTURE_SPEC §1, CC-2, C-1, D-CC2, V-7):

    The LLM decides WHAT to compute; deterministic tools compute it.
    The model NEVER emits a number.

Phase 4 puts a real LLM in the loop, so this must be proven, not asserted. It holds *by
construction*: the only objects the LLM can return are the three closed structured-output models
(``PlannerOutput`` / ``IntentVerdict`` / ``OutputVerdict``), and NONE of them carries a
trial-count field — so a model literally cannot hand back a count. Every number the user sees is
inserted by CODE from a tool result (the aggregation core / the code-templated title + answer).
"""

from __future__ import annotations

from typing import Any

from app.llm.planner import PlannerOutput
from app.llm.reviewers import IntentVerdict, OutputVerdict
from app.plan.models import Plan
from app.viz.spec import build_envelope

# The ONLY integer fields any LLM-output model may legitimately carry are the inclusive year
# bounds of a filter — configuration the user typed, never a computed trial tally.
_ALLOWED_INT_FIELDS = {"start_year", "end_year"}
# Field names that would smell like a model emitting a tally.
_BANNED_COUNT_NAMES = {"count", "total", "trial_count", "count_trials", "count_mentions", "n", "total_count"}


def _leaf_number_fields(schema: dict, defs: dict, path: str = "") -> list[tuple[str, str]]:
    """Walk a Pydantic JSON schema; return ``(field_name, json_type)`` for every leaf typed
    ``integer``/``number`` (resolving ``$ref``/``anyOf``). Closed models only — no recursion risk."""
    out: list[tuple[str, str]] = []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return out
    for name, sub in props.items():
        out.extend(_field_number_types(name, sub, defs))
    return out


def _field_number_types(name: str, sub: dict, defs: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    candidates: list[Any] = [sub]
    # Follow anyOf (nullable optionals) and $ref (nested closed models).
    candidates.extend(sub.get("anyOf", []))
    for cand in list(candidates):
        ref = cand.get("$ref") if isinstance(cand, dict) else None
        if ref:
            target = defs.get(ref.split("/")[-1], {})
            out.extend(_leaf_number_fields(target, defs))
    for cand in candidates:
        if isinstance(cand, dict) and cand.get("type") in ("integer", "number"):
            out.append((name, cand["type"]))
    return out


def test_llm_output_models_carry_no_trial_count() -> None:
    """None of the three structured-output models the LLM can return carries a numeric
    trial-count field — the invariant made structural. The only ints allowed are year bounds."""
    for model in (PlannerOutput, IntentVerdict, OutputVerdict):
        schema = model.model_json_schema()
        defs = schema.get("$defs", {})
        number_fields = {name for name, _ in _leaf_number_fields(schema, defs)}
        stray = number_fields - _ALLOWED_INT_FIELDS
        assert not stray, f"{model.__name__} exposes numeric field(s) the LLM could fill: {stray}"
        assert not (number_fields & _BANNED_COUNT_NAMES), (
            f"{model.__name__} exposes a count-like field: {number_fields & _BANNED_COUNT_NAMES}"
        )


def test_number_in_spec_is_inserted_by_code_from_the_tool() -> None:
    """The displayed number is the tool's computed total, inserted by the CODE builder — not
    anything the LLM produced. A single_value plan + a tool_result carrying 42 -> a datum of 42."""
    plan = Plan(query_class="single_value", entities={"condition": "melanoma"}, chart_type="single_value")
    tool_results = [{
        "tool": "count_trials", "total_count": 42, "kind": "visualization",
        "citations": [{"nct_id": "NCT00000001", "field_path":
                       "protocolSection.identificationModule.nctId", "value": "NCT00000001",
                       "excerpt": "NCT00000001"}],
    }]
    spec = build_envelope(
        plan=plan, tool_results=tool_results, status="ok", question="How many melanoma trials?"
    )
    assert spec.kind == "visualization"
    assert spec.visualization is not None
    assert spec.visualization.type.value == "single_value"
    assert spec.visualization.data[0].count_trials == 42  # the code-inserted tool number
    assert spec.meta.count_basis.trials == 42
