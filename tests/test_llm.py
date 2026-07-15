"""Self-verification for the Phase-0 LLM layer (adapter, planner stub, reviewer
stubs -- ARCHITECTURE_SPEC §3.1/§3.2/§3.4/§3.8).

Three guarantees:

1. ``get_adapter`` defaults to a real, $0, no-network ``StubAdapter`` whose
   ``capabilities()`` is a realistic Claude-like descriptor; the Phase-4
   provider names are wired to a documented ``NotImplementedError``, not
   silently swallowed.
2. ``plan_request`` / ``review_intent`` / ``review_output_llm`` all route
   through the adapter (C-99) and come back as validated Plan / verdict
   instances, matching the canned Phase-0 content.
3. ``propose`` always hands back a real, schema-validated instance of the
   requested ``response_model`` -- canned or not -- never a raw dict.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.api.schemas import ChartType, Meta, VisualizeResponse
from app.llm.adapter import (
    AnthropicAdapter,
    CapabilityDescriptor,
    LLMAdapter,
    OpenAIAdapter,
    StubAdapter,
    get_adapter,
)
from app.llm.planner import plan_request
from app.llm.reviewers import IntentVerdict, OutputVerdict, review_intent, review_output_llm
from app.plan.models import Plan

# --- get_adapter -----------------------------------------------------------


def test_get_adapter_default_is_stub_adapter() -> None:
    adapter = get_adapter()
    assert isinstance(adapter, StubAdapter)
    assert isinstance(adapter, LLMAdapter)


def test_get_adapter_explicit_stub() -> None:
    assert isinstance(get_adapter("stub"), StubAdapter)


@pytest.mark.parametrize(
    "provider,adapter_cls",
    [("openai", OpenAIAdapter), ("openrouter", OpenAIAdapter), ("anthropic", AnthropicAdapter)],
)
def test_get_adapter_real_providers_return_lazy_adapter(
    provider: str, adapter_cls: type[LLMAdapter]
) -> None:
    # Phase 4: the real providers are now implemented. Construction is lazy — no
    # key or socket is required to obtain the adapter (the key is read only on the
    # first propose/verify call).
    adapter = get_adapter(provider)
    assert isinstance(adapter, adapter_cls)
    assert isinstance(adapter, LLMAdapter)


def test_get_adapter_unknown_provider_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        get_adapter("bogus-provider")


# --- capabilities ------------------------------------------------------------


def test_capabilities_returns_realistic_descriptor() -> None:
    caps = get_adapter().capabilities()
    assert isinstance(caps, CapabilityDescriptor)
    assert caps.supports_forced_tool_choice is True
    assert caps.supports_parallel_tool_calls is True
    assert caps.supports_native_structured_output is True
    assert caps.system_prompt_style == "system_param"
    assert isinstance(caps.json_schema_dialect, str) and caps.json_schema_dialect
    assert caps.max_context == 200_000


# --- planner stub ------------------------------------------------------------


def test_plan_request_returns_valid_distribution_plan() -> None:
    plan = plan_request(
        get_adapter(), {"condition": "pancreatic cancer", "interventional_only": True}
    )
    assert isinstance(plan, Plan)
    assert plan.query_class == "distribution"
    assert plan.field == "phase"
    assert plan.chart_type == ChartType.BAR
    assert plan.entities.get("condition") == "pancreatic cancer"
    assert plan.interventional_only is True
    assert ChartType.HISTOGRAM in plan.alternates


def test_plan_request_defaults_condition_when_absent() -> None:
    plan = plan_request(get_adapter(), {})
    assert plan.entities.get("condition") == "pancreatic cancer"
    assert plan.interventional_only is False


# --- reviewer stubs ------------------------------------------------------------


def test_review_intent_approves() -> None:
    plan = plan_request(get_adapter(), {"condition": "melanoma"})
    verdict = review_intent(
        get_adapter(), "How are melanoma trials distributed by phase?", plan
    )
    assert isinstance(verdict, IntentVerdict)
    assert verdict.decision == "approve"


def test_review_output_llm_approves() -> None:
    spec = VisualizeResponse(
        status="empty", kind="answer", answer="No trials found.", meta=Meta()
    )
    verdict = review_output_llm(get_adapter(), "how many trials?", spec)
    assert isinstance(verdict, OutputVerdict)
    assert verdict.decision == "approve"


# --- propose always returns a validated response_model instance --------------


class _TinyModel(BaseModel):
    ok: bool = True


class _RequiredFieldModel(BaseModel):
    name: str
    count: int
    chart: ChartType


def test_propose_returns_response_model_instance_without_canned() -> None:
    """Even with no ``canned`` payload, propose must still return a real,
    validated instance of the requested response_model (never None/dict)."""
    result = get_adapter().propose(
        system="test",
        messages=[{"role": "user", "content": "hi"}],
        response_model=_TinyModel,
    )
    assert isinstance(result, _TinyModel)
    assert result.ok is True


def test_propose_best_effort_default_satisfies_required_fields() -> None:
    result = get_adapter().propose(
        system="test",
        messages=[{"role": "user", "content": "hi"}],
        response_model=_RequiredFieldModel,
    )
    assert isinstance(result, _RequiredFieldModel)
    assert isinstance(result.name, str)
    assert isinstance(result.count, int)
    assert isinstance(result.chart, ChartType)


def test_propose_with_canned_validates_into_response_model() -> None:
    result = get_adapter().propose(
        system="test",
        messages=[{"role": "user", "content": "hi"}],
        response_model=_RequiredFieldModel,
        canned={"name": "phase", "count": 3, "chart": "bar"},
    )
    assert isinstance(result, _RequiredFieldModel)
    assert result.name == "phase"
    assert result.count == 3
    assert result.chart == ChartType.BAR


def test_verify_always_returns_response_model_instance() -> None:
    result = get_adapter().verify(
        system="test",
        messages=[{"role": "user", "content": "hi"}],
        response_model=_TinyModel,
    )
    assert isinstance(result, _TinyModel)
