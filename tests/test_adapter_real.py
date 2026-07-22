"""Offline tests for the Phase-4 real provider adapters (Interface Contract v4 §1).

Everything here is network-free: the OpenAI/Anthropic adapters accept an injected
fake ``client`` (their documented offline seam), so no key is read and no socket
opens. We exercise:

* ``_strict_schema`` — ``additionalProperties: false`` + all-required on every
  object node, with an ``X | None`` field kept nullable (and still required).
* ``OpenAIAdapter`` — native json_schema call returns a validated model; the
  bounded re-ask on invalid-then-valid JSON; re-ask exhaustion raises
  ``AdapterError``; ``reasoning_effort`` fail-soft on a 400; provider errors are
  redacted; no ``temperature``/``max_tokens`` are ever sent.
* ``AnthropicAdapter`` — native ``output_config`` path, forced-tool fallback when
  native 400s, and the re-ask; no forbidden sampling params are sent.
* ``get_adapter`` — lazy (no key/socket) for the real providers, unchanged for
  stub, and ``NotImplementedError`` for an unknown provider.
"""

from __future__ import annotations

import json
import types
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.llm.adapter import (
    AdapterError,
    AnthropicAdapter,
    LLMAdapter,
    OpenAIAdapter,
    StubAdapter,
    _strict_schema,
    get_adapter,
)

# --------------------------------------------------------------------------- #
# Closed response models used as the structured-output target under test.      #
# --------------------------------------------------------------------------- #


class _Inner(BaseModel):
    label: str
    score: int | None = None  # optional -> nullable schema, still required in strict mode


class _Decision(BaseModel):
    action: Literal["go", "stop"]
    inner: _Inner
    tags: list[str] = []
    note: str | None = None


def _decision_json(**over: object) -> str:
    payload = {"action": "go", "inner": {"label": "x", "score": 3}, "tags": ["a"], "note": None}
    payload.update(over)
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# _strict_schema                                                                #
# --------------------------------------------------------------------------- #


def _object_nodes(node: object) -> list[dict]:
    """Collect every JSON-schema object node (has a ``properties`` dict)."""
    found: list[dict] = []
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            found.append(node)
        for value in node.values():
            found.extend(_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_object_nodes(item))
    return found


def test_strict_schema_closes_and_requires_every_object_node() -> None:
    schema = _strict_schema(_Decision)
    nodes = _object_nodes(schema)
    # Top-level _Decision + the nested _Inner ($defs) are both objects.
    assert len(nodes) >= 2
    for obj in nodes:
        assert obj["additionalProperties"] is False
        # strict mode: every declared property is required.
        assert set(obj["required"]) == set(obj["properties"].keys())


def test_strict_schema_keeps_optional_field_nullable_and_required() -> None:
    schema = _strict_schema(_Decision)
    note = schema["properties"]["note"]
    # `note: str | None` stays nullable (anyOf includes a null branch)...
    assert any(branch.get("type") == "null" for branch in note["anyOf"])
    # ...and is nonetheless listed as required (strict = present-but-nullable).
    assert "note" in schema["required"]


# --------------------------------------------------------------------------- #
# Fakes                                                                         #
# --------------------------------------------------------------------------- #


def _make_400(exc_cls: type[BaseException]) -> BaseException:
    req = httpx.Request("POST", "https://example.test/v1")
    return exc_cls("bad request", response=httpx.Response(400, request=req), body=None)


class _FakeOpenAIClient:
    """Mimics ``client.chat.completions.create(**kwargs) -> ChatCompletion``."""

    def __init__(
        self,
        contents: list[str] | None = None,
        *,
        raise_on_reasoning: bool = False,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._contents = list(contents or [])
        self._raise_on_reasoning = raise_on_reasoning
        self._raise_exc = raise_exc
        self.calls: list[dict] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._raise_on_reasoning and "reasoning_effort" in kwargs:
            from openai import BadRequestError

            raise _make_400(BadRequestError)
        content = self._contents.pop(0)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _text_block(text: str) -> object:
    return types.SimpleNamespace(type="text", text=text)


def _tool_block(payload: dict) -> object:
    return types.SimpleNamespace(type="tool_use", input=payload)


class _FakeAnthropicClient:
    """Mimics ``client.messages.create(**kwargs) -> Message``."""

    def __init__(
        self,
        *,
        native_texts: list[str] | None = None,
        tool_inputs: list[dict] | None = None,
        reject_native: bool = False,
    ) -> None:
        self._native_texts = list(native_texts or [])
        self._tool_inputs = list(tool_inputs or [])
        self._reject_native = reject_native
        self.calls: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if "output_config" in kwargs:
            if self._reject_native:
                from anthropic import BadRequestError

                raise _make_400(BadRequestError)
            return types.SimpleNamespace(content=[_text_block(self._native_texts.pop(0))])
        return types.SimpleNamespace(content=[_tool_block(self._tool_inputs.pop(0))])


# --------------------------------------------------------------------------- #
# OpenAIAdapter                                                                  #
# --------------------------------------------------------------------------- #


def test_openai_propose_returns_validated_model() -> None:
    client = _FakeOpenAIClient([_decision_json()])
    adapter = OpenAIAdapter(client=client)
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    assert result.action == "go"
    assert result.inner.label == "x"
    # Exactly one call; correct token param; no temperature / max_tokens leaked in.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["max_completion_tokens"] == 1500
    assert "temperature" not in call
    assert "max_tokens" not in call
    # Native structured output with strict json_schema.
    rf = call["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "_Decision"
    # System prompt is prepended.
    assert call["messages"][0] == {"role": "system", "content": "s"}


def test_openai_uses_reviewer_model_on_verify() -> None:
    client = _FakeOpenAIClient([_decision_json()])
    adapter = OpenAIAdapter(client=client, planner_model="planner-x", reviewer_model="reviewer-y")
    adapter.verify(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert client.calls[0]["model"] == "reviewer-y"


def test_openai_bounded_reask_on_invalid_then_valid() -> None:
    # First reply is schema-invalid JSON, second is valid -> one re-ask, success.
    bad = json.dumps({"action": "sideways", "inner": {"label": "x"}})  # bad Literal
    client = _FakeOpenAIClient([bad, _decision_json()])
    adapter = OpenAIAdapter(client=client)
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    assert len(client.calls) == 2
    # The re-ask carries the assistant echo + a correction user turn.
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-1]["role"] == "user"
    assert "corrected JSON" in retry_messages[-1]["content"]


def test_openai_reask_exhausted_raises_adapter_error() -> None:
    client = _FakeOpenAIClient(["not json", "still not json"])
    adapter = OpenAIAdapter(client=client)
    with pytest.raises(AdapterError) as excinfo:
        adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert excinfo.value.code == "invalid_response"
    assert len(client.calls) == 2


def test_openai_reasoning_effort_fail_soft_on_400() -> None:
    # NOTE ``reasoning_effort`` is reachable ONLY by constructing the adapter directly,
    # as here: ``get_adapter`` never passes it and no env var feeds it, so nothing in the
    # shipped request path can set it. This test covers the fail-soft, not a live knob.
    client = _FakeOpenAIClient([_decision_json()], raise_on_reasoning=True)
    adapter = OpenAIAdapter(client=client, reasoning_effort="high")
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    # First attempt carried reasoning_effort (rejected); retry dropped it.
    assert len(client.calls) == 2
    assert client.calls[0].get("reasoning_effort") == "high"
    assert "reasoning_effort" not in client.calls[1]


def test_openai_provider_error_is_redacted() -> None:
    client = _FakeOpenAIClient(raise_exc=RuntimeError("boom sk-secret-value"))
    adapter = OpenAIAdapter(client=client)
    with pytest.raises(AdapterError) as excinfo:
        adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert excinfo.value.code == "provider_error"
    # The redacted message must not echo the underlying (secret-bearing) error.
    assert "secret" not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# AnthropicAdapter                                                               #
# --------------------------------------------------------------------------- #


def test_anthropic_native_output_config_path() -> None:
    client = _FakeAnthropicClient(native_texts=[_decision_json()])
    adapter = AnthropicAdapter(client=client)
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert "output_config" in call
    # Anthropic token param + no forbidden sampling params.
    assert call["max_tokens"] == 1500
    assert "max_completion_tokens" not in call
    assert "temperature" not in call
    assert "top_p" not in call
    assert "budget_tokens" not in call


def test_anthropic_forced_tool_fallback_when_native_rejected() -> None:
    client = _FakeAnthropicClient(reject_native=True, tool_inputs=[json.loads(_decision_json())])
    adapter = AnthropicAdapter(client=client)
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    assert len(client.calls) == 2
    # First tried native, second used a forced single tool.
    assert "output_config" in client.calls[0]
    fallback = client.calls[1]
    assert fallback["tools"][0]["name"] == "emit"
    assert fallback["tool_choice"] == {"type": "tool", "name": "emit"}


def test_anthropic_bounded_reask_on_invalid_then_valid() -> None:
    bad = json.dumps({"action": "nope", "inner": {"label": "x"}})
    client = _FakeAnthropicClient(native_texts=[bad, _decision_json()])
    adapter = AnthropicAdapter(client=client)
    result = adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert isinstance(result, _Decision)
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "corrected JSON" in retry_messages[-1]["content"]


def test_anthropic_missing_key_redacted_when_no_client() -> None:
    # No injected client + no env key -> lazy build refuses with a redacted error.
    adapter = AnthropicAdapter(api_key_env="DEFINITELY_UNSET_KEY_ENV_XYZ")
    with pytest.raises(AdapterError) as excinfo:
        adapter.propose(system="s", messages=[{"role": "user", "content": "q"}], response_model=_Decision)
    assert excinfo.value.code == "missing_api_key"


# --------------------------------------------------------------------------- #
# get_adapter                                                                   #
# --------------------------------------------------------------------------- #


def test_get_adapter_openai_is_lazy_no_key_no_socket() -> None:
    # Constructing the adapter must not read a key or open a socket.
    adapter = get_adapter("openai")
    assert isinstance(adapter, OpenAIAdapter)
    assert isinstance(adapter, LLMAdapter)


def test_get_adapter_openrouter_configures_base_url_and_key_env() -> None:
    adapter = get_adapter("openrouter")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter._base_url == "https://openrouter.ai/api/v1"
    assert adapter._api_key_env == "OPENROUTER_API_KEY"


def test_get_adapter_anthropic_is_lazy() -> None:
    assert isinstance(get_adapter("anthropic"), AnthropicAdapter)


def test_get_adapter_stub_unchanged() -> None:
    assert isinstance(get_adapter("stub"), StubAdapter)


def test_get_adapter_bogus_raises() -> None:
    with pytest.raises(NotImplementedError):
        get_adapter("bogus")


def test_stub_adapter_still_validates_canned() -> None:
    # StubAdapter behavior is unchanged: canned validates straight into the model.
    result = StubAdapter().propose(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        response_model=_Decision,
        canned=json.loads(_decision_json()),
    )
    assert isinstance(result, _Decision)
    assert result.action == "go"


def test_stub_adapter_best_effort_without_canned() -> None:
    result = StubAdapter().verify(
        system="s", messages=[{"role": "user", "content": "q"}], response_model=_Inner
    )
    assert isinstance(result, _Inner)


def test_decision_model_rejects_bad_literal() -> None:
    # Sanity: the "bad" payloads above genuinely fail validation.
    with pytest.raises(ValidationError):
        _Decision.model_validate({"action": "sideways", "inner": {"label": "x"}})
