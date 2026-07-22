"""The provider-agnostic LLM adapter (ARCHITECTURE_SPEC §3.1, §B.1).

This is the ONLY seam through which any graph node talks to a language model
(C-99): the planner, the Intent Reviewer, and the Output Reviewer all call
``propose``/``verify`` on an ``LLMAdapter`` instance — never a provider SDK
directly. That is what lets the planner run a strong model and the reviewers a
cheaper one (``propose`` resolves the planner model, ``verify`` the reviewer
model), and what lets Claude/OpenAI/OpenRouter swap in without touching a
single graph node.

Three concrete adapters ship, all in this file:

* ``StubAdapter`` — the default (``LLM_PROVIDER`` unset ⇒ ``"stub"``). It never
  opens a network connection and never reads a real provider key. When a caller
  supplies ``canned``, it validates that dict straight into the requested
  ``response_model``; when it doesn't, the adapter still has to return
  *something*, so it constructs a best-effort default instance and validates
  that instead. Either way the "always validate" discipline holds — every
  successful return is a real, Pydantic-validated ``response_model`` instance,
  never a raw dict; the no-``canned`` path raises ``ValueError``/``ValidationError``
  instead of guessing when the schema cannot be satisfied by defaults (e.g. a
  required, constrained, or recursive field) — see ``_best_effort_instance``.
* ``OpenAIAdapter`` — real OpenAI-compatible calls; also backs the
  ``openrouter`` provider (same class, different ``base_url``/key env var).
* ``AnthropicAdapter`` — real Anthropic calls.

Both real adapters are lazy (no client, socket, or key read at import or
``__init__``), redact provider/network failures into ``AdapterError``, and
normalize STRUCTURED OUTPUT only — there is no canonical tool-call/result type
in this module (see ``LLMAdapter.propose`` for what ``tools`` actually does).
``get_adapter`` resolves ``stub``/``openai``/``openrouter``/``anthropic``; the
``NotImplementedError`` it can raise is now only for an UNKNOWN provider name.

One disclosed gap: neither real adapter sets a request timeout, so the provider
SDK's own default applies. The graph's wall-clock deadline is checked before
``plan`` and inside ``execute`` (``app.graph.guards``), never around a reviewer
call — a slow provider is bounded by the SDK, not by this system.
"""

from __future__ import annotations

import json
import logging
import os
import types
import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

_LOG = logging.getLogger(__name__)

# The two flavors `typing.get_origin` reports for a Union, depending on whether it
# was written as `typing.Union[...]`/`typing.Optional[...]` or the PEP 604 `X | Y`
# spelling. Both must be recognized when walking an annotation tree.
_UNION_ORIGINS = (typing.Union, types.UnionType)


class CapabilityDescriptor(BaseModel):
    """What a given adapter/model *declares* it can do (ARCHITECTURE_SPEC §B.1).

    The design intent was that nodes query this instead of ever branching on a
    provider name. As shipped that is aspirational, not wired: nothing in
    ``app/`` reads it (repo-wide, the only caller is ``tests/test_llm.py``), and
    both real adapters return one hardcoded descriptor regardless of which model
    they were pointed at — ``max_context=200_000`` in particular is a nominal
    figure, not a per-model lookup. Read it as the declared shape of the
    capability seam. The one place a capability really does vary at runtime
    (Anthropic native structured output vs the forced-tool fallback) is settled
    by catching the provider's rejection, not by consulting this object.
    """

    supports_forced_tool_choice: bool
    supports_parallel_tool_calls: bool
    supports_native_structured_output: bool
    system_prompt_style: Literal["system_param", "first_message"]
    json_schema_dialect: str
    max_context: int


class LLMAdapter(ABC):
    """Provider-agnostic interface every graph node calls through (C-99).

    Two entry points, matching the two shapes of LLM work in this system
    (ARCHITECTURE_SPEC §3.1/§3.4/§3.8):

    * ``propose`` — the planner's structured-output call: turn a question into a
      typed object (a ``Plan``).
    * ``verify`` — a bounded structured judgment call used by the two reviewers:
      turn a question + already-computed artifact into a typed verdict.

    Both always return a schema-validated instance of ``response_model``; a
    caller never receives a raw dict or an unvalidated object.
    """

    @abstractmethod
    def capabilities(self) -> CapabilityDescriptor:
        """Describe what this adapter/model combination can do (§B.1)."""
        raise NotImplementedError

    @abstractmethod
    def propose(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        tools: list[dict] | None = None,
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        """Structured-output entry point (§3.1). Returns a schema-validated
        instance of ``response_model``.

        ``tools`` is a pass-through, not a normalized surface: the shipped
        planner always passes ``None`` (``app.llm.planner.plan_request``), and
        this module defines no canonical tool-call/result type. When tools ARE
        passed, ``OpenAIAdapter`` forwards them verbatim with
        ``tool_choice="auto"`` and ``AnthropicAdapter.propose`` drops them.
        """
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        """A bounded structured judgment call (used by the Intent/Output
        reviewers, §3.4/§3.8). Returns a schema-validated instance of
        ``response_model``."""
        raise NotImplementedError


# Small, deliberate cap on how deep `_default_for_annotation`/`_best_effort_instance`
# will recurse into nested BaseModel fields before giving up. Paired with `_seen`
# (the set of model types already on the current recursion path), this turns an
# infinite `RecursionError` on a self-referential/mutually-recursive required
# field into a clean, catchable `ValueError`.
_MAX_DEFAULT_DEPTH = 8


def _default_for_annotation(
    annotation: Any, *, _depth: int = 0, _seen: frozenset[type] = frozenset()
) -> Any:
    """Best-effort default value for a single type annotation.

    Recurses into Optional/Union, Literal, containers, Enums, and nested
    BaseModel fields. Deliberately simple — this only exists so ``StubAdapter``
    can hand back *some* schema-valid instance when no ``canned`` payload was
    given; it is never asked to be a clever guesser. ``_depth``/``_seen`` are
    internal recursion-guard bookkeeping (see ``_best_effort_instance``) — never
    passed by callers.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in _UNION_ORIGINS:
        # Optional[X] / X | None -> None is the obvious "nothing to say" default.
        if type(None) in args:
            return None
        return _default_for_annotation(args[0], _depth=_depth, _seen=_seen)

    if origin is Literal:
        return args[0]

    if origin in (list, set, frozenset):
        return origin()
    if origin is tuple:
        return ()
    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return next(iter(annotation))
        if issubclass(annotation, BaseModel):
            return _best_effort_instance(annotation, _depth=_depth + 1, _seen=_seen)
        if annotation is str:
            return ""
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False

    # Unrecognized/unparameterized annotation (e.g. bare `Any`) — None is the
    # least-committal placeholder; model_validate will reject it if the field
    # actually requires something more specific, which is the correct failure.
    return None


def _best_effort_instance(
    response_model: type[BaseModel], *, _depth: int = 0, _seen: frozenset[type] = frozenset()
) -> BaseModel:
    """Construct + validate a default instance of ``response_model``.

    Fields with a Pydantic default are left alone (the default applies).
    Required fields get a type-appropriate placeholder from
    ``_default_for_annotation``. The result is always passed back through
    ``model_validate`` — never hand-assembled and returned unchecked — so the
    "always validate" discipline holds even on the no-``canned`` path: this
    returns a validated instance, or raises ``ValueError`` (a schema too deep
    or self-/mutually-recursive to default-fill, see ``_MAX_DEFAULT_DEPTH``) or
    ``ValidationError`` (a constrained field no placeholder satisfies) if the
    schema cannot be satisfied by defaults.

    ``_depth``/``_seen`` are the recursion guard: ``_seen`` is the set of model
    types already on the current recursion path (a required self-referential
    or mutually-recursive field re-encounters its own type almost immediately);
    ``_depth`` is a backstop for pathological deep-but-non-cyclic nesting. Both
    are internal — never passed by callers.
    """
    if response_model in _seen or _depth > _MAX_DEFAULT_DEPTH:
        raise ValueError(
            f"cannot build a default for recursive/over-deep schema {response_model.__name__!r} "
            f"(no `canned` payload was supplied for it)"
        )
    _seen = _seen | {response_model}
    data: dict[str, Any] = {}
    for name, field in response_model.model_fields.items():
        if field.is_required():
            data[name] = _default_for_annotation(field.annotation, _depth=_depth + 1, _seen=_seen)
    return response_model.model_validate(data)


class StubAdapter(LLMAdapter):
    """The default adapter: zero network, zero provider key, fully deterministic.

    ``propose``/``verify`` never call out anywhere. If the caller passes
    ``canned``, that dict is validated straight into ``response_model``. If not,
    a best-effort default instance is constructed and validated instead — every
    caller that supplies ``canned`` gets back a real validated object rather
    than ``None`` or a bare dict; without it, the caller gets a validated
    object or a raised ``ValueError``/``ValidationError`` if the schema cannot
    be satisfied by defaults (see ``_best_effort_instance``).
    """

    _CAPABILITIES = CapabilityDescriptor(
        supports_forced_tool_choice=True,
        supports_parallel_tool_calls=True,
        supports_native_structured_output=True,
        system_prompt_style="system_param",
        json_schema_dialect="draft2020-12",
        max_context=200_000,
    )

    def capabilities(self) -> CapabilityDescriptor:
        return self._CAPABILITIES

    def propose(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        tools: list[dict] | None = None,
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del system, messages, tools, model  # unused in Phase 0 — no call is made
        if canned is not None:
            return response_model.model_validate(canned)
        return _best_effort_instance(response_model)

    def verify(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del system, messages, model  # unused in Phase 0 — no call is made
        if canned is not None:
            return response_model.model_validate(canned)
        return _best_effort_instance(response_model)


# --------------------------------------------------------------------------- #
# The real provider adapters (OpenAI / OpenRouter / Anthropic)                   #
# --------------------------------------------------------------------------- #


class AdapterError(RuntimeError):
    """A redacted, machine-readable adapter failure.

    Provider/network exceptions can carry request context; we never let them
    reach a caller verbatim (they could echo prompt content and are noisy). The
    real exception is logged server-side via ``_LOG``; callers see a fixed
    ``message`` plus a stable ``code`` they can branch on. The provider API key
    is never read into a message here or anywhere else.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


# One bounded correction turn: when the model returns unparseable or
# schema-invalid JSON, we echo it back with the concrete error and ask once
# more. This is the only place a validation failure is "retried".
_REASK_TEMPLATE = (
    "Your previous reply could not be parsed/validated against the required JSON "
    "schema (error: {error}). Reply with ONLY a corrected JSON object that "
    "validates against the schema. Do not include any prose or code fences."
)

# ~1500 output tokens is comfortable for the closed decision objects this system
# asks for (a plan / a verdict) without risking a truncated JSON body.
_OPENAI_MAX_COMPLETION_TOKENS = 1500
_ANTHROPIC_MAX_TOKENS = 1500

# Name of the single forced tool used for Anthropic's structured-output fallback.
_EMIT_TOOL_NAME = "emit"


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from either an SDK response object (attribute) or a plain
    dict (key). Lets the parsing code treat real SDK objects and test fakes
    uniformly without caring which shape it got."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _openai_bad_request() -> type[BaseException]:
    """Lazily import the OpenAI 400 class (keeps the SDK out of import time)."""
    from openai import BadRequestError

    return BadRequestError


def _anthropic_bad_request() -> type[BaseException]:
    """Lazily import the Anthropic 400 class (keeps the SDK out of import time)."""
    from anthropic import BadRequestError

    return BadRequestError


def _strict_schema(response_model: type[BaseModel]) -> dict:
    """Return a strict-mode-safe JSON schema for ``response_model``.

    Both OpenAI (``strict: True``) and Anthropic native structured output want
    every object node to (a) forbid extra keys (``additionalProperties: false``)
    and (b) list *every* property in ``required`` — strict mode treats optional
    fields as "present but nullable", not "absent". Pydantic already emits the
    nullable/``anyOf``-with-``null`` schema for an ``X | None`` field, which
    satisfies "required + nullable"; we do NOT strip that null. ``$ref``/``$defs``
    are left inline (both providers accept them).

    The response models this system uses are designed closed — no bare
    ``dict[str, Any]``, no recursion — so this walk is mechanical and total.
    """
    schema = response_model.model_json_schema()
    _strictify(schema)
    return schema


# JSON-Schema keywords whose value is a `{name: subschema}` MAP rather than a schema node.
# Their keys are author-chosen names (a model's field names, a `$defs` entry's class name),
# so the keyword handling in `_strictify` must not be applied to them directly — see the
# `default`-named-field note in that docstring.
_SCHEMA_MAP_KEYWORDS = ("properties", "$defs")


def _strictify(node: Any) -> None:
    """In-place: on every object node, set ``additionalProperties: false`` and
    make all declared properties ``required``; strip the ``default`` keyword;
    recurse into every child.

    ``default`` MUST be removed everywhere: OpenAI strict mode rejects it, and
    Pydantic emits it as a *sibling of ``$ref``* for a nested-model field that has
    a default (``{"$ref": "#/$defs/X", "default": {...}}``) — which OpenAI rejects
    outright (``$ref cannot have keywords {'default'}``). Stripping it also makes
    "optional" mean "required + nullable" uniformly (strict mode's contract), since
    a defaulted field is no longer advertised as skippable. (Caught by a LIVE
    provider call — an offline fake-client test can't exercise the real validator.)

    Recursion descends through ``properties``/``$defs`` a level at a time because those
    are ``{name: subschema}`` maps, not schema nodes: walking them as nodes deleted a
    field literally *named* ``default`` from ``properties`` while ``required`` (computed
    first) still listed it — an invalid schema. No shipped response model has such a
    field, so this was latent, not live; the split keeps the walk total either way.
    """
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        elif node.get("type") == "object":
            # An OPEN object node (a `dict[str, Any]` with no declared properties) — Pydantic
            # leaves `additionalProperties: true`, which OpenAI strict mode rejects. Our response
            # models are all closed, so this never fires today; it's the defensive close for the
            # "next strict-schema bug" class (the `default`-on-`$ref` bug's sibling) — a future
            # open-dict field can't silently break a live strict call.
            node["additionalProperties"] = False
        # Snapshot items() so the keys we just added don't perturb iteration.
        for key, value in list(node.items()):
            if key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
                for subschema in list(value.values()):
                    _strictify(subschema)
            else:
                _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


class OpenAIAdapter(LLMAdapter):
    """Real OpenAI (and OpenAI-compatible, e.g. OpenRouter) adapter.

    Lazy: no client, socket, or API key is touched at import or ``__init__`` —
    the client is built on the first ``propose``/``verify`` call and cached. An
    optional ``client`` may be injected (offline unit tests pass a fake with no
    network). Structured output uses native ``response_format`` json_schema with
    ``strict: True``; ``max_completion_tokens`` (gpt-5.x rejects ``max_tokens``);
    ``temperature`` is never sent. The key is read only inside this adapter and
    is never logged or echoed (errors are redacted to ``AdapterError``).

    ``reasoning_effort`` is fail-soft (sent, then dropped and retried once on a
    400) but is a direct-construction knob ONLY: ``get_adapter`` never passes it
    and no env var feeds it, so on the wired path it is always ``None``. The
    model/base-url knobs by contrast ARE env-wired (``LLM_MODEL_PLANNER`` /
    ``LLM_MODEL_REVIEWER`` / ``LLM_BASE_URL``).
    """

    _DEFAULT_PLANNER_MODEL = "gpt-5.4"
    _DEFAULT_REVIEWER_MODEL = "gpt-5.4-mini"

    def __init__(
        self,
        *,
        client: Any | None = None,
        planner_model: str | None = None,
        reviewer_model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client  # injection seam; None => build lazily on first call
        self._planner_model = (
            planner_model or os.environ.get("LLM_MODEL_PLANNER") or self._DEFAULT_PLANNER_MODEL
        )
        self._reviewer_model = (
            reviewer_model or os.environ.get("LLM_MODEL_REVIEWER") or self._DEFAULT_REVIEWER_MODEL
        )
        # An explicit base_url arg wins; otherwise fall back to LLM_BASE_URL
        # (None => the OpenAI default endpoint).
        self._base_url = base_url if base_url is not None else os.environ.get("LLM_BASE_URL")
        self._api_key_env = api_key_env
        self._reasoning_effort = reasoning_effort

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            supports_forced_tool_choice=True,
            supports_parallel_tool_calls=True,
            supports_native_structured_output=True,
            system_prompt_style="system_param",
            json_schema_dialect="draft2020-12",
            max_context=200_000,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                # Redacted: names the env var, never a value.
                raise AdapterError(
                    f"LLM credentials not configured (set {self._api_key_env})",
                    code="missing_api_key",
                )
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def propose(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        tools: list[dict] | None = None,
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del canned  # real adapter ignores the stub's offline answer
        return self._run(
            system=system,
            messages=messages,
            response_model=response_model,
            tools=tools,
            model=model or self._planner_model,
        )

    def verify(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del canned
        return self._run(
            system=system,
            messages=messages,
            response_model=response_model,
            tools=None,
            model=model or self._reviewer_model,
        )

    def _run(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        tools: list[dict] | None,
        model: str,
    ) -> BaseModel:
        try:
            client = self._get_client()
            schema = _strict_schema(response_model)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
            base_messages = [{"role": "system", "content": system}, *messages]

            content = self._complete(client, base_messages, response_format, tools, model)
            try:
                return response_model.model_validate(json.loads(content))
            except (ValidationError, ValueError) as exc:
                _LOG.warning(
                    "openai structured output failed validation; re-asking once (%s)",
                    type(exc).__name__,
                )
                retry_messages = [
                    *base_messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": _REASK_TEMPLATE.format(error=exc)},
                ]
                content2 = self._complete(client, retry_messages, response_format, tools, model)
                try:
                    return response_model.model_validate(json.loads(content2))
                except (ValidationError, ValueError) as exc2:
                    raise AdapterError(
                        "model response did not satisfy the requested schema",
                        code="invalid_response",
                    ) from exc2
        except AdapterError:
            raise
        except Exception:
            # Redact provider/network failures; log the real cause server-side.
            _LOG.exception("openai adapter call failed")
            raise AdapterError("LLM provider call failed", code="provider_error") from None

    def _complete(
        self,
        client: Any,
        messages: list[dict],
        response_format: dict,
        tools: list[dict] | None,
        model: str,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "max_completion_tokens": _OPENAI_MAX_COMPLETION_TOKENS,
        }
        if tools:
            # Forward-compatible path (Phase-4 planner passes tools=None).
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._create(client, kwargs)
        return self._extract(resp)

    def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """Make the create call, fail-soft on ``reasoning_effort``: gpt-5.x may
        400 on it, in which case we drop it and retry once. ``temperature`` is
        never sent."""
        if self._reasoning_effort is not None:
            try:
                return client.chat.completions.create(
                    reasoning_effort=self._reasoning_effort, **kwargs
                )
            except _openai_bad_request() as exc:
                _LOG.warning(
                    "openai reasoning_effort=%r rejected (HTTP 400: %s); retrying without it",
                    self._reasoning_effort,
                    type(exc).__name__,
                )
        return client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract(resp: Any) -> str:
        choices = _field(resp, "choices") or []
        if not choices:
            return ""
        message = _field(choices[0], "message")
        return _field(message, "content") or ""


class AnthropicAdapter(LLMAdapter):
    """Real Anthropic adapter.

    Lazy client (same discipline as ``OpenAIAdapter``). Structured output prefers
    native ``output_config`` json_schema; if the SDK/model rejects it, it falls
    back to a single forced tool whose ``input_schema`` is the response schema,
    reading ``tool_use.input``. That switch is EXCEPTION-driven, not
    capability-driven: ``_obtain`` tries native and catches the provider's 400
    (``BadRequestError``) or the SDK's ``TypeError`` on an unknown param —
    ``capabilities()`` is never consulted (it reports one hardcoded descriptor;
    see ``CapabilityDescriptor``). Which path ran is logged. ``max_tokens`` (Anthropic name,
    NOT ``max_completion_tokens``); ``temperature``/``top_p``/``budget_tokens``
    are never sent (they 400 on Opus 4.8 / Haiku 4.5). Same bounded re-ask and
    redaction as OpenAI.
    """

    _DEFAULT_PLANNER_MODEL = "claude-opus-4-8"
    _DEFAULT_REVIEWER_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        *,
        client: Any | None = None,
        planner_model: str | None = None,
        reviewer_model: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self._client = client  # injection seam; None => build lazily on first call
        self._planner_model = (
            planner_model or os.environ.get("LLM_MODEL_PLANNER") or self._DEFAULT_PLANNER_MODEL
        )
        self._reviewer_model = (
            reviewer_model or os.environ.get("LLM_MODEL_REVIEWER") or self._DEFAULT_REVIEWER_MODEL
        )
        self._api_key_env = api_key_env

    def capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            supports_forced_tool_choice=True,
            supports_parallel_tool_calls=True,
            supports_native_structured_output=True,
            system_prompt_style="system_param",
            json_schema_dialect="draft2020-12",
            max_context=200_000,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise AdapterError(
                    f"LLM credentials not configured (set {self._api_key_env})",
                    code="missing_api_key",
                )
            self._client = Anthropic(api_key=api_key)
        return self._client

    def propose(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        tools: list[dict] | None = None,
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del tools, canned  # native structured output; stub answer ignored
        return self._run(
            system=system,
            messages=messages,
            response_model=response_model,
            model=model or self._planner_model,
        )

    def verify(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        model: str | None = None,
        canned: dict | None = None,
    ) -> BaseModel:
        del canned
        return self._run(
            system=system,
            messages=messages,
            response_model=response_model,
            model=model or self._reviewer_model,
        )

    def _run(
        self,
        *,
        system: str,
        messages: list[dict],
        response_model: type[BaseModel],
        model: str,
    ) -> BaseModel:
        try:
            client = self._get_client()
            schema = _strict_schema(response_model)
            base_messages = list(messages)

            raw = self._obtain(client, system, base_messages, model, schema)
            try:
                return response_model.model_validate(json.loads(raw))
            except (ValidationError, ValueError) as exc:
                _LOG.warning(
                    "anthropic structured output failed validation; re-asking once (%s)",
                    type(exc).__name__,
                )
                retry_messages = [
                    *base_messages,
                    {"role": "assistant", "content": raw or ""},
                    {"role": "user", "content": _REASK_TEMPLATE.format(error=exc)},
                ]
                raw2 = self._obtain(client, system, retry_messages, model, schema)
                try:
                    return response_model.model_validate(json.loads(raw2))
                except (ValidationError, ValueError) as exc2:
                    raise AdapterError(
                        "model response did not satisfy the requested schema",
                        code="invalid_response",
                    ) from exc2
        except AdapterError:
            raise
        except Exception:
            _LOG.exception("anthropic adapter call failed")
            raise AdapterError("LLM provider call failed", code="provider_error") from None

    def _obtain(
        self,
        client: Any,
        system: str,
        messages: list[dict],
        model: str,
        schema: dict,
    ) -> str:
        """Return the model's JSON payload as a string, trying native structured
        output first and falling back to a forced single tool if native is
        unavailable. Logs which path ran."""
        try:
            resp = self._messages_call(client, system, messages, model, schema, native=True)
        except (_anthropic_bad_request(), TypeError) as exc:
            _LOG.warning(
                "anthropic native output_config unavailable (%s); using forced-tool fallback",
                type(exc).__name__,
            )
            resp = self._messages_call(client, system, messages, model, schema, native=False)
            _LOG.info("anthropic structured output via forced-tool fallback")
            return self._tool_json(resp)
        _LOG.info("anthropic structured output via native output_config")
        return self._text_json(resp)

    @staticmethod
    def _messages_call(
        client: Any,
        system: str,
        messages: list[dict],
        model: str,
        schema: dict,
        *,
        native: bool,
    ) -> Any:
        # No temperature/top_p/budget_tokens — they 400 on Opus 4.8 / Haiku 4.5.
        if native:
            return client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=_ANTHROPIC_MAX_TOKENS,
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        return client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            tools=[
                {
                    "name": _EMIT_TOOL_NAME,
                    "description": "Emit the structured result as this tool's input.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": _EMIT_TOOL_NAME},
        )

    @staticmethod
    def _text_json(resp: Any) -> str:
        for block in _field(resp, "content") or []:
            if _field(block, "type") == "text":
                return _field(block, "text") or ""
        return ""

    @staticmethod
    def _tool_json(resp: Any) -> str:
        for block in _field(resp, "content") or []:
            if _field(block, "type") == "tool_use":
                return json.dumps(_field(block, "input") or {})
        return ""


def get_adapter(provider: str | None = None) -> LLMAdapter:
    """Adapter factory (§3.1).

    Provider resolution: explicit ``provider`` arg, else ``LLM_PROVIDER`` env var,
    else ``"stub"``. The real adapters read their provider key only inside the
    adapter, lazily on first call — so constructing one here opens no socket and
    needs no key (§A(e), the system's one secret; never logged, redacted on
    error). An unknown provider is a clear ``NotImplementedError``.
    """
    resolved = provider or os.environ.get("LLM_PROVIDER", "stub")

    if resolved == "stub":
        return StubAdapter()
    if resolved == "openai":
        return OpenAIAdapter()
    if resolved == "openrouter":
        return OpenAIAdapter(
            base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY"
        )
    if resolved == "anthropic":
        return AnthropicAdapter()

    raise NotImplementedError(f"unknown LLM provider {resolved!r}")
