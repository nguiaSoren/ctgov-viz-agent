"""Structured-event logging + a defense-in-depth secret-redaction backstop.

Security posture (ARCHITECTURE_SPEC §A(i), SEC-46/47/48). Registry data is
public, but two things must never reach the logs at info level:

* the **provider API key** — the one secret in the system (``OPENAI_API_KEY`` /
  ``ANTHROPIC_API_KEY`` / ``OPENROUTER_API_KEY``, read only by
  ``app/llm/adapter.py``); and
* the **raw user query / free-text argument values** — the query reveals what a
  researcher is investigating, so it is treated as sensitive.

Two independent guards enforce this, so a mistake in one is caught by the other:

1. **A field-name allowlist at the emit site** (:func:`log_event`). A structured
   event can only carry keys from a closed allowlist of *structural* fields
   (the status/kind enums, recipe/query_class, entity TYPE, metric, validated
   tokens, counts, timings, ``retrieved_at``, cache hit/miss, …). A raw
   ``query="..."`` or ``drug_name="..."`` has **no allowlisted home**, so it is
   dropped before serialization — the raw query is structurally unable to leak,
   not merely scrubbed after the fact.

2. **A record-rewriting redaction filter** (:class:`RedactionFilter`) attached to
   every root handler. It runs on **every** record at **every** level as a
   backstop: even a stray ``logger.info("... sk-ant-XXXX ...")`` from anywhere in
   the codebase (or a future call site that forgets the allowlist) has its
   key-shaped substrings and any live provider-key value replaced with
   :data:`REDACTED` before the record is formatted or emitted.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

# The single redaction token — both guards replace secrets with this exact string.
REDACTED = "«redacted-key»"

# --- Guard 2: the redaction filter ---------------------------------------

# (a) Provider-key SHAPES. The three documented prefixes (Anthropic ``sk-ant-``,
# OpenRouter ``sk-or-``, bare OpenAI ``sk-``). The bare ``sk-`` pattern already
# subsumes the two prefixed forms (``-`` is in the char class), but all three are
# listed explicitly so the guard reads exactly like the spec it enforces and so a
# tightened bare-``sk-`` rule later can't silently stop covering the vendor forms.
_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),
    re.compile(r"sk-or-[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
)

# (b) The env vars whose LIVE value is redacted verbatim — a defence against keys
# that don't match the ``sk-`` shape (a project key, a proxied credential, …).
_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)

# Only redact an env value long enough to be a real key — never a trivially-short
# placeholder (e.g. ``"x"``) whose blanket replacement would corrupt normal text.
_MIN_ENV_KEY_LEN = 8


def _scrub(text: str) -> str:
    """Return ``text`` with every provider-key value / key-shaped substring redacted.

    Env-value literals are replaced first (they may not match the ``sk-`` shape),
    then the shape patterns catch anything key-formatted that remains.
    """
    for name in _KEY_ENV_VARS:
        val = os.environ.get(name)
        if val and len(val) >= _MIN_ENV_KEY_LEN:
            text = text.replace(val, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class RedactionFilter(logging.Filter):
    """Scrub provider secrets from every log record, at every level (SEC-47).

    This is a *backstop*, not the primary control (:func:`log_event`'s allowlist
    is). It formats the record's args into the final message string, scrubs that
    string, then bakes it back onto the record (``record.msg`` = scrubbed string,
    ``record.args = ()``) so no downstream handler can re-expose the un-scrubbed
    template or re-run ``%``-formatting.

    Contract of a logging filter: it must be **total** — a filter that raises
    breaks logging for the whole process. So every path returns ``True`` (this
    filter scrubs, it never drops), and any unexpected error leaves the record
    untouched rather than propagating.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # getMessage() applies %-args and str()-ifies a non-string msg for us;
            # a broken format string raises here and is swallowed below.
            message = record.getMessage()
            scrubbed = _scrub(message)
            record.msg = scrubbed
            record.args = ()
        except Exception:  # never raise from a filter — logging would break process-wide
            return True
        return True


# --- Guard 1: the structured-event emitter -------------------------------

# The CLOSED allowlist of structural field names a structured event may carry.
# A key absent from this set is DROPPED before serialization — which is precisely
# why the raw user query / free-text arg values cannot leak through log_event:
# they have no allowlisted home (``query``, ``drug_name``, ``condition``, …, and
# any model reasoning are all absent by construction). Extend deliberately, and
# never with a free-text field.
#
# The set is deliberately WIDER than today's emit sites — an allowlist is a
# vocabulary of what may be logged, not a schema of what currently is. Only
# ``app/main.py`` and ``app/graph/nodes.py`` call ``log_event`` at all, and
# between them they emit ``status``/``kind``/``query_class``/``cache``/
# ``count_total``; every other member names a value the pipeline really computes
# and could log without a new decision. What is NOT here is ``request_id``: v1 is
# a single-process service with one graph run per request and no correlation ID
# anywhere in the codebase, so allowlisting one would name a value that does not
# exist. Add it the day a request ID does.
_ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "event",
        "status",
        "kind",  # the envelope's visualization|answer|clarification discriminator
        "query_class",
        "recipe",
        "entity_type",
        "metric",
        "field",
        "date_field",
        "grain",
        "count_total",
        "distinct_trials",
        "n_buckets",
        "n_nodes",
        "n_edges",
        "tool",
        "duration_ms",
        "retrieved_at",
        "cache",
        "escalation_count",
        "iter_count",
        "reason_code",
    }
)

# Even an allowlisted string value is capped, so a structural field can't smuggle
# a large free-text blob (a bounded value is a bounded disclosure).
_MAX_STR_LEN = 80


def _cap(value: Any) -> Any:
    """Cap an over-long string value; pass non-strings through untouched."""
    if isinstance(value, str) and len(value) > _MAX_STR_LEN:
        return value[:_MAX_STR_LEN]
    return value


def log_event(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Emit ONE structured event as a compact-JSON INFO line.

    Serializes ``{"event": event, **allowlisted_fields}`` where only keys in
    :data:`_ALLOWED_FIELDS` survive and every string value is capped at
    ~``_MAX_STR_LEN`` chars. A disallowed key (e.g. ``query`` or ``drug_name``)
    is silently dropped — that omission is the point: raw free text has no
    allowlisted home, so it can never be logged here (SEC-46/48).

    ``event`` is positional-only so a caller can still pass ``event=`` as a data
    field name without colliding with the event label.
    """
    payload: dict[str, Any] = {"event": _cap(event)}
    for key, value in fields.items():
        if key == "event" or key not in _ALLOWED_FIELDS:
            continue
        payload[key] = _cap(value)
    try:
        message = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        # Serialization must never break a log call; fall back to the label alone.
        message = json.dumps({"event": str(event)[:_MAX_STR_LEN]}, separators=(",", ":"))
    logger.info(message)


# --- Configuration --------------------------------------------------------


def configure_logging(level: str | None = None) -> None:
    """Attach a :class:`RedactionFilter` to the root handlers and set the level.

    Level resolution: the ``level`` arg wins, else ``$LOG_LEVEL``, else ``info``.
    If the root logger has no handlers, one ``StreamHandler`` is added so events
    are actually emitted. Idempotent — repeated calls never double-add a handler
    or stack a second filter, so it is safe to call at every startup / import.
    """
    root = logging.getLogger()

    name = (level or os.environ.get("LOG_LEVEL") or "info").strip().upper()
    level_num = logging.getLevelNamesMapping().get(name, logging.INFO)
    root.setLevel(level_num)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level_num)
        root.addHandler(handler)

    for handler in root.handlers:
        if not any(isinstance(f, RedactionFilter) for f in handler.filters):
            handler.addFilter(RedactionFilter())
