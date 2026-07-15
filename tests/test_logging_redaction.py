"""Tests for the structured-event logger + secret-redaction backstop.

Two guards are proven independently (ARCHITECTURE_SPEC §A(i), SEC-46/47/48):

* :func:`log_event`'s field-name allowlist — a raw query / free-text arg has no
  allowlisted home, so it is dropped and can never be logged (the key PII test).
* :class:`RedactionFilter` — a stray key-shaped substring or a live provider-key
  value is scrubbed on every record, at every level, as a backstop.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.logging_setup import REDACTED, RedactionFilter, configure_logging, log_event


class _CaptureHandler(logging.Handler):
    """A handler that stores each record's FINAL (post-filter) message.

    A ``logging.Handler`` runs its attached filters in ``handle()`` before
    ``emit()``, so a ``RedactionFilter`` added here mutates the record exactly as
    it would in production; ``record.getMessage()`` in ``emit`` sees the scrubbed
    text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _make_logger(name: str, *, redact: bool = False) -> tuple[logging.Logger, _CaptureHandler]:
    """A fresh, isolated logger (no propagation to root / caplog) + its capture handler."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CaptureHandler()
    if redact:
        handler.addFilter(RedactionFilter())
    logger.addHandler(handler)
    return logger, handler


# --- Guard 1: log_event structured events + allowlist ---------------------


def test_log_event_emits_safe_structural_fields():
    logger, handler = _make_logger("test.event.safe")
    log_event(logger, "tool_completed", tool="aggregate_by", duration_ms=184, count_total=3950)

    out = handler.messages[-1]
    payload = json.loads(out)
    assert payload == {
        "event": "tool_completed",
        "tool": "aggregate_by",
        "duration_ms": 184,
        "count_total": 3950,
    }
    # compact JSON, safe fields present
    assert "aggregate_by" in out
    assert "184" in out
    assert "3950" in out


def test_log_event_drops_disallowed_free_text_fields():
    """The key PII test: raw ``query`` / ``drug_name`` have no allowlisted home."""
    logger, handler = _make_logger("test.event.pii")
    log_event(
        logger,
        "planned",
        query="pancreatic cancer secret research topic",
        drug_name="foo",
    )

    out = handler.messages[-1]
    assert "pancreatic cancer" not in out
    assert "foo" not in out
    # the disallowed keys are dropped entirely, only the structural label survives
    assert json.loads(out) == {"event": "planned"}


def test_log_event_keeps_only_allowlisted_keys_from_a_mixed_call():
    logger, handler = _make_logger("test.event.mixed")
    log_event(
        logger,
        "aggregated",
        status="ok",  # allowed
        recipe="phase_distribution",  # allowed
        n_buckets=4,  # allowed
        query="raw sensitive text",  # dropped
        reasoning="model chain of thought",  # dropped
    )

    payload = json.loads(handler.messages[-1])
    assert payload == {
        "event": "aggregated",
        "status": "ok",
        "recipe": "phase_distribution",
        "n_buckets": 4,
    }


def test_log_event_caps_long_string_values():
    logger, handler = _make_logger("test.event.cap")
    log_event(logger, "reviewed", reason_code="x" * 200)

    payload = json.loads(handler.messages[-1])
    assert len(payload["reason_code"]) == 80


# --- Guard 2: RedactionFilter backstop ------------------------------------


def test_provider_key_pattern_is_redacted():
    logger, handler = _make_logger("test.redact.pattern", redact=True)
    logger.info("calling provider with key sk-ant-ABC123SECRETXYZ")

    out = handler.messages[-1]
    assert "sk-ant-ABC123SECRETXYZ" not in out
    assert REDACTED in out


@pytest.mark.parametrize(
    "line",
    [
        "openrouter key sk-or-DEADBEEF0011 in use",
        "bare openai key sk-PLAINKEY99887766 loaded",
    ],
)
def test_all_provider_key_shapes_are_redacted(line: str):
    logger, handler = _make_logger("test.redact.shapes", redact=True)
    logger.info(line)

    out = handler.messages[-1]
    assert REDACTED in out
    assert "sk-or-DEADBEEF0011" not in out
    assert "sk-PLAINKEY99887766" not in out


def test_live_env_key_value_is_redacted(monkeypatch):
    """The task's specified case: OPENAI_API_KEY monkeypatched to sk-FAKEKEYVALUE123."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKEKEYVALUE123")
    logger, handler = _make_logger("test.redact.env", redact=True)
    logger.info("booting adapter with OPENAI_API_KEY=sk-FAKEKEYVALUE123 present")

    out = handler.messages[-1]
    assert "sk-FAKEKEYVALUE123" not in out
    assert REDACTED in out


def test_live_env_key_value_redacted_even_without_sk_prefix(monkeypatch):
    """Branch (b) proof: a key that does NOT match the ``sk-`` shape is still
    redacted via its live env value, so a non-standard credential can't leak."""
    secret = "PROJKEY_ABCDEF0123456789"  # no ``sk-`` prefix → only the env-value branch can catch it
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    logger, handler = _make_logger("test.redact.env.noprefix", redact=True)
    logger.info(f"resolved credential {secret} for provider")

    out = handler.messages[-1]
    assert secret not in out
    assert REDACTED in out


def test_short_env_value_is_not_treated_as_a_key(monkeypatch):
    """A trivially-short env value (< 8 chars) is left alone — blanket-replacing it
    would corrupt ordinary text."""
    monkeypatch.setenv("OPENAI_API_KEY", "abc")  # 3 chars, below the min-key threshold
    logger, handler = _make_logger("test.redact.short", redact=True)
    logger.info("the alphabet starts with abc and so on")

    out = handler.messages[-1]
    assert out == "the alphabet starts with abc and so on"
    assert REDACTED not in out


def test_filter_scrubs_across_percent_args():
    """The key rides in via a ``%s`` arg — the filter formats then scrubs."""
    logger, handler = _make_logger("test.redact.args", redact=True)
    logger.info("provider %s key %s", "anthropic", "sk-ant-XYZ987SECRET")

    out = handler.messages[-1]
    assert "anthropic" in out  # non-secret arg survives, proving args were formatted in
    assert "sk-ant-XYZ987SECRET" not in out
    assert REDACTED in out


def test_filter_never_raises_and_always_returns_true():
    f = RedactionFilter()

    # normal %-args
    rec_ok = logging.LogRecord("n", logging.INFO, __file__, 1, "value %s and %d", ("x", 5), None)
    assert f.filter(rec_ok) is True

    # non-string msg (a dict) — getMessage() str()-ifies it, filter must cope
    rec_nonstr = logging.LogRecord("n", logging.INFO, __file__, 1, {"a": 1}, None, None)
    assert f.filter(rec_nonstr) is True

    # a BROKEN format string (%d given a non-int) raises inside getMessage —
    # the filter must swallow it and still return True, leaving the record intact
    rec_broken = logging.LogRecord("n", logging.INFO, __file__, 1, "val %d", ("notint",), None)
    assert f.filter(rec_broken) is True


# --- configure_logging ----------------------------------------------------


@pytest.fixture
def _restore_root():
    """Snapshot and restore the global root logger so configure_logging tests
    don't leak handler/filter/level state into the rest of the suite."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_filters = {id(h): list(h.filters) for h in saved_handlers}
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for h in saved_handlers:
            h.filters[:] = saved_filters.get(id(h), h.filters)


def test_configure_logging_attaches_filter_and_is_idempotent(_restore_root):
    root = _restore_root
    configure_logging("info")
    configure_logging("info")  # second call must not double-add

    assert root.handlers, "configure_logging must ensure at least one handler"
    for handler in root.handlers:
        redaction_filters = [f for f in handler.filters if isinstance(f, RedactionFilter)]
        assert len(redaction_filters) == 1


def test_configure_logging_sets_level_from_arg(_restore_root):
    root = _restore_root
    configure_logging("warning")
    assert root.level == logging.WARNING


def test_configure_logging_reads_log_level_env(monkeypatch, _restore_root):
    root = _restore_root
    monkeypatch.setenv("LOG_LEVEL", "debug")
    configure_logging()  # no explicit arg → falls back to $LOG_LEVEL
    assert root.level == logging.DEBUG
