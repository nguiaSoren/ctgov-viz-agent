"""Hardening tests (D4 — SSE single-run, error redaction, doctor, adapter recursion guard).

Companion to ``tests/test_main.py``; covers regressions fixed in this pass:

1. ``/visualize/stream`` must run the graph exactly ONCE (no second, divergent
   ``run_sync``/``graph.invoke`` for the terminal envelope).
2. A mid-stream failure must never leak raw exception text to the client.
3. ``app.doctor.main()`` (the skeleton self-check) is exercised inside the suite.
4. The adapter's no-``canned`` default-builder must raise a clean ``ValueError``
   on a recursive/over-deep schema, never a bare ``RecursionError`` -- and the
   real Phase-0 models must still build fine with no ``canned`` payload.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.main import app

client = TestClient(app)


def _happy_request() -> dict:
    return {
        "query": "Phase distribution of pancreatic cancer trials",
        "condition": "pancreatic cancer",
    }


def _terminal_envelope(raw: str) -> dict:
    """Pull the last SSE `data: {...}` line (the terminal `result` event) and parse it."""
    result_line = [ln for ln in raw.splitlines() if ln.startswith("data: {")][-1]
    return json.loads(result_line[len("data: ") :])


# --- FIX 1: the stream must run the graph exactly once ----------------------


def test_stream_never_calls_run_sync(monkeypatch):
    """The happy path must derive both the status events and the terminal
    envelope from the ONE `graph.stream(...)` call -- not from a second,
    independent `run_sync`. Proven by making `run_sync` explode if invoked."""

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("run_sync must not be called from the stream path")

    monkeypatch.setattr("app.main.run_sync", _must_not_be_called)

    with client.stream("POST", "/visualize/stream", json=_happy_request()) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    # status events precede the terminal result (planning happens at the offline
    # `plan` node, before execute ever touches the network)
    status_idx = raw.index("event: status")
    result_idx = raw.index("event: result")
    assert status_idx < result_idx
    assert "data: planning" in raw

    # The terminal result proves run_sync was NEVER called: had it been, its
    # AssertionError would be caught by the stream's own `except` and surfaced as
    # code "stream_error". A live rate-limit/outage instead surfaces code
    # "upstream_error" (from execute) -- an acceptable transient, NOT a run_sync call.
    envelope = _terminal_envelope(raw)
    assert envelope["meta"]["source"] == "clinicaltrials.gov"
    if envelope["status"] == "error":
        assert envelope["error"]["code"] == "upstream_error", envelope["error"]
    else:
        assert envelope["status"] == "ok"


# --- FIX 2: exception text must never reach the client -----------------------


def test_stream_error_is_redacted(monkeypatch):
    """A mid-stream failure must surface a FIXED generic message on the wire,
    never `str(exc)` (which could embed upstream URLs/params, §A(e)/§3.11)."""

    def _boom():
        raise RuntimeError("secret /Users/example/leak")

    monkeypatch.setattr("app.main.build_graph", _boom)

    with client.stream("POST", "/visualize/stream", json=_happy_request()) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    envelope = _terminal_envelope(raw)
    assert envelope["status"] == "error"
    message = envelope["error"]["message"]
    assert "secret" not in message
    assert "/Users/example" not in message
    assert message == "internal error while streaming"


# --- doctor covered inside the suite ------------------------------------------


def test_doctor_passes():
    import app.doctor

    assert app.doctor.main() == 0


# --- FIX 4: adapter recursion guard -------------------------------------------


def test_adapter_recursion_guard_raises_clean_value_error():
    """A required, self-referential field with no `canned` payload must raise
    a clean `ValueError` -- never a bare `RecursionError`."""
    from app.llm.adapter import StubAdapter

    class SelfRef(BaseModel):
        name: str
        child: SelfRef

    SelfRef.model_rebuild()

    adapter = StubAdapter()
    try:
        adapter.propose(system="s", messages=[], response_model=SelfRef)
    except RecursionError:
        pytest.fail("adapter raised RecursionError instead of a clean ValueError")
    except ValueError:
        pass
    else:
        pytest.fail("expected ValueError for a recursive schema with no canned payload")


def test_real_models_still_build_without_canned():
    """The recursion guard must not change behavior for the real (non-recursive)
    Phase-0 models -- they must still default-build fine with no `canned`."""
    from app.llm.adapter import StubAdapter
    from app.llm.reviewers import IntentVerdict, OutputVerdict
    from app.plan.models import Plan

    adapter = StubAdapter()
    for model in (Plan, IntentVerdict, OutputVerdict):
        instance = adapter.propose(system="s", messages=[], response_model=model)
        assert isinstance(instance, model)
