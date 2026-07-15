"""FastAPI surface tests (in-process via TestClient — no server is ever started)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _skip_if_transient_upstream(body: dict) -> None:
    """``execute``'s default path is a LIVE ClinicalTrials.gov call now. When the
    API is unreachable or rate-limited the pipeline returns a clean, redacted
    ``status:"error"`` / ``code:"upstream_error"`` envelope (that routing is what
    the offline tests already prove). Such a transient upstream failure is not
    what these structure tests are asserting, so they skip on it — a real logic
    regression surfaces a DIFFERENT code (e.g. ``reconciliation_failed``) or an
    ``ok`` envelope with wrong structure, both of which still fail."""
    if body.get("status") == "error" and (body.get("error") or {}).get("code") == "upstream_error":
        pytest.skip("live ClinicalTrials.gov API unavailable/rate-limited")


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_visualize_happy_path_returns_valid_envelope():
    r = client.post("/visualize", json={
        "query": "Phase distribution of interventional pancreatic cancer trials",
        "condition": "pancreatic cancer",
        "interventional_only": True,
    })
    assert r.status_code == 200
    body = r.json()
    _skip_if_transient_upstream(body)
    assert body["status"] == "ok"
    assert body["kind"] == "visualization"
    assert body["visualization"]["type"] == "bar"
    assert len(body["visualization"]["data"]) >= 1
    assert body["meta"]["source"] == "clinicaltrials.gov"


def test_visualize_rejects_empty_query_with_422():
    r = client.post("/visualize", json={"query": "   "})
    assert r.status_code == 422


def test_visualize_rejects_unknown_field_with_422():
    r = client.post("/visualize", json={"query": "trials", "not_a_field": 1})
    assert r.status_code == 422


def test_visualize_rejects_oversized_condition_with_422():
    r = client.post("/visualize", json={"query": "trials", "condition": "x" * 5000})
    assert r.status_code == 422


def test_stream_emits_status_events_then_terminal_envelope():
    with client.stream("POST", "/visualize/stream", json={
        "query": "Phase distribution of pancreatic cancer trials",
        "condition": "pancreatic cancer",
    }) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    # ordered high-level status events, then a terminal result carrying the full envelope
    assert "event: status" in raw
    assert "data: planning" in raw
    assert "event: result" in raw
    # the terminal result parses as a full envelope
    result_line = [ln for ln in raw.splitlines() if ln.startswith("data: {")][-1]
    envelope = json.loads(result_line[len("data: "):])
    assert envelope["status"] in {"ok", "empty", "too_large", "error"}
    assert envelope["meta"]["source"] == "clinicaltrials.gov"
