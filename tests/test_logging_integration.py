"""End-to-end log hygiene (§A(i) · SEC-46/47): a real request through the app
never emits the raw user query at info level (the unit-level key redaction lives
in tests/test_logging_redaction.py). Offline — the "this drug" query clarifies
before execute, so no network is touched."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.logging_setup import configure_logging
from app.main import app


def test_request_does_not_log_the_raw_query(caplog):
    configure_logging()
    distinctive = "zzsecretresearchtopicqq for this drug"  # 'this drug' → offline clarification
    with caplog.at_level(logging.INFO):
        client = TestClient(app)
        resp = client.post("/visualize", json={"query": distinctive})
    assert resp.status_code == 200
    assert resp.json()["kind"] == "clarification"  # took the offline path
    logged = " ".join(r.getMessage() for r in caplog.records)
    # The raw query (the sensitive part) never appears in any log line at info level.
    assert "zzsecretresearchtopicqq" not in logged
    # But a structured event WAS emitted (the decided shape), proving logging is live.
    assert any("visualize_complete" in r.getMessage() for r in caplog.records)
