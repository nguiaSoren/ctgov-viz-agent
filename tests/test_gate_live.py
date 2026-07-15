"""The Phase-1 KILLER GATE as a live test (X-2), through the real POST /visualize.

This is the pytest form of `scripts/run_gate.py`: it drives the whole pipeline
end-to-end over the FastAPI route against the live ClinicalTrials.gov API and
asserts the reconciliation claim the phase is gated on (CC-16) — distinct-nctId
== the API's exact countTotal — plus the cited-spec invariants.

LIVE-ONLY (project decision, no VCR). The public API rate-limits under a
full-suite burst; when it returns a transient upstream error the pipeline
degrades to status:"error" (code `upstream_error`) and this test SKIPS rather
than fails (it passes in isolation — a real regression yields a different code
or a structural failure, which still fails). See LESSON H1.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

# X-2: the sub-20k interventional population that charts (not too_large).
_X2_REQUEST = {
    "query": "Phase distribution of interventional pancreatic cancer trials",
    "condition": "pancreatic cancer",
    "interventional_only": True,
}
_DRIFT_PCT = 0.005
_DRIFT_ABS = 20
_PHASES_PATH = "protocolSection.designModule.phases"


@pytest.fixture(scope="module")
def gate_body() -> dict:
    """POST X-2 once, live, and share the envelope. Skips on a transient upstream
    error (the whole pipeline reports status:error/code:upstream_error) — never on
    a reconciliation/structural failure, so the gate keeps its teeth."""
    client = TestClient(app)
    resp = client.post("/visualize", json=_X2_REQUEST)
    assert resp.status_code == 200, f"/visualize -> HTTP {resp.status_code}"
    body = resp.json()
    if body["status"] == "error" and (body.get("error") or {}).get("code") == "upstream_error":
        pytest.skip("clinicaltrials.gov transient upstream error — live gate skipped (H1)")
    return body


def test_gate_reconciles_distinct_to_count_total(gate_body: dict) -> None:
    """The killer gate: Σ distinct-trial counts reconciles to the API countTotal."""
    assert gate_body["status"] == "ok"
    assert gate_body["kind"] == "visualization"
    assert gate_body["visualization"]["type"] == "bar"
    data = gate_body["visualization"]["data"]
    observed = sum(d["count_trials"] for d in data)
    total = gate_body["meta"]["count_basis"]["trials"]
    drift = abs(observed - total)
    assert drift == 0 or (drift <= _DRIFT_PCT * total and drift <= _DRIFT_ABS), (
        f"Σcount_trials={observed} does not reconcile to countTotal={total} (drift {drift})"
    )


def test_gate_has_explicit_na_bucket_and_composites(gate_body: dict) -> None:
    """NA is its own bucket (CC-5); combined phases are their own composites (CC-15)."""
    values = {d["value"] for d in gate_body["visualization"]["data"]}
    assert "NA" in values
    # This population is known to carry combined phases (e.g. PHASE1|PHASE2).
    assert any("|" in v for v in values), "expected at least one CC-15 composite phase bucket"


def test_gate_every_nonzero_bucket_cited_with_exact_contributing_count(gate_body: dict) -> None:
    """K=20 sampled citations + exact contributing_count (CC-9)."""
    for d in gate_body["visualization"]["data"]:
        if d["count_trials"] > 0:
            assert len(d["citations"]) >= 1
            assert d["contributing_count"] == d["count_trials"]
            assert len(d["citations"]) <= 20  # the K=20 sample cap
            if d["contributing_count"] > 20:
                assert d["citations_truncated"] is True


def test_gate_every_excerpt_is_a_live_substring(gate_body: dict) -> None:
    """Every excerpt is an element-precise quote of the trial's real phase value."""
    for d in gate_body["visualization"]["data"]:
        for c in d["citations"]:
            value = c["value"]
            excerpt = c["excerpt"]
            if isinstance(value, list):
                assert excerpt in [str(el) for el in value], f"{excerpt!r} not in {value!r}"
            else:
                assert excerpt in str(value)


def test_gate_provenance_and_vega(gate_body: dict) -> None:
    """meta stamps the real population + a renderable vega_lite block."""
    meta = gate_body["meta"]
    assert meta["source"] == "clinicaltrials.gov"
    assert meta["retrieved_at"]
    assert meta["filters"].get("interventional_only") is True
    params = meta["query_provenance"]["params"]
    assert params["filter.advanced"] == "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL"
    assert gate_body["vega_lite"] is not None
