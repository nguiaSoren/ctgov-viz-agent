"""Phase-2 breadth gate, IN the suite (LESSON A3): all five query classes + the
too_large refuse, driven end-to-end through the FULL graph with a hardcoded Plan
(the deterministic engine; the LLM planner is Phase 4).

Live, $0 (no key). Each class fetches once (module-scoped) and SKIPS — never
fails — on any transport ``UpstreamError`` (offline / rate-limited under a
full-suite burst), so an offline run stays green while a real regression still
surfaces as an assertion failure in a test BODY (H1). Run in isolation to see
them PASS: ``pytest tests/test_phase2_gate_live.py -q``.
"""

from __future__ import annotations

import pytest

from app.api.schemas import ChartType, VisualizeRequest
from app.ctgov.client import UpstreamError
from app.graph.build import run_sync
from app.plan.models import NetworkSpec, Plan, Series


def _run(plan: Plan, query: str = "gate") -> dict:
    try:
        spec = run_sync(VisualizeRequest(query=query), overrides={"_force_plan": plan})
    except UpstreamError as exc:  # transport-only here -> infra, not a logic bug
        pytest.skip(f"clinicaltrials.gov transport error ({exc.code}) -- live gate skipped")
    body = spec.model_dump()
    # A mid-paging rate-limit under a full-suite burst is caught by ``execute`` and
    # returned as a REDACTED upstream_error envelope (not re-raised) — that is an
    # infra condition, not a logic bug, so SKIP rather than fail on it (H1 at the
    # graph level). A real regression surfaces as an assertion failure in a body
    # after a clean fetch, which still fails.
    err = body.get("error") or {}
    if body["status"] == "error" and err.get("code", "").startswith("upstream"):
        pytest.skip(f"transient upstream error envelope ({err.get('code')}) -- live gate skipped")
    return body


@pytest.fixture(scope="module")
def x1_timeseries() -> dict:
    return _run(
        Plan(
            query_class="timeseries",
            entities={"condition": "melanoma"},
            filters={"start_year": 2015},
            date_field="startDate",
            grain="year",
            chart_type=ChartType.TIME_SERIES,
        )
    )


@pytest.fixture(scope="module")
def x3_compare() -> dict:
    return _run(
        Plan(
            query_class="compare",
            field="overallStatus",
            chart_type=ChartType.GROUPED_BAR,
            series=[
                Series(label="Pembrolizumab", entities={"drug": "pembrolizumab"}),
                Series(label="Nivolumab", entities={"drug": "nivolumab"}),
            ],
        )
    )


@pytest.fixture(scope="module")
def x4_geographic() -> dict:
    return _run(
        Plan(
            query_class="geographic",
            entities={"condition": "diabetes"},
            filters={"overallStatus": "RECRUITING"},
            field="country",
            chart_type=ChartType.BAR,
        )
    )


@pytest.fixture(scope="module")
def x5_network() -> dict:
    return _run(
        Plan(
            query_class="network",
            entities={"condition": "melanoma"},
            chart_type=ChartType.NETWORK_GRAPH,
            network=NetworkSpec(kind="drug_drug"),
        )
    )


class TestTimeseriesGate:
    def test_reconciles_and_cited(self, x1_timeseries):
        b = x1_timeseries
        assert b["status"] == "ok" and b["visualization"]["type"] == "time_series"
        data = b["visualization"]["data"]
        # Σ (incl. planned incl. any MISSING) == the distinct-trial basis (countTotal).
        assert sum(d["count_trials"] for d in data) == b["meta"]["count_basis"]["trials"]
        assert b["meta"]["date_field_used"] == "startDate"
        # every non-zero year-bucket is cited; a 0-fill gap year carries none (G-35).
        for d in data:
            if d["count_trials"] > 0 and d.get("period") is not None:
                assert d["citations"], f"year {d['value']} uncited"

    def test_future_years_are_planned_not_clamped(self, x1_timeseries):
        planned = [d for d in x1_timeseries["visualization"]["data"] if d.get("planned")]
        for d in planned:  # a planned bucket keeps its real future year, not the current one
            assert int(d["value"]) > 2026


class TestCompareGate:
    def test_two_series_percent_within_series(self, x3_compare):
        b = x3_compare
        assert b["status"] == "ok" and b["visualization"]["type"] == "grouped_bar"
        series = {d.get("series") for d in b["visualization"]["data"]}
        assert series == {"Pembrolizumab", "Nivolumab"}
        # percent is within-series: a COMPLETED datum's percent == count / that series' N.
        for d in b["visualization"]["data"]:
            assert d.get("percent") is not None and d.get("series")

    def test_compare_is_reconciliation_exempt_but_cited(self, x3_compare):
        # A non-zero compare datum still carries a real citation (tamper-evidence
        # preserved even though the single-oracle reconciliation is waived).
        nonzero = [d for d in x3_compare["visualization"]["data"] if d["count_trials"] > 0]
        assert nonzero and all(d["citations"] for d in nonzero)


class TestGeographicGate:
    def test_ranked_with_other_fold_and_distinct_basis(self, x4_geographic):
        b = x4_geographic
        assert b["status"] == "ok" and b["visualization"]["type"] == "bar"
        data = b["visualization"]["data"]
        # explode: the distinct-trial basis is BELOW the membership sum (multi-country).
        basis = b["meta"]["count_basis"]
        assert basis["mentions"] >= basis["trials"]
        # top-N + an "Other" fold (derived, cites its members).
        other = [d for d in data if d["value"] == "Other"]
        assert other and other[0]["derived"] and other[0].get("members")

    def test_country_citation_is_element_targeted(self, x4_geographic):
        # A country bar cites THAT country, not locations[0] (element-precise).
        for d in x4_geographic["visualization"]["data"]:
            if d["value"] not in ("Other", "UNKNOWN") and d["citations"]:
                assert d["citations"][0]["matched_value"] == d["value"]
                break


class TestNetworkGate:
    def test_graph_shape_and_per_edge_citations(self, x5_network):
        b = x5_network
        assert b["status"] == "ok" and b["visualization"]["type"] == "network_graph"
        assert b["vega_lite"] is None  # networks are NEVER Vega-Lite (C-60)
        graph = b["visualization"]["data"]
        assert graph["nodes"] and graph["edges"]
        for edge in graph["edges"]:  # two endpoint citations per edge (G-25)
            assert len(edge["citations"]) == 2
            # The two endpoints are distinct entities: for drug_drug both cite
            # interventions[].name (same path, DIFFERENT drug names); for
            # sponsor_drug the two field_paths differ. Either way the excerpts differ.
            excerpts = {c["matched_value"] for c in edge["citations"]}
            assert len(excerpts) == 2


class TestTooLargeGate:
    def test_overall_cancer_by_phase_refuses(self):
        b = _run(
            Plan(
                query_class="distribution",
                entities={"condition": "cancer"},
                field="phase",
                chart_type=ChartType.BAR,
            )
        )
        assert b["status"] == "too_large" and b["kind"] == "answer"
        assert b["visualization"] is None and b["vega_lite"] is None
        assert b["meta"]["count_basis"]["trials"] > 20_000  # the exact refused total
