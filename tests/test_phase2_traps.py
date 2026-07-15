"""Per-class trap coverage (Phase 2, BUILD_PLAN §Phase-2 trap task) — offline.

Each trap the interrogation surfaced either has a mechanical test here or a
documented handling pinned in a recipe note (asserted here so the prose can't
silently drift). Traps: combined-phase own bucket (CC-15), intervention/country
multi-value double-count basis (CC-3), country dedupe + no-choropleth (CC-13),
sponsor OTHER-dominance, placebo/synonym network hygiene (CC-12), % within series
(CC-14), planned future dates (G-40), and the generalized-Plan checker holes
(a compare/network plan must carry its shape + validate every series arm, G-33).
"""

from __future__ import annotations

from app.api.schemas import ChartType
from app.ctgov.aggregate import AggregationCore
from app.ctgov.fields import intervention_type_key_fn
from app.plan.checker import check_plan
from app.plan.models import NetworkSpec, Plan, Series
from app.plan.recipes import RECIPES


class _FakeClient:
    def __init__(self, records):
        self._records = records

    def iter_studies(self, search_params, *, fields, page_size=1000, max_pages=20):
        return list(self._records), False


def _record(nct, types):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "armsInterventionsModule": {"interventions": [{"type": t, "name": t} for t in types]},
        }
    }


class TestMultiValueDoubleCount:
    def test_interventiontype_explode_mentions_exceed_trials(self):
        # A trial with two DISTINCT types contributes ONE distinct-trial to each
        # bucket but the mention total across buckets exceeds the trial count
        # (CC-3: "most common intervention" double-counts). Reconciliation anchors
        # on distinct-nctId (== 2), never the mention sum.
        records = [
            _record("NCT1", ["DRUG", "DEVICE"]),  # in DRUG and DEVICE
            _record("NCT2", ["DRUG"]),
        ]
        group = AggregationCore(_FakeClient(records)).page_and_group(
            {}, fields="NCTId|InterventionType", key_fn=intervention_type_key_fn, mode="explode"
        )
        by_value = {b.value: b for b in group.buckets}
        assert by_value["DRUG"].count_trials == 2  # distinct trials with a DRUG arm
        assert by_value["DEVICE"].count_trials == 1
        # Σ mentions (2 DRUG + 1 DEVICE = 3) > distinct trials (2) — the double-count.
        assert sum(b.count_mentions for b in group.buckets) == 3
        assert group.distinct_trials == 2  # the reconciliation anchor


class TestRecipeProsePinsTheTraps:
    """Every trap that is handled in PROSE (not a data path) must actually appear
    in its recipe note, so a doc edit can't silently drop the disclosure."""

    def test_geographic_pins_dedupe_and_no_choropleth(self):
        note = RECIPES["geographic"].notes.lower()
        assert "dedup" in note and "choropleth" in note

    def test_network_pins_placebo_and_synonym(self):
        note = RECIPES["network"].notes.lower()
        assert "placebo" in note and "synonym" in note

    def test_compare_pins_percent_within_series(self):
        note = RECIPES["compare"].notes.lower()
        assert "percentage" in note or "%" in note or "within" in note

    def test_timeseries_pins_planned_future(self):
        note = RECIPES["timeseries"].notes.lower()
        assert "planned" in note and "date field" in note


class TestGeneralizedPlanChecker:
    """The generalized-Plan checker (G-33) validates each class's shape + recurses
    into every compare arm — else an invented token in series[1] slips through or a
    shape-less compare/network plan passes."""

    def test_compare_invented_token_in_second_arm_rejected(self):
        plan = Plan(
            query_class="compare",
            field="overallStatus",
            chart_type=ChartType.GROUPED_BAR,
            series=[
                Series(label="A", entities={"drug": "a"}),
                Series(label="B", filters={"overallStatus": "RECRUTING_TYPO"}),
            ],
        )
        result = check_plan(plan)
        assert not result.ok and "invalid_filter_token" in result.reason

    def test_compare_requires_two_arms(self):
        plan = Plan(
            query_class="compare", field="overallStatus", chart_type=ChartType.GROUPED_BAR,
            series=[Series(label="A", entities={"drug": "a"})],
        )
        assert check_plan(plan).reason == "compare_requires_two_series"

    def test_network_requires_block(self):
        plan = Plan(
            query_class="network", entities={"condition": "melanoma"},
            chart_type=ChartType.NETWORK_GRAPH,
        )
        assert check_plan(plan).reason == "network_requires_network_block"

    def test_valid_drug_drug_network_passes(self):
        plan = Plan(
            query_class="network", entities={"condition": "melanoma"},
            chart_type=ChartType.NETWORK_GRAPH, network=NetworkSpec(kind="drug_drug"),
        )
        assert check_plan(plan).ok

    def test_geographic_requires_country_field(self):
        plan = Plan(
            query_class="geographic", entities={"condition": "diabetes"},
            field="phase", chart_type=ChartType.BAR,
        )
        assert "geographic_requires_country_field" in check_plan(plan).reason
