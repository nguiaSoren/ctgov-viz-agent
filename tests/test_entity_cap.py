"""MAX_COMPARE_ENTITIES cap on compare arms (E-21) — first-N + a dropped note."""

from __future__ import annotations

from app import config
from app.graph import nodes
from app.plan.models import Plan, Series


def test_compare_caps_arms_and_discloses_dropped(monkeypatch):
    monkeypatch.setattr(nodes, "count_trials", lambda q, f: 100)  # no arm is empty/too_large
    captured = {}

    def _fake_compare(series, field):
        captured["n_series"] = len(series)
        return {"tool": "compare", "mode": "compare", "buckets": [], "notes": []}

    monkeypatch.setattr(nodes, "compare", _fake_compare)

    n = config.MAX_COMPARE_ENTITIES + 3
    arms = [Series(label=f"Drug{i}", entities={"drug": f"drug{i}"}, filters={}) for i in range(n)]
    plan = Plan(
        query_class="compare",
        entities={},
        filters={},
        field="phase",
        chart_type="grouped_bar",
        series=arms,
    )

    out = nodes._execute_compare(plan, "2026-07-16T00:00:00")
    result = out["tool_results"][0]
    notes = " ".join(result.get("notes") or [])

    assert captured["n_series"] == config.MAX_COMPARE_ENTITIES  # only the first N reached the tool
    assert f"first {config.MAX_COMPARE_ENTITIES} of {n}" in notes  # the drop is disclosed
    assert f"Drug{config.MAX_COMPARE_ENTITIES}" in notes  # a dropped label is named
    assert f"Drug{n - 1}" in notes


def test_compare_under_cap_has_no_dropped_note(monkeypatch):
    monkeypatch.setattr(nodes, "count_trials", lambda q, f: 100)
    monkeypatch.setattr(
        nodes, "compare", lambda series, field: {"tool": "compare", "mode": "compare", "buckets": []}
    )
    arms = [Series(label=f"Drug{i}", entities={"drug": f"drug{i}"}, filters={}) for i in range(2)]
    plan = Plan(
        query_class="compare", entities={}, filters={}, field="phase",
        chart_type="grouped_bar", series=arms,
    )
    out = nodes._execute_compare(plan, "2026-07-16T00:00:00")
    notes = " ".join(out["tool_results"][0].get("notes") or [])
    assert "dropped" not in notes.lower()
