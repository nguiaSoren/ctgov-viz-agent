"""``aggregate_by_counts`` — the exact per-token count-based distribution (the over-budget path
that charts instead of refusing). Offline: a fake client returns canned counts + samples, so the
combine Missing-residual, the explode double-count, and the citation shape are tested
deterministically (no rate-limit flakiness). The Essie counts are live-verified elsewhere."""

from __future__ import annotations

import app.ctgov.tools as tools
from app.viz.review import record_grounded_reverify
from app.viz.spec import build_envelope


class _FakeClient:
    """A CTGovClient stand-in. ``count`` returns the population total (no field filter) or a
    per-token count; ``get`` returns that token's count + a canned sample record carrying the
    token at the field path so the citation round-trips."""

    def __init__(self, total: int, per_token: dict[str, int], status_path: bool) -> None:
        self._total = total
        self._per_token = per_token
        self._status_path = status_path  # overallStatus (scalar) vs interventionType (list)

    def _token_of(self, params: dict) -> str | None:
        # The per-token filter rides on filter.overallStatus or in filter.advanced.
        if "filter.overallStatus" in params:
            return params["filter.overallStatus"]
        adv = params.get("filter.advanced", "")
        for tok in self._per_token:
            if tok in adv:
                return tok
        return None

    def count(self, params: dict) -> int:
        tok = self._token_of(params)
        return self._per_token.get(tok, 0) if tok else self._total

    def get(self, path: str, params: dict) -> dict:
        tok = self._token_of(params)
        n = self._per_token.get(tok, 0)
        if n <= 0:
            return {"totalCount": 0, "studies": []}
        if self._status_path:
            rec = {"protocolSection": {"identificationModule": {"nctId": f"NCT{tok[:8]:0>8}"},
                                       "statusModule": {"overallStatus": tok}}}
        else:
            rec = {"protocolSection": {"identificationModule": {"nctId": f"NCT{tok[:8]:0>8}"},
                                       "armsInterventionsModule": {"interventions": [{"type": tok}]}}}
        return {"totalCount": n, "studies": [rec]}


def test_combine_field_missing_residual_reconciles(monkeypatch) -> None:
    """A single-value field (overallStatus): Σ(known tokens) + a derived Missing residual ==
    total, so the combine reconciliation (Σ==distinct==countTotal) holds; every non-Missing
    bucket is cited; the citation round-trips against the fetched record."""
    # Include the genuine "UNKNOWN" status token — it collides with key_fn's missing-sentinel,
    # so a naive value-strip would empty its citation and hard-fail citation_invalid (the exact
    # live bug this regresses).
    fake = _FakeClient(total=100, per_token={"RECRUITING": 30, "COMPLETED": 50, "UNKNOWN": 5},
                       status_path=True)
    monkeypatch.setattr(tools, "CTGovClient", lambda: fake)

    result = tools.aggregate_by_counts({"cond": "x"}, {}, "overallStatus")
    assert result["mode"] == "combine"
    assert result["distinct_trials"] == 100
    assert result["truncated"] is False  # exact, not a paged prefix
    by_value = {b["value"]: b for b in result["buckets"]}
    assert by_value["RECRUITING"]["count_trials"] == 30
    assert by_value["COMPLETED"]["count_trials"] == 50
    # covered = 85 -> a derived Missing residual of 15 keeps Σ == total.
    assert by_value["Missing"]["count_trials"] == 15 and by_value["Missing"]["derived"] is True
    assert sum(b["count_trials"] for b in result["buckets"]) == 100
    # every non-zero real bucket is cited — INCLUDING the UNKNOWN-status bucket (the collision).
    assert by_value["RECRUITING"]["citations"] and by_value["COMPLETED"]["citations"]
    assert by_value["UNKNOWN"]["citations"], "UNKNOWN status bucket must be cited, not emptied"

    # It reconciles through the real builder (Σ==count_basis) + the record-grounded re-verify.
    from app.plan.models import Plan
    plan = Plan(query_class="distribution", entities={"condition": "x"}, field="overallStatus",
                chart_type="bar")
    spec = build_envelope(plan=plan, tool_results=[result], status="ok", question="q")
    assert spec.status == "ok"
    assert sum(d.count_trials for d in spec.visualization.data) == 100 == spec.meta.count_basis.trials
    rg = record_grounded_reverify(spec, result["record_index"])
    assert rg.ok and not rg.hard_fail


def test_explode_field_double_counts_but_distinct_is_total(monkeypatch) -> None:
    """interventionType (explode): a trial with two types counts in each, so Σbars ≥ distinct,
    and distinct == the population total (the CC-16 anchor); no Missing bucket."""
    fake = _FakeClient(total=100, per_token={"DRUG": 60, "DEVICE": 50}, status_path=False)
    monkeypatch.setattr(tools, "CTGovClient", lambda: fake)

    result = tools.aggregate_by_counts({"cond": "x"}, {}, "interventionType")
    assert result["mode"] == "explode"
    assert result["distinct_trials"] == 100
    assert not any(b["value"] == "Missing" for b in result["buckets"])  # explode has no residual
    assert sum(b["count_trials"] for b in result["buckets"]) == 110  # 60+50 ≥ distinct 100
    assert all(b["citations"] for b in result["buckets"])  # every type bucket cited
