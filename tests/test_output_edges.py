"""Phase-5 OUTPUT edge cases — envelope shaping that works TODAY.

All offline + deterministic: the graph paths are driven by the ``initial_state``
test sentinels (``_force_error`` / ``_force_too_large`` / ``_force_canned``)
which short-circuit before any network / LLM call; the ``build_envelope`` shaping
is exercised with arbitrary computed values (a pure call); and the categorical
bucket paths run the real pure functions
(``overall_status_key_fn``) or the real ``aggregate_by`` over a fake in-memory
client (``monkeypatch``), so no ClinicalTrials.gov request is ever made.

Complementary to ``tests/test_graph.py`` / ``tests/test_hardening_graph.py``
(which prove the sentinel ROUTING): this file pins the envelope CONTRACT — the
"never a half-viz" error shape, the exact-total ``too_large`` refuse, the empty
shape + note, canned citations, and the ``build_envelope`` shaping for arbitrary
totals — plus the UNKNOWN-status labeled bucket, the country top-N / "Other"
fold, and ``citations_truncated`` at the pure-function / tool layer.

Note: the empty-envelope shape is pinned via ``build_envelope`` directly (the
stable pure surface) rather than the full-graph ``_force_empty`` sentinel — that
routing is already owned by ``test_graph.py`` and the sentinel's two-pass re-plan
interacts with the in-flight ``plan`` stall detector, so the pure builder keeps
this file's empty-contract coverage deterministic and non-duplicative.
"""

from __future__ import annotations

from app import config
from app.api.schemas import ChartType, VisualizeRequest, VisualizeResponse
from app.ctgov import tools
from app.ctgov.fields import overall_status_key_fn
from app.graph.build import run_sync
from app.plan.models import Plan
from app.viz.spec import build_envelope


def _distribution_plan() -> Plan:
    return Plan(
        query_class="distribution",
        entities={"condition": "pancreatic cancer"},
        field="phase",
        chart_type=ChartType.BAR,
    )


# --- sentinel-driven envelope contracts -------------------------------------


def test_force_error_never_ships_half_viz() -> None:
    """``_force_error`` → a pure error envelope: code AND message populated, and
    NOTHING viz-shaped (no visualization, no vega_lite, no answer, no
    count_basis) — the "never a half-built viz" guarantee (API-22)."""
    resp = run_sync(VisualizeRequest(query="x"), overrides={"_force_error": True})

    assert resp.status == "error"
    assert resp.kind == "answer"
    assert resp.error is not None
    assert resp.error.code and resp.error.message  # both populated, non-empty
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.answer is None
    assert resp.meta.count_basis is None
    assert resp.citations == {}
    # Round-trips as a valid wire envelope.
    assert VisualizeResponse.model_validate(resp.model_dump()).status == "error"


def test_force_too_large_answer_carries_exact_total() -> None:
    """``_force_too_large`` → an ``answer`` refuse: no viz/vega, ``meta.partial``
    stays null (refusing to chart is not truncating, G-39), count_basis.trials is
    the EXACT total and count_basis.mentions is null, and the code-templated
    answer surfaces that total."""
    resp = run_sync(VisualizeRequest(query="y"), overrides={"_force_too_large": True})

    assert resp.status == "too_large"
    assert resp.kind == "answer"
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.meta.partial is None
    assert resp.meta.count_basis is not None
    assert resp.meta.count_basis.trials == 142_411
    assert resp.meta.count_basis.mentions is None
    assert resp.answer is not None
    assert "142,411" in resp.answer  # the exact total, code-templated into the prose
    assert VisualizeResponse.model_validate(resp.model_dump()).status == "too_large"


def test_force_canned_is_a_valid_bar_with_citations() -> None:
    """``_force_canned`` → a valid BAR whose every datum carries citations, and
    every datum ``source_id`` resolves to a real Citation (inline or the
    top-level dedup index, G-4)."""
    resp = run_sync(
        VisualizeRequest(query="phase distribution", condition="pancreatic cancer"),
        overrides={"_force_canned": True},
    )

    assert resp.status == "ok"
    assert resp.kind == "visualization"
    assert resp.visualization is not None
    assert resp.visualization.type == ChartType.BAR
    assert len(resp.visualization.data) >= 1
    assert resp.citations  # top-level dedup index is populated

    top_level_ids = set(resp.citations)
    for datum in resp.visualization.data:
        assert datum.citations, "every canned datum carries an inline citation"
        allowed = {c.nct_id for c in datum.citations} | top_level_ids
        for source_id in datum.source_ids:
            assert source_id in allowed


# --- build_envelope shaping, generic values (pure, no graph) ----------------


def test_build_envelope_too_large_uses_the_computed_total() -> None:
    """``build_envelope(status='too_large')`` shapes the refuse from an ARBITRARY
    computed total (not just the sentinel's 142,411): count_basis.trials == that
    total, partial null, no viz/vega, and the total in the answer."""
    resp = build_envelope(
        plan=None,
        tool_results=[{"tool": "count_trials", "total_count": 50_000}],
        status="too_large",
        question="phase distribution of diabetes trials",
    )
    assert resp.status == "too_large"
    assert resp.kind == "answer"
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.meta.partial is None
    assert resp.meta.count_basis.trials == 50_000
    assert "50,000" in resp.answer


def test_build_envelope_error_populates_error_obj_only() -> None:
    """``build_envelope(status='error')`` carries the ErrorObj and nothing
    viz-shaped; count_basis is null (no retrieval happened)."""
    resp = build_envelope(
        plan=None,
        tool_results=[],
        status="error",
        question="q",
        error={"code": "upstream_timeout", "message": "provider timed out"},
    )
    assert resp.status == "error"
    assert resp.kind == "answer"
    assert resp.error.code == "upstream_timeout"
    assert resp.error.message == "provider timed out"
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.meta.count_basis is None


def test_build_envelope_empty_distribution_shape() -> None:
    """``build_envelope(status='empty')`` for a distribution → an empty-data BAR
    (still ``kind:'visualization'``) with the 'no trials matched' note."""
    resp = build_envelope(
        plan=_distribution_plan(),
        tool_results=[{"tool": "aggregate_by", "buckets": []}],
        status="empty",
        question="phase distribution",
    )
    assert resp.status == "empty"
    assert resp.kind == "visualization"
    assert resp.visualization.data == []
    assert any("no trials" in note.lower() for note in resp.meta.notes)


def test_build_envelope_single_value_answer_shape() -> None:
    """A ``single_value`` plan with ``answer_kind='answer'`` → a code-templated
    yes/no ``answer`` envelope: no visualization, no vega_lite (CC-7)."""
    plan = Plan(
        query_class="single_value",
        entities={"condition": "melanoma"},
        chart_type=ChartType.SINGLE_VALUE,
        answer_kind="answer",
    )
    resp = build_envelope(
        plan=plan,
        tool_results=[
            {"tool": "count_trials", "total_count": 42, "kind": "answer", "citations": []}
        ],
        status="ok",
        question="are there melanoma trials?",
    )
    assert resp.status == "ok"
    assert resp.kind == "answer"
    assert resp.visualization is None
    assert resp.vega_lite is None
    assert resp.answer is not None
    assert "42" in resp.answer


# --- UNKNOWN-status labeled bucket (pure function) --------------------------


def test_overall_status_unknown_bucket_is_labeled() -> None:
    """A record with no status (or a malformed protocolSection) maps to the
    single labeled UNKNOWN bucket — never a raise (K1/B2 totality)."""
    no_status = {"protocolSection": {"identificationModule": {"nctId": "NCT01"}}}
    assert overall_status_key_fn(no_status) == [("UNKNOWN", "Unknown status")]
    assert overall_status_key_fn({"protocolSection": None}) == [("UNKNOWN", "Unknown status")]
    assert overall_status_key_fn({}) == [("UNKNOWN", "Unknown status")]


def test_overall_status_known_and_titlecased_labels() -> None:
    """Known tokens get their curated label; an unseen-but-present token
    title-cases to its own honest bucket (never dropped)."""

    def _rec(status: str) -> dict:
        return {"protocolSection": {"statusModule": {"overallStatus": status}}}

    assert overall_status_key_fn(_rec("RECRUITING")) == [("RECRUITING", "Recruiting")]
    assert overall_status_key_fn(_rec("NOT_YET_RECRUITING")) == [
        ("NOT_YET_RECRUITING", "Not yet recruiting")
    ]
    # An unknown/future token falls back to a title-cased label on its own bucket.
    assert overall_status_key_fn(_rec("FOO_BAR")) == [("FOO_BAR", "Foo Bar")]


# --- high-cardinality top-N + "Other" fold + citations_truncated ------------


class _FakeAggClient:
    """A ``CTGovClient`` stand-in for ``aggregate_by``: ``iter_studies`` returns
    canned records, no network. Mirrors the real keyword signature."""

    def __init__(self, records: list[dict], *, truncated: bool = False) -> None:
        self._records = records
        self._truncated = truncated

    def iter_studies(self, search_params, *, fields, page_size=1000, max_pages=20):
        return list(self._records), self._truncated


def _country_record(nct_id: str, country: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "contactsLocationsModule": {"locations": [{"country": country}]},
        }
    }


def test_country_top_n_folds_lowcount_tail_into_other(monkeypatch) -> None:
    """(TOP_N_CATEGORIES + 3) distinct, disjoint countries with strictly-descending
    counts → the ``country`` spec keeps the top ``TOP_N_CATEGORIES`` and folds the 3
    lowest-count countries into ONE derived 'Other' bucket that cites its members and
    sorts last; the explode reconciliation (Σ bars == distinct == total) still holds.
    N-agnostic (P5-TOPN reconciled the cap 15 → 50 — this test tracks the config)."""
    top = config.TOP_N_CATEGORIES
    n = top + 3  # exactly 3 beyond the cap → exactly 3 fold into "Other"
    records: list[dict] = []
    serial = 0
    expected_counts: dict[str, int] = {}
    for i in range(n):
        country = f"Country{i:03d}"  # not an alias-table entry → no folding by canonicalization
        count = n - i + 1  # strictly descending, all >= 2, distinct
        expected_counts[country] = count
        for _ in range(count):
            serial += 1
            records.append(_country_record(f"NCT{serial:08d}", country))
    total = sum(expected_counts.values())

    monkeypatch.setattr(tools, "CTGovClient", lambda: _FakeAggClient(records))
    result = tools.aggregate_by({"cond": "cancer"}, {}, "country")

    by_value = {b["value"]: b for b in result["buckets"]}
    real = [b for b in result["buckets"] if b["value"] != "Other"]
    assert len(real) == top  # exactly the top-N kept
    assert "Other" in by_value

    folded = [f"Country{i:03d}" for i in range(top, n)]  # the 3 lowest-count, count-desc order
    other = by_value["Other"]
    assert other["derived"] is True
    assert other["members"] == folded
    assert other["label"] == "Other (3 countries)"
    assert other["count_trials"] == sum(expected_counts[c] for c in folded)
    assert other["citations"], "the Other fold cites its member records"

    # Other is a sentinel → pinned to the end of the ranked bar.
    assert result["buckets"][-1]["value"] == "Other"
    # Explode reconciliation: disjoint countries → Σ bars == distinct == countTotal.
    assert result["distinct_trials"] == total
    assert sum(b["count_trials"] for b in result["buckets"]) == total


def test_country_facet_discloses_normalization_and_folds_variants(monkeypatch) -> None:
    """E-20 honesty: the country facet BOTH folds spelling variants (USA + United
    States → one bucket) AND discloses in meta.notes that spellings are normalized —
    the disclosure the pre-fix docstring promised but never emitted."""
    records = [
        _country_record("NCT00000001", "USA"),
        _country_record("NCT00000002", "United States"),
        _country_record("NCT00000003", "U.S."),
    ]
    monkeypatch.setattr(tools, "CTGovClient", lambda: _FakeAggClient(records))
    result = tools.aggregate_by({"cond": "x"}, {}, "country")

    notes = " ".join(result.get("notes") or [])
    assert "normalized" in notes.lower() and "United States" in notes  # the disclosure fires

    non_unknown = [b for b in result["buckets"] if b["value"] != "UNKNOWN"]
    assert len(non_unknown) == 1  # all three variants folded into ONE bucket
    assert non_unknown[0]["value"] == "United States"
    assert non_unknown[0]["count_trials"] == 3


def test_bucket_citations_truncated_over_k(monkeypatch) -> None:
    """A single country with 25 contributing trials → its bucket caps the
    citation sample at K=20 and flags ``citations_truncated`` while reporting the
    exact ``contributing_count``."""
    records = [_country_record(f"NCT{i:08d}", "United States") for i in range(25)]
    monkeypatch.setattr(tools, "CTGovClient", lambda: _FakeAggClient(records))

    result = tools.aggregate_by({"cond": "cancer"}, {}, "country")
    bucket = next(b for b in result["buckets"] if b["value"] == "United States")

    assert bucket["count_trials"] == 25
    assert bucket["contributing_count"] == 25
    assert bucket["citations_truncated"] is True
    assert len(bucket["citations"]) == 20  # capped at K
