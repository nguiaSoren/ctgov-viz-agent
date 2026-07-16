"""Offline unit tests for ``app.ctgov.compare.union_series`` (W1d).

Pure functions — no network. Buckets are synthetic ``aggregate_by``-shaped dicts
carrying real ``Citation`` objects so the verbatim provenance pass-through is
exercised. Covers: two-series union, 0-fill for a category present in only one
series, within-series percentage (each series' own N as denominator, CC-14),
series-label attachment, deterministic ordering across re-runs, and TOTAL
behavior on empty/malformed series.
"""

from __future__ import annotations

from app.api.schemas import Citation
from app.ctgov.compare import union_series

FIELD_PATH = "protocolSection.statusModule.overallStatus"


def _citation(nct_id: str, value: str) -> Citation:
    """A real Citation (excerpt == the status token, string-extracted upstream)."""
    return Citation(nct_id=nct_id, field_path=FIELD_PATH, value=value, matched_value=value)


def _bucket(value: str, label: str, count: int, *, nct: str | None = None) -> dict:
    """An ``aggregate_by``-shaped bucket-dict with a single citation."""
    nct = nct or f"NCT{abs(hash(value)) % 10**8:08d}"
    return {
        "value": value,
        "label": label,
        "count_trials": count,
        "count_mentions": count,
        "source_ids": [nct],
        "citations": [_citation(nct, value)],
        "citations_truncated": False,
        "contributing_count": count,
    }


def _two_series() -> list[dict]:
    """Pembrolizumab (N=903) and Nivolumab (N=612).

    ``RECRUITING`` count is 50 in BOTH so within-series percent must still differ.
    ``TERMINATED`` appears only in Pembrolizumab → Nivolumab must be 0-filled for it.
    """
    return [
        {
            "label": "Pembrolizumab",
            "N": 903,
            "buckets": [
                _bucket("RECRUITING", "Recruiting", 50),
                _bucket("COMPLETED", "Completed", 400),
                _bucket("TERMINATED", "Terminated", 30),
            ],
        },
        {
            "label": "Nivolumab",
            "N": 612,
            "buckets": [
                _bucket("RECRUITING", "Recruiting", 50),
                _bucket("COMPLETED", "Completed", 300),
            ],
        },
    ]


def _index(datums: list[dict]) -> dict[tuple[str, str], dict]:
    """Index datums by (category value, series label)."""
    return {(d["value"], d["series"]): d for d in datums}


def test_two_series_union_all_categories_present_in_both() -> None:
    """Every (category × series) pair is emitted — 3 categories × 2 series = 6 datums."""
    datums, _notes = union_series(_two_series())

    by_key = _index(datums)
    assert len(datums) == 6
    for value in ("RECRUITING", "COMPLETED", "TERMINATED"):
        assert (value, "Pembrolizumab") in by_key
        assert (value, "Nivolumab") in by_key


def test_zero_fill_for_category_present_in_only_one_series() -> None:
    """TERMINATED exists only in Pembrolizumab → Nivolumab gets a 0-fill datum."""
    datums, _notes = union_series(_two_series())
    fill = _index(datums)[("TERMINATED", "Nivolumab")]

    assert fill["count_trials"] == 0
    assert fill["count_mentions"] == 0
    assert fill["percent"] == 0.0
    assert fill["citations"] == []  # legit empty — no authored citation (G-35)
    assert fill["source_ids"] == []
    assert fill["contributing_count"] == 0
    assert fill["series"] == "Nivolumab"


def test_percent_is_within_series_not_union_total() -> None:
    """Same raw count (50) in both series yields DIFFERENT within-series percents."""
    datums, _notes = union_series(_two_series())
    by_key = _index(datums)

    pembro = by_key[("RECRUITING", "Pembrolizumab")]
    nivo = by_key[("RECRUITING", "Nivolumab")]

    assert pembro["count_trials"] == nivo["count_trials"] == 50  # identical raw count
    assert pembro["percent"] == round(100 * 50 / 903, 1) == 5.5
    assert nivo["percent"] == round(100 * 50 / 612, 1) == 8.2
    assert pembro["percent"] != nivo["percent"]  # denominator is each series' own N


def test_series_label_attached_to_every_datum() -> None:
    """No datum leaves without its series label."""
    datums, _notes = union_series(_two_series())
    assert all(d["series"] in {"Pembrolizumab", "Nivolumab"} for d in datums)
    assert all(isinstance(d["series"], str) and d["series"] for d in datums)


def test_citations_passed_through_verbatim() -> None:
    """A present bucket's Citation objects survive unchanged (never re-authored)."""
    datums, _notes = union_series(_two_series())
    completed = _index(datums)[("COMPLETED", "Pembrolizumab")]

    assert len(completed["citations"]) == 1
    citation = completed["citations"][0]
    assert isinstance(citation, Citation)
    assert citation.field_path == FIELD_PATH
    assert citation.matched_value == "COMPLETED"


def test_deterministic_order_across_two_runs() -> None:
    """Re-running yields byte-identical (value, series) row order."""
    run_a = [(d["value"], d["series"]) for d in union_series(_two_series())[0]]
    run_b = [(d["value"], d["series"]) for d in union_series(_two_series())[0]]
    assert run_a == run_b


def test_category_order_is_first_seen_union() -> None:
    """Categories appear in first-seen order across series; series is the inner loop.

    Pembrolizumab introduces RECRUITING, COMPLETED, TERMINATED (in that order);
    Nivolumab introduces nothing new → union order is exactly Pembro's order.
    """
    datums, _notes = union_series(_two_series())
    # Category-outer, series-inner: each category's two series bars are adjacent.
    seen_categories = []
    for datum in datums:
        if datum["value"] not in seen_categories:
            seen_categories.append(datum["value"])
    assert seen_categories == ["RECRUITING", "COMPLETED", "TERMINATED"]


def test_new_category_from_second_series_appended_last() -> None:
    """A category first seen in series B lands after all of series A's categories."""
    series = _two_series()
    series[1]["buckets"].append(_bucket("WITHDRAWN", "Withdrawn", 12))

    datums, _notes = union_series(series)
    order: list[str] = []
    for datum in datums:
        if datum["value"] not in order:
            order.append(datum["value"])
    assert order == ["RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN"]

    # And Pembrolizumab (series A) is 0-filled for the B-only category.
    fill = _index(datums)[("WITHDRAWN", "Pembrolizumab")]
    assert fill["count_trials"] == 0
    assert fill["percent"] == 0.0


def test_notes_disclose_each_series_n_and_within_series_basis() -> None:
    """Notes carry the per-series N list + the within-series-percentage disclosure."""
    _datums, notes = union_series(_two_series())
    assert "Pembrolizumab N=903; Nivolumab N=612" in notes
    assert any("within-series" in note for note in notes)


def test_empty_bucket_list_series_does_not_crash() -> None:
    """A series with zero buckets is fully 0-filled against the other series' union."""
    series = [
        {
            "label": "Pembrolizumab",
            "N": 903,
            "buckets": [_bucket("RECRUITING", "Recruiting", 50)],
        },
        {"label": "Nivolumab", "N": 612, "buckets": []},
    ]
    datums, _notes = union_series(series)

    nivo = [d for d in datums if d["series"] == "Nivolumab"]
    assert len(nivo) == 1
    assert nivo[0]["value"] == "RECRUITING"
    assert nivo[0]["count_trials"] == 0
    assert nivo[0]["percent"] == 0.0
    assert nivo[0]["citations"] == []


def test_total_on_missing_n_gives_zero_percent_no_raise() -> None:
    """A series missing N (or N=0) never divides by zero — percent falls back to 0.0."""
    series = [
        {"label": "A", "buckets": [_bucket("RECRUITING", "Recruiting", 40)]},  # no N
        {"label": "B", "N": 0, "buckets": [_bucket("RECRUITING", "Recruiting", 10)]},
    ]
    datums, notes = union_series(series)
    by_key = _index(datums)

    assert by_key[("RECRUITING", "A")]["percent"] == 0.0
    assert by_key[("RECRUITING", "A")]["count_trials"] == 40  # raw count still honest
    assert by_key[("RECRUITING", "B")]["percent"] == 0.0
    assert "A N=unknown; B N=unknown" in notes


def test_total_on_malformed_series_and_buckets() -> None:
    """Non-dict series entries, non-dict buckets, and value-less buckets are skipped."""
    series = [
        None,  # not a dict
        "garbage",  # not a dict
        {"label": "Real", "N": 100, "buckets": [
            _bucket("RECRUITING", "Recruiting", 20),
            "not-a-bucket",
            {"label": "no value here", "count_trials": 5},  # missing "value"
            {"value": 123, "label": "non-str value", "count_trials": 5},  # non-str value
        ]},
    ]
    datums, _notes = union_series(series)

    # Only the one well-formed bucket survives; malformed entries drop silently.
    assert len(datums) == 1
    assert datums[0]["value"] == "RECRUITING"
    assert datums[0]["series"] == "Real"
    assert datums[0]["percent"] == 20.0  # 20/100


def test_non_list_input_returns_empty_datums_no_raise() -> None:
    """A wholly malformed (non-list) input yields no datums but still the standing note."""
    datums, notes = union_series(None)  # type: ignore[arg-type]
    assert datums == []
    assert any("within-series" in note for note in notes)
