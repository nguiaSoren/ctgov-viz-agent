"""Egress-boundary tests: ``params.build_search_params`` + the live HTTP client.

Two layers:

* **Pure unit** (no network) — the SSRF invariant of ``build_search_params``:
  user free-text lands ONLY in a ``query.<area>`` value slot, and an unknown
  area is rejected. Plus the code-generated ``filter.advanced`` / status token.
* **Live** (``$0``, no key) — per the project's LIVE-ONLY decision (no VCR), we
  hit the real ClinicalTrials.gov v2 API: the interventional pancreatic-cancer
  count reconciles to ~3950, and a 2-page walk returns distinct nctIds. If the
  network is unreachable the live tests ``skip`` (not fail), so an offline run
  never produces a false red — a wrong count or logic bug still fails loudly.
"""

from __future__ import annotations

import pytest

from app.ctgov.client import CTGovClient, UpstreamError
from app.ctgov.params import build_search_params

# The X-2 gate population: interventional pancreatic-cancer trials.
_PANCREATIC_QUERY = {"cond": "pancreatic cancer"}
_INTERVENTIONAL_FILTERS = {"interventional_only": True}
_INTERVENTIONAL_ESSIE = "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL"
_EXPECTED_COUNT = 3950  # live-verified 2026-07-15; registry drifts a little.


# --- Pure unit: build_search_params SSRF invariant --------------------------


class TestBuildSearchParamsSSRF:
    def test_user_text_only_in_value_slot(self):
        """User free-text is a param VALUE, never a param name or an area code."""
        params = build_search_params({"cond": "x"}, {})
        assert params == {"query.cond": "x"}
        # The literal user text is the value; the only key is a code-generated name.
        assert list(params.keys()) == ["query.cond"]
        assert params["query.cond"] == "x"

    def test_unknown_area_raises(self):
        with pytest.raises(ValueError):
            build_search_params({"evil": "x"}, {})

    def test_value_passed_through_unencoded(self):
        """The client URL-encodes; pre-encoding here would double-encode."""
        params = build_search_params({"cond": "a & b"}, {})
        assert params["query.cond"] == "a & b"

    def test_interventional_only_emits_filter_advanced(self):
        params = build_search_params(_PANCREATIC_QUERY, _INTERVENTIONAL_FILTERS)
        assert params["query.cond"] == "pancreatic cancer"
        assert params["filter.advanced"] == _INTERVENTIONAL_ESSIE

    def test_status_token_validated_and_emitted(self):
        params = build_search_params({}, {"status": "RECRUITING"})
        assert params["filter.overallStatus"] == "RECRUITING"

    def test_unknown_status_token_raises(self):
        with pytest.raises(ValueError):
            build_search_params({}, {"status": "definitely-not-a-token"})

    def test_unknown_study_type_token_raises(self):
        with pytest.raises(ValueError):
            build_search_params({}, {"studyType": "bogus"})

    def test_study_type_and_interventional_only_do_not_duplicate(self):
        params = build_search_params(
            {}, {"interventional_only": True, "studyType": "INTERVENTIONAL"}
        )
        assert params["filter.advanced"] == _INTERVENTIONAL_ESSIE

    def test_year_range_uses_ints_and_open_bounds(self):
        both = build_search_params({}, {"start_year": 2015, "end_year": 2020})
        assert both["filter.advanced"] == "AREA[StartDate]RANGE[2015-01-01,2020-12-31]"
        open_end = build_search_params({}, {"start_year": 2015})
        assert open_end["filter.advanced"] == "AREA[StartDate]RANGE[2015-01-01,MAX]"
        open_start = build_search_params({}, {"end_year": 2020})
        assert open_start["filter.advanced"] == "AREA[StartDate]RANGE[MIN,2020-12-31]"

    def test_composed_and_joined_expression(self):
        params = build_search_params(
            _PANCREATIC_QUERY, {"interventional_only": True, "start_year": 2015}
        )
        assert params["filter.advanced"] == (
            "AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL AND "
            "AREA[StartDate]RANGE[2015-01-01,MAX]"
        )

    def test_bad_year_type_and_range_raise(self):
        with pytest.raises(ValueError):
            build_search_params({}, {"start_year": "2015"})  # str, not int
        with pytest.raises(ValueError):
            build_search_params({}, {"start_year": True})  # bool is not a year
        with pytest.raises(ValueError):
            build_search_params({}, {"start_year": 1800})  # out of range

    def test_no_filters_yields_no_filter_params(self):
        assert build_search_params({"term": "cancer"}, {}) == {"query.term": "cancer"}


# --- Live: real ClinicalTrials.gov v2 API ($0, no key) ----------------------


def _skip_if_transient(exc: Exception) -> None:
    """Skip (never fail) on ANY transport-level ``UpstreamError`` — offline
    (`upstream_unreachable`) OR a retry-exhausted rate-limit (`upstream_status_429`/`5xx`)
    that the public API returns under a full-suite burst (LESSON H1). Safe without
    losing teeth: these live tests raise ``UpstreamError`` ONLY from the client
    call inside the ``try`` — a transport condition, never a correctness bug — so
    a real regression still surfaces as an assertion failure below the ``try``.
    Anything that is not an ``UpstreamError`` re-raises."""
    if isinstance(exc, UpstreamError):
        pytest.skip(f"clinicaltrials.gov transport error ({exc.code}) — live test skipped")
    raise exc


@pytest.fixture(scope="module")
def client() -> CTGovClient:
    return CTGovClient()


@pytest.fixture(scope="module")
def interventional_params() -> dict:
    return build_search_params(_PANCREATIC_QUERY, _INTERVENTIONAL_FILTERS)


class TestLiveClient:
    def test_count_reconciles_to_expected(self, client, interventional_params):
        """Interventional pancreatic-cancer count is ~3950 (allow live drift)."""
        try:
            n = client.count(interventional_params)
        except Exception as exc:  # noqa: BLE001 — narrow to offline-skip below
            _skip_if_transient(exc)
            raise
        assert isinstance(n, int)
        assert abs(n - _EXPECTED_COUNT) <= 200, f"count={n} drifted far from {_EXPECTED_COUNT}"

    def test_two_page_walk_returns_distinct_nct_ids(self, client, interventional_params):
        try:
            records, truncated = client.iter_studies(
                interventional_params, fields="NCTId|Phase", page_size=100, max_pages=2
            )
        except Exception as exc:  # noqa: BLE001 — narrow to offline-skip below
            _skip_if_transient(exc)
            raise
        assert len(records) == 200, f"expected a full 2×100 walk, got {len(records)}"
        ids = [r["protocolSection"]["identificationModule"]["nctId"] for r in records]
        assert len(set(ids)) == len(ids), "paging returned duplicate nctIds across pages"
        # 3950 rows >> 200, so a 2-page budget must report truncated.
        assert truncated is True

    def test_page_size_clamped_to_cap(self, client, interventional_params):
        """A >1000 page_size is clamped to the API's hard cap without error."""
        try:
            records, _ = client.iter_studies(
                interventional_params, fields="NCTId", page_size=5000, max_pages=1
            )
        except Exception as exc:  # noqa: BLE001 — narrow to offline-skip below
            _skip_if_transient(exc)
            raise
        assert len(records) <= 1000
