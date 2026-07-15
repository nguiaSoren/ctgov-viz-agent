"""Regression tests for the schema-guard + golden-fixture hardening pass.

Two guarantees added here that ``tests/test_schemas.py`` didn't cover:

1. ``Datum.source_ids`` / ``Edge.source_ids`` **resolve** — every sampled
   nctId cited by a datum/edge is backed by a real ``Citation``, either
   inline in that datum's/edge's own ``citations[]`` or in the top-level
   ``VisualizeResponse.citations{}`` dedup index (G-4).
2. ``Visualization``'s ``type``↔``data`` shape invariant is a real guarantee,
   not just a structural union: ``network_graph`` MUST carry ``NetworkData``,
   every other ``type`` MUST carry ``list[Datum]`` — a mismatched mark raises.

Plus the presence-semantics acceptance criteria (kind/status field presence)
for ``golden_answer`` / ``golden_error`` / ``golden_too_large``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ChartType,
    Datum,
    NetworkData,
    Visualization,
    VisualizeResponse,
)

FIXTURES = Path(__file__).parent / "fixtures"

GOLDENS = [
    "golden_distribution.json",
    "golden_timeseries.json",
    "golden_network.json",
    "golden_answer.json",
    "golden_error.json",
    "golden_too_large.json",
]


# --- source_ids resolve ----------------------------------------------------


@pytest.mark.parametrize("filename", GOLDENS)
def test_source_ids_resolve(filename: str) -> None:
    """Every source_id on every datum/edge resolves to a real Citation —
    either inline in that datum's/edge's own citations[] or in the top-level
    citations{} dedup index (G-4)."""
    payload = json.loads((FIXTURES / filename).read_text())
    resp = VisualizeResponse.model_validate(payload)
    top_level_ids = set(resp.citations.keys())

    if resp.visualization is None:
        pytest.skip(f"{filename}: kind={resp.kind!r} carries no visualization")

    data = resp.visualization.data
    items = data if isinstance(data, list) else data.edges

    for item in items:
        inline_ids = {c.nct_id for c in item.citations}
        allowed = inline_ids | top_level_ids
        for source_id in item.source_ids:
            assert source_id in allowed, (
                f"{filename}: source_id {source_id!r} resolves to no Citation "
                f"(inline={inline_ids}, top-level={top_level_ids})"
            )


# --- type<->shape guard -----------------------------------------------------


def _sample_datum() -> Datum:
    return Datum(value="PHASE1", label="Phase 1", count_trials=1)


def _sample_network_data() -> NetworkData:
    return NetworkData(nodes=[], edges=[])


def test_network_type_with_row_data_raises() -> None:
    """type=network_graph with a row list (not NetworkData) is rejected."""
    with pytest.raises(ValidationError):
        Visualization(
            type=ChartType.NETWORK_GRAPH,
            title="x",
            encoding={},
            data=[_sample_datum()],
        )


def test_bar_type_with_network_data_raises() -> None:
    """type=bar with a NetworkData payload (not rows) is rejected."""
    with pytest.raises(ValidationError):
        Visualization(
            type=ChartType.BAR,
            title="x",
            encoding={},
            data=_sample_network_data(),
        )


def test_bar_type_with_row_data_validates() -> None:
    """The correct pairing (bar + rows) validates."""
    viz = Visualization(
        type=ChartType.BAR,
        title="x",
        encoding={},
        data=[_sample_datum()],
    )
    assert isinstance(viz.data, list)


def test_network_type_with_network_data_validates() -> None:
    """The correct pairing (network_graph + NetworkData) validates."""
    viz = Visualization(
        type=ChartType.NETWORK_GRAPH,
        title="x",
        encoding={},
        data=_sample_network_data(),
    )
    assert isinstance(viz.data, NetworkData)


# --- presence semantics (the Phase-0 acceptance) ---------------------------


def test_golden_answer_presence_semantics() -> None:
    payload = json.loads((FIXTURES / "golden_answer.json").read_text())
    resp = VisualizeResponse.model_validate(payload)
    assert resp.answer is not None
    assert resp.visualization is None


def test_golden_error_presence_semantics() -> None:
    payload = json.loads((FIXTURES / "golden_error.json").read_text())
    resp = VisualizeResponse.model_validate(payload)
    assert resp.error is not None
    assert resp.visualization is None


def test_golden_too_large_presence_semantics() -> None:
    payload = json.loads((FIXTURES / "golden_too_large.json").read_text())
    resp = VisualizeResponse.model_validate(payload)
    assert resp.meta.partial is None
    assert resp.meta.count_basis is not None
    assert resp.meta.count_basis.trials is not None
    assert resp.vega_lite is None
