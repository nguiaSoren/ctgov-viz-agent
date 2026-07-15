"""The Vega-Lite projection (ARCHITECTURE_SPEC §3.7) -- Phase 0.

The custom canonical ``Visualization`` (``app.api.schemas``) is the source of
truth; this module derives a **convenience** Vega-Lite v5 spec for the
standard chart types a frontend can render directly. It is intentionally
small: Phase 0 has exactly one real producer (the distribution recipe's bar
chart), so this only needs to be *correct* for the standard marks, not
exhaustive.

``network_graph`` is never projected (C-60/G-41d) -- a node-link graph has no
Vega-Lite mark, and the custom schema is the only shape that can carry it.
``single_value`` and ``table`` also have no natural Vega-Lite mark, so both
return ``None`` alongside ``network_graph``.
"""

from __future__ import annotations

from app.api.schemas import ChartType, Visualization

# The chart types this Phase-0 projector knows how to render as Vega-Lite v5,
# and the mark each one maps to. Anything not in this table (network_graph /
# single_value / table) returns None from `to_vega_lite`.
_STANDARD_MARKS: dict[ChartType, str] = {
    ChartType.BAR: "bar",
    ChartType.GROUPED_BAR: "bar",
    ChartType.TIME_SERIES: "line",
    ChartType.HISTOGRAM: "bar",
    ChartType.SCATTER: "point",
}


def to_vega_lite(viz: Visualization) -> dict | None:
    """Project ``viz`` to a minimal, valid Vega-Lite v5 spec, or ``None``.

    Returns ``None`` for ``network_graph``/``single_value``/``table`` (no
    natural Vega-Lite mark; networks specifically must NEVER be expressed as
    Vega-Lite, C-60/G-41d). For the standard types, builds one inline-data row
    per ``Datum`` and reads the x/y channel field names off
    ``viz.encoding`` -- so the projection stays in sync with whatever fields
    the viz-builder actually populated, rather than hardcoding a shape.
    """
    mark = _STANDARD_MARKS.get(viz.type)
    if mark is None:
        return None

    # Defensive: a standard chart type should never carry NetworkData, but
    # this function must not assume that upstream invariant holds silently.
    if not isinstance(viz.data, list):
        return None

    x_channel = viz.encoding.get("x")
    y_channel = viz.encoding.get("y")
    color_channel = viz.encoding.get("color")  # present only for grouped_bar (compare series)
    x_field = x_channel.field if x_channel is not None else "value"
    y_field = y_channel.field if y_channel is not None else "count_trials"
    color_field = color_channel.field if color_channel is not None else None

    # A time series reads best on an ordinal (sortable) x; categorical charts use nominal.
    x_type = "ordinal" if viz.type is ChartType.TIME_SERIES else "nominal"

    def _row(datum: object) -> dict:
        row = {x_field: getattr(datum, x_field, None), y_field: getattr(datum, y_field, None)}
        if color_field is not None:
            row[color_field] = getattr(datum, color_field, None)
        return row

    values = [_row(datum) for datum in viz.data]

    encoding: dict = {
        "x": {
            "field": x_field,
            "type": x_type,
            "title": (x_channel.label if x_channel is not None else x_field) or x_field,
        },
        "y": {
            "field": y_field,
            "type": "quantitative",
            "title": (y_channel.label if y_channel is not None else y_field) or y_field,
        },
    }
    if color_field is not None:
        # Grouped bar: color splits the series and xOffset places the bars side by side.
        color_spec = {
            "field": color_field,
            "type": "nominal",
            "title": (color_channel.label if color_channel is not None else color_field) or color_field,
        }
        encoding["color"] = color_spec
        encoding["xOffset"] = {"field": color_field, "type": "nominal"}

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": viz.title,
        "data": {"values": values},
        "mark": mark,
        "encoding": encoding,
    }
