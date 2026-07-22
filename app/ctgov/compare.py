"""Union of two-or-more per-series aggregations into grouped-bar datums (W1d).

Pure module — no I/O, no client, no LLM. It consumes the per-series bucket-dicts
already produced by ``app.ctgov.tools.aggregate_by`` (each a
``{value, label, count_trials, count_mentions, source_ids, citations,
citations_truncated, contributing_count}``) and folds them into ONE flat list of
grouped-bar ``Datum``-dicts.

Two invariants make this honest:

* **0-fill (G-35).** A category present in one series but absent from another
  still yields a ``(category, series)`` datum with ``count_trials=0`` /
  ``percent=0.0`` and NO citations — a legitimately empty bar, not a hidden gap.
* **Within-series percentage (CC-14).** The ``percent`` channel divides each
  bar's raw count by *that series' own N*, never the union total, so a large-N
  series cannot visually swamp a small-N one. Raw ``count_trials`` is retained
  per bar for the tooltip.

Compare is MULTI-population (two ``countTotal``s), so it is exempt from the
single-oracle Σ==countTotal precheck — each series is self-reconciled upstream by
its own ``aggregate_by``. The per-series citations are passed through verbatim
(never authored here), so the excerpt tamper-evidence check still holds.

Every function here is TOTAL over live/malformed input: it never raises on a
missing key, an empty series, a series with zero buckets, or a series whose N is
missing/zero. Output ordering is deterministic (categories in first-seen union
order; series in input order) so a reviewer re-running it gets byte-identical rows.
"""

from __future__ import annotations


def _is_number(value: object) -> bool:
    """True for a real int/float (``bool`` excluded — it is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_count(value: object) -> int:
    """Coerce a bucket count to a non-negative-safe int; non-numbers → 0 (TOTAL)."""
    return int(value) if _is_number(value) else 0


def union_series(series_results: list[dict]) -> tuple[list[dict], list[str]]:
    """Union ≥2 per-series aggregations into grouped-bar ``Datum``-dicts.

    The count basis is always within-series percentage (CC-14) — there is no basis
    parameter. (An unused ``count_basis="pct_within_series"`` keyword used to sit in
    this signature; nothing ever passed it and the body never read it, so it was
    removed rather than left as a knob that does nothing.)

    Parameters
    ----------
    series_results:
        ``[{"label": "Pembrolizumab", "N": 903, "buckets": [aggregate_by-bucket]}, ...]``
        — one entry per series (≥2 in practice; fewer never raises). ``buckets``
        are the per-bucket dicts ``aggregate_by`` returns.

    Returns
    -------
    ``(datum_dicts, notes)`` where each datum is
    ``{value, label, series, count_trials, count_mentions, percent, source_ids,
    citations, citations_truncated, contributing_count}`` and ``notes`` discloses
    each series' N plus the within-series-percentage convention.
    """
    if not isinstance(series_results, list):
        series_results = []

    # --- 1. Normalize each series + collect the union of categories -----------
    # normalized: list of (series_label, N-or-None, {category_value: bucket_dict}).
    normalized: list[tuple[str, float | None, dict[str, dict]]] = []
    # category_labels: value -> first-seen display label. Insertion order == the
    # deterministic first-seen union order across series (dicts preserve it).
    category_labels: dict[str, str] = {}

    for index, series in enumerate(series_results):
        if not isinstance(series, dict):
            # Genuine garbage (not a dict) is not a series — drop it rather than
            # coerce it into a phantom 0-filled bar. A legitimate but empty
            # ``{"label", "N", "buckets": []}`` series is kept and 0-filled below.
            continue

        label = series.get("label")
        if not isinstance(label, str) or not label:
            label = f"series_{index + 1}"

        n_raw = series.get("N")
        series_n = n_raw if _is_number(n_raw) and n_raw > 0 else None

        raw_buckets = series.get("buckets")
        if not isinstance(raw_buckets, list):
            raw_buckets = []

        bucket_map: dict[str, dict] = {}
        for bucket in raw_buckets:
            if not isinstance(bucket, dict):
                continue
            value = bucket.get("value")
            if not isinstance(value, str) or not value:
                continue
            if value in bucket_map:
                continue  # first occurrence within a series wins (deterministic)
            bucket_map[value] = bucket
            if value not in category_labels:
                cat_label = bucket.get("label")
                category_labels[value] = (
                    cat_label if isinstance(cat_label, str) and cat_label else value
                )

        normalized.append((label, series_n, bucket_map))

    # --- 2. Emit datums: category-outer (union order), series-inner -----------
    datums: list[dict] = []
    for value, label in category_labels.items():
        for series_label, series_n, bucket_map in normalized:
            bucket = bucket_map.get(value)
            if bucket is None:
                # 0-fill: legitimately empty bar (G-35) — no citations authored.
                datums.append(
                    {
                        "value": value,
                        "label": label,
                        "series": series_label,
                        "count_trials": 0,
                        "count_mentions": 0,
                        "percent": 0.0,
                        "source_ids": [],
                        "citations": [],
                        "citations_truncated": False,
                        "contributing_count": 0,
                    }
                )
                continue

            count = _as_count(bucket.get("count_trials"))
            percent = round(100 * count / series_n, 1) if series_n else 0.0
            datums.append(
                {
                    "value": value,
                    "label": label,
                    "series": series_label,
                    "count_trials": count,
                    "count_mentions": bucket.get("count_mentions"),
                    "percent": percent,
                    # Provenance passed through verbatim — never authored here.
                    "source_ids": list(bucket.get("source_ids") or []),
                    "citations": list(bucket.get("citations") or []),
                    "citations_truncated": bool(bucket.get("citations_truncated", False)),
                    "contributing_count": bucket.get("contributing_count"),
                }
            )

    # --- 3. Notes: disclose each series' N + the within-series convention ------
    notes: list[str] = []
    n_parts = [
        f"{series_label} N={int(series_n)}" if series_n else f"{series_label} N=unknown"
        for series_label, series_n, _bucket_map in normalized
    ]
    if n_parts:
        notes.append("; ".join(n_parts))
    notes.append(
        "Percent is within-series (denominator = each series' own N), so a large-N "
        "series does not visually swamp a small-N one; raw trial counts are retained "
        "per bar."
    )

    return datums, notes
