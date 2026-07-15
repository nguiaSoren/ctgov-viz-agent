"""Study-duration histogram binning (ARCHITECTURE_SPEC §3.5 / R-16, G-35).

One pure function, :func:`bin_durations`, that bins trials by their start→completion
duration in months. It is deliberately built on the two dated fields
(``startDateStruct`` / ``completionDateStruct``), NOT on the unverified enrollment
field (R-16): duration is a real, checkable magnitude; enrollment is not.

Duration is measured at month precision (``year*12 + month``, day ignored; a
year-only date such as ``"2020"`` is treated as January). A trial missing either
date, or one whose completion precedes its start (a negative, impossible
duration), is routed to an explicit ``Unknown (undated)`` bucket — kept for
reconciliation and honesty, never silently dropped. The function is **TOTAL**: a
malformed record never raises; its date simply fails to resolve and it lands in
the undated bucket (or, id-less, is not counted at all — a record with no nctId
has no identity to reconcile against, mirroring the aggregation core).
"""

from __future__ import annotations

from app.ctgov.citations import build_bucket_citations
from app.ctgov.dates import parse_ct_date

_NCT_PATH = ("protocolSection", "identificationModule", "nctId")

_DEFAULT_START_PATH = "protocolSection.statusModule.startDateStruct.date"
_DEFAULT_END_PATH = "protocolSection.statusModule.completionDateStruct.date"

# Default duration bins in MONTHS, half-open ``[lo, hi)`` (a 6-month duration falls
# in "6–12 mo", not "0–6 mo"); the final bin's ``None`` upper edge is open (10+ yr).
_DEFAULT_BINS: list[tuple[int, int | None]] = [
    (0, 6),
    (6, 12),
    (12, 24),
    (24, 48),
    (48, 120),
    (120, None),
]
# En-dash label constants (module-level literals — no escape sequences in f-strings).
_DEFAULT_LABELS: list[str] = ["0–6 mo", "6–12 mo", "1–2 yr", "2–4 yr", "4–10 yr", "10+ yr"]

# The explicit sentinel bucket for undatable / negative-duration trials.
_UNDATED_VALUE = "UNDATED"
_UNDATED_LABEL = "Unknown (undated)"


def _resolve(record: dict, field_path: str) -> object:
    """Walk a dotted ``a.b.c`` path, isinstance-guarding every descent (TOTAL)."""
    current: object = record
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _nct_id(record: dict) -> str | None:
    """Read the nctId, or ``None`` if the path is absent/malformed."""
    current: object = record
    for part in _NCT_PATH:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, str) else None


def _months(record: dict, field_path: str) -> int | None:
    """Resolve the date at ``field_path`` to a month index (``year*12 + month``).

    A year-only date has ``month=None`` and is treated as month 1 (January). Returns
    ``None`` when the value is absent, non-string, or unparseable — never raises.
    """
    raw = _resolve(record, field_path)
    if not isinstance(raw, str):
        return None
    try:
        year, month = parse_ct_date(raw)
    except ValueError:
        return None
    return year * 12 + (month if month is not None else 1)


def _duration_months(record: dict, start_path: str, end_path: str) -> int | None:
    """Months from start to completion, or ``None`` when undatable or negative."""
    start = _months(record, start_path)
    end = _months(record, end_path)
    if start is None or end is None:
        return None
    duration = end - start
    if duration < 0:
        return None
    return duration


def _bin_index(duration: int, edges: list[tuple[int, int | None]]) -> int | None:
    """Index of the half-open ``[lo, hi)`` bin containing ``duration``, else ``None``."""
    for index, (lo, hi) in enumerate(edges):
        low = lo if lo is not None else 0
        if duration >= low and (hi is None or duration < hi):
            return index
    return None


def bin_durations(
    records: list[dict],
    *,
    start_path: str = _DEFAULT_START_PATH,
    end_path: str = _DEFAULT_END_PATH,
    bins: list[tuple[int, int | None]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Bin ``records`` by start→completion duration into a histogram (COMBINE mode).

    Each duration bin datum carries ``bin_start``/``bin_end`` (months), the dual
    counts (equal in combine), and per-bucket citations at the START date field
    path (the field that anchors the bin membership). Empty bins are kept (0 count,
    no citations) so the histogram shows the full range honestly. A trial missing
    either date or with a negative duration goes to the ``Unknown (undated)``
    bucket, appended last only when non-empty.

    ``count_trials`` is a distinct-nctId count (a duplicate page row is deduped per
    bucket, K3); an id-less record is not counted (no identity to reconcile).

    Returns ``(datum_dicts, notes)``.
    """
    edges = bins if bins is not None else _DEFAULT_BINS
    labels = _DEFAULT_LABELS if bins is None else [_range_label(lo, hi) for lo, hi in edges]

    # Per-bin and undated accumulators: nctId -> record (first-seen dedup, K3).
    bin_records: list[dict[str, dict]] = [{} for _ in edges]
    undated_records: dict[str, dict] = {}

    for record in records:
        nct = _nct_id(record)
        if nct is None:
            continue  # no identity to reconcile against — dropped, like the core
        duration = _duration_months(record, start_path, end_path)
        if duration is None:
            undated_records.setdefault(nct, record)
            continue
        index = _bin_index(duration, edges)
        if index is None:
            # Duration is valid but outside every provided bin — keep it honest.
            undated_records.setdefault(nct, record)
            continue
        bin_records[index].setdefault(nct, record)

    datums: list[dict] = []
    for index, (lo, hi) in enumerate(edges):
        contributing = list(bin_records[index].values())
        citations, contributing_count, truncated = build_bucket_citations(
            contributing, start_path, k=20
        )
        datums.append(
            {
                "value": labels[index],
                "label": labels[index],
                "bin_start": float(lo) if lo is not None else None,
                "bin_end": float(hi) if hi is not None else None,
                "count_trials": len(contributing),
                "count_mentions": len(contributing),
                "source_ids": [citation.nct_id for citation in citations],
                "citations": citations,
                "citations_truncated": truncated,
                "contributing_count": contributing_count,
            }
        )

    notes = [
        "Duration measured start→completion at month precision (day ignored; a "
        "year-only date is treated as January). Derived from the two dated status "
        "fields, not the unverified enrollment field (R-16)."
    ]
    if undated_records:
        undated = list(undated_records.values())
        citations, contributing_count, truncated = build_bucket_citations(
            undated, start_path, k=20
        )
        datums.append(
            {
                "value": _UNDATED_VALUE,
                "label": _UNDATED_LABEL,
                "bin_start": None,
                "bin_end": None,
                "count_trials": len(undated),
                "count_mentions": len(undated),
                "source_ids": [citation.nct_id for citation in citations],
                "citations": citations,
                "citations_truncated": truncated,
                "contributing_count": contributing_count,
            }
        )
        notes.append(
            f"{len(undated)} trial(s) are undated or have an implausible negative "
            f"duration (completion before start) and are grouped in an "
            f"'{_UNDATED_LABEL}' bucket, kept for reconciliation and excluded from "
            f"the duration bins."
        )

    return datums, notes


def _range_label(lo: int | None, hi: int | None) -> str:
    """A generic label for a caller-supplied bin range (months)."""
    low = lo if lo is not None else 0
    if hi is None:
        return f"{low}+ mo"
    return f"{low}–{hi} mo"
