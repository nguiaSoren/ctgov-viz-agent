"""Time-series binning + finalization (ARCHITECTURE_SPEC §3.5 / CC-4, G-35, G-40).

Two pure pieces the ``timeseries`` tool composes over the aggregation core:

* :func:`year_key_fn` — a COMBINE-mode ``key_fn`` factory: one record → the single
  year bucket its date falls in (mixed precision ``"1998"`` / ``"2021-02"`` /
  ``"2019-04-12"`` all normalize to the year). It is **TOTAL**: ``parse_ct_date``
  raises ``ValueError`` on a bad/missing date, and this module CATCHES it → the
  explicit ``MISSING`` bucket, never propagating (one bad record must not sink the
  whole chart, LESSON K1).
* :func:`finalize_timeseries` — turns the core's raw year buckets into the chart's
  ordered datum rows: it separates the ``MISSING`` bucket (kept, ``period=None``,
  for reconciliation), gap-fills every empty year inside the observed range with a
  0-count/no-citation bucket (G-35 — a line must not connect across a silently
  missing year), and routes genuine future years to a flagged ``planned`` bucket
  rather than clamping them into the current year (G-40 — clamping inflates the
  present, dropping breaks ``countTotal`` reconciliation).

Reconciliation invariant: ``Σ count_trials`` over ALL emitted datums (real +
gap-filled + planned + MISSING) equals the distinct-trial total equals the API's
exact ``countTotal`` for a combine field — the ``MISSING`` datum is what keeps the
sum exact when some trials carry no date.
"""

from __future__ import annotations

from collections.abc import Callable

from app.ctgov.dates import is_future, parse_ct_date

# The single explicit "no date" bucket key (CC-5): a missing/unparseable date is
# an honest, separate bucket, never silently dropped.
_MISSING_KEY: tuple[str, str] = ("MISSING", "Missing (no start date)")
_MISSING_VALUE = "MISSING"


def _resolve(record: dict, field_path: str) -> object:
    """Walk a dotted ``a.b.c`` path, isinstance-guarding every descent.

    Returns ``None`` the moment a segment is absent or the current node is not a
    dict (a present-but-``None`` struct such as ``startDateStruct: null`` resolves
    to ``None``, not a crash). The date paths this module reads carry no ``[]``
    list segments, so a plain dict walk is sufficient and stays TOTAL.
    """
    current: object = record
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def year_key_fn(date_field_path: str) -> Callable[[dict], list[tuple[str, str]]]:
    """Build a COMBINE-mode ``key_fn`` that buckets a record by the year at ``date_field_path``.

    The returned function maps one record to exactly one ``(value, label)`` key:
    the string year of the date at ``date_field_path`` (``[("2019", "2019")]``), or
    :data:`_MISSING_KEY` when the date is absent, empty, non-string, or
    unparseable. Mixed wire precision (``"1998"`` / ``"2021-02"`` /
    ``"2019-04-12"``) all normalize to the year via :func:`parse_ct_date`.

    TOTAL by construction: ``parse_ct_date`` raises ``ValueError`` on bad input and
    a non-string value never reaches it (isinstance guard), so the key function
    never raises on a live/malformed record — the batch survives one bad row.
    """

    def _key(record: dict) -> list[tuple[str, str]]:
        raw = _resolve(record, date_field_path)
        if not isinstance(raw, str):
            return [_MISSING_KEY]
        try:
            year, _month = parse_ct_date(raw)
        except ValueError:
            return [_MISSING_KEY]
        return [(str(year), str(year))]

    return _key


def _year_of(value: object) -> int | None:
    """Parse a bucket value into an int year, or ``None`` if it is not a plain year."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _gap_bucket(year: int) -> dict:
    """A gap-filled 0-count year bucket (G-35): a legitimately empty year ships NO
    citations — there is no trial to cite, and inventing one would be a lie."""
    return {
        "value": str(year),
        "label": str(year),
        "period": str(year),
        "count_trials": 0,
        "count_mentions": 0,
        "source_ids": [],
        "citations": [],
        "contributing_count": 0,
    }


def finalize_timeseries(
    buckets: list[dict],
    *,
    current_year: int,
    grain: str = "year",
) -> tuple[list[dict], list[str], bool]:
    """Turn the core's raw year buckets into ordered, gap-filled chart datum rows.

    Parameters
    ----------
    buckets:
        The aggregation core's per-year bucket dicts (value/label/counts/citations/
        contributing_count), value being the string year or ``"MISSING"``. These
        dicts already carry their citations (built by the tools layer); this
        function never authors an excerpt — it only reorders, gap-fills, and flags.
    current_year:
        The current calendar year — the boundary that routes a bucket to
        ``planned`` (``year > current_year``, via :func:`app.ctgov.dates.is_future`).
    grain:
        Time granularity; only ``"year"`` is supported today (kept in the signature
        so the tools layer can pass it through without a later break).

    Returns
    -------
    (datum_dicts, notes, degrade)
        ``datum_dicts`` — real years (with ``period`` set) + gap-filled 0-count
        years + planned future years, sorted ascending by year, with the ``MISSING``
        datum (``period=None``) appended last. ``notes`` — the gap-fill, planned,
        and missing-bucket disclosures (each only when it applies). ``degrade`` —
        ``True`` iff at most ONE bucket with a plottable year has a non-zero count
        (the caller degrades the time series to a bar, since a single point is not
        a trend). "Plottable year" means the bucket value parsed as an integer year,
        which INCLUDES future/planned years and gap-filled ones; only the
        ``MISSING`` (period=None) bucket is excluded. So a series consisting of one
        real year plus one planned year does NOT degrade.

    Range caveat: the gap-fill spans ``min(non-future year) .. max(non-future
    year)`` with no width cap, and ``parse_ct_date`` applies no calendar fence, so a
    single record carrying an implausible early year (``"0001-05"`` parses to year
    1) would emit ~2000 zero-count buckets. Filter years are fenced to [1900, 2100]
    in ``app.ctgov.params``; nothing equivalent guards the PARSE side, and no
    row-count cap exists downstream. Not observed on live data — stated as a known
    edge, not a defended one.
    """
    # Partition input: numeric-year buckets vs everything without a plottable year
    # (the MISSING bucket, and any unexpected non-year value — both kept, period=None).
    year_buckets: dict[int, dict] = {}
    no_period: list[dict] = []
    has_missing = False
    for bucket in buckets:
        value = bucket.get("value")
        year = _year_of(value)
        if year is None:
            no_period.append(bucket)
            if value == _MISSING_VALUE:
                has_missing = True
            continue
        year_buckets[year] = bucket

    # Gap-fill inside the observed NON-FUTURE range only: a future/planned year sits
    # above the current year and must not drag a decade of empty gap buckets behind
    # it (golden: 2024 → 2030 planned, with 2025..2029 NOT filled).
    non_future = [y for y in year_buckets if not is_future(y, current_year)]
    gap_years: list[int] = []
    if non_future:
        lo, hi = min(non_future), max(non_future)
        for year in range(lo, hi + 1):
            if year not in year_buckets:
                year_buckets[year] = _gap_bucket(year)
                gap_years.append(year)

    # Emit year buckets ascending; flag future years planned (NOT clamped, NOT dropped)
    # and the CURRENT year partial (data only through the retrieval date — legitimately
    # short, so its dip is an artifact, not a real decline; BUILD_PLAN timeseries task).
    datums: list[dict] = []
    planned_years: list[int] = []
    current_partial = False
    for year in sorted(year_buckets):
        datum = dict(year_buckets[year])  # copy — never mutate the caller's bucket
        datum["period"] = str(year)
        if is_future(year, current_year):
            datum["planned"] = True
            datum["label"] = f"{year} (planned)"
            planned_years.append(year)
        elif year == current_year and datum.get("count_trials", 0) > 0:
            datum["partial_year"] = True
            datum["label"] = f"{year} (partial)"
            current_partial = True
        datums.append(datum)

    # The MISSING datum(s) last, excluded from the line but kept for reconciliation.
    for bucket in no_period:
        datum = dict(bucket)
        datum["period"] = None
        datums.append(datum)

    # Degrade when at most one YEAR-valued bucket is non-zero — one point is not a
    # trend. "Year-valued" spans every plottable year including planned/future ones;
    # only the period=None MISSING bucket is out of scope here.
    non_zero_real = sum(1 for year, b in year_buckets.items() if b.get("count_trials", 0) > 0)
    degrade = non_zero_real <= 1

    notes = _build_notes(planned_years, gap_years, has_missing, no_period, current_partial, current_year)
    return datums, notes, degrade


def _build_notes(
    planned_years: list[int],
    gap_years: list[int],
    has_missing: bool,
    no_period: list[dict],
    current_partial: bool = False,
    current_year: int | None = None,
) -> list[str]:
    """Assemble the interpretation notes (each only when it actually applies)."""
    notes: list[str] = []
    if current_partial and current_year is not None:
        notes.append(
            f"The current year {current_year} is a PARTIAL year (trials only through the "
            f"retrieval date), so its count is legitimately short — read its dip as incomplete "
            f"data, not a real decline."
        )
    if planned_years:
        years = ", ".join(str(y) for y in planned_years)
        notes.append(
            f"Bucket(s) {years} hold genuine future/estimated dates and are flagged "
            f"'planned' — kept in the series, not clamped into the current year "
            f"(which would inflate it) and not dropped (which would break "
            f"reconciliation) (G-40)."
        )
    if gap_years:
        years = ", ".join(str(y) for y in gap_years)
        notes.append(
            f"Gap year(s) {years} filled with a 0-count bucket so the line does not "
            f"connect across a silently-missing year; a legitimately-empty bucket "
            f"ships no citations (G-35)."
        )
    if has_missing:
        missing_count = sum(int(b.get("count_trials", 0) or 0) for b in no_period)
        notes.append(
            f"{missing_count} trial(s) have no parseable date and are kept in a "
            f"separate 'Missing' bucket (period=None) for reconciliation, excluded "
            f"from the line."
        )
    return notes
