"""Pure date-handling primitives (SPEC_INTERROGATION §C / CC-4, G-40).

ClinicalTrials.gov date strings carry mixed precision (``"2015"`` /
``"2015-05"`` / ``"2015-05-10"``) and ``startDate`` legitimately includes
future/estimated years. These are the small, real, pure functions the
timeseries tool (Phase 1) will build binning and the "planned" bucket on top
of. **Do not clamp a future date into the current year** — the caller routes
it to a flagged "planned" bucket instead (G-40): clamping would inflate the
current year, and silently dropping it would break countTotal reconciliation.
"""

from __future__ import annotations


def parse_ct_date(s: str | None) -> tuple[int, int | None]:
    """Parse a ClinicalTrials.gov date string into ``(year, month | None)``.

    Handles the three precisions seen on the wire: ``"2015"`` -> ``(2015,
    None)``, ``"2015-05"`` -> ``(2015, 5)``, ``"2015-05-10"`` -> ``(2015, 5)``
    (day is not returned — nothing downstream bins finer than month).

    Raises ``ValueError`` — clearly and intentionally, never a bare
    ``AttributeError``/unhandled crash — on ``None``, an empty/whitespace
    string, a non-numeric component, or a month outside ``1..12``. A bad date
    on the wire must fail loudly here rather than silently mis-bin downstream.

    The YEAR is not range-checked: ``"0001-05"`` parses to ``(1, 5)`` and a far-future
    year parses too. Only the month is fenced. That is deliberate — future start
    dates are legitimate registry data (G-40) — but it means an implausible early
    year propagates as a real bucket. See ``timeseries.finalize_timeseries`` for the
    consequence (an uncapped gap-fill range). Filter years supplied by the plan ARE
    fenced to [1900, 2100], but in ``app.ctgov.params``, not here.
    """
    if s is None or not s.strip():
        raise ValueError(f"parse_ct_date: expected a non-empty date string, got {s!r}")

    parts = s.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else None
    except ValueError as exc:
        raise ValueError(f"parse_ct_date: malformed date string {s!r}") from exc

    if month is not None and not (1 <= month <= 12):
        raise ValueError(f"parse_ct_date: month out of range 1..12 in {s!r}")

    return (year, month)


def is_future(year: int, current_year: int) -> bool:
    """Is ``year`` strictly after ``current_year``? The genuine-future-date test
    that routes a bucket to "planned" (G-40) instead of clamping or dropping it."""
    return year > current_year


def precision_of(s: str | None) -> str:
    """Return the precision of a ClinicalTrials.gov date string: ``"year"``,
    ``"month"``, or ``"day"`` — driven purely by how many ``-``-separated
    components are present.

    Raises ``ValueError`` on ``None``/empty/whitespace input rather than
    silently returning ``"year"`` for garbage — consistent with
    :func:`parse_ct_date`, which rejects the same inputs.
    """
    if s is None or not s.strip():
        raise ValueError(f"precision_of: expected a non-empty date string, got {s!r}")

    parts = s.split("-")
    if len(parts) == 1:
        return "year"
    if len(parts) == 2:
        return "month"
    return "day"
