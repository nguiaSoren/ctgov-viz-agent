"""Small, disclosed country-alias normalization (E-20).

ClinicalTrials.gov's ``country`` is **free text with no ISO codes** — a v1
documented limitation. The same place appears as ``"USA"``, ``"U.S."`` and
``"United States of America"`` across records, which fragments a country
distribution into near-duplicate buckets.

This module is a deliberately NARROW fix for that fragmentation, NOT a
geopolitical or ISO-3166 resolver: it folds only a handful of common English
abbreviations/variants onto one canonical *display* form. It does not do ISO
codes, multilingual names, historical/disputed territories, or subdivisions —
those are explicitly out of scope. The map is small (≤ ~20 entries) and every
fold is auditable in :data:`COUNTRY_ALIASES` below. A *listed* variant IS
rewritten to its canonical spelling (that is the normalization — ``USA`` →
``United States``); a name NOT listed passes through unchanged. The rewriting is
never silent: the country facet's ``meta.notes`` (built in
``app.ctgov.tools.aggregate_by``) states that spellings are normalized (E-20), so
a reader knows a canonical bucket may aggregate variant spellings.

Everything here is pure and total: :func:`canonical_country` never raises, and
is safe for ``None`` / empty / non-``str`` input.
"""

from __future__ import annotations

# Keys are the NORMALIZED lookup form produced by ``_normalize_key`` (stripped,
# internal whitespace collapsed, casefolded, trailing period(s) removed). Values
# are the canonical DISPLAY form returned to the caller. Scope is intentionally
# narrow — see the module docstring; this is not an ISO/geopolitical resolver.
COUNTRY_ALIASES: dict[str, str] = {
    # United States
    "usa": "United States",
    "us": "United States",
    "u.s": "United States",  # "U.S." -> trailing period stripped
    "u.s.a": "United States",  # "U.S.A." -> trailing period stripped
    "united states of america": "United States",
    # United Kingdom
    "uk": "United Kingdom",
    "u.k": "United Kingdom",  # "U.K." -> trailing period stripped
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    # South Korea
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "s. korea": "South Korea",
    "south korea": "South Korea",
    # Russia
    "russia": "Russia",
    "russian federation": "Russia",
    # Netherlands
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    # Czechia
    "czechia": "Czechia",
    "czech republic": "Czechia",
}


def _normalize_key(name: str) -> str:
    """Normalize a country string to its :data:`COUNTRY_ALIASES` lookup key.

    Strips, collapses internal whitespace to single spaces, casefolds, and
    removes trailing period(s) — so ``"  U.S.  "`` and ``"u.s."`` both map to
    the key ``"u.s"``. Internal periods (``"s. korea"``) are preserved.
    """
    collapsed = " ".join(name.split())  # strip + collapse internal whitespace
    return collapsed.casefold().rstrip(".")


def canonical_country(name: str) -> tuple[str, bool]:
    """Return ``(canonical_display_name, was_aliased)`` for a country string.

    TOTAL and non-raising. The input is normalized for *lookup* only (see
    :func:`_normalize_key`); on a hit the canonical DISPLAY value is returned
    with ``was_aliased=True``. On a miss the input passes through as
    ``(name.strip(), False)`` — free text is never rewritten. ``None`` / empty /
    non-``str`` input yields ``("", False)``.
    """
    if not isinstance(name, str):
        return ("", False)
    stripped = name.strip()
    if not stripped:
        return ("", False)
    canonical = COUNTRY_ALIASES.get(_normalize_key(stripped))
    if canonical is not None:
        return (canonical, True)
    return (stripped, False)
