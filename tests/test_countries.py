"""Tests for the disclosed country-alias normalization (E-20).

Verifies :mod:`app.ctgov.countries` folds the common English abbreviations onto
one canonical display form, returns ``was_aliased`` correctly, passes genuine
free text through unchanged, and is TOTAL (never raises) for edge inputs.
"""

from __future__ import annotations

import pytest

from app.ctgov.countries import COUNTRY_ALIASES, canonical_country


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("USA", "United States"),
        ("US", "United States"),
        ("U.S.", "United States"),
        ("U.S.A.", "United States"),
        ("United States of America", "United States"),
        ("us", "United States"),
        ("UK", "United Kingdom"),
        ("U.K.", "United Kingdom"),
        ("Britain", "United Kingdom"),
        ("Great Britain", "United Kingdom"),
        ("South Korea", "South Korea"),
        ("Republic of Korea", "South Korea"),
        ("Korea, Republic of", "South Korea"),
        ("S. Korea", "South Korea"),
        ("Russia", "Russia"),
        ("Russian Federation", "Russia"),
        ("The Netherlands", "Netherlands"),
        ("Holland", "Netherlands"),
        ("Czechia", "Czechia"),
        ("Czech Republic", "Czechia"),
    ],
)
def test_known_aliases_canonicalize_and_flag(raw: str, expected: str) -> None:
    assert canonical_country(raw) == (expected, True)


def test_spec_headline_cases() -> None:
    assert canonical_country("USA") == ("United States", True)
    assert canonical_country("U.S.") == ("United States", True)
    assert canonical_country("us") == ("United States", True)
    assert canonical_country("UK") == ("United Kingdom", True)
    assert canonical_country("South Korea") == ("South Korea", True)
    assert canonical_country("Korea, Republic of") == ("South Korea", True)


def test_free_text_passes_through_unchanged() -> None:
    assert canonical_country("Freedonia") == ("Freedonia", False)
    # A real CT.gov country with no alias must NOT be rewritten.
    assert canonical_country("Botswana") == ("Botswana", False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  usa  ", "United States"),  # surrounding whitespace
        ("u.s.a.", "United States"),  # already-lower with periods
        ("great   britain", "United Kingdom"),  # collapsed internal whitespace
        ("SOUTH KOREA", "South Korea"),  # uppercase
        ("czech republic", "Czechia"),  # lowercase variant
    ],
)
def test_normalization_variants(raw: str, expected: str) -> None:
    assert canonical_country(raw) == (expected, True)


def test_free_text_is_stripped_but_not_aliased() -> None:
    canonical, aliased = canonical_country("  Freedonia  ")
    assert canonical == "Freedonia"
    assert aliased is False


def test_none_and_empty_are_safe() -> None:
    assert canonical_country(None) == ("", False)  # type: ignore[arg-type]
    assert canonical_country("") == ("", False)
    assert canonical_country("   ") == ("", False)


def test_non_str_is_safe() -> None:
    # TOTAL: exotic inputs return a clean tuple, never raise.
    assert canonical_country(123) == ("", False)  # type: ignore[arg-type]
    assert canonical_country(["USA"]) == ("", False)  # type: ignore[arg-type]


def test_alias_map_is_small_and_disclosed() -> None:
    # Guards the "narrow, disclosed scope" invariant (E-20): keep it a small
    # curated map, not a creeping geopolitical resolver.
    assert len(COUNTRY_ALIASES) <= 20
    # Keys must already be in normalized (lowercased, no trailing period) form.
    for key in COUNTRY_ALIASES:
        assert key == key.casefold()
        assert not key.endswith(".")
