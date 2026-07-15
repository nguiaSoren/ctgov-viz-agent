"""Tests for the NCT-id path-injection guard (G-8 / R-20 / SEC-25 / C-45).

Verifies :mod:`app.ctgov.identifiers` is TOTAL (never raises on any input),
accepts only the exact ``NCT`` + 8-digit format, and — critically for the
security framing — never echoes a rejected/injection payload verbatim into its
error message.
"""

from __future__ import annotations

import pytest

from app.ctgov.identifiers import NCT_ID_RE, is_valid_nct_id, validate_nct_id

VALID = "NCT01234567"

# Every shape the guard must reject, and WHY it is malformed.
INVALID: list[object] = [
    "NCT1234567",  # 7 digits (too short)
    "NCT123456789",  # 9 digits (too long)
    "nct01234567",  # lowercase prefix
    "NCT0123456A",  # non-digit tail character
    "  NCT01234567  ",  # surrounding whitespace
    "NCT01234567\n",  # trailing newline ($-anchor gotcha)
    "NCT 1234567",  # embedded space
    None,  # None
    123,  # non-str (int)
    12345678,  # non-str numeric that "looks" id-shaped
    "",  # empty string
    "NCT01234567; DROP",  # extra chars
]


def test_regex_is_compiled_and_anchored() -> None:
    assert NCT_ID_RE.pattern == r"^NCT[0-9]{8}$"  # ASCII digits only (not \d — Unicode gap)
    assert NCT_ID_RE.fullmatch(VALID) is not None


def test_valid_id_is_accepted() -> None:
    assert is_valid_nct_id(VALID) is True
    assert validate_nct_id(VALID) == VALID


@pytest.mark.parametrize("bad", INVALID)
def test_invalid_ids_rejected_by_predicate(bad: object) -> None:
    assert is_valid_nct_id(bad) is False


@pytest.mark.parametrize("bad", INVALID)
def test_invalid_ids_raise_from_validator(bad: object) -> None:
    with pytest.raises(ValueError):
        validate_nct_id(bad)


def test_predicate_is_total_and_returns_bool() -> None:
    # Exotic inputs must return a clean bool, never raise.
    for weird in (object(), [], {}, 3.14, b"NCT01234567", ("NCT01234567",)):
        assert is_valid_nct_id(weird) is False


def test_validator_error_does_not_echo_injection_payload() -> None:
    payload = "NCT" + "../../../../etc/passwd" + ("A" * 5000)
    with pytest.raises(ValueError) as exc:
        validate_nct_id(payload)
    message = str(exc.value)
    # The huge/traversal payload must NOT ride into the error message: the full
    # payload is absent, the 5000-char bulk never appears, the message is
    # bounded, and it carries the truncation marker instead.
    assert payload not in message
    assert ("A" * 100) not in message  # the 5000-char bulk is not echoed
    assert "<truncated>" in message
    assert len(message) < 200
    # It should still name the expected format so the failure is actionable.
    assert "NCT" in message


def test_validator_message_is_bounded_for_long_str() -> None:
    long_bad = "X" * 10000
    with pytest.raises(ValueError) as exc:
        validate_nct_id(long_bad)
    assert long_bad not in str(exc.value)
    assert len(str(exc.value)) < 200


def test_rejects_unicode_digits_ascii_only():
    """Adversarial finding (Phase-5 review): ``\\d`` matched Unicode digits, so a
    non-ASCII digit could satisfy the pattern and NFKC-fold to a different real id.
    ``[0-9]`` closes it — only ASCII digits pass."""
    from app.ctgov.identifiers import is_valid_nct_id, validate_nct_id

    assert is_valid_nct_id("NCT0123456٩") is False  # Arabic-Indic 9
    assert is_valid_nct_id("NCT1234567８") is False  # fullwidth 8
    for bad in ("NCT0123456٩", "NCT1234567８"):
        try:
            validate_nct_id(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert is_valid_nct_id("NCT01234567") is True  # plain ASCII still passes
