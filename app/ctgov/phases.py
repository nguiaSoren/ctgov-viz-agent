"""Human trial-phase string → wire phase tokens (E-16 / CC-1).

``trial_phase`` is a STRUCTURED request field with a CLOSED vocabulary (the 6
real ``PHASE_TOKENS``), unlike a free-text entity name. So a value that cannot be
normalized to a real token is a MALFORMED structured input → the request layer
rejects it with 422 + the valid list (P5-INPUT/E-16), NOT a 200-empty (that is
reserved for a mistyped free-text ENTITY, E-18/ENG-22). This module is the single
normalizer both the request validator (to reject) and the planner (to apply the
field authoritatively, CC-1) call, so "what counts as a valid phase" lives in one
place.

Accepted forms (case-insensitive): the wire tokens themselves (``PHASE1``,
``EARLY_PHASE1``, ``NA`` …); ``"Phase 1"`` / ``"Phase-1"`` / ``"phase I"`` /
``"1"``; roman numerals ``I..IV``; ``"Early Phase 1"`` / ``"Phase 0"`` →
``EARLY_PHASE1``; ``"N/A"`` / ``"not applicable"`` → ``NA``; and combinations
``"1/2"`` / ``"Phase 2/3"`` / ``"1-2"`` / ``"1 and 2"`` → a token list (CC-15's
composite phase).

Intentionally LENIENT on glued/spelled-out multi-phase input (``"phase1phase2"``
→ ``["PHASE1","PHASE2"]``): every unit still maps to a REAL wire token (no
invalid token can be produced — that is the anti-hallucination guarantee), and
the Plan Checker is the final gate, so accepting an ugly-but-unambiguous spelling
is safer than a brittle 422. What is rejected is a unit that names NO real phase.
"""

from __future__ import annotations

import re

from app.ctgov.enums import PHASE_TOKENS

# A single phase "unit" (after stripping the words "phase"/"early") → its token.
_UNIT_TO_TOKEN: dict[str, str] = {
    "0": "EARLY_PHASE1",  # "Phase 0" is the early-phase designation
    "1": "PHASE1",
    "2": "PHASE2",
    "3": "PHASE3",
    "4": "PHASE4",
}
_ROMAN_TO_DIGIT: dict[str, str] = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}

# Human display list for the 422 message (stable order).
_VALID_DISPLAY = "Early Phase 1, Phase 1, Phase 2, Phase 3, Phase 4, NA (combinations like '1/2' allowed)"


class InvalidTrialPhase(ValueError):
    """A ``trial_phase`` string that does not normalize to any real phase token."""

    def __init__(self, value: object) -> None:
        super().__init__(
            f"unrecognized trial_phase {value!r}; valid phases are: {_VALID_DISPLAY}"
        )


def normalize_trial_phase(text: object) -> list[str]:
    """Normalize a human ``trial_phase`` string to ≥1 real wire ``PHASE_TOKENS``.

    Returns a de-duplicated, order-preserving list (a combined phase like ``"1/2"``
    → ``["PHASE1", "PHASE2"]``, CC-15). Raises :class:`InvalidTrialPhase` (a
    ``ValueError`` subclass → 422 at the request layer) on anything that names no
    real phase. TOTAL: a non-str / empty input raises cleanly, never anything else.
    """
    raw = str(text).strip()
    if not raw:
        raise InvalidTrialPhase(text)

    # (a) already a wire token (case-insensitive, spaces/hyphens normalized to _)
    as_token = re.sub(r"[\s-]+", "_", raw).upper()
    if as_token in PHASE_TOKENS:
        return [as_token]

    low = raw.lower()

    # (b) NA family
    if re.sub(r"[.\s/]+", "", low) in ("na", "notapplicable"):
        return ["NA"]

    early = "early" in low
    body = low.replace("early", " ").replace("phase", " ")
    # split on whitespace / , / & / - / and the word "and" — the hyphen makes the
    # common human form "Phase-1" (and the range "1-2") normalize instead of 422-ing.
    parts = [p for p in re.split(r"[\s,/&-]+|\band\b", body) if p]

    tokens: list[str] = []
    for part in parts:
        unit = part.strip(".")
        unit = _ROMAN_TO_DIGIT.get(unit, unit)  # roman numeral → digit
        if unit not in _UNIT_TO_TOKEN:
            raise InvalidTrialPhase(text)
        token = "EARLY_PHASE1" if (early and unit in ("0", "1")) else _UNIT_TO_TOKEN[unit]
        if token not in tokens:
            tokens.append(token)

    if not tokens:
        raise InvalidTrialPhase(text)
    return tokens
