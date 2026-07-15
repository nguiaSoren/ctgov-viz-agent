"""Dangling-reference detection → the ``kind:"clarification"`` outcome (E-13).

A well-formed request whose natural-language query names a dimension with a
DEMONSTRATIVE referent ("this drug", "that condition") but supplies NO antecedent
— neither a structured field nor an entity the planner could resolve — is
*incomplete intent*, not a malformed request. The honest answer is to ASK ("Which
drug are you referring to?"), not to guess a drug (dishonest) and not to 422 (the
HTTP request is valid). This is the input-handling hybrid: 422 for structured-field
errors, a first-class ``clarification`` for an unresolvable NL referent.

Deterministic by design (code, not prompt vigilance — the ROGUE principle): a
tight demonstrative regex (``this/that/these/those`` + a dimension noun), gated by
"is that dimension actually resolved?", so a query that DOES name the entity
("trials of that drug, pembrolizumab" → the planner resolved ``drug=pembrolizumab``)
does NOT trip. The clarifying ``question`` is a fixed code-owned string (never
LLM-authored, never carries a number — the §1 invariant holds for it too).
"""

from __future__ import annotations

import re

from app.plan.models import Plan

# Demonstrative referents (NOT the weak "the …", which is non-referential too often)
# to a dimension the system can scope a search on. Each maps to a resolvable
# dimension + a fixed clarifying question.
_REFERENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "drug": re.compile(
        r"\b(?:this|that|these|those)\s+"
        r"(?:drug|drugs|medication|medications|treatment|treatments|therapy|therapies|"
        r"intervention|interventions|compound|compounds|agent|agents)\b"
    ),
    "condition": re.compile(
        r"\b(?:this|that|these|those)\s+"
        r"(?:condition|conditions|disease|diseases|disorder|disorders|indication|"
        r"indications|illness|illnesses)\b"
    ),
    "sponsor": re.compile(
        r"\b(?:this|that|these|those)\s+"
        r"(?:sponsor|sponsors|company|companies|organization|organizations|"
        r"organisation|organisations|manufacturer|manufacturers)\b"
    ),
    # "trial" nouns only (NOT "study"/"studies" — those collide with "study type" /
    # "study population" and wrongly clarified a legit distribution query); all four
    # demonstratives, symmetric with the other dimensions.
    "trial": re.compile(r"\b(?:this|that|these|those)\s+(?:trial|trials)\b"),
}

# An NCT id already in the query resolves a "this trial" reference — don't ask for
# what the user already supplied.
_NCT_IN_QUERY = re.compile(r"NCT[0-9]{8}", re.IGNORECASE)

# Which request field / plan-entity dimension resolves each referent.
_DIMENSION_FIELD: dict[str, str] = {"drug": "drug_name", "condition": "condition", "sponsor": "sponsor"}

_QUESTIONS: dict[str, str] = {
    "drug": "Which drug do you mean? Please name the drug (e.g. in the drug_name field).",
    "condition": "Which condition do you mean? Please name the condition (e.g. in the condition field).",
    "sponsor": "Which sponsor do you mean? Please name the sponsor (e.g. in the sponsor field).",
    "trial": "Which trial do you mean? Please provide its NCT id (e.g. NCT01234567).",
}


def detect_dangling_reference(merged_inputs: dict | None, plan: Plan | None) -> str | None:
    """Return a fixed clarifying question if the query makes a demonstrative
    reference to a dimension it never resolved, else ``None``. TOTAL: never raises.

    Resolution check: a dimension counts as resolved if a structured field supplies
    it OR the planner placed an entity on it (so "trials for that drug, Keytruda" —
    where the planner resolved ``drug=Keytruda`` — is NOT dangling). A ``this/that
    trial`` reference asks UNLESS an ``NCT`` id is already in the query. **Caveat:**
    for drug/condition/sponsor, resolution rides on the planner's entity extraction,
    so an inline apposition ("this drug, pembrolizumab") is only treated as resolved
    when the planner actually populated ``entities`` — under a degraded/offline
    planner a named-but-unextracted entity can still be asked about. That is an
    accepted false-positive (asking is safe; guessing is not), not a silent wrong
    answer.
    """
    merged_inputs = merged_inputs or {}
    query = str(merged_inputs.get("query") or "")
    if not query.strip():
        return None
    low = query.lower()
    entities = (plan.entities if plan is not None else None) or {}

    for dimension, pattern in _REFERENT_PATTERNS.items():
        if not pattern.search(low):
            continue
        if dimension == "trial":
            if _NCT_IN_QUERY.search(query):
                continue  # the NCT id is supplied — the reference is resolved, don't ask
            return _QUESTIONS["trial"]
        field_key = _DIMENSION_FIELD[dimension]
        has_field = merged_inputs.get(field_key) not in (None, "")
        # An entity value that is ITSELF a demonstrative referent ("this drug") is NOT a
        # resolution. The REAL LLM planner can extract the demonstrative phrase AS the
        # entity (query.intr="this drug"), which the offline StubAdapter never did — so
        # the plain ``bool(entities.get(dimension))`` check passed live while the query
        # was still dangling (a wired-isn't-run gap: this detector had only ever been
        # exercised against a stub that left the entity empty). Re-apply the same
        # demonstrative pattern to the extracted value: if the "resolution" is itself a
        # referent, it resolves nothing, so we still ask.
        entity_val = str(entities.get(dimension) or "")
        has_entity = bool(entity_val) and not pattern.search(entity_val.lower())
        if not (has_field or has_entity):
            return _QUESTIONS[dimension]
    return None
