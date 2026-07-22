"""SSRF/Essie-safe search-param builder (ARCHITECTURE_SPEC §A(d)) — Wave 1A.

The single boundary where a validated plan becomes ClinicalTrials.gov *search-
selecting* wire params (``query.*`` + ``filter.*``). The transport params
(``pageSize``/``countTotal``/``fields``/``pageToken``) are never added here; they
are attached downstream — normally by ``app.ctgov.client.CTGovClient``
(``count``/``iter_studies``), and in one place by the caller itself:
``app.ctgov.tools.aggregate_by_counts`` builds its own
``countTotal``/``pageSize``/``fields`` and calls ``client.get`` directly, because
it needs a count AND a citation sample from the SAME request. The invariant that
matters is unchanged either way: whatever adds them, no user text reaches a
transport param — they are all code-generated constants or config ints.

SSRF/Essie invariant (the reason this module exists):
    User free-text may appear ONLY as a ``query.<area>`` VALUE. Every param
    NAME, every area code, every status token, every year integer, and the
    whole ``filter.advanced`` Essie expression is CODE-GENERATED here from
    tokens that are validated against ``app.ctgov.enums``. User text can never
    become a param name, a filter expression, or an area code.

    It CAN, however, reach a ``query.<area>`` VALUE — and CT.gov FULLY PARSES the
    Essie query language on those values after url-decode, so an un-neutralized
    value can smuggle an operator (``cancer OR HIV`` runs a union, not a phrase
    search) or even a cross-field selector (``AREA[Phase]PHASE1`` selects a whole
    different population). ``neutralize_query_value`` closes that hole (SEC-24 /
    G-31, live-verified 2026-07-16): a CLEAN value (no Essie metacharacter, no
    standalone UPPERCASE operator keyword) passes through UNCHANGED for full
    recall; a DIRTY value is wrapped as an inert Essie StringLiteral (leading +
    trailing ``"``, internal ``\\`` / ``"`` escaped) so the parser reads it as a
    plain phrase, never as syntax. Values still pass through UN-encoded at the
    HTTP layer: the httpx client URL-encodes params, and pre-encoding here would
    double-encode.
"""

from __future__ import annotations

import re

from app.ctgov.enums import (
    INTERVENTION_TYPE_TOKENS,
    OVERALL_STATUS_TOKENS,
    PHASE_TOKENS,
    QUERY_AREAS,
    SPONSOR_CLASS_TOKENS,
    STUDY_TYPE_TOKENS,
)

# Sane calendar fence for a code-generated year integer (never user text).
_MIN_YEAR = 1900
_MAX_YEAR = 2100


def _validate_year(value: object, *, label: str) -> int:
    """Return ``value`` iff it is a real int year in ``[1900, 2100]``.

    Rejects ``bool`` (a ``bool`` is an ``int`` subclass in Python — ``True``
    would otherwise pass as year 1) and anything non-int, so only a genuine
    code-supplied integer can reach the Essie ``RANGE`` clause.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int year, got {type(value).__name__}")
    if not (_MIN_YEAR <= value <= _MAX_YEAR):
        raise ValueError(f"{label} out of range [{_MIN_YEAR}, {_MAX_YEAR}]: {value}")
    return value


# --- Essie neutralization (SEC-24 / G-31) ----------------------------------
# The Essie metacharacters that let a ``query.<area>`` value break out of a plain
# phrase into grouping / a cross-field ``AREA[...]`` selector / a ``RANGE`` / a
# quoted sub-literal. Any of these means the value MUST be quoted as a StringLiteral.
#
# Known gap: backslash is NOT in this set, so a clean value containing one passes
# through unescaped (``neutralize_query_value("a\\b")`` is unchanged). That is only
# safe if a bare backslash carries no meaning to Essie OUTSIDE a string literal --
# which we never verified live. Inside the quoting branch backslash IS escaped, so
# the gap is limited to otherwise-clean values. Adding "\\" here would be the
# conservative fix; it was left out to avoid quoting values on a character that has
# no demonstrated effect.
_ESSIE_METACHARACTERS = frozenset('[]()"')

# Essie operator / function KEYWORDS. Essie is CASE-SENSITIVE: only the UPPERCASE
# token is an operator (``OR`` unions, ``or``/``Or`` is a literal term), so we match
# them at WORD BOUNDARIES and case-sensitively — ``ANDROGEN`` / ``ORganon`` (a
# substring, not a standalone token) must NOT trigger, only a standalone ``AND``/``OR``.
_ESSIE_OPERATOR_KEYWORDS = (
    "AND", "OR", "NOT", "SEARCH", "AREA", "RANGE", "DISTANCE", "COVERAGE",
    "EXPANSION", "TILT", "ALL", "MISSING", "MIN", "MAX",
)
_ESSIE_OPERATOR_RE = re.compile(r"\b(?:" + "|".join(_ESSIE_OPERATOR_KEYWORDS) + r")\b")


def neutralize_query_value(value: str) -> str:
    """Make a user ``query.<area>`` value inert against the Essie query language.

    CT.gov fully parses Essie on ``query.*`` values after url-decode, so a raw
    user value can inject operators (``cancer OR HIV`` → a union) or a cross-field
    selector (``AREA[Phase]PHASE1`` → a different population). This is the single
    place that neutralizes that (SEC-24 / G-31, live-verified 2026-07-16).

    * A **clean** value — no Essie metacharacter (``[ ] ( ) "``) and no standalone
      UPPERCASE operator keyword — is returned UNCHANGED (full recall; the 99%
      case: ``pancreatic cancer``, ``pembrolizumab``, ``Crohn's disease``,
      lowercase ``or``, brand names, ``ANDROGEN``/``Organon`` substrings).
    * A **dirty** value is wrapped as an Essie ``StringLiteral``: a leading and
      trailing ``"`` with internal ``\\`` escaped to ``\\\\`` and internal ``"``
      escaped to ``\\"`` (the ``\\"`` escape form is API-accepted, live-verified —
      no need to strip the quote). The parser then reads it as a plain phrase, so
      ``"AREA[Phase]PHASE1"`` matches literally (→ 0 trials) instead of selecting.

    TOTAL by contract: a non-``str`` is coerced to ``str`` (``None`` → ``""``) and
    an empty value returns ``""`` — this never raises.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if not value:
        return ""
    is_dirty = any(ch in _ESSIE_METACHARACTERS for ch in value) or bool(
        _ESSIE_OPERATOR_RE.search(value)
    )
    if not is_dirty:
        return value  # clean → full recall, unchanged
    # Escape backslash FIRST (so we don't double-escape the ones we add for quotes),
    # then the quote, then wrap as an Essie StringLiteral.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_search_params(query: dict[str, str], filters: dict) -> dict[str, str]:
    """Build the SSRF-safe search-selecting wire params from a validated plan.

    Parameters
    ----------
    query:
        ``{area_code: value}`` where ``area_code`` must be one of
        ``enums.QUERY_AREAS`` (``term``/``cond``/``intr``/``spons``/``locn``).
        The value is the ONLY place user free-text is allowed.
    filters:
        Validated filter dict. This builder consumes exactly these keys:
        ``overallStatus``/``status`` (enum token → ``filter.overallStatus``),
        ``interventional_only`` (bool), ``studyType`` (enum token),
        ``start_year``/``end_year`` (int), ``phase`` (a token or list of tokens →
        an OR group), ``sponsorClass`` (enum token), and ``interventionType``
        (enum token). ``overallStatus``/``status`` becomes its own
        ``filter.overallStatus`` param; every OTHER consumed key is AND-composed
        into the single ``filter.advanced`` Essie expression.
        Every token is re-validated here (the SSRF/Essie boundary), so
        an unvalidated token can never reach the wire. Count and paging both route
        through this same builder, so a filter is applied to BOTH populations or
        neither — the two can never desync (the reconciliation invariant depends
        on this).

        Two asymmetries worth stating plainly, because they are easy to misread:

        * **Unknown KEYS are silently ignored; unknown VALUES raise.**
          ``build_search_params({"cond": "x"}, {"country": "France", "bogus": 1})``
          returns just ``{"query.cond": "x"}``. This is fail-closed (an unrecognized
          filter can never widen or narrow the wire query) but it is silent — a
          key the planner emits and this builder does not know about simply has no
          effect. ``filters["country"]`` is exactly that case today: the planner can
          emit it and the Plan Checker whitelists it, but nothing consumes it here;
          a country restriction reaches the wire as an ENTITY mapped to
          ``query.locn`` by the graph layer, not as a filter.
        * **``status`` is a vestigial alias.** No producer emits it (``PlannerFilters``
          and ``VisualizeRequest`` both lack the field), and it must already be a
          real ``OVERALL_STATUS_TOKENS`` token: a human hint like ``"recruiting"``
          passes the Plan Checker and then raises ``ValueError`` here, which the
          execute node converts into a redacted ``upstream_error`` rather than a
          clean re-plan. Kept only as a defensive alias for ``overallStatus``.

    Returns
    -------
    ``{"query.<area>": value, "filter.overallStatus": TOKEN,
       "filter.advanced": "<AND-joined Essie expr>"}`` — only the keys that apply.

    Raises
    ------
    ValueError
        On an unknown query area, an unknown status/studyType token, or an
        out-of-range / non-int year. Failing loud is the point: an unvalidated
        token must never reach the wire.
    """
    params: dict[str, str] = {}

    # --- query.<area> = value : the ONLY slot user free-text may occupy -----
    # Neutralize each value against the Essie query language (SEC-24 / G-31):
    # clean values pass through for full recall, dirty ones are quoted as inert
    # StringLiterals so a user can't smuggle an operator or a cross-field selector.
    for area, value in query.items():
        if area not in QUERY_AREAS:
            raise ValueError(f"unknown query area {area!r}; allowed: {sorted(QUERY_AREAS)}")
        params[f"query.{area}"] = neutralize_query_value(value)

    # --- filter.overallStatus : a single validated enum token ---------------
    # ``status`` is a vestigial alias with no producer today; see the docstring.
    status = filters.get("overallStatus", filters.get("status"))
    if status is not None:
        if status not in OVERALL_STATUS_TOKENS:
            raise ValueError(f"unknown overallStatus token {status!r}")
        params["filter.overallStatus"] = status

    # --- filter.advanced : ONE AND-joined Essie expr, from tokens/ints only --
    clauses: list[str] = []

    if filters.get("interventional_only") is True:
        clauses.append("AREA[StudyType]COVERAGE[FullMatch]INTERVENTIONAL")

    study_type = filters.get("studyType")
    if study_type is not None:
        if study_type not in STUDY_TYPE_TOKENS:
            raise ValueError(f"unknown studyType token {study_type!r}")
        clause = f"AREA[StudyType]COVERAGE[FullMatch]{study_type}"
        if clause not in clauses:  # de-dupe an explicit INTERVENTIONAL + interventional_only
            clauses.append(clause)

    # --- phase filter : one or many tokens -> an OR group over the Phase array ---
    # (``Phase`` is a multi-value field; a trial matches if ANY of its phases is in
    # the set, so this is a parenthesized OR, not a FullMatch. Live-verified:
    # interventional pancreatic 3950 -> +AREA[Phase](PHASE1 OR PHASE2) 2601.)
    phase = filters.get("phase")
    if phase is not None:
        tokens = [str(p) for p in (phase if isinstance(phase, list) else [phase])]
        for token in tokens:
            if token not in PHASE_TOKENS:
                raise ValueError(f"unknown phase token {token!r}")
        if tokens:
            clauses.append(f"AREA[Phase]({' OR '.join(tokens)})")

    # --- lead-sponsor class : a single validated enum token ----------------------
    sponsor_class = filters.get("sponsorClass")
    if sponsor_class is not None:
        if sponsor_class not in SPONSOR_CLASS_TOKENS:
            raise ValueError(f"unknown sponsorClass token {sponsor_class!r}")
        clauses.append(f"AREA[LeadSponsorClass]COVERAGE[FullMatch]{sponsor_class}")

    # --- intervention type : a single validated enum token -----------------------
    intervention_type = filters.get("interventionType")
    if intervention_type is not None:
        if intervention_type not in INTERVENTION_TYPE_TOKENS:
            raise ValueError(f"unknown interventionType token {intervention_type!r}")
        clauses.append(f"AREA[InterventionType]COVERAGE[FullMatch]{intervention_type}")

    start_year = filters.get("start_year")
    end_year = filters.get("end_year")
    if start_year is not None or end_year is not None:
        if start_year is not None:
            start_bound = f"{_validate_year(start_year, label='start_year')}-01-01"
        else:
            start_bound = "MIN"
        if end_year is not None:
            end_bound = f"{_validate_year(end_year, label='end_year')}-12-31"
        else:
            end_bound = "MAX"
        clauses.append(f"AREA[StartDate]RANGE[{start_bound},{end_bound}]")

    if clauses:
        params["filter.advanced"] = " AND ".join(clauses)

    return params
