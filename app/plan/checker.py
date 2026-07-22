"""The Plan Checker (ARCHITECTURE_SPEC §3.3 / §B.3) — mechanical, deterministic
validation of a planner-emitted Plan.

Code, not a prompt. Runs before any LLM review and before any API call — the
real anti-hallucination gate: the planner cannot invent a capability the tool
set doesn't expose, and cannot pass an invalid token/field/chart-type through.
Rejects on the first violation with a precise machine ``reason`` the planner
can escalate on (one bounded re-plan); when every check passes it returns
``ok=True`` with the plan echoed back UNCHANGED as ``normalized_plan`` — this
module validates, it never rewrites (the field name is a leftover; see
``CheckResult``).
"""

from __future__ import annotations

from app.api.schemas import ChartType
from app.ctgov.enums import (
    DATE_FIELDS,
    FIELD_ALIASES,
    INTERVENTION_TYPE_TOKENS,
    OVERALL_STATUS_TOKENS,
    PHASE_TOKENS,
    QUERY_AREAS,
    SPONSOR_CLASS_TOKENS,
    STUDY_TYPE_TOKENS,
)
from app.plan.models import CheckResult, Plan
from app.plan.recipes import get_recipe

# Derived (computed) aggregation fields that are NOT a raw JSON path in
# FIELD_ALIASES — the value is computed by the executor, not read off one field.
# ``study_duration`` = completionDate − startDate (the histogram recipe, R-16),
# binned by ``app.ctgov.histogram.bin_durations``. The checker allows a plan
# ``field`` to be either a real alias OR one of these derived names.
DERIVED_FIELDS: frozenset[str] = frozenset({"study_duration"})

# Filter key -> the real token set its value(s) must be drawn from
# (ARCHITECTURE_SPEC §B.3: "status/studyType/sponsor-class/intervention-type ∈
# their real token sets"). Keys match the convention used in meta.filters
# throughout the golden fixtures (e.g. "phase", "overallStatus", "studyType").
_FILTER_TOKEN_SETS: dict[str, frozenset[str]] = {
    "phase": PHASE_TOKENS,
    "overallStatus": OVERALL_STATUS_TOKENS,
    "studyType": STUDY_TYPE_TOKENS,
    "sponsorClass": SPONSOR_CLASS_TOKENS,
    "interventionType": INTERVENTION_TYPE_TOKENS,
}

# The complete filter-key vocabulary this codebase actually uses. The checker is
# the anti-hallucination gate (module docstring above): a typo'd/invented filter
# key (e.g. "phaze") must fail mechanical validation rather than pass silently.
# Covers the token-set-checked keys above (phase/overallStatus/studyType/
# sponsorClass/interventionType) plus the keys whose value has no fixed token
# vocabulary:
#   * "start_year"/"end_year" (inclusive year-range bounds) and
#     "interventional_only" (the CC-5/E-38 toggle) — really emitted by the planner
#     (``PlannerFilters``) and really consumed by ``build_search_params``.
#   * "country" — the planner CAN emit ``filters.country`` (``PlannerFilters.country``)
#     and it is accepted here, but NOTHING consumes it: ``build_search_params``
#     reads no country key, and a country only reaches the wire as an ENTITY mapped
#     to ``query.locn`` in ``app.graph.nodes``. A ``filters.country`` is therefore
#     accepted and then silently ignored.
#   * "status" — a human status hint (e.g. "recruiting"), NOT a wire token, and NOT
#     a field of ``VisualizeRequest`` (which has study_type/trial_phase/country/
#     start_year/end_year/interventional_only, no status). No producer emits it:
#     ``PlannerFilters`` has no status field either. It is vestigial and slightly
#     sharp — ``build_search_params`` does read ``filters["status"]`` as a fallback
#     for "overallStatus" and RAISES on a non-token, so if a producer ever appeared,
#     a human hint would clear this checker and then fail at the param boundary as a
#     redacted upstream error instead of a clean re-plan.
# Extend as the filter vocabulary grows — add a row here (and to _FILTER_TOKEN_SETS
# if the new key has a fixed token set) whenever a new filter key is wired up;
# never let the checker silently accept an unlisted key.
KNOWN_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "phase",
        "status",
        "overallStatus",
        "studyType",
        "sponsorClass",
        "interventionType",
        "country",
        "start_year",
        "end_year",
        "interventional_only",
    }
)

# Self-consistency guard: every token-set-checked key must also be a known
# filter key, so the two tables can never drift apart. This is an import-time
# consistency check on our own tables, never a user-input check — and like every
# `assert`, it is stripped under `python -O`.
assert set(_FILTER_TOKEN_SETS) <= KNOWN_FILTER_KEYS

# Internal entity-dimension name -> the real ClinicalTrials.gov query area it
# targets (SPEC_INTERROGATION §C: query.term/cond/intr/spons/locn). Plan.entities
# / Series.entities are keyed by dimension name (e.g. "condition", "drug"), not
# the raw query-area code, so this is what the checker validates entity keys
# against to confirm "query areas used ⊆ QUERY_AREAS".
_ENTITY_DIMENSION_AREAS: dict[str, str] = {
    "term": "term",
    "condition": "cond",
    "drug": "intr",
    "sponsor": "spons",
    "country": "locn",
}

# Self-consistency guard (not a user-input check; also stripped under `python -O`):
# every mapped area must actually be a whitelisted query area, so the two tables
# can't drift apart.
assert set(_ENTITY_DIMENSION_AREAS.values()) <= QUERY_AREAS


def _is_malformed_token_shape(value: object) -> bool:
    """Is ``value`` a shape a real filter token can never legally be (a dict, or a
    list containing a dict/list)?

    Two defenses guard the same hole (FIX 2 — a checker must never raise): the
    membership test below coerces with ``str(token)``, so an unhashable value can
    never blow up a frozenset lookup, and this predicate runs first so such a value
    is reported as a precise ``malformed_filter_value`` rather than stringified into
    a misleading ``invalid_filter_token`` reason."""
    if isinstance(value, dict):
        return True
    if isinstance(value, list):
        return any(isinstance(item, (dict, list)) for item in value)
    return False


def _check_filter_tokens(filters: dict) -> str | None:
    """Return a machine reason if any filter key is unknown, or if a key that HAS
    a fixed token set carries a malformed shape (dict/nested-list — unhashable
    against that frozenset) or an invented token.

    Note the ordering below: a known key with NO token set (start_year/end_year/
    interventional_only/status/country) ``continue``s *before* the shape check, so
    a malformed value under one of those keys is accepted here —
    ``{"country": {"a": 1}}`` passes. Nothing downstream reads ``filters["country"]``
    at all; a malformed ``start_year`` is caught by ``_validate_year`` at the
    param boundary instead. Tightening this means moving the shape check above the
    ``continue``.

    Never raises for a Pydantic-constructed Plan: that is the checker's whole
    contract (``check_plan`` must always return a ``CheckResult``, never propagate
    a ``TypeError``/``KeyError``)."""
    for key, raw in filters.items():
        if key not in KNOWN_FILTER_KEYS:
            return f"unknown_filter_key:{key!r}"

        token_set = _FILTER_TOKEN_SETS.get(key)
        if token_set is None:
            # A known key with no fixed token vocabulary (start_year/end_year/
            # interventional_only/status/country) — nothing to check against here.
            # NOTE this `continue` also skips the shape check below, so a dict under
            # one of these keys passes the checker (see the docstring).
            continue

        if _is_malformed_token_shape(raw):
            return f"malformed_filter_value:{key!r}"

        tokens = raw if isinstance(raw, list) else [raw]
        for token in tokens:
            if str(token) not in token_set:
                return f"invalid_filter_token:{key}={token!r}"
    return None


def _check_entity_dimensions(entities: dict) -> str | None:
    """Return a machine reason if an entity key names an unknown dimension
    (i.e. one that doesn't map to a whitelisted query area)."""
    for key in entities:
        if key not in _ENTITY_DIMENSION_AREAS:
            return f"unknown_entity_dimension:{key!r}"
    return None


def check_plan(plan: Plan) -> CheckResult:
    """Mechanically validate ``plan`` (ARCHITECTURE_SPEC §3.3 / §B.3).

    Eight checks, in order (first violation wins):

    1. ``chart_type`` is a real ``ChartType`` member. Defensive: Pydantic already
       guarantees it for any Plan it constructed, so this can only fire on a Plan
       whose attribute was mutated after construction.
    2. ``date_field``, when present, is one of the 5 real date fields — defensive
       for the same reason (it is a ``Literal`` on the model).
    3. ``field``, when present, resolves via the known field-alias allowlist or is
       one of ``DERIVED_FIELDS`` (rejects an invented aggregation field).
    4. Every entity-dimension key (own + each ``compare`` arm) maps to a
       whitelisted query area.
    5. Every filter key (own + each ``compare`` arm) is a known key
       (``KNOWN_FILTER_KEYS``); every key with a fixed token vocabulary carries
       only real token(s) from that set, in a legal shape (never a dict/
       nested-list). See ``_check_filter_tokens`` for what is *not* shape-checked.
    6. The plan's ``query_class`` has a registered recipe, and ``chart_type``
       is one the recipe allows (its default, an alternate, or its degeneracy
       fallback). The ``KeyError`` branch is likewise mutation-only: every
       ``QueryClass`` member has a row in ``RECIPES``.
    7. Data-shape <-> chart sanity: ``network_graph`` only for
       ``query_class == "network"``; ``time_series`` requires a ``date_field``.
       The ``network_graph`` half is unreachable defense-in-depth as the recipe
       table stands — ``NETWORK_GRAPH`` is allowed by the network recipe only, so
       check 6 already rejects it for every other class.
    8. ``_check_class_shape`` — the G-33 per-class structural check (a compare plan
       really carries ≥2 ``series``, a network plan really carries a ``network``
       block, a timeseries really carries a ``grain``, …). This is the check that
       stops a shape-less plan from reaching ``execute``.

    On pass, returns ``ok=True`` with the *same* plan object as
    ``normalized_plan`` — nothing is rewritten here (see ``CheckResult``).

    What this checker deliberately does NOT do, despite §B.3's wording:

    * It does not normalize anything, and it never mutates the plan. §B.3/CC-8 as
      written have the checker *override* an unsupportable ``chart_type``; the
      shipped checker REJECTS instead (check 6) and lets the planner re-choose in
      the one bounded re-plan. Token/field normalization likewise runs upstream in
      the planner (``_normalize_field``, ``normalize_trial_phase``), not here.
    * It does not check the year range or ordering: ``{"start_year": 2020,
      "end_year": 2010}`` passes. A request-supplied range is ordered by a
      ``VisualizeRequest`` validator, and ``_validate_year`` in ``app.ctgov.params``
      fences each year to [1900, 2100] and RAISES otherwise (surfacing as a redacted
      upstream error, not a clean re-plan) — but nothing rejects an *inverted* pair
      the planner invented: it reaches the wire as a reversed
      ``AREA[StartDate]RANGE[...]`` clause, unchallenged by any layer.
    * It does not validate tools or tool args — there is no tool call to validate
      in a single-shot planner; the executor dispatches on ``query_class`` and
      ``Recipe.allowed_tools`` is a build-time-tested declaration (see
      ``app.plan.recipes``).
    * It does not validate ``plan.alternates``; only ``plan.chart_type`` is checked
      against the recipe.

    Never raises for a Plan that Pydantic constructed — the contract the graph
    depends on. ``Plan`` sets no ``validate_assignment``, so mutating an attribute
    after construction (``plan.chart_type = "network_graph"``) bypasses the
    Literal/Enum guards and CAN raise here; that path is unreachable from the LLM,
    which only ever produces a freshly-validated Plan.
    """
    if plan.chart_type not in ChartType:
        return CheckResult(ok=False, reason=f"invalid_chart_type:{plan.chart_type!r}")

    if plan.date_field is not None and plan.date_field not in DATE_FIELDS:
        return CheckResult(ok=False, reason=f"invalid_date_field:{plan.date_field!r}")

    if plan.field is not None and plan.field not in FIELD_ALIASES and plan.field not in DERIVED_FIELDS:
        return CheckResult(ok=False, reason=f"unknown_field:{plan.field!r}")

    reason = _check_entity_dimensions(plan.entities)
    if reason:
        return CheckResult(ok=False, reason=reason)
    for series in plan.series or []:
        reason = _check_entity_dimensions(series.entities)
        if reason:
            return CheckResult(ok=False, reason=reason)

    reason = _check_filter_tokens(plan.filters)
    if reason:
        return CheckResult(ok=False, reason=reason)
    for series in plan.series or []:
        reason = _check_filter_tokens(series.filters)
        if reason:
            return CheckResult(ok=False, reason=reason)

    try:
        # Mutation-only branch: every ``QueryClass`` Literal member has a RECIPES row,
        # so a Pydantic-constructed Plan can never miss (defense-in-depth, check 6).
        recipe = get_recipe(plan.query_class)
    except KeyError:
        return CheckResult(ok=False, reason=f"unknown_query_class:{plan.query_class!r}")

    allowed_charts = {recipe.chart_type, *recipe.alternates}
    if recipe.degeneracy_fallback is not None:
        allowed_charts.add(recipe.degeneracy_fallback)
    if plan.chart_type not in allowed_charts:
        return CheckResult(
            ok=False,
            reason=(
                f"chart_type_not_allowed_for_recipe:{plan.query_class}:{plan.chart_type.value}"
            ),
        )

    # Unreachable as the recipe table stands (only the network recipe allows
    # NETWORK_GRAPH, so check 6 rejects it first for every other class) — kept as
    # defense-in-depth against a future row that widens an allowed set.
    if plan.chart_type is ChartType.NETWORK_GRAPH and plan.query_class != "network":
        return CheckResult(ok=False, reason="network_graph_requires_network_query_class")

    if plan.chart_type is ChartType.TIME_SERIES and plan.date_field is None:
        return CheckResult(ok=False, reason="time_series_requires_date_field")

    reason = _check_class_shape(plan)
    if reason:
        return CheckResult(ok=False, reason=reason)

    return CheckResult(ok=True, normalized_plan=plan)


def _check_class_shape(plan: Plan) -> str | None:
    """Per-class structural completeness (G-33) — the generalized-Plan recursion.

    The Phase-1 checker validated tokens/fields/areas but not that a plan carried
    the *shape* its query_class needs; a compare/network plan could pass with a
    missing ``series``/``network`` block (an anti-hallucination hole) or be
    rejected outright so X-3/X-5 never run. This dispatches on ``query_class`` and
    requires the class's mandatory slots — while the token/area/field validators
    above have ALREADY recursed into every ``series[]`` element's entities+filters,
    so an invented status token in ``series[1]`` is caught upstream, not here.

    * distribution / geographic → an aggregation ``field`` is required (geographic
      is specifically ``country``). That equality is load-bearing coupling, not
      belt-and-braces: the geographic executor aggregates on the literal
      ``"country"`` and never reads ``plan.field``, so relaxing this check would
      silently make the executor aggregate the wrong dimension.
    * timeseries → a ``date_field`` (already enforced when chart is time_series) and
      a ``grain``.
    * compare → ``series`` with ≥2 arms AND an aggregation ``field`` (the dimension
      each arm is bucketed on).
    * network → a ``network`` block (its ``kind`` is Literal-constrained by the model).
    * single_value → NO aggregation ``field`` and NO series/network: it is a single
      ``count_trials`` over the plan's entities/filters (the no-viz / scalar path,
      CC-7). Accepted unconditionally — an unscoped count ("how many trials in
      existence") is allowed through; ``execute`` refuses it via the too_large gate
      if the population is huge (G-39). Nothing structural to require.
    """
    query_class = plan.query_class
    if query_class == "distribution":
        if plan.field is None:
            return "distribution_requires_field"
    elif query_class == "geographic":
        if plan.field != "country":
            return f"geographic_requires_country_field:{plan.field!r}"
    elif query_class == "timeseries":
        if plan.date_field is None:
            return "timeseries_requires_date_field"
        if plan.grain is None:
            return "timeseries_requires_grain"
    elif query_class == "compare":
        if not plan.series or len(plan.series) < 2:
            return "compare_requires_two_series"
        if plan.field is None:
            return "compare_requires_field"
    elif query_class == "network":
        if plan.network is None:
            return "network_requires_network_block"
    elif query_class == "single_value":
        # A scalar count over the plan's entities/filters — no field, no series/
        # network. Even an unscoped count is accepted; the too_large gate handles a
        # huge population at execute time. Nothing structural to require.
        return None
    return None
