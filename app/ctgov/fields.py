"""Field -> bucketing spec (the per-field aggregation recipe) -- Wave 2.

``AggregationCore.page_and_group`` is field-agnostic: it takes a ``key_fn`` that
maps one fetched record to its ``(value, label)`` bucket key(s) and a ``mode``
(``"combine"`` vs ``"explode"``). This module holds the *per-field* half of that
contract -- the ``FieldSpec`` the tools layer selects by alias.

Phase 1 needed exactly one field: ``phase`` (COMBINE mode -- a trial belongs to
exactly one phase bucket, and a combined phase like ``["PHASE1","PHASE2"]`` is
its OWN composite bucket, never split into two bars, CC-15). Phase 2 adds two
EXPLODE fields -- ``country`` and ``interventionType`` -- where one trial can
carry several distinct values (a distinct-trial count to each), plus per-field
``sort_key`` (count-desc, sentinels pinned last) and ``top_n`` (the "Other"-fold
threshold the tools layer applies). The ``phase`` spec's ordering is unchanged.

Design notes
------------
* The composite bucket VALUE (``"PHASE1|PHASE2"``) is our own aggregation output
  -- it is never sent back to the API -- so a ``"|"``-joined token string is a
  safe, deterministic identity. The API only ever sees the validated single
  tokens in ``PHASE_TOKENS`` (via ``params.build_search_params``).
* ``phase_key_fn`` always returns EXACTLY ONE ``(value, label)`` per record
  (combine): missing / NA / single / composite each map to one bucket. It never
  raises on unknown/garbage tokens -- they still produce a deterministic bucket.
* ``phase_sort_key`` gives the deterministic bucket order the tools layer sorts
  by: ``MISSING, NA, EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4``, with each
  composite placed immediately after its lead (lowest-rank) token.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app import config
from app.ctgov.countries import canonical_country
from app.ctgov.enums import FIELD_ALIASES

# --- Phase display labels + rank order (Interface Contract v1) --------------

# Single-token display labels (matches the contract snippet). ``MISSING`` is not
# here on purpose: it is emitted by ``phase_key_fn`` with its own explicit label
# and never routed through ``_phase_label``.
PHASE_LABELS: dict[str, str] = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "NA (not applicable)",
}

# The explicit-sentinel labels (missing key != NA, CC-5).
_MISSING_LABEL = "Missing (not reported)"

# Deterministic bucket-sort order. Composites are placed after their lead token
# by ``phase_sort_key`` (below), not by an entry here.
PHASE_RANK: tuple[str, ...] = (
    "MISSING",
    "NA",
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE2",
    "PHASE3",
    "PHASE4",
)
_RANK_INDEX: dict[str, int] = {token: i for i, token in enumerate(PHASE_RANK)}
_UNKNOWN_RANK = len(PHASE_RANK)  # any unknown/garbage token sorts after all known ones

# Plain numbered phases (for the "Phase 1/2" composite label). EARLY_PHASE1 is
# deliberately excluded -- it has no clean numeric form.
_PLAIN_PHASE_NUM: dict[str, str] = {"PHASE1": "1", "PHASE2": "2", "PHASE3": "3", "PHASE4": "4"}

_PHASE_FIELD_PATH = FIELD_ALIASES["phase"]  # protocolSection.designModule.phases


def _rank_of(token: str) -> int:
    return _RANK_INDEX.get(token, _UNKNOWN_RANK)


def _phase_label(token: str) -> str:
    """Human label for a single phase token; the token itself for unknowns."""
    return PHASE_LABELS.get(token, token)


def _composite_label(tokens: list[str]) -> str:
    """Human label for a combined phase (CC-15). ``["PHASE1","PHASE2"] -> "Phase 1/2"``.

    Falls back to slash-joined full labels when a token isn't a plain numbered
    phase (e.g. ``EARLY_PHASE1`` or an unknown token) so it stays readable and
    never crashes.
    """
    if all(token in _PLAIN_PHASE_NUM for token in tokens):
        return "Phase " + "/".join(_PLAIN_PHASE_NUM[token] for token in tokens)
    return "/".join(_phase_label(token) for token in tokens)


def phase_key_fn(record: dict) -> list[tuple[str, str]]:
    """Map one record to EXACTLY ONE ``(value, label)`` phase bucket (combine).

    * no ``phases`` key / empty list -> ``("MISSING", "Missing (not reported)")``
    * ``["NA"]``                     -> ``("NA", "NA (not applicable)")``
    * single ``["PHASE2"]``          -> ``("PHASE2", "Phase 2")``
    * combined ``["PHASE1","PHASE2"]`` -> ``("PHASE1|PHASE2", "Phase 1/2")`` --
      its own composite bucket, tokens ordered by phase-rank (CC-15).

    Unknown/garbage tokens still produce a deterministic bucket rather than
    raising (ties break by token name so bucket identity is order-independent).

    TOTAL by construction (LESSON B2 + K1): a ``key_fn`` runs over live, sometimes
    malformed registry records, and a raise here sinks the WHOLE chart (one bad
    record -> ``status:"error"``). So every descent is ``isinstance``-guarded, a
    non-dict ``protocolSection``/``designModule`` is treated as absent, a bare-
    string ``phases`` is ONE token (never iterated character-by-character into a
    garbage bucket), a non-list/non-string ``phases`` is MISSING, and every token
    is stringified + de-duplicated so a stray non-string / repeated token can't
    crash the sort/join or split one trial across two bars.
    """
    ps = record.get("protocolSection")
    design = ps.get("designModule") if isinstance(ps, dict) else None
    phases = design.get("phases") if isinstance(design, dict) else None

    # Normalize to a list of string tokens (see the totality note above).
    if isinstance(phases, str):
        phases = [phases]
    elif not isinstance(phases, list):
        phases = None
    if not phases:
        return [("MISSING", _MISSING_LABEL)]
    tokens = list(dict.fromkeys(str(token) for token in phases))  # stringify + dedup

    # Deterministic token order: by phase-rank, then by name for a stable tie-break.
    tokens = sorted(tokens, key=lambda token: (_rank_of(token), token))
    if len(tokens) == 1:
        token = tokens[0]
        return [(token, _phase_label(token))]

    value = "|".join(tokens)
    return [(value, _composite_label(tokens))]


def phase_sort_key(value: str) -> tuple[int, int, tuple[int, ...], str]:
    """Deterministic sort key for a phase bucket value (single or composite).

    Primary: the lead (lowest-rank) token's rank -- so a composite sorts next to
    its lead single bucket. Secondary: composites (1) after singles (0) of the
    same lead rank. Then the full rank tuple and the raw value break any
    remaining ties, so the order is total and stable.
    """
    tokens = value.split("|")
    lead_rank = min((_rank_of(token) for token in tokens), default=_UNKNOWN_RANK)
    is_composite = 1 if len(tokens) > 1 else 0
    return (lead_rank, is_composite, tuple(_rank_of(token) for token in tokens), value)


# --- Country + intervention-type explode fields (Phase 2) ------------------

# The explicit "unknown/unreported" bucket both explode key_fns emit when a trial
# carries no usable value (no locations / no interventions). A total key_fn never
# returns ``[]`` (K1), so a value-less trial still lands in an honest bucket.
_UNKNOWN_VALUE = "UNKNOWN"
_UNKNOWN_LABEL = "Unknown"

# Sentinel bucket VALUES that ``count_desc_sort_key`` pins to the END of a
# count-ranked bar regardless of size, so a large Unknown / folded-Other tail can
# never top the ranking. ``"Other"`` is the derived fold bucket value the tools
# layer (W2) emits when a ``top_n`` spec folds its tail (see the ``country`` spec
# note); ``"NA"``/``"MISSING"`` are the other explicit sentinels used elsewhere.
_SENTINEL_VALUES: frozenset[str] = frozenset({"UNKNOWN", "NA", "MISSING", "Other"})

# Public aliases for the tools layer (aggregate_by's top-N/"Other" fold + explode
# citation branch) — one source of truth for the sentinel vocabulary + the Unknown
# value, so fields.py and tools.py can never drift apart on what "sentinel"/"Unknown"
# mean.
SENTINEL_VALUES = _SENTINEL_VALUES
UNKNOWN_VALUE = _UNKNOWN_VALUE
OTHER_VALUE = "Other"


def count_desc_sort_key(bucket: dict) -> tuple[int, int, str]:
    """Sort key for a count-ranked bar: ``-count_trials`` then value, sentinels last.

    ``aggregate_by`` calls ``spec.sort_key(bucket_dict)`` per bucket. Returns
    ``(is_sentinel, -count_trials, value)`` so real categories rank by descending
    trial count (ties broken by value for a deterministic, total order) and every
    sentinel bucket (``UNKNOWN``/``NA``/``MISSING``/``Other``) sorts AFTER them no
    matter how large it is. TOTAL -- a missing / non-int ``count_trials`` and a
    non-str ``value`` are coerced, so a malformed bucket dict can't raise mid-sort.
    """
    value = bucket.get("value", "")
    if not isinstance(value, str):
        value = str(value)
    count = bucket.get("count_trials", 0)
    if not isinstance(count, int):
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 0
    is_sentinel = 1 if value in _SENTINEL_VALUES else 0
    return (is_sentinel, -count, value)


def country_key_fn(record: dict) -> list[tuple[str, str]]:
    """Map one record to its DISTINCT location countries (explode, CC-13).

    Reads ``protocolSection.contactsLocationsModule.locations[].country`` (a free
    string, e.g. ``"United States"``, ``"Turkey (Türkiye)"`` -- no ISO codes) and
    returns one ``(value, label)`` per DISTINCT country the trial lists, so a
    trial with US, US, UK contributes ``[("United States", ...), ("United Kingdom",
    ...)]`` -- one distinct-trial count to each, never two to US (CC-13). Each raw
    country string is run through the small disclosed alias table
    (``canonical_country``, E-20) so common variants (``USA``/``US``/``U.S.`` ->
    ``United States``) fold into ONE bucket instead of splitting the count; the
    normalization is narrow (common English abbreviations only, no ISO/geopolitical
    resolution) and the country facet's ``meta.notes`` states that spellings are
    normalized (``tools.aggregate_by`` — fires on every country facet, since the
    alias table is always applied). No locations / no usable country ->
    ``[("UNKNOWN", "Unknown")]``.

    TOTAL (K1/B2): a non-dict ``protocolSection``/``contactsLocationsModule``, a
    non-list ``locations`` (including present-but-``None`` -- ``.get`` does NOT
    apply its default when the key is present with a ``None`` value), a non-dict
    location element, and a missing / non-string / empty ``country`` all degrade
    to skip-or-Unknown, never a raise. ``dict``-keyed accumulation dedupes while
    keeping first-seen order.
    """
    ps = record.get("protocolSection")
    module = ps.get("contactsLocationsModule") if isinstance(ps, dict) else None
    locations = module.get("locations") if isinstance(module, dict) else None
    if not isinstance(locations, list):
        return [(_UNKNOWN_VALUE, _UNKNOWN_LABEL)]

    distinct: dict[str, tuple[str, str]] = {}
    for location in locations:
        if not isinstance(location, dict):
            continue
        raw = location.get("country")
        if not isinstance(raw, str) or not raw:
            continue
        country, _ = canonical_country(raw)  # fold common variants (E-20); free text passes through
        if country and country not in distinct:
            distinct[country] = (country, country)

    if not distinct:
        return [(_UNKNOWN_VALUE, _UNKNOWN_LABEL)]
    return list(distinct.values())


# protocolSection.armsInterventionsModule.interventions[].type -> human label.
# Keys are the 11 real INTERVENTION_TYPE_TOKENS (enums.py); an unknown/garbage
# token falls back to itself (see ``intervention_type_key_fn``), so an unseen
# future token still renders as its own honest bucket.
INTERVENTION_TYPE_LABELS: dict[str, str] = {
    "DRUG": "Drug",
    "DEVICE": "Device",
    "BIOLOGICAL": "Biological",
    "PROCEDURE": "Procedure",
    "RADIATION": "Radiation",
    "BEHAVIORAL": "Behavioral",
    "GENETIC": "Genetic",
    "DIETARY_SUPPLEMENT": "Dietary Supplement",
    "DIAGNOSTIC_TEST": "Diagnostic Test",
    "COMBINATION_PRODUCT": "Combination Product",
    "OTHER": "Other",
}


def intervention_type_key_fn(record: dict) -> list[tuple[str, str]]:
    """Map one record to its DISTINCT intervention types (explode).

    Reads ``protocolSection.armsInterventionsModule.interventions[].type`` (a
    token like ``DRUG``/``DEVICE``) and returns one ``(value, label)`` per DISTINCT
    type, deduped within the trial (a trial repeating DRUG counts once). value ==
    the raw token; label == a human string via ``INTERVENTION_TYPE_LABELS`` (an
    unknown token maps to itself). No interventions -> ``[("UNKNOWN", "Unknown")]``.

    TOTAL (K1/B2): a non-dict ``protocolSection``/``armsInterventionsModule``, a
    non-list ``interventions`` (including present-but-``None``), a non-dict
    element, and a missing / non-string / empty ``type`` all degrade -- never a
    raise. ``dict``-keyed accumulation dedupes while keeping first-seen order.
    """
    ps = record.get("protocolSection")
    module = ps.get("armsInterventionsModule") if isinstance(ps, dict) else None
    interventions = module.get("interventions") if isinstance(module, dict) else None
    if not isinstance(interventions, list):
        return [(_UNKNOWN_VALUE, _UNKNOWN_LABEL)]

    distinct: dict[str, tuple[str, str]] = {}
    for intervention in interventions:
        if not isinstance(intervention, dict):
            continue
        itype = intervention.get("type")
        if not isinstance(itype, str) or not itype:
            continue
        if itype not in distinct:
            distinct[itype] = (itype, INTERVENTION_TYPE_LABELS.get(itype, itype))

    if not distinct:
        return [(_UNKNOWN_VALUE, _UNKNOWN_LABEL)]
    return list(distinct.values())


_COUNTRY_FIELD_PATH = FIELD_ALIASES["country"]  # ...contactsLocationsModule.locations[].country
_INTERVENTION_TYPE_FIELD_PATH = FIELD_ALIASES["interventionType"]  # ...interventions[].type
_OVERALL_STATUS_FIELD_PATH = FIELD_ALIASES["overallStatus"]  # ...statusModule.overallStatus
_SPONSOR_CLASS_FIELD_PATH = FIELD_ALIASES["sponsorClass"]  # ...leadSponsor.class


def _titlecase_token(token: str) -> str:
    """Human-ish label for an ``UPPER_SNAKE`` API token: ``NOT_YET_RECRUITING``
    -> ``"Not Yet Recruiting"``. Used as the fallback display for status/sponsor
    tokens without an explicit label."""
    return " ".join(part.capitalize() for part in token.split("_"))


# protocolSection.statusModule.overallStatus is SINGLE-valued (never missing per the
# API brief) -> one combine bucket per trial. Labels for the common tokens; any
# other real token title-cases via _titlecase_token.
_STATUS_LABELS: dict[str, str] = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "UNKNOWN": "Unknown status",
}


def overall_status_key_fn(record: dict) -> list[tuple[str, str]]:
    """Map one record to its single ``overallStatus`` bucket (combine).

    TOTAL: a non-dict ``protocolSection``/``statusModule`` or a missing / non-string
    status -> ``[("UNKNOWN", "Unknown status")]`` (the enum note says status is never
    missing, but a total key_fn defends anyway)."""
    ps = record.get("protocolSection")
    module = ps.get("statusModule") if isinstance(ps, dict) else None
    status = module.get("overallStatus") if isinstance(module, dict) else None
    if not isinstance(status, str) or not status:
        return [("UNKNOWN", "Unknown status")]
    return [(status, _STATUS_LABELS.get(status, _titlecase_token(status)))]


def sponsor_class_key_fn(record: dict) -> list[tuple[str, str]]:
    """Map one record to its lead-sponsor class bucket (combine).

    ``leadSponsor.class`` is single-valued (OTHER/INDUSTRY/NIH/…). TOTAL: a missing
    class -> ``[("UNKNOWN", "Unknown")]``. Note: OTHER (academic/hospital) dominates
    the registry -- the recipe surfaces that in a note (E-71)."""
    ps = record.get("protocolSection")
    module = ps.get("sponsorCollaboratorsModule") if isinstance(ps, dict) else None
    lead = module.get("leadSponsor") if isinstance(module, dict) else None
    klass = lead.get("class") if isinstance(lead, dict) else None
    if not isinstance(klass, str) or not klass:
        return [(_UNKNOWN_VALUE, _UNKNOWN_LABEL)]
    return [(klass, _titlecase_token(klass))]


# --- The field->bucketing spec table ---------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """How one aggregation field is paged, keyed, and projected.

    ``key_fn`` maps a record to its bucket key(s); ``mode`` picks the counting
    convention (``"combine"`` = 1 key/record; ``"explode"`` = >=1 key/record);
    ``fields_projection`` is the pipe-separated ``fields=`` the client requests
    for this field (kept minimal so paging stays cheap).

    ``sort_key`` / ``top_n`` drive the tools layer's post-aggregation ordering
    (both optional; default = the Phase-1 behavior of "core first-seen order,
    no fold"):

    * ``sort_key(bucket_dict) -> object`` -- the order ``aggregate_by`` sorts the
      bucket dicts by (``None`` keeps the core's first-seen order).
    * ``top_n`` -- when set, ``aggregate_by`` keeps the top-N NON-sentinel buckets
      after sorting and folds the remaining non-sentinel tail into ONE derived
      ``"Other"`` bucket (see the ``country`` spec note for the exact fold
      contract); ``None`` keeps every bucket.
    """

    alias: str
    field_path: str
    mode: str  # "combine" | "explode"
    key_fn: Callable[[dict], list[tuple[str, str]]]
    fields_projection: str
    sort_key: Callable[[dict], object] | None = None
    top_n: int | None = None


# Phase 1 registered only ``phase`` (combine). Phase 2 adds the ``country`` and
# ``interventionType`` explode fields here without touching the core.
FIELD_SPEC: dict[str, FieldSpec] = {
    "phase": FieldSpec(
        alias="phase",
        field_path=_PHASE_FIELD_PATH,
        mode="combine",
        key_fn=phase_key_fn,
        fields_projection="NCTId|Phase|BriefTitle",
        # Byte-identical Phase-1 behavior: phase-rank order, no top_n fold. The
        # X-2 live gate (Σ count_trials == countTotal == 3950) depends on this
        # exact ordering; the lambda reproduces tools.py's hardcoded sort.
        sort_key=lambda bucket: phase_sort_key(bucket["value"]),
        top_n=None,
    ),
    "country": FieldSpec(
        alias="country",
        field_path=_COUNTRY_FIELD_PATH,
        mode="explode",
        key_fn=country_key_fn,
        fields_projection="NCTId|LocationCountry|BriefTitle",
        sort_key=count_desc_sort_key,
        top_n=config.TOP_N_CATEGORIES,  # 50 (§B.4 response-size cap; P5-TOPN — was 15 pre-Phase-5)
        # --- "Other"-fold contract (aggregate_by / W2 implements the fold; this
        # spec only SETS top_n + sort_key) -----------------------------------
        # After sorting by ``count_desc_sort_key`` (count-desc, sentinels last),
        # aggregate_by keeps the top ``TOP_N_CATEGORIES`` NON-sentinel country buckets
        # and folds the remaining non-sentinel tail into ONE derived bucket:
        #   {value: "Other", label: "Other (N countries)",
        #    count_trials: Σ DISTINCT nctIds across the tail (deduped, K3),
        #    count_mentions: Σ tail mentions, derived: True, members: [folded values],
        #    citations: sampled up to k=20 from the tail records}
        # The Unknown sentinel bucket is ALWAYS kept (never folded). Because the
        # fold sums DISTINCT-trial counts (a trial listed in two folded countries
        # is ONE trial in Other, not two), the explode reconciliation still holds:
        #   Σ(top-15 + Other + Unknown) count_trials == distinct_trials == countTotal.
    ),
    "interventionType": FieldSpec(
        alias="interventionType",
        field_path=_INTERVENTION_TYPE_FIELD_PATH,
        mode="explode",
        key_fn=intervention_type_key_fn,
        fields_projection="NCTId|InterventionType|BriefTitle",
        sort_key=count_desc_sort_key,
        top_n=None,  # small, bounded token set -- show every type, no fold
    ),
    "overallStatus": FieldSpec(
        alias="overallStatus",
        field_path=_OVERALL_STATUS_FIELD_PATH,
        mode="combine",  # single-valued: exactly one status per trial (Σ == distinct == countTotal)
        key_fn=overall_status_key_fn,
        fields_projection="NCTId|OverallStatus|BriefTitle",
        sort_key=count_desc_sort_key,
        top_n=None,  # 14 real values, bounded -- no fold
    ),
    "sponsorClass": FieldSpec(
        alias="sponsorClass",
        field_path=_SPONSOR_CLASS_FIELD_PATH,
        mode="combine",  # single-valued lead-sponsor class (Σ == distinct == countTotal)
        key_fn=sponsor_class_key_fn,
        fields_projection="NCTId|LeadSponsorClass|BriefTitle",
        sort_key=count_desc_sort_key,
        top_n=None,  # 9 real values, bounded -- no fold
    ),
}
