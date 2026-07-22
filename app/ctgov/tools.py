"""The high-level deterministic task tools (ARCHITECTURE_SPEC §3.5).

Each tool performs its **full deterministic job internally** (paging, counting,
dedupe, sort, Unknown-bucketing, dual counts, citations) and returns *computed*
results. The LLM never pages or tallies anything itself (ARCHITECTURE_SPEC §1, the
governing invariant).

How a tool is actually selected in v1 — state this precisely, it is not the
tool-calling shape the spec sketches. The planner makes ONE structured-output call
with ``tools=None`` (``app.llm.planner``); it emits a ``query_class``, not a tool
name. The recipe registry maps that class to a tool name
(``app.plan.recipes``), and the execute node dispatches with a hand-written
if/elif on ``plan.query_class`` (``app.graph.nodes``). So :data:`TOOL_REGISTRY`
below is the name→callable inventory that recipes and tests are checked against —
it is NOT the live dispatch table, and nothing in ``app/`` looks a tool up through
it at runtime.

Inventory (count carefully — three different numbers are defensible):

* 8 entries in :data:`TOOL_REGISTRY`.
* 6 of them are LIVE: ``count_trials``, ``aggregate_by``, ``timeseries``,
  ``compare``, ``build_network``, ``study_duration_histogram``.
* 2 are documented ``NotImplementedError`` stubs: ``get_trial`` (its
  path-injection guard IS wired and tested) and ``resolve_entity``.
* Plus one live callable that is deliberately NOT registered:
  :func:`aggregate_by_counts`, the exact over-budget path the execute node selects
  itself. It is an implementation of the ``distribution`` class, not a tool the
  planner could ever name.

The surface is read-only and GET-only via ``app.ctgov.client.CTGovClient`` — see
ARCHITECTURE_SPEC §A(a) for the per-tool least-privilege matrix.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable

from app import config
from app.api.schemas import Citation
from app.ctgov.aggregate import AggregationCore, _nct_id
from app.ctgov.citations import brief_title, build_bucket_citations
from app.ctgov.client import CTGovClient
from app.ctgov.compare import union_series
from app.ctgov.enums import (
    FIELD_ALIASES,
    INTERVENTION_TYPE_TOKENS,
    OVERALL_STATUS_TOKENS,
    SPONSOR_CLASS_TOKENS,
)
from app.ctgov.fields import FIELD_SPEC, SENTINEL_VALUES, UNKNOWN_VALUE, FieldSpec
from app.ctgov.histogram import bin_durations
from app.ctgov.identifiers import validate_nct_id
from app.ctgov.network import build_graph
from app.ctgov.params import build_search_params
from app.ctgov.timeseries import finalize_timeseries, year_key_fn

# ``fields=`` projection piece for each of the 5 real date fields (the wire token
# that projects the corresponding ``...DateStruct`` — verified live: NCTId|StartDate
# projects protocolSection.statusModule.startDateStruct). PUBLIC so the execute node
# stamps the SAME wire token into meta.query_provenance that the tool actually fetches:
# the plan-internal field name ("startDate") is NOT the wire projection token
# ("StartDate"), and stamping the former would publish a params echo the API never saw.
DATE_PROJECTION: dict[str, str] = {
    "startDate": "StartDate",
    "primaryCompletionDate": "PrimaryCompletionDate",
    "completionDate": "CompletionDate",
    "studyFirstPostDate": "StudyFirstPostDate",
    "lastUpdatePostDate": "LastUpdatePostDate",
}


def count_trials(query: dict, filters: dict) -> int:
    """Return the exact matching trial count (the ``trial_count`` primitive).

    One ``countTotal=true&pageSize=1`` call — cheap, always exact, never
    partial. This is the correctness oracle (CC-16): every other aggregation
    reconciles against this number.
    """
    search_params = build_search_params(query, filters)
    return CTGovClient().count(search_params)


def aggregate_by(query: dict, filters: dict, field: str) -> dict:
    """Page under budget and bucket the matching trials by ``field``.

    Internally: page → bucket by ``field`` (via its ``FieldSpec.key_fn``) →
    explicit Missing/NA buckets (CC-5) → dual counts, distinct-trial +
    trial×value mentions (CC-3) → combined multi-value tokens get their own
    bucket (CC-15) → per-bucket citations (CC-9). ``field`` is a short alias
    (e.g. ``"phase"``) resolved via ``app.ctgov.fields.FIELD_SPEC``.

    Returns the tool-result dict the executor appends to ``tool_results`` and
    the viz-spec builder reads (``viz.spec._bucket_to_datum``): the per-bucket
    dicts carry ``value, label, count_trials, count_mentions, source_ids,
    citations, citations_truncated, contributing_count``. For a ``combine``
    field, ``Σ count_trials`` over buckets reconciles to the API's exact
    ``countTotal`` (CC-16) -- the whole point of this unit.
    """
    spec = FIELD_SPEC.get(field)
    if spec is None:
        raise ValueError(f"unknown aggregation field {field!r}; known: {sorted(FIELD_SPEC)}")

    core = AggregationCore(CTGovClient())
    search_params = build_search_params(query, filters)
    group = core.page_and_group(
        search_params,
        fields=spec.fields_projection,
        key_fn=spec.key_fn,
        mode=spec.mode,
    )

    # Carry each bucket's contributing records through sort + (optional) top-N
    # fold; citations are built LAST (so the "Other" fold can element-target them).
    raw = [
        {
            "value": bucket.value,
            "label": bucket.label,
            "count_trials": bucket.count_trials,
            "count_mentions": bucket.count_mentions,
            "records": bucket.records,
        }
        for bucket in group.buckets
    ]
    if spec.sort_key is not None:
        raw.sort(key=spec.sort_key)

    other: dict | None = None
    if spec.top_n is not None:
        raw, other = _split_top_n(raw, spec.top_n)

    buckets = [_finalize_bucket(bucket, spec) for bucket in raw]
    if other is not None:
        buckets.append(_finalize_other(other, spec))
    if spec.sort_key is not None:
        # Re-sort finalized dicts so a folded "Other" (a sentinel) lands after the
        # top-N and beside the Unknown bucket, keeping the ranked order total.
        buckets.sort(key=spec.sort_key)

    # Disclosures (surfaced onto meta.notes) so the reader sees how the data was
    # shaped (LESSON I1): the explode double-count basis and the top-N/"Other" fold.
    notes: list[str] = []
    if spec.mode == "explode":
        noun = "countries" if spec.alias == "country" else f"{spec.alias} values"
        notes.append(
            f"Each bar counts the DISTINCT trials studied in that {spec.alias}; a trial spanning "
            f"multiple {noun} is counted once per {spec.alias}, so bar totals sum to MORE than the "
            f"trial count -- the headline count_basis.trials is the distinct-trial total (CC-3)."
        )
    if other is not None:
        notes.append(
            f"Ranked bar, top {spec.top_n} by trial count; {len(other['tail'])} lower-count "
            f"{spec.alias} values are folded into the derived 'Other' bucket (which cites its members)."
        )
    if spec.alias == "country":
        notes.append(
            "Country is a free-text display name with no ISO code, so this is a ranked bar, "
            "never a choropleth; a multi-country trial is deduped to one count per country (CC-13). "
            "Common spelling variants are normalized to a canonical form before counting "
            "(e.g. USA / U.S. → United States), so variant spellings are merged into one bar (E-20)."
        )

    return {
        "tool": "aggregate_by",
        "field": field,
        "field_path": spec.field_path,
        "mode": spec.mode,
        "distinct_trials": group.distinct_trials,
        "truncated": group.truncated,
        "buckets": buckets,
        "notes": notes,
        "record_index": _bounded_record_index([bucket.records for bucket in group.buckets]),
    }


def _nct_sort_key(record: dict) -> str:
    """Deterministic nctId sort key (missing id sorts as '' — never raises)."""
    nct = _nct_id(record)
    return nct if isinstance(nct, str) else ""


# Bounds on the record index a tool surfaces for the Phase-4 record-grounded citation
# re-verify (viz.review.record_grounded_reverify): ``per_list_k`` sampled records per list
# (for a per-bucket call, the SAME first-K-by-nctId the citation sample uses -> the index
# covers each bucket's cited records exactly) and a hard ``cap`` on the total, so the index
# stays small + JSON-serialisable in graph state even for a wide result.
#
# The cap does NOT cover every path. A ranked bar can cite up to K x #buckets nctIds, and
# with K=20 and the top-N fold at 50 that is up to 51 x 20 = 1020 > the 500 cap. Measured on
# the shipped examples/run_06_geographic_ranked_bar.json: 51 buckets, 836 citation objects,
# 566 unique cited nctIds — i.e. ~66 cited records are genuinely absent from the index. A
# network's cited nctIds span the whole paged set and can exceed it by more. The re-verify is
# therefore best-effort on wide boards: viz.review skips a citation whose record is not in
# the index rather than failing it, so the check never produces a false negative — it just
# covers fewer citations than a naive reading of "every citation is re-verified" implies.
# (This comment used to claim full coverage; that was true only while TOP_N_CATEGORIES was 15.)
_RECORD_INDEX_K = config.CITATION_SAMPLE_K  # ≤K cited nctIds/datum (§B.4/ENG-32), operator-tunable
_RECORD_INDEX_CAP = config.RECORD_INDEX_CAP  # bounded per-request record index for re-verify


def _bounded_record_index(
    record_lists: list[list[dict]], per_list_k: int = _RECORD_INDEX_K
) -> dict[str, dict]:
    """Build a bounded ``{nct_id: record}`` index from the records a tool paged.

    Additive provenance surface (changes NO count / existing key): the Output Reviewer
    re-verifies each citation's matched_value against the ACTUAL fetched record here (an
    independent ground truth, not the citation's own stored ``value``), giving
    ``is_substring_at`` a runtime caller (LESSON M3) and defending against any future
    non-code path that could author a citation. Deterministic (first-``per_list_k``-by-nctId per
    list) and total (a record with no nctId is skipped, never raises). Per-bucket calls pass the
    default ``per_list_k`` (== the citation cap, so the index == the cited records); a flat-list
    call (network / histogram — one list backs all citations) passes a larger ``per_list_k`` to
    cover as many cited records as the ``cap`` allows."""
    index: dict[str, dict] = {}
    for records in record_lists:
        for record in sorted(records or [], key=_nct_sort_key)[:per_list_k]:
            if len(index) >= _RECORD_INDEX_CAP:
                return index
            nct = _nct_id(record)
            if isinstance(nct, str) and nct and nct not in index:
                index[nct] = record
    return index


def _explode_citations(
    records: list[dict], spec: FieldSpec, *, bucket_value: str | None, member_set: set | None = None
) -> tuple[list[Citation], int, bool]:
    """Element-targeted per-bucket citations for an EXPLODE field (country /
    interventionType). The ``matched_value`` quotes the BUCKET'S OWN value (the
    specific country/type), which the record provably carries (the core bucketed it
    here via ``key_fn``) — so a "France" bar cites "France", never ``locations[0]``
    (the loose "first list value" a plain string-extract would return, which for a
    multi-value field is often a DIFFERENT element). Round-trips via
    ``is_substring_at`` (the value EQUALS one list element) and passes the Output
    Reviewer's element-precise check. ``excerpt`` stays what it is everywhere else:
    the trial's brief title.

    * ``member_set`` (the "Other" fold): ``matched_value`` is the record's OWN value
      that falls in the folded set, resolved via ``spec.key_fn`` — precise even
      though "Other" spans many countries.
    * The ``UNKNOWN`` bucket has no value to quote: an *absence* citation
      (``matched_value=""``, ``value=[]``), valid only against a genuinely-absent
      value.

    The sample is capped at ``config.CITATION_SAMPLE_K`` records per bucket (the
    same cap the combine path uses via ``build_bucket_citations``).

    **Known latent defect — the canonicalizing key_fn breaks verbatim round-trip.**
    ``country_key_fn`` normalizes spellings (``canonical_country``, E-20), so for the
    ``country`` field both ``matched_value`` (the bucket value) and ``value`` (from
    ``spec.key_fn``) are CANONICAL, not what the record literally says. If a record's
    raw ``locations[].country`` is an alias-table KEY whose canonical form differs —
    "USA", "UK", "Russian Federation", "Czech Republic", … — then
    ``is_substring_at(record, field_path, "United States")`` is False against a
    record that says "USA". ``deterministic_precheck`` still passes (it compares
    ``matched_value`` to the equally-canonical ``value``), but
    ``record_grounded_reverify`` compares against the RAW fetched record and would
    hard-fail the whole chart with ``citation_invalid``. Reproduced offline with a
    hand-built "USA" record.

    Not reached on live data as measured (2026-07-22, 200 pancreatic-cancer records,
    31 distinct country strings): CT.gov already emits canonical-ish names, and the
    only alias hits — "Russia", "Czechia", "South Korea" — are IDENTITY mappings, so
    ``matched_value`` equals the raw string. It stays latent only as long as that
    holds. The fix is not local to this function: ``matched_value`` and ``value``
    must move together (they must stay mutually consistent for
    ``deterministic_precheck``), so it needs a per-field "raw spelling that folded
    into this bucket" hook on ``FieldSpec`` rather than a patch here.
    """
    contributing = len(records)
    k = config.CITATION_SAMPLE_K
    sample = sorted(records, key=_nct_sort_key)[:k]
    citations: list[Citation] = []
    for record in sample:
        nct = _nct_id(record) or ""
        # ``value`` = the record's REAL distinct values at this field (the list the
        # key_fn extracted from the record), NOT a copy of the excerpt. This is what
        # gives the Output Reviewer's citation check TEETH for explode fields: the
        # matched_value (the bucket's value) is verified to be a genuine element of
        # the record's own list, so a FABRICATED bucket value ("Atlantis") fails the
        # element-precise check instead of passing against a copy of itself.
        record_values = [value for value, _ in spec.key_fn(record) if value != UNKNOWN_VALUE]
        if bucket_value == UNKNOWN_VALUE:
            # Genuine absence: no value at the path -> empty matched_value against [].
            citations.append(Citation(nct_id=nct, field_path=spec.field_path, value=[], matched_value="", excerpt=brief_title(record) or ""))
            continue
        if member_set is not None:
            # Other fold: quote THIS record's own folded value.
            excerpt = next((value for value in record_values if value in member_set), None)
        else:
            excerpt = bucket_value
        citations.append(
            Citation(
                nct_id=nct,
                field_path=spec.field_path,
                value=record_values,
                matched_value=excerpt if excerpt is not None else "",
                excerpt=brief_title(record) or (excerpt or ""),
            )
        )
    return citations, contributing, contributing > k


def _finalize_bucket(bucket: dict, spec: FieldSpec) -> dict:
    """Build one bucket's citation surface. The ``matched_value`` is string-extracted
    at ``field_path`` for combine fields and element-targeted to the bucket's own value
    for explode fields; ``excerpt`` is the trial's brief title on both paths."""
    records = bucket["records"]
    if spec.mode == "combine":
        # A composite combine bucket (e.g. phase "PHASE1|PHASE2", CC-15) must cite
        # EVERY member token, not just the first (Phase-3 Task 5). The composite value
        # is our own "|"-joined output (never a wire token — the "|" is a safe
        # discriminator; single-valued combine fields like status carry no "|"), so
        # split it back into the member tokens the citation core verifies per-token.
        value = bucket["value"]
        member_tokens = value.split("|") if "|" in value else None
        citations, contributing, truncated = build_bucket_citations(
            records, spec.field_path, k=config.CITATION_SAMPLE_K, member_tokens=member_tokens
        )
    else:
        citations, contributing, truncated = _explode_citations(
            records, spec, bucket_value=bucket["value"]
        )
    return {
        "value": bucket["value"],
        "label": bucket["label"],
        "count_trials": bucket["count_trials"],
        "count_mentions": bucket["count_mentions"],
        "source_ids": [citation.nct_id for citation in citations],
        "citations": citations,
        "citations_truncated": truncated,
        "contributing_count": contributing,
    }


def _split_top_n(raw: list[dict], top_n: int) -> tuple[list[dict], dict | None]:
    """Split ``raw`` (already sorted count-desc, sentinels last) into the kept
    buckets (top-N non-sentinel + ALL sentinels) and the folded tail. Returns
    ``(kept, tail_marker | None)`` where ``tail_marker`` carries the tail buckets to
    fold, or ``None`` when the tail is empty (nothing to fold)."""
    kept: list[dict] = []
    tail: list[dict] = []
    n_regular = 0
    for bucket in raw:
        if bucket["value"] in SENTINEL_VALUES:
            kept.append(bucket)  # Unknown/NA/Missing are always kept, never folded
        elif n_regular < top_n:
            kept.append(bucket)
            n_regular += 1
        else:
            tail.append(bucket)
    if not tail:
        return kept, None
    return kept, {"tail": tail}


def _finalize_other(other: dict, spec: FieldSpec) -> dict:
    """Build the derived "Other" fold bucket from the tail buckets (G-35 derived).

    ``count_trials`` = DISTINCT trials appearing in ANY tail category (union-dedup
    by nctId — a trial in two folded countries is ONE trial in Other), so the bar
    is an honest "everything-else" count consistent with the per-category bars.
    Citations are element-targeted to each record's own folded value. This does not
    touch ``group.distinct_trials`` (the reconciliation anchor), so explode
    reconciliation is unaffected by the fold."""
    tail = other["tail"]
    members = [bucket["value"] for bucket in tail]
    member_set = set(members)
    seen: set[str] = set()
    records: list[dict] = []
    for bucket in tail:
        for record in bucket["records"]:
            nct = _nct_id(record)
            if nct is None or nct in seen:
                continue
            seen.add(nct)
            records.append(record)
    citations, contributing, truncated = _explode_citations(
        records, spec, bucket_value=None, member_set=member_set
    )
    noun = "countries" if spec.alias == "country" else "categories"
    return {
        "value": "Other",
        "label": f"Other ({len(members)} {noun})",
        "count_trials": len(records),
        "count_mentions": sum(int(bucket.get("count_mentions") or 0) for bucket in tail),
        "source_ids": [citation.nct_id for citation in citations],
        "citations": citations,
        "citations_truncated": truncated,
        "contributing_count": contributing,
        "derived": True,
        "members": members,
    }


# Categorical distribution fields whose buckets can be counted EXACTLY by ONE count query per
# token (a bounded token set), so a broad-condition distribution that would blow the paging
# budget (diabetes ~24k, cancer ~142k) returns an exact, cited answer instead of a too_large
# refuse. `phase` is EXCLUDED (its CC-15 composite buckets aren't a single-token count) and
# `country` is EXCLUDED (an unbounded free-text token set) — those stay paged / too_large.
_COUNT_AGGREGATABLE: dict[str, frozenset[str]] = {
    "overallStatus": OVERALL_STATUS_TOKENS,
    "sponsorClass": SPONSOR_CLASS_TOKENS,
    "interventionType": INTERVENTION_TYPE_TOKENS,
}
# Citation-sample records pulled per bucket, in the SAME call as that bucket's count. It is
# deliberately NOT ``config.CITATION_SAMPLE_K``: this number is the ``pageSize`` of a
# countTotal request, so raising it makes ~14 real requests heavier, whereas CITATION_SAMPLE_K
# is a pure output cap. It is a local literal rather than a config knob — i.e. this one number
# is NOT operator-tunable, unlike the rest of the safety envelope in ``app.config``.
_COUNT_SAMPLE_K = 10


def is_count_aggregatable(field: str | None) -> bool:
    """Can this distribution field be aggregated EXACTLY via per-token count queries (so an
    over-budget population charts instead of refusing)? True only for the bounded-token
    categorical fields (status / sponsorClass / interventionType), not phase or country."""
    return field in _COUNT_AGGREGATABLE


def aggregate_by_counts(query: dict, filters: dict, field: str) -> dict:
    """Exact per-category distribution via ONE ``countTotal`` query per token — no paging, no bias.

    The over-budget path for a bounded-token categorical field (``is_count_aggregatable``): the
    count endpoint is always exact regardless of population size, so a broad distribution
    (diabetes ~24k, cancer ~142k) returns an EXACT, cited answer instead of a ``too_large`` refuse
    — the biased-sorted-prefix objection (§B.7) doesn't apply because nothing is paged. Each
    per-token call also pulls a bounded citation sample (``pageSize=K``, same call), so every
    non-zero bucket is cited. ``distinct_trials`` == the population total; for a single-value
    (combine) field Σ(buckets incl a derived ``Missing`` residual) == total, for the explode
    ``interventionType`` Σbars ≥ distinct (a multi-type trial counts per type). Returns the SAME
    dict shape ``aggregate_by`` does, so the viz-builder + Output Reviewer are unchanged. TOTAL:
    a malformed page element is skipped, never raises.

    Three ways this path differs from ``aggregate_by``, all visible in the output — know them
    before demoing ``examples/run_09_exact_at_scale_status.json``:

    1. **Bucket ORDER is alphabetical, not ranked.** Buckets are appended in ``sorted(tokens)``
       order and nothing re-sorts them (this function ignores ``spec.sort_key``, and
       ``app.viz.spec`` never sorts data). So the same question answered below the budget comes
       back count-desc, and above it comes back A→Z. Ranking here would need a second pass after
       all counts are known; it is not done today.
    2. **Labels are plain title-case**, ``token.replace("_"," ").title()`` — so status
       ``NOT_YET_RECRUITING`` renders "Not Yet Recruiting", where the paged path's curated
       ``fields._STATUS_LABELS`` renders "Not yet recruiting". Same bucket, different casing,
       depending on which path served it.
    3. **Zero-count tokens are dropped**, not shown as empty bars (``if n <= 0: continue``). That
       is why a 14-token status chart can ship 13 bars: in run_09 ``WITHHELD`` matched 0 trials.

    It also assembles its own transport params (``countTotal``/``pageSize``/``fields``) and calls
    ``client.get`` directly, rather than going through ``count``/``iter_studies`` — it needs the
    count and the citation sample from ONE request."""
    spec = FIELD_SPEC[field]
    tokens = _COUNT_AGGREGATABLE[field]
    client = CTGovClient()

    total = count_trials(query, filters)  # the population total = the distinct-trial anchor (CC-16)
    buckets: list[dict] = []
    record_index: dict[str, dict] = {}
    covered = 0
    for token in sorted(tokens):
        params = build_search_params(query, {**filters, field: token})
        body = client.get(
            "/studies",
            {**params, "countTotal": "true", "pageSize": _COUNT_SAMPLE_K,
             "fields": spec.fields_projection},
        )
        n = int(body.get("totalCount") or 0)
        if n <= 0:
            continue  # a token with no matching trials is dropped, not shown as a 0-height bar
        covered += n
        sample = [rec for rec in (body.get("studies") or []) if isinstance(rec, dict)]
        citations: list[Citation] = []
        for record in sorted(sample, key=_nct_sort_key)[:_COUNT_SAMPLE_K]:
            nct = _nct_id(record)
            if not isinstance(nct, str) or not nct:
                continue
            record_index[nct] = record
            # value = the record's REAL field value(s) via key_fn (teeth: matched_value=token
            # must be a genuine element, since the record matched the field=token filter), never
            # a copy of matched_value. Keep ALL values — do NOT strip the UNKNOWN sentinel: for overallStatus,
            # "UNKNOWN" is itself a real enum token (it collides with key_fn's missing-sentinel),
            # and a record in the UNKNOWN bucket genuinely carries status "UNKNOWN", so stripping
            # it would empty the value and fail the citation against its own real record.
            real_values = [value for value, _ in spec.key_fn(record)]
            citations.append(
                Citation(nct_id=nct, field_path=spec.field_path, value=real_values, matched_value=token, excerpt=brief_title(record) or token)
            )
        buckets.append({
            "value": token,
            # Plain title-case, NOT fields._STATUS_LABELS — see the divergence note in the
            # docstring. The curated labels are keyed off a record via ``key_fn``; here we
            # only have the token, so this path renders it directly.
            "label": token.replace("_", " ").title(),
            "count_trials": n,
            "count_mentions": n,
            "source_ids": [c.nct_id for c in citations],
            "citations": citations,
            "citations_truncated": n > len(citations),
            "contributing_count": n,
        })

    notes: list[str] = [
        f"Computed via exact per-category count queries (not paged): the {total:,}-trial "
        "population exceeds the paging budget, so each bar is one exact countTotal call — an "
        "exact distribution at any scale, no biased sampling.",
    ]
    if spec.mode == "combine":
        # Single-value field: every trial has exactly one value, so Σ(known tokens) + a derived
        # Missing residual == total (keeps the combine Σ==distinct==countTotal reconciliation).
        missing = total - covered
        if missing > 0:
            buckets.append({
                "value": "Missing", "label": "Missing (not reported)",
                "count_trials": missing, "count_mentions": missing,
                "source_ids": [], "citations": [], "citations_truncated": False,
                "contributing_count": missing, "derived": True,
            })
    else:  # explode (interventionType)
        notes.append(
            "Each bar counts the DISTINCT trials studying that intervention type; a trial with "
            "multiple types is counted once per type, so bar totals sum to MORE than the trial "
            "count -- the headline is the distinct-trial total (CC-3)."
        )

    return {
        "tool": "aggregate_by_counts",
        "field": field,
        "field_path": spec.field_path,
        "mode": spec.mode,
        "distinct_trials": total,
        "truncated": False,  # counts are EXACT + complete (not a paged prefix)
        "buckets": buckets,
        "notes": notes,
        "record_index": record_index,
    }


def timeseries(query: dict, filters: dict, date_field: str, grain: str = "year") -> dict:
    """Page and bin the matching trials by ``date_field`` at the given ``grain``.

    Internally: page (one combine bucket per trial via ``year_key_fn``) → bin by the
    chosen date field (CC-4) → genuine future dates go into a flagged "planned"
    bucket, never clamped (G-40) → mixed precision (``"2015"``/``"2015-05"``/
    ``"2015-05-10"``) normalized to ``grain`` → gap periods within the observed
    range filled with an explicit 0-count bucket → a ``MISSING`` bucket for trials
    with no date (kept for reconciliation: Σ incl. planned incl. missing ==
    distinct == countTotal). Each real year-bucket carries its contributing-nctId
    citations (a time bucket is a first-class citable datum).

    Raises ``ValueError`` on an unrecognized ``date_field`` — the same clean failure
    ``aggregate_by`` gives for an unknown aggregation field. The Plan Checker
    validates ``plan.date_field`` upstream, so this is a contract guard for direct
    callers, not a routine path.
    """
    if date_field not in DATE_PROJECTION or date_field not in FIELD_ALIASES:
        raise ValueError(
            f"unknown date field {date_field!r}; known: {sorted(DATE_PROJECTION)}"
        )
    date_path = FIELD_ALIASES[date_field]
    projection = f"NCTId|{DATE_PROJECTION[date_field]}|BriefTitle"
    core = AggregationCore(CTGovClient())
    search_params = build_search_params(query, filters)
    group = core.page_and_group(
        search_params, fields=projection, key_fn=year_key_fn(date_path), mode="combine"
    )

    raw: list[dict] = []
    for bucket in group.buckets:
        citations, contributing, truncated = build_bucket_citations(bucket.records, date_path, k=config.CITATION_SAMPLE_K)
        raw.append(
            {
                "value": bucket.value,
                "label": bucket.label,
                "count_trials": bucket.count_trials,
                "count_mentions": bucket.count_mentions,
                "source_ids": [citation.nct_id for citation in citations],
                "citations": citations,
                "citations_truncated": truncated,
                "contributing_count": contributing,
            }
        )

    current_year = _dt.datetime.now(_dt.UTC).year
    datums, notes, degrade = finalize_timeseries(raw, current_year=current_year, grain=grain)
    return {
        "tool": "timeseries",
        "field": date_field,
        "field_path": date_path,
        "mode": "combine",
        "distinct_trials": group.distinct_trials,
        "truncated": group.truncated,
        "buckets": datums,
        "notes": notes,
        "degrade": degrade,
        "date_field": date_field,
        "grain": grain,
        "record_index": _bounded_record_index([bucket.records for bucket in group.buckets]),
    }


def compare(series: list[dict], field: str) -> dict:
    """Run one aggregation per series on ``field`` and union their categories (G-24).

    ``series`` is the generalized ≥2-arm shape (each ``{label, query, filters}``) —
    the G-24 generalization of the ARCHITECTURE_SPEC §3.5 two-arg sketch. Internally:
    one ``aggregate_by`` per arm → union categories with an explicit 0-fill on any
    side → percentage-WITHIN-series as the default count basis, each arm's own N
    labelled (CC-14) — raw counts remain per bucket alongside the percentage. Each
    arm is self-reconciled by its own ``aggregate_by`` (distinct-nctId ==
    countTotal); the union spans two populations, so it is exempt from the single-
    oracle Output-Reviewer reconciliation (the caller passes ``reconcile=False``).
    """
    results: list[dict] = []
    truncated = False
    record_index: dict[str, dict] = {}
    for arm in series:
        agg = aggregate_by(arm["query"], arm.get("filters", {}), field)
        truncated = truncated or agg["truncated"]
        for nct, record in (agg.get("record_index") or {}).items():
            if len(record_index) < _RECORD_INDEX_CAP and nct not in record_index:
                record_index[nct] = record
        # % denominator = the arm's exact countTotal (passed in by the executor's
        # per-arm count call), NOT the paged distinct — so the within-series % is
        # honest even if paging drifted mid-walk. Falls back to paged distinct if the
        # executor didn't supply a count (defensive).
        paged = agg["distinct_trials"]
        arm_total = arm.get("count_total", paged)
        results.append(
            {"label": arm["label"], "N": arm_total, "buckets": agg["buckets"], "paged_distinct": paged}
        )

    datums, notes = union_series(results)
    # Per-arm reconciliation disclosure: arms are budget-gated (<=20k -> fully paged),
    # so paged distinct should EQUAL the countTotal denominator; disclose any drift
    # (a mid-run registry mutation) rather than silently biasing the %.
    for result in results:
        if result["N"] and result["paged_distinct"] != result["N"]:
            notes.append(
                f"{result['label']}: paged {result['paged_distinct']:,} of {result['N']:,} matching "
                f"trials (the % denominator is the full countTotal; this drift is disclosed, not hidden)."
            )
    return {
        "tool": "compare",
        "field": field,
        "mode": "compare",
        "truncated": truncated,
        "buckets": datums,
        "notes": notes,
        "series_meta": [{"label": result["label"], "N": result["N"]} for result in results],
        "record_index": record_index,
    }


# The projection the network pager requests: sponsor name + each intervention's
# type/name/otherNames (for the DRUG filter, synonym merge, and placebo drop).
NETWORK_FIELDS = "NCTId|LeadSponsorName|InterventionName|InterventionType|InterventionOtherName|BriefTitle"

# The two verified date fields the study-duration histogram needs (R-16) — no
# dependency on the unverified enrollment field (G-28).
_DURATION_START_PATH = FIELD_ALIASES["startDate"]
_DURATION_END_PATH = FIELD_ALIASES["completionDate"]
_DURATION_FIELDS = "NCTId|StartDate|CompletionDate|BriefTitle"


def study_duration_histogram(query: dict, filters: dict) -> dict:
    """Page the matching trials and bin them by study duration (completionDate −
    startDate, R-16) — the histogram recipe (G-28). One bin per trial (combine),
    so Σ(bins incl. the undated bucket) == distinct == countTotal."""
    client = CTGovClient()
    search_params = build_search_params(query, filters)
    records, truncated = client.iter_studies(search_params, fields=_DURATION_FIELDS, max_pages=config.PAGE_BUDGET_PAGES)
    distinct_trials = len({nct for nct in (_nct_id(record) for record in records) if nct is not None})
    datums, notes = bin_durations(
        records, start_path=_DURATION_START_PATH, end_path=_DURATION_END_PATH
    )
    return {
        "tool": "study_duration_histogram",
        "field": "study_duration",
        "field_path": _DURATION_START_PATH,
        "mode": "combine",
        "distinct_trials": distinct_trials,
        "truncated": truncated,
        "buckets": datums,
        "notes": notes,
        "record_index": _bounded_record_index([records], per_list_k=_RECORD_INDEX_CAP),
    }


def build_network(
    query: dict, kind: str, filters: dict | None = None,
    entity_a: str | None = None, entity_b: str | None = None
) -> dict:
    """Page the matching trials and build a sponsor↔drug or drug↔drug graph.

    Internally: page (sponsor name + intervention type/name/otherNames) → delegate
    to ``app.ctgov.network.build_graph``: extract entities (canonicalized lead
    sponsor / DRUG interventions) → edges are trial-pairs (CC-3's counting convention
    extended to graphs) → **alias-only** drug-name synonym merge (an ``otherName``
    merges two drugs ONLY when it is itself another drug's primary ``name`` — brand↔
    generic, no transitive over-merge) + approximate sponsor-name canonicalization
    (case/punctuation/legal-suffix + a tiny same-org alias table, no parent/subsidiary
    fold) → placebo/standard-of-care nodes dropped (avoids a false mega-hub) → top-N
    nodes by degree + a per-kind minimum edge weight (drug_drug ≥2, sponsor_drug ≥1;
    CC-12) → every edge carries its contributing nctIds and two citations (one per
    endpoint field_path, G-25). A degenerate graph (≤1 node / no co-occurrence, G-41e)
    carries a cited ``fallback`` drug-frequency bar the executor renders instead.
    ``entity_a``/``entity_b`` are accepted for the anchored-network extension (unused
    in the v1 baseline)."""
    client = CTGovClient()
    search_params = build_search_params(query, filters or {})
    records, truncated = client.iter_studies(search_params, fields=NETWORK_FIELDS, max_pages=config.PAGE_BUDGET_PAGES)
    graph = build_graph(records, kind=kind)
    return {
        "tool": "build_network",
        "kind": kind,
        "mode": "network",
        "truncated": truncated,
        "distinct_trials": int(graph.get("distinct_trials", 0)),
        "graph": graph,
        # Edge citations reference nctIds across the WHOLE paged set (one flat list), so index up
        # to the cap; a dense network may still cite trials beyond it -> best-effort re-verify.
        "record_index": _bounded_record_index([records], per_list_k=_RECORD_INDEX_CAP),
    }


def get_trial(nct_id: str) -> dict:
    """STUB — always raises ``NotImplementedError`` after validating ``nct_id``.

    Intended shape (drill-down by NCT id): a single ``GET /studies/{nctId}`` call
    where the raw record never reaches the LLM directly; only a bounded,
    string-extracted excerpt would (ARCHITECTURE_SPEC §A(c)). The fetch is not built.

    The ``nct_id`` is format-validated (``^NCT\\d{8}$``, ``validate_nct_id``) BEFORE
    it would ever be interpolated into the ``/studies/{nctId}`` path — the one
    non-value-slot token that reaches a URL PATH rather than a query VALUE slot
    (G-8/R-20/SEC-25). A malformed id raises ``ValueError`` here and is never sent.
    The guard is wired + tested now (``tests/test_identifiers.py``); the fetch body
    itself is a documented Phase-6 stretch (no user-controlled nctId reaches a path
    today — nctIds come from CT.gov records — so this is a forward-compatible gate).
    """
    validate_nct_id(nct_id)  # path-injection guard (fires even though the fetch is a stub)
    raise NotImplementedError(
        "get_trial fetch is a Phase-6 stretch (drill-down); the nctId path-injection "
        "guard (validate_nct_id) is wired + tested"
    )


def resolve_entity(name: str, kind: str) -> list[str]:
    """STUB — always raises ``NotImplementedError``.

    Intended shape: light alias canonicalization of a drug/condition/sponsor/country
    to its display label(s) (e.g. ``"USA"`` -> ``"United States"``). The API already
    resolves search-*recall* synonyms (``Keytruda`` ≡ ``pembrolizumab``), so this
    tool would only normalize *display* — it would not change what a search matches.
    The country half of that idea does ship, just not as a tool:
    ``app.ctgov.countries.canonical_country`` is called inside ``country_key_fn``.
    """
    raise NotImplementedError("resolve_entity is a Phase-2/3 stretch (display canon)")


# Recipes reference tools by name (ARCHITECTURE_SPEC §B.6). This is the name -> callable
# inventory those names are checked against — at BUILD time, by tests/test_ctgov_plan.py
# (every ``Recipe.allowed_tools`` must be a subset of TOOL_NAMES). No runtime code in app/
# resolves a tool through it: the execute node dispatches with an if/elif on
# plan.query_class. ``aggregate_by_counts`` is intentionally absent — it is an
# implementation of the distribution class, not a name a planner could select.
TOOL_REGISTRY: dict[str, Callable] = {
    "count_trials": count_trials,
    "aggregate_by": aggregate_by,
    "timeseries": timeseries,
    "compare": compare,
    "build_network": build_network,
    "study_duration_histogram": study_duration_histogram,
    "get_trial": get_trial,
    "resolve_entity": resolve_entity,
}

TOOL_NAMES = frozenset(TOOL_REGISTRY)
