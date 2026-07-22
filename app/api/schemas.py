"""API I/O contracts — the request model and the full response envelope.

This is the **lowest layer** of the system: at module level it imports stdlib +
pydantic only, nothing from ``app.*`` (the one ``app`` dependency —
``app.ctgov.phases.normalize_trial_phase`` — is imported *inside* the
``trial_phase`` validator, so importing this module never pulls in a higher
layer). Every other module (planner, checker, executor, viz-spec builder, the
FastAPI transport) imports its wire types from here, so the names in this file are
a **frozen interface contract** — do not rename them.

What lives here:

* The status / kind discriminators (``Status``, ``Kind``) and the closed chart
  enum (``ChartType``) — ARCHITECTURE_SPEC §B.3 / CC-10.
* The shared viz value objects (``Citation``, ``Datum``, ``Node``, ``Edge``,
  ``NetworkData``, ``EncodingChannel``, ``Visualization``).
* The response envelope (``VisualizeResponse``) — ARCHITECTURE_SPEC §6 — plus its
  ``meta`` sub-objects (``CountBasis``, ``Partial``, ``ErrorObj``, ``Meta``).
* The request model (``VisualizeRequest``) — the documented, per-field-validated
  input schema (A-22 "document the request schema").

Design invariant this schema enforces structurally (ARCHITECTURE_SPEC §1, G-30):
the LLM never emits a number. Numbers reach the user only through ``Datum`` /
``CountBasis`` fields that the deterministic aggregation core fills, and the prose
fields ``title``, the scalar ``answer`` and ``question`` are code-templated.
``meta.notes`` is the honest exception: an entry there CAN be LLM prose (a planner
interpretation note, an Output-Reviewer flag reason), so it is *gated* rather than
templated — every entry runs through the deterministic digit post-check
``app.viz.review.note_number_safe`` and is dropped for a fixed code-owned caveat if
it names a number the engine did not compute (``app.graph.nodes.build_spec``).
The schema cannot enforce any of this on its own, but it is the single place those
fields are declared, so the guarantee has one home.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Discriminators -------------------------------------------------------

# Top-level response status (API-8/API-9). ``too_large`` is the §B.7 over-budget
# refuse contract: a match set above the page budget is refused, not truncated.
Status = Literal["ok", "empty", "too_large", "error"]

# Whether the payload is a chart spec, a natural-language answer, or a request for
# more information (API-10 / CC-7 / E-13). ``answer`` covers yes/no + scalar
# questions and the ``too_large`` refuse. ``clarification`` is a first-class
# outcome (P5-INPUT) for a well-formed request whose NL names an *unresolvable
# referent* ("this drug", no ``drug_name``): the envelope carries a code-templated
# ``question`` and no data, so a frontend can prompt for the missing detail rather
# than the backend either guessing (dishonest) or 422-ing a syntactically-valid
# request (wrong — the HTTP request is fine; the intent is incomplete).
Kind = Literal["visualization", "answer", "clarification"]


class ChartType(str, Enum):  # noqa: UP042 — frozen contract mandates (str, Enum), not StrEnum
    """The closed set of renderable marks (ARCHITECTURE_SPEC §B.3 checker enum).

    Closed on purpose: the Plan Checker rejects any ``chart_type`` outside this
    enum, and ``table`` is the universal fallback when no richer mark fits.
    """

    BAR = "bar"  # categorical counts (phase / status / intervention-type distribution)
    GROUPED_BAR = "grouped_bar"  # two labelled series side by side (compare)
    TIME_SERIES = "time_series"  # one metric binned over a date field
    HISTOGRAM = "histogram"  # binned continuous magnitude (e.g. study duration)
    SCATTER = "scatter"  # two continuous axes (deferred; kept in the enum for completeness)
    NETWORK_GRAPH = "network_graph"  # node-link graph (sponsor↔drug / drug↔drug)
    SINGLE_VALUE = "single_value"  # stat card for a scalar / yes-no answer
    TABLE = "table"  # generic fallback


# --- Shared value objects -------------------------------------------------


class Citation(BaseModel):
    """A single per-datum provenance record (CC-9 / A-45/A-46).

    It carries assignment §5's *two* readings of "an exact text excerpt from the
    API response (or a specific field/value)" in two separate fields, because the
    readable one and the checkable one come from different paths:

    * ``matched_value`` — the **checkable** anchor: the exact value at
      ``field_path`` that decided membership (``"PHASE1"``, ``"1997-05"``,
      ``"France"``), string-extracted from the record.
    * ``excerpt`` — the **readable** one: the trial's brief title, string-extracted
      at the fixed path ``protocolSection.identificationModule.briefTitle``
      (``app.ctgov.citations.brief_title``), which is NOT ``field_path``. When a
      record didn't project ``BriefTitle`` it falls back to ``matched_value``, so a
      citation always ships something human-readable.

    Neither is ever authored by the LLM — both are walked out of the fetched record,
    and no citation text is ever synthesized. Three code paths build them to that one
    contract: ``app.ctgov.citations.build_citation`` (combine buckets),
    ``app.ctgov.tools._explode_citations`` (multi-value fields — the anchor is the
    record's OWN element in this bucket, not the loose first list item), and
    ``app.ctgov.network._endpoint_citation`` (one per edge endpoint).

    What the Output Reviewer actually verifies (``app.viz.review``): only
    ``matched_value`` and every ``matched_tokens`` member, twice each — element-
    precise against this citation's own ``value`` (``_citation_valid``) and
    round-tripped against the fetched record via ``is_substring_at``
    (``record_grounded_reverify``). ``excerpt`` is **not** re-verified by the
    reviewer: it is code-extracted from a fixed path rather than proven, which is the
    weaker of the two guarantees and is why membership rests on ``matched_value``,
    not on the title.

    Composite buckets (CC-15): when a bucket is formed from MULTIPLE source tokens
    (e.g. the ``PHASE1|PHASE2`` bucket built from ``["PHASE1","PHASE2"]``),
    ``matched_tokens`` carries every member token verified present in THAT record,
    in token order, and ``matched_value`` is the FIRST member token (display /
    back-compat). ``matched_tokens`` is ``None`` for a single-value bucket.

    Absence citations: for a genuinely-absent value (the ``UNKNOWN`` bucket),
    ``matched_value`` is ``""`` and ``value`` is ``None``/``[]``/``""`` — the empty
    anchor is valid *only* against a genuine absence (``_excerpt_in_value``).
    ``excerpt`` still carries the brief title when the record has one.
    """

    nct_id: str  # e.g. "NCT01234567"
    # Readable supporting excerpt: the trial's brief title, string-extracted from
    # identificationModule.briefTitle (NOT from field_path); falls back to
    # ``matched_value`` when the record carried no briefTitle. Not reviewer-verified.
    excerpt: str = ""
    field_path: str  # the JSON path whose value placed this trial in the datum
    value: Any  # the record's real value(s) at field_path (may be an array, e.g. ["PHASE1"])
    # The anti-fabrication anchor: the exact value at ``field_path`` that decided
    # membership — element-precise and round-trip verified by the Output Reviewer.
    matched_value: str = ""
    # For a composite bucket: every verified matched member value (each a verbatim element at
    # field_path), in token order. None for single-value buckets.
    matched_tokens: list[str] | None = None


class Datum(BaseModel):
    """One row of a chart's ``data`` array.

    Carries the categorical identity (``value``/``label``), any class-specific
    channel fields (``period`` for time series, ``series`` for compare,
    ``bin_start``/``bin_end`` for histograms), the **dual counts** (CC-3), and its
    inline provenance list (G-25/G-34 — the load-bearing citation surface; the
    top-level ``citations{}`` map is only a dedup index).

    ``extra="allow"`` lets a recipe attach class-specific channels without a schema
    change, while the named fields below stay a stable contract. It is **load-bearing,
    not merely convenient**: of the seven channels the spec builder forwards
    (``app.viz.spec._CHANNEL_KEYS``) three are undeclared here and survive only on
    ``extra`` — ``percent`` (grouped_bar's y channel, ``app.ctgov.compare``),
    ``planned`` and ``partial_year`` (time-series flags). Tightening this to
    ``extra="forbid"`` would break grouped bars. The cost of the looseness is real
    and unguarded: ``app.viz.vega`` reads channels with ``getattr(datum, field, None)``,
    so a typo'd or missing ``percent`` emits ``y: null`` silently rather than raising.

    Unlike ``Datum``, ``Node``/``Edge``/``NetworkData`` keep pydantic's default
    ``extra="ignore"``, so a network payload has no extension channel — an unknown
    key there is dropped, not carried.
    """

    model_config = ConfigDict(extra="allow")

    # Raw bucket token, e.g. "PHASE1", or a PIPE-joined composite "PHASE1|PHASE2"
    # (CC-15 combine, app.ctgov.fields.phase_key_fn). The pipe is load-bearing:
    # app.ctgov.tools splits on it to recover the member tokens each citation
    # verifies. The SLASH form ("Phase 1/2") is the human ``label``, never the value.
    value: str
    label: str  # display label, e.g. "Phase 1" (composite: "Phase 1/2")

    # Optional per-class channel fields (null when the class doesn't use them):
    period: str | None = None  # time-series bucket, e.g. "2023"
    series: str | None = None  # compare series label, e.g. "Pembrolizumab"
    bin_start: float | None = None  # histogram bin lower edge
    bin_end: float | None = None  # histogram bin upper edge

    # Measures (the numbers — always code-computed, never LLM-authored):
    count_trials: int  # distinct-trial count; reconciles against countTotal (CC-3)
    count_mentions: int | None = None  # trial×value mention count (honest per-membership tally)

    # Provenance:
    source_ids: list[str] = []  # ≤K sampled nctIds; each resolves to a Citation either inline in
    # this datum's citations[] OR in the top-level citations{} dedup index (G-4)
    citations: list[Citation] = []  # inline per-datum citation list (G-25/G-34 — authoritative)
    citations_truncated: bool = False  # true when the citation sample was capped at K
    contributing_count: int | None = None  # exact size of the contributing set (may exceed len(citations))
    # True for a bucket the core COMPUTED rather than read off one field value. Two
    # producers ship it (``app.ctgov.tools``): the top-N "Other" fold (still fully cited —
    # each citation quotes that record's OWN folded value) and the exact-count "Missing"
    # residual (total − Σ covered tokens: a subtraction, so it carries NO citations).
    derived: bool = False
    # For the "Other" fold: the folded CATEGORY values it stands for (e.g. country names),
    # not nctIds — the contributing trials are in ``source_ids``/``citations``.
    members: list[str] | None = None


class Node(BaseModel):
    """A network graph node (a sponsor or a drug)."""

    id: str  # stable node id, e.g. "sponsor:merck" or "drug:pembrolizumab"
    label: str  # display label, e.g. "Pembrolizumab"
    kind: str  # node kind, e.g. "sponsor" / "drug"
    degree: int | None = None  # number of incident edges (for degree-based sizing)


class Edge(BaseModel):
    """A network graph edge. ``weight`` is a derived count of contributing trials;
    it carries **two** citations (one per endpoint ``field_path``, G-25)."""

    source: str  # source node id
    target: str  # target node id
    weight: int  # #trials the edge is derived from (derived value → cites members)
    source_ids: list[str] = []  # contributing nctIds (the trials that formed this edge)
    citations: list[Citation] = []  # two entries: one per endpoint field_path (G-25)


class NetworkData(BaseModel):
    """The ``data`` payload for a ``network_graph`` — nodes + edges, not a flat
    row array. Edges reference existing node ids."""

    nodes: list[Node]
    edges: list[Edge]


class EncodingChannel(BaseModel):
    """A rich encoding channel spec (CC-10): which field maps to a visual channel,
    plus optional display hints. Keyed in ``Visualization.encoding`` by channel
    name (``x``/``y``/``color``/``nodes``/``edges``…)."""

    field: str  # the datum field this channel reads, e.g. "count_trials"
    label: str | None = None  # axis/legend label
    unit: str | None = None  # unit string, e.g. "trials"
    sort: str | None = None  # sort directive, e.g. "-count_trials" or an explicit order
    scale: str | None = None  # scale hint, e.g. "linear" / "log"


class Visualization(BaseModel):
    """The custom canonical viz spec (CC-10) — the source of truth for a chart.

    ``data`` is a **structurally-resolved union with a checked type↔shape
    invariant (enforced by a model_validator)**: a standard chart carries
    ``list[Datum]`` (a row array); a network carries ``NetworkData``
    (``{nodes, edges}``). Pydantic v2's smart union distinguishes the two
    structurally — a JSON list validates to ``list[Datum]``, a JSON object
    validates to ``NetworkData`` — but shape alone doesn't guarantee the shape
    matches ``type``. The ``_data_matches_type`` validator below closes that
    gap: ``type=="network_graph"`` MUST carry ``NetworkData``, every other
    ``type`` MUST carry a ``list``. The custom schema (not Vega-Lite) is
    canonical because it carries citations, dual counts, and the graph shape,
    which Vega-Lite cannot express.
    """

    type: ChartType  # the mark; also selects the data shape (rows vs network)
    title: str  # code-templated from the validated Plan + computed data (G-30 — never LLM-authored)
    encoding: dict[str, EncodingChannel]  # channel-name → spec, e.g. {"x": ..., "y": ...}
    data: list[Datum] | NetworkData  # rows for charts, {nodes, edges} for a network

    @model_validator(mode="after")
    def _data_matches_type(self) -> Visualization:
        """Enforce the type↔shape invariant the union alone can't check.

        ``network_graph`` MUST carry a ``NetworkData`` (``{nodes, edges}``);
        every other ``type`` MUST carry a ``list`` (rows). Without this, a
        mismatched mark (e.g. ``type="bar"`` with a ``{nodes,edges}`` payload,
        or ``type="network_graph"`` with a row array) validates silently.
        """
        if self.type is ChartType.NETWORK_GRAPH:
            if not isinstance(self.data, NetworkData):
                raise ValueError(
                    "type='network_graph' requires data to be a NetworkData "
                    "({nodes, edges}), not a row list"
                )
        else:
            if not isinstance(self.data, list):
                raise ValueError(
                    f"type={self.type.value!r} requires data to be a list[Datum] "
                    "(rows), not a NetworkData"
                )
        return self


# --- meta sub-objects -----------------------------------------------------


class CountBasis(BaseModel):
    """The dual-count basis (CC-3). ``trials`` is the distinct-trial count that
    reconciles against ``countTotal``; ``mentions`` is the honest trial×value tally
    for multi-value (explode) fields. For ``status:"too_large"``, ``trials`` is the
    exact matching ``countTotal`` (G-39)."""

    trials: int  # distinct-trial total
    mentions: int | None = None  # trial×value mention total (None when a field isn't multi-value)


class Partial(BaseModel):
    """A genuine, defensible truncation (API-19). Present iff a partial was
    actually shipped; it never claims completeness. NULL — not this — for
    ``too_large``, because refusing to chart is not truncating (G-39)."""

    truncated: bool  # true when the shipped data is an honest partial
    of_total: int | None = None  # the true total the partial is a subset of


class ErrorObj(BaseModel):
    """A top-level error object (API-22) for ``status:"error"`` — never folded
    into a half-populated viz (no retrieval happened)."""

    code: str  # machine error code, e.g. "upstream_timeout"
    message: str  # human-readable message (provider key redacted; never leaks internals)


class Meta(BaseModel):
    """The provenance + interpretation envelope (ARCHITECTURE_SPEC §6, CC-18).

    Every field except ``source`` is optional so the same object serves every
    status; ``source`` defaults to the registry name (A-33/G-2).
    """

    count_basis: CountBasis | None = None  # dual counts (CC-3); the exact total on too_large (G-39)
    date_field_used: str | None = None  # which of the 5 date fields a time series used (CC-4)
    time_granularity: str | None = None  # time-series grain, e.g. "year" / "month" (G-3/E-31)
    filters: dict[str, Any] = {}  # the effective, validated filters applied
    query_provenance: dict[str, Any] = {}  # endpoint + effective params, for reproducibility (CC-18)
    retrieved_at: str | None = None  # ISO-8601 retrieval timestamp (CC-18 live-stamp)
    source: str = "clinicaltrials.gov"  # authoritative data source (A-33/G-2)
    partial: Partial | None = None  # a genuine truncation; NULL for too_large (G-39)
    notes: list[str] = []  # interpretation, overrides, and messy-data disclosures (CC-1)


# --- The response envelope ------------------------------------------------


class VisualizeResponse(BaseModel):
    """THE response envelope (ARCHITECTURE_SPEC §6) — every endpoint returns this.

    Field presence is keyed off ``status`` / ``kind``:

    * ``kind:"visualization"`` → ``visualization`` populated; ``answer`` null.
    * ``kind:"answer"`` → ``visualization`` null; ``answer`` populated.
    * ``status:"error"`` → ``error`` populated; ``visualization`` null (never a
      half-viz).
    * ``status:"too_large"`` → ``kind:"answer"``, ``answer`` populated,
      ``visualization``/``vega_lite`` null, ``meta.partial`` null (G-39).
    * ``kind:"clarification"`` → ``question`` populated; ``visualization``/
      ``answer``/``vega_lite`` null; ``status:"empty"`` (well-formed request, an
      unresolvable NL referent, nothing queried — P5-INPUT/E-13).
    * ``vega_lite`` is a convenience projection for the standard marks only —
      ``app.viz.vega._STANDARD_MARKS`` holds bar / grouped_bar / time_series /
      histogram / scatter (scatter is deferred: no recipe emits it). It is null for
      the three marks with no natural Vega-Lite equivalent — ``network_graph``
      (G-41d: a node-link graph must NEVER be expressed as Vega-Lite),
      ``single_value`` and ``table`` — and for every envelope that carries no
      visualization at all (answer / clarification / too_large). So a
      ``single_value`` stat card is a POPULATED ``kind:"visualization"`` with a
      null ``vega_lite`` (shipped: ``examples/run_12_cc1_field_vs_query_conflict.json``).

    These presence rules are conventions the pipeline upholds, not schema
    constraints — the envelope keeps every field optional so a single type serves
    every status.
    """

    status: Status  # ok | empty | too_large | error
    kind: Kind  # visualization | answer | clarification
    visualization: Visualization | None = None  # the custom canonical spec; null on answer/clarification
    # Vega-Lite projection for the standard marks; null for network_graph/single_value/table
    # and for any envelope with no visualization (see the class docstring)
    vega_lite: dict | None = None
    answer: str | None = None  # natural-language answer for kind:"answer" (API-21); code-templated
    question: str | None = None  # the disambiguating question for kind:"clarification" (P5-INPUT/E-13);
    # code-templated (never LLM-authored data); null for every other kind
    error: ErrorObj | None = None  # top-level error for status:"error" (API-22)
    citations: dict[str, Citation] = {}  # top-level dedup index keyed by nctId (API-13/G-4); optional
    meta: Meta  # provenance + interpretation (always present)


# --- The request model ----------------------------------------------------


class VisualizeRequest(BaseModel):
    """The request schema (A-14..A-23, API-4..7, CC-5) — documented + per-field validated.

    Fields
    ------
    query : str (REQUIRED)
        The natural-language question. Non-empty after strip, ≤ 500 chars
        (API-4/E-25/SEC-35). A ``field_validator`` strips surrounding whitespace
        and rejects an empty/whitespace-only value (E-17 → 422).
    drug_name, condition, sponsor, country : str | None (optional, ≤ 200 chars each)
        Structured dimension fields. Authoritative for the dimension they name
        (CC-1 precedence — resolved later by the planner). Each is capped at 200
        chars (G-41b): without a cap they are an unbounded DoS / Essie-injection
        surface — only ``query`` was bounded before.
    trial_phase : str | None (optional, ≤ 100 chars)
        A human phase string ("Phase 1", "1/2", "phase I", "Early Phase 1", "NA").
        Because ``trial_phase`` is a STRUCTURED field with a CLOSED vocabulary, a
        value that names no real phase is a malformed structured input and is
        REJECTED with 422 + the valid list (E-16 / P5-INPUT) — distinct from a
        mistyped free-text ENTITY (drug/condition), which stays a 200-empty result
        (ENG-22). The recognized forms are normalized to wire tokens by
        ``app.ctgov.phases.normalize_trial_phase`` (the single normalizer the
        planner also applies for CC-1 field precedence).
    start_year, end_year : int | None (optional, 1900..2100)
        Year range. When both are present, ``start_year <= end_year`` is enforced
        (else → 422).
    study_type : str | None (optional, ≤ 100 chars)
        A study-type hint (e.g. "interventional"); tokenized downstream.
    interventional_only : bool (default False)
        The CC-5/E-38 toggle — when true a phase distribution can offer the
        interventional denominator (observational trials legitimately have no phase).

    Policy: ``extra="forbid"`` — unknown request fields are rejected (→ 422). This
    is a deliberate choice (fail-closed on a malformed request) over silently
    ignoring typo'd field names.

    Where the caps really live: every length cap here is a ``max_length`` literal on
    the field below and nothing else enforces it. ``app.config`` declares
    ``MAX_QUERY_CHARS``/``MAX_STRUCTURED_FIELD_CHARS`` with the same 500/200 numbers
    (and ``.env.example`` lists them), but no module reads those constants, and the
    100-char ``trial_phase``/``study_type`` caps have no config twin at all — this
    file stays stdlib+pydantic-only at import time, so setting those env vars does
    NOT move these caps. Treat them as documentation of the values, not as knobs.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., max_length=500)  # REQUIRED; non-empty after strip (see validator)
    drug_name: str | None = Field(default=None, max_length=200)  # authoritative drug dimension (CC-1)
    condition: str | None = Field(default=None, max_length=200)  # authoritative condition dimension
    sponsor: str | None = Field(default=None, max_length=200)  # authoritative sponsor dimension
    country: str | None = Field(default=None, max_length=200)  # authoritative country dimension
    trial_phase: str | None = Field(default=None, max_length=100)  # human phase string; tokenized later
    study_type: str | None = Field(default=None, max_length=100)  # study-type hint; tokenized later
    start_year: int | None = Field(default=None, ge=1900, le=2100)  # inclusive lower year bound
    end_year: int | None = Field(default=None, ge=1900, le=2100)  # inclusive upper year bound
    interventional_only: bool = False  # CC-5/E-38 interventional-denominator toggle

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, v: str) -> str:
        """Strip surrounding whitespace and reject an empty/whitespace-only query (E-17)."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must be a non-empty string")
        return stripped

    @field_validator("trial_phase")
    @classmethod
    def _trial_phase_known(cls, v: str | None) -> str | None:
        """Reject a ``trial_phase`` that names no real phase (E-16 → 422 with the
        valid list). Uses the single normalizer (``app.ctgov.phases``); a deferred
        import keeps this module's top-level imports stdlib+pydantic-only (the wire
        layer stays the lowest layer). Returns the original human string on success
        — the planner re-normalizes it to authoritative tokens for CC-1 precedence."""
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            return None  # whitespace-only → treat as unset, not an error
        from app.ctgov.phases import normalize_trial_phase  # deferred (avoid layer inversion)

        normalize_trial_phase(stripped)  # raises InvalidTrialPhase (ValueError) → 422
        return stripped

    @model_validator(mode="after")
    def _year_range_ordered(self) -> VisualizeRequest:
        """Reject an inverted year range when both bounds are given (→ 422).

        Scope caveat: this is the ONLY ordering guard in the system, and it only sees
        the typed request fields. A ``start_year``/``end_year`` pair the planner reads
        out of the NL instead is not re-checked — ``app.plan.checker`` skips year keys
        (they have no token set) and ``app.ctgov.params._validate_year`` checks type
        and the [1900, 2100] range but not ordering — so an inverted planner-emitted
        pair ships as an inverted ``AREA[StartDate]RANGE[...]`` clause instead of a 422.
        """
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must be <= end_year")
        return self
