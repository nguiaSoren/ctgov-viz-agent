"""API I/O contracts — the request model and the full response envelope.

This is the **lowest layer** of the system: it imports stdlib + pydantic only and
nothing from ``app.*``. Every other module (planner, checker, executor, viz-spec
builder, the FastAPI transport) imports its wire types from here, so the names in
this file are a **frozen interface contract** — do not rename them.

What lives here:

* The status / kind discriminators (``Status``, ``Kind``) and the closed chart
  enum (``ChartType``) — ARCHITECTURE_SPEC §B.3 / CC-10.
* The shared viz value objects (``CategoryValue``, ``Citation``, ``Datum``,
  ``Node``, ``Edge``, ``NetworkData``, ``EncodingChannel``, ``Visualization``).
* The response envelope (``VisualizeResponse``) — ARCHITECTURE_SPEC §6 — plus its
  ``meta`` sub-objects (``CountBasis``, ``Partial``, ``ErrorObj``, ``Meta``).
* The request model (``VisualizeRequest``) — the documented, per-field-validated
  input schema (A-22 "document the request schema").

Design invariant this schema enforces structurally (ARCHITECTURE_SPEC §1, G-30):
the LLM never emits a number. Numbers reach the user only through ``Datum`` /
``CountBasis`` fields that the deterministic aggregation core fills, and the
prose fields (``title``, scalar ``answer``, ``meta.notes``) are code-templated.
The schema cannot enforce "code-templated" on its own, but it is the single place
those fields are declared, so the guarantee has one home.
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


class CategoryValue(BaseModel):
    """A ``{value, label}`` pair (CC-10): identity/sort on the raw ``value``,
    display on the ``label``. The backend owns the token→label map so the wire
    value never drifts from what the API returned."""

    value: str  # raw registry token, e.g. "PHASE1"
    label: str  # human display, e.g. "Phase 1"


class Citation(BaseModel):
    """A single per-datum provenance record (CC-9 / A-45/A-46).

    ``excerpt`` is **string-extracted** from the fetched record at ``field_path``,
    never authored by the LLM — the Output Reviewer asserts it is a real substring
    of the source at that path. It proves *membership* (this NCT is in this
    bucket); the count is proven by the cardinality of the contributing set.

    Citation invariant: ``excerpt`` is always a verbatim substring of the source
    record at ``field_path``; when a bucket is formed from MULTIPLE source tokens
    (a composite bucket, e.g. ``PHASE1|PHASE2``, CC-15), the additional verified
    literals ride in ``excerpt_tokens`` (each also a verbatim element at
    ``field_path``); no synthesized excerpt text is ever generated. ``excerpt``
    stays the FIRST member token for display/back-compat.
    """

    nct_id: str  # e.g. "NCT01234567"
    field_path: str  # JSON path that decided membership, e.g. "protocolSection.designModule.phases"
    value: Any  # the literal field value at that path (may be an array, e.g. ["PHASE1"])
    excerpt: str  # a real substring of the source at field_path (round-trip verifiable)
    # For a composite bucket: every verified member literal (each a verbatim element at
    # field_path), in token order. None for single-value buckets (back-compat).
    excerpt_tokens: list[str] | None = None


class Datum(BaseModel):
    """One row of a chart's ``data`` array.

    Carries the categorical identity (``value``/``label``), any class-specific
    channel fields (``period`` for time series, ``series`` for compare,
    ``bin_start``/``bin_end`` for histograms), the **dual counts** (CC-3), and its
    inline provenance list (G-25/G-34 — the load-bearing citation surface; the
    top-level ``citations{}`` map is only a dedup index).

    ``extra="allow"`` lets a recipe attach class-specific channels (e.g. a
    ``planned: true`` flag on a future time-series bucket) without a schema change,
    while the named fields below stay a stable contract.
    """

    model_config = ConfigDict(extra="allow")

    value: str  # raw bucket token, e.g. "PHASE1" or a composite "PHASE1/2" (CC-15 combine)
    label: str  # display label, e.g. "Phase 1"

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
    derived: bool = False  # a computed value (rate/edge weight) — cites its members, not an excerpt
    members: list[str] | None = None  # for derived data: the member nctIds the value was derived from


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
    * ``vega_lite`` is a convenience projection for standard charts only; it is
      null for network / answer / clarification / too_large (G-41d).

    These presence rules are conventions the pipeline upholds, not schema
    constraints — the envelope keeps every field optional so a single type serves
    every status.
    """

    status: Status  # ok | empty | too_large | error
    kind: Kind  # visualization | answer | clarification
    visualization: Visualization | None = None  # the custom canonical spec; null on kind:"answer"
    vega_lite: dict | None = None  # Vega-Lite projection for standard charts; null for network/answer/too_large
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
    trial_phase : str | None (optional)
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
    study_type : str | None (optional)
        A study-type hint (e.g. "interventional"); tokenized downstream.
    interventional_only : bool (default False)
        The CC-5/E-38 toggle — when true a phase distribution can offer the
        interventional denominator (observational trials legitimately have no phase).

    Policy: ``extra="forbid"`` — unknown request fields are rejected (→ 422). This
    is a deliberate choice (fail-closed on a malformed request) over silently
    ignoring typo'd field names.
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
        """Reject an inverted year range when both bounds are given (→ 422)."""
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must be <= end_year")
        return self
