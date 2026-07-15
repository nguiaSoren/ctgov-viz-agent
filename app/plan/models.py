"""Internal typed Plan models — the planner's structured output and the objects
that flow between the graph nodes.

These are **internal** contracts (planner → checker → executor), distinct from the
external wire schema in ``app.api.schemas``. The only cross-import is ``ChartType``
(the chart enum is shared, never duplicated — CC-10 identity).

What lives here:

* ``Plan`` — the typed plan the ReAct planner emits (ARCHITECTURE_SPEC §3.2). It
  generalizes across all five query classes: single-set classes use
  ``entities``/``filters``; ``compare`` uses ``series`` (G-24); ``network`` uses
  ``network`` (G-24).
* ``Observation`` — the **computed, bounded** summary a tool returns to the planner
  (ARCHITECTURE_SPEC §B.2). Never raw records, never stack traces — this is what
  holds the "LLM never counts" invariant *through* the ReAct loop.
* ``CheckResult`` — the Plan Checker's verdict (§3.3), carrying the normalized Plan
  when it passes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.api.schemas import ChartType

# The query classes the recipe registry is keyed on (ARCHITECTURE_SPEC §B.6). Six as of
# Phase 4: the five chart classes + ``single_value`` (the no-viz / scalar path, CC-7 — a
# scalar count as a stat card, or a yes/no as ``kind:"answer"``; assignment step-3 "identify
# IF a visualization is needed"). Adding a class stays a config row, not a code branch.
QueryClass = Literal[
    "distribution", "timeseries", "compare", "geographic", "network", "single_value"
]

# The five real date fields on the v2 API (SPEC_INTERROGATION §C). The planner
# selects one per intent (CC-4); the Plan Checker rejects anything outside this set.
DateField = Literal[
    "startDate",
    "primaryCompletionDate",
    "completionDate",
    "studyFirstPostDate",
    "lastUpdatePostDate",
]


class Series(BaseModel):
    """One arm of a ``compare`` plan (G-24) — a labelled, independently-filtered
    query set. A compare plan carries ≥2 of these."""

    label: str  # display label for the series, e.g. "Pembrolizumab"
    entities: dict[str, str] = {}  # dimension → value for this series, e.g. {"drug": "pembrolizumab"}
    filters: dict[str, Any] = {}  # series-specific filters (status, year range, …)


class NetworkSpec(BaseModel):
    """The ``network`` block of a network plan (G-24) — declares which graph to
    build and (optionally) which entities anchor it."""

    kind: Literal["sponsor_drug", "drug_drug"]  # bipartite sponsor↔drug or drug↔drug co-occurrence
    entity_a: str | None = None  # optional anchor entity (e.g. a condition or sponsor)
    entity_b: str | None = None  # optional second anchor entity


class Plan(BaseModel):
    """The typed Plan the planner emits (ARCHITECTURE_SPEC §3.2).

    The planner's job is two moves: pick ``query_class`` (classify) and fill that
    recipe's slots. The Plan Checker (§3.3) then confirms every filled field is
    *legal* before any LLM review or API call. Shapes by class:

    * distribution / timeseries / geographic → ``entities`` + ``filters`` (+
      ``field`` / ``date_field`` / ``grain`` as the class needs).
    * compare → ``series`` (the per-arm sets).
    * network → ``network`` (the graph spec).
    """

    query_class: QueryClass  # which recipe (ARCHITECTURE_SPEC §B.6)
    entities: dict[str, str] = {}  # resolved dimensions, e.g. {"condition": "melanoma"}
    filters: dict[str, Any] = {}  # validated filters (status, year range, study type, …)
    field: str | None = None  # aggregation field, e.g. "phase" / "country" / "interventionType"
    date_field: DateField | None = None  # the chosen date field for a time series (CC-4)
    grain: Literal["year", "month"] | None = None  # time-series bucket grain
    chart_type: ChartType  # the proposed mark (must be in the recipe's allowed set, CC-8)
    alternates: list[ChartType] = []  # other apt marks the frontend may offer (CC-8)
    series: list[Series] | None = None  # compare arms (G-24); None for non-compare classes
    network: NetworkSpec | None = None  # network spec (G-24); None for non-network classes
    interventional_only: bool = False  # CC-5/E-38 interventional-denominator toggle
    # single_value only (CC-7): "visualization" -> a scalar stat card; "answer" -> a yes/no
    # kind:"answer". None for every other class. The executor reads it to shape the no-viz path.
    answer_kind: Literal["visualization", "answer"] | None = None
    notes: list[str] = []  # planner interpretation notes (e.g. a CC-1 override echo)


class Observation(BaseModel):
    """What a tool returns to the planner (ARCHITECTURE_SPEC §B.2).

    A **computed, bounded** summary — never raw records, never a stack trace. This
    is the object that keeps the LLM structurally away from the numbers: the
    planner sees a total and a small preview, never the underlying rows.
    """

    tool: str  # the tool that produced this observation, e.g. "aggregate_by"
    ok: bool  # whether the tool succeeded
    error_code: str | None = None  # a typed actionable-semantic error (e.g. "zero_results")
    total_count: int | None = None  # the exact matching countTotal, when known
    buckets_preview: list[dict] | None = None  # a small, bounded preview of the top buckets
    truncated: bool | None = None  # whether the underlying aggregation hit the page budget
    date_field_used: str | None = None  # which date field a time series actually used (CC-4)
    note: str | None = None  # a short human note (e.g. "no new data" on a memoized repeat)


class CheckResult(BaseModel):
    """The Plan Checker's verdict (ARCHITECTURE_SPEC §3.3).

    On ``ok``, ``normalized_plan`` carries the Plan with tokens/fields normalized
    (e.g. phase strings → wire tokens). On reject, ``reason`` is a precise machine
    message the planner can escalate on (one bounded re-plan)."""

    ok: bool  # did the plan pass mechanical validation?
    reason: str | None = None  # precise machine reason on reject (drives the re-plan)
    normalized_plan: Plan | None = None  # the normalized plan when ok
