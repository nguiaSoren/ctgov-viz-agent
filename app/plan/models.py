"""Internal typed Plan models — the planner's structured output and the objects
that flow between the graph nodes.

These are **internal** contracts (planner → checker → executor), distinct from the
external wire schema in ``app.api.schemas``. The only cross-import is ``ChartType``
(the chart enum is shared, never duplicated — CC-10 identity).

What lives here:

* ``Plan`` — the typed plan the planner emits (ARCHITECTURE_SPEC §3.2). It
  generalizes across all six query classes: the single-set classes
  (``distribution``/``timeseries``/``geographic``/``single_value``) use
  ``entities``/``filters``; ``compare`` uses ``series`` (G-24); ``network`` uses
  ``network`` (G-24).
* ``Observation`` — the typed tool→planner summary from ARCHITECTURE_SPEC §B.2:
  a **computed, bounded** result, never raw records, never a stack trace.
  **Declared here and instantiated nowhere** (see its class docstring): the v1
  planner is single-shot, so no tool result is ever handed back to it.
* ``CheckResult`` — the Plan Checker's verdict (§3.3). On pass it carries back the
  *same* Plan object it was given; nothing is normalized at that seam (see its
  class docstring).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.api.schemas import ChartType

# The query classes the recipe registry is keyed on (ARCHITECTURE_SPEC §B.6). Six as of
# Phase 4: the five chart classes + ``single_value`` (the no-viz / scalar path, CC-7 — a
# scalar count as a stat card, or a yes/no as ``kind:"answer"``; assignment step-3 "identify
# IF a visualization is needed"). Adding a class is mostly a config row (a RECIPES entry
# fixes its marks, fallback and conventions), but not only that: it also needs a shape
# branch in ``_check_class_shape`` and a dispatch branch in the executor — the data table
# holds the policy, the if/elif holds the wiring.
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
    filters: dict[str, Any] = {}  # series-specific filters (overallStatus, year range, …)


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
    * single_value → ``entities`` + ``filters`` only (a scalar ``count_trials``,
      CC-7): ``field``/``series``/``network`` are never read on this path. Note the
      checker requires nothing of this class, so a stray ``field`` is ignored rather
      than rejected.
    """

    query_class: QueryClass  # which recipe (ARCHITECTURE_SPEC §B.6)
    entities: dict[str, str] = {}  # resolved dimensions, e.g. {"condition": "melanoma"}
    filters: dict[str, Any] = {}  # validated filters (overallStatus, year range, studyType, …)
    field: str | None = None  # aggregation field, e.g. "phase" / "country" / "interventionType"
    date_field: DateField | None = None  # the chosen date field for a time series (CC-4)
    grain: Literal["year", "month"] | None = None  # time-series bucket grain
    chart_type: ChartType  # the proposed mark (must be in the recipe's allowed set, CC-8)
    # Other apt marks the frontend may offer (CC-8). Planner-side only: the Plan
    # Checker validates ``chart_type`` but NOT this list, and it is not on the wire
    # (VisualizeResponse/Visualization carry no ``alternates`` field, and the cache
    # key deliberately excludes it) — so CC-8's "return chosen + alternates" is
    # half-implemented: chosen ships, alternates stay internal.
    alternates: list[ChartType] = []
    series: list[Series] | None = None  # compare arms (G-24); None for non-compare classes
    network: NetworkSpec | None = None  # network spec (G-24); None for non-network classes
    interventional_only: bool = False  # CC-5/E-38 interventional-denominator toggle
    # single_value only (CC-7): "visualization" -> a scalar stat card; "answer" -> a yes/no
    # kind:"answer". None for every other class. The executor reads it to shape the no-viz path.
    answer_kind: Literal["visualization", "answer"] | None = None
    notes: list[str] = []  # planner interpretation notes (e.g. a CC-1 override echo)


class Observation(BaseModel):
    """The designed tool→planner summary (ARCHITECTURE_SPEC §B.2) — **declared
    here and instantiated nowhere**: repo-wide, ``Observation`` appears only in
    this module (no importer in ``app/``, ``tests/`` or ``scripts/``).

    It is the shape a tool would hand back to a *multi-turn* planner: a computed,
    bounded summary — a total and a small preview, never the underlying rows,
    never a stack trace. The shipped v1 planner never observes one, because it
    never takes a second turn: ``plan_request`` makes a single structured-output
    call with ``tools=None``, the executor sends tool results straight to the spec
    builder, and the one bounded re-plan carries its feedback as a plain string
    (``plan_feedback``). So this class does not hold the "LLM never counts"
    invariant — that is held by the executor inserting every number into the
    envelope from a tool result (CC-16) and by the Output Reviewer re-verifying
    it. Kept as the typed contract a re-opened loop would bind to, not as live
    code.
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

    On ``ok``, ``normalized_plan`` is the *same object* that was passed in —
    ``check_plan`` validates and echoes, it never rewrites a token or a field
    (``check_plan(plan).normalized_plan is plan`` → True). The name is
    aspirational: normalization really happens one layer upstream in the planner
    (``_normalize_field`` for the aggregation-field spelling, ``normalize_trial_phase``
    inside ``_apply_field_precedence`` for phase strings → wire tokens), before the
    plan ever reaches the checker. Nothing in ``app/`` reads the field either: the
    graph stores the whole ``CheckResult`` in ``state["validation"]`` and keeps
    using ``state["plan"]``; only tests assert on it.

    On reject, ``reason`` is a precise machine message the planner can escalate on
    (one bounded re-plan)."""

    ok: bool  # did the plan pass mechanical validation?
    reason: str | None = None  # precise machine reason on reject (drives the re-plan)
    normalized_plan: Plan | None = None  # on ok: the SAME Plan object, unmodified
