"""The recipe / skill registry (ARCHITECTURE_SPEC §B.6) — a data-only registry.

Rather than let the LLM improvise *how* to handle each of the 6 query classes,
each class gets a predefined **recipe**: a fixed, deterministic procedure
(allowed tools, default + alternate chart types, whether a date field must be
disclosed, the counting convention, a degeneracy fallback). The registry is
**data (a config table), not code branches** — the per-class *policy* is a row
here rather than a branch scattered through the codebase, which is the literal
reading of "single coherent approach, no one-off hacks" (CC-11). A new class is
not a pure config change, though: it also needs a shape branch in the checker's
``_check_class_shape`` and a dispatch branch in the executor.

The planner's job shrinks to two moves: classify the question -> ``query_class``,
then fill that recipe's slots. The Plan Checker (``app.plan.checker``) then
confirms the filled Plan satisfies the chosen recipe's constraints.

How much of a row is actually *executed* (be precise about this — the table reads
more load-bearing than it is):

* ``chart_type`` / ``alternates`` / ``degeneracy_fallback`` — read at runtime by
  ``check_plan``, which builds the allowed-chart set from exactly those three.
  This is the only runtime consumer of a Recipe.
* ``allowed_tools`` — no runtime reader. It is enforced at BUILD time by
  ``tests/test_ctgov_plan.py`` (every name must be in ``tools.TOOL_NAMES``), and
  the executor's real dispatch is a hand-written ``if/elif`` on ``query_class``.
* ``date_field_disclosed`` / ``count_basis_rule`` — no reader at all, runtime or
  test. Disclosure is driven straight off ``query_class`` in the spec builder and
  the count basis is computed from the buckets. They document the convention; they
  do not enforce it.
* ``notes`` — read by ``tests/test_phase2_traps.py``, which pins the prose
  disclosures (dedupe/no-choropleth, placebo/synonym, %-within-series, planned
  dates) so a doc edit can't silently drop one.

Not every tool a class can reach is listed in its ``allowed_tools``: every class
first calls ``count_trials`` as the exact-count oracle, a ``study_duration``
distribution runs ``study_duration_histogram``, and an over-budget categorical
distribution runs ``aggregate_by_counts``. Treat the field as "the recipe's
primary tool", not as a complete capability grant.

The registry is also narrower than the chart enum by one mark: no recipe emits
``ChartType.SCATTER``. It is a deferred member kept for completeness (G-20 — trials
have no generic pair of continuous axes to plot), so a plan can never legally
select it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.schemas import ChartType
from app.plan.models import QueryClass


class Recipe(BaseModel):
    """One row of the recipe registry, keyed by ``query_class``.

    ``chart_type`` is the default mark; ``alternates`` are other apt marks the
    frontend may offer (CC-8). ``degeneracy_fallback``, when set, is the mark
    a recipe degrades to when its data is too sparse for the default (e.g. a
    1-node network falling back to a bar, CC-12). Those three are the fields the
    Plan Checker actually reads; the rest are declarative (see the module
    docstring for exactly who reads what).
    """

    query_class: QueryClass
    # The recipe's primary tool(s), by TOOL_REGISTRY name. Checked at build time by
    # tests (⊆ TOOL_NAMES); no runtime reader — dispatch is an if/elif on query_class.
    allowed_tools: list[str]
    chart_type: ChartType
    alternates: list[ChartType] = Field(default_factory=list)
    # Documentation of the convention, not its enforcement: the spec builder decides
    # date-field disclosure from query_class == "timeseries", and derives the count
    # basis from the buckets. Neither field has a reader.
    date_field_disclosed: bool = False
    count_basis_rule: str
    degeneracy_fallback: ChartType | None = None
    notes: str = ""  # prose disclosures; pinned by tests/test_phase2_traps.py


RECIPES: dict[str, Recipe] = {
    "distribution": Recipe(
        query_class="distribution",
        allowed_tools=["aggregate_by"],
        chart_type=ChartType.BAR,
        alternates=[ChartType.HISTOGRAM, ChartType.TABLE],
        date_field_disclosed=False,
        count_basis_rule="distinct+mentions",
        degeneracy_fallback=None,
        notes=(
            "Categorical counts (e.g. phase, intervention type). Always carries an "
            "explicit NA/Unknown bucket (CC-5) rather than silently dropping "
            "unphased/untyped trials; a combined multi-value token (e.g. "
            '["PHASE1","PHASE2"]) gets its own composite bucket (CC-15), never split '
            "into two bars. Distinct-trial count reconciles against countTotal; the "
            "mention count is the honest per-membership tally for multi-value fields."
        ),
    ),
    "timeseries": Recipe(
        query_class="timeseries",
        allowed_tools=["timeseries"],
        chart_type=ChartType.TIME_SERIES,
        alternates=[ChartType.BAR],
        date_field_disclosed=True,
        count_basis_rule="distinct+mentions",
        degeneracy_fallback=ChartType.BAR,
        notes=(
            "The planner picks one of the 5 real date fields per intent (started -> "
            "startDate, registered -> studyFirstPostDate, completed -> "
            "primaryCompletionDate) and the choice is always disclosed via "
            "meta.date_field_used (CC-4). Genuine future/estimated dates are flagged "
            'as a "planned" bucket, never clamped into the current year (G-40). A '
            "single-period series degrades to the bar fallback rather than a "
            "1-point line."
        ),
    ),
    "compare": Recipe(
        query_class="compare",
        allowed_tools=["compare"],
        chart_type=ChartType.GROUPED_BAR,
        alternates=[ChartType.BAR],
        date_field_disclosed=False,
        count_basis_rule="% within series",
        degeneracy_fallback=None,
        notes=(
            "Two independently-filtered series (Plan.series, >=2 arms) aggregated on "
            "the same field, categories unioned with an explicit 0-fill on either "
            "side. Default view is percentage-within-each-series so a large-N series "
            "does not visually swamp a small-N one (CC-14); each series' own total N "
            "is labelled, and raw counts remain available per bucket."
        ),
    ),
    "geographic": Recipe(
        query_class="geographic",
        allowed_tools=["aggregate_by"],
        chart_type=ChartType.BAR,
        alternates=[ChartType.TABLE],
        date_field_disclosed=False,
        count_basis_rule="distinct+mentions",
        degeneracy_fallback=None,
        notes=(
            "Ranked horizontal bar (top-N + Other), never a choropleth — country is a "
            "free-text display string with no ISO code (226 unique values, ~60k "
            "missing). A trial with multiple locations is deduped to one count per "
            "distinct (trial, country) pair before bucketing (CC-13) — the trap this "
            "recipe exists to close."
        ),
    ),
    "network": Recipe(
        query_class="network",
        allowed_tools=["build_network"],
        chart_type=ChartType.NETWORK_GRAPH,
        alternates=[ChartType.BAR],
        date_field_disclosed=False,
        count_basis_rule="trial_pairs",
        degeneracy_fallback=ChartType.BAR,
        notes=(
            "Bipartite sponsor<->drug or drug<->drug co-occurrence graph; an edge "
            "weight is the count of trials pairing its two endpoints (CC-3's counting "
            "convention extended to graphs). Drug-name synonyms are merged "
            "CONSERVATIVELY (P3-MERGE), not by unioning shared otherName tokens: an "
            "otherName merges two interventions only when it is itself some other "
            "drug's primary name (the brand->generic case), a combination/regimen "
            "intervention never merges its own components, and a pair must be attested "
            "by >=2 distinct trials before it merges — the earlier union-any-shared-"
            "token rule over-merged distinct drugs through a shared protocol code. "
            "Under-merging is the safe direction and is disclosed in meta.notes. "
            "placebo/standard-of-care nodes are excluded (avoids "
            "a false mega-hub), nodes are capped top-N by degree with a minimum edge "
            "weight (CC-12). A degenerate result (≤1 node OR no co-occurring edges, "
            "G-41e) falls back to a cited bar of individual drug frequencies with a "
            '"too sparse to graph" note; never emitted as Vega-Lite.'
        ),
    ),
    "single_value": Recipe(
        query_class="single_value",
        allowed_tools=["count_trials"],
        chart_type=ChartType.SINGLE_VALUE,
        alternates=[ChartType.TABLE],
        date_field_disclosed=False,
        count_basis_rule="distinct",  # one exact countTotal, no bucketing
        degeneracy_fallback=None,
        notes=(
            "The no-viz / scalar path (assignment step-3 'identify IF a visualization "
            "is needed', CC-7). A scalar count question (\"how many X trials?\") or a "
            "yes/no (\"are there any X trials?\"). The single count_trials tool returns "
            "the exact countTotal over the plan's entities/filters — no aggregation "
            "field, no bucketing, no series/network. A scalar count renders as a stat "
            "card (kind:visualization, a SINGLE_VALUE mark); a yes/no renders as "
            "kind:answer. Either way the NUMBER is the exact countTotal inserted by "
            "code from the count_trials result (CC-16), never LLM-authored; the "
            "citation sample proves membership in the counted set (this nctId was "
            "counted) — a legitimate 'this trial is in the count' provenance, not an "
            "excerpt of an aggregation field."
        ),
    ),
}


def get_recipe(query_class: str) -> Recipe:
    """Look up the recipe for ``query_class``. Raises ``KeyError`` (with a
    precise message) if no recipe is registered — a query_class the registry
    doesn't know about is itself a mechanical validation failure upstream."""
    try:
        return RECIPES[query_class]
    except KeyError as exc:
        raise KeyError(f"no recipe registered for query_class={query_class!r}") from exc
