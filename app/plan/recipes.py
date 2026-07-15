"""The recipe / skill registry (ARCHITECTURE_SPEC §B.6) — a real, data-only registry.

Rather than let the LLM improvise *how* to handle each of the 5 query classes,
each class gets a predefined **recipe**: a fixed, deterministic procedure
(allowed tools, default + alternate chart types, whether a date field must be
disclosed, the counting convention, a degeneracy fallback). The registry is
**data (a config table), not code branches** — adding a query class means
adding a row here, which is the literal reading of "single coherent approach,
no one-off hacks" (CC-11).

The planner's job shrinks to two moves: classify the question -> ``query_class``,
then fill that recipe's slots. The Plan Checker (``app.plan.checker``) then
confirms the filled Plan satisfies the chosen recipe's constraints.
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
    1-node network falling back to a bar, CC-12).
    """

    query_class: QueryClass
    allowed_tools: list[str]
    chart_type: ChartType
    alternates: list[ChartType] = Field(default_factory=list)
    date_field_disclosed: bool = False
    count_basis_rule: str
    degeneracy_fallback: ChartType | None = None
    notes: str = ""


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
            "convention extended to graphs). Drug names are synonym-merged "
            "(name + otherNames), placebo/standard-of-care nodes are excluded (avoids "
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
