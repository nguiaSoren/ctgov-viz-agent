"""The planner — the ReAct agent whose classify->fill loop lives here (ARCHITECTURE_SPEC §3.2).

Two moves, no more: **classify** the question into exactly one of six query classes,
then **fill** that recipe's slots (entities / filters / field / date_field / chart). The
model emits a single typed :class:`PlannerOutput` (a CLOSED structured object — no free-form
``dict[str, Any]``, so a made-up filter key literally cannot be represented); code maps that
to the internal :class:`~app.plan.models.Plan` and the deterministic tools compute every
number. **The LLM never counts, pages, or aggregates** — it only decides *what* to compute.

Why a closed :class:`PlannerOutput` rather than emitting a ``Plan`` directly: the LLM-facing
schema is the anti-hallucination boundary. Its filter vocabulary is a fixed set of typed keys
(``phase``/``overall_status``/…), so the model can only choose real filters; ``to_plan``
re-spells those keys into the exact strings the Plan Checker's ``KNOWN_FILTER_KEYS`` expects.

CC-1 field precedence: the structured request fields (``drug_name``/``condition``/… ) are
authoritative for the dimension they name; the NL query supplies intent and gap-fills. When a
typed field and the query disagree on a dimension, the field wins and an override note is
echoed into ``Plan.notes``. This is re-asserted in code (``_apply_field_precedence``) after the
LLM returns, so precedence holds regardless of what the model did — with one correctness guard:
a typed value is only written into a *token-constrained* filter (phase / study_type) when it is
already a legal wire token, never a raw human string that the Plan Checker would reject.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.api.schemas import ChartType
from app.ctgov.enums import (
    DATE_FIELDS,
    INTERVENTION_TYPE_TOKENS,
    OVERALL_STATUS_TOKENS,
    PHASE_TOKENS,
    SPONSOR_CLASS_TOKENS,
    STUDY_TYPE_TOKENS,
)
from app.ctgov.phases import normalize_trial_phase
from app.llm.adapter import LLMAdapter
from app.plan.models import DateField, NetworkSpec, Plan, Series

# Snake_case planner filter key -> the EXACT checker key spelling (KNOWN_FILTER_KEYS).
# Module-level (not a model attribute) so it stays a plain constant, never a Pydantic
# field / private-attr descriptor.
_FILTER_KEY_MAP: dict[str, str] = {
    "phase": "phase",
    "overall_status": "overallStatus",
    "study_type": "studyType",
    "sponsor_class": "sponsorClass",
    "intervention_type": "interventionType",
    "start_year": "start_year",
    "end_year": "end_year",
    "country": "country",
}

# Aggregation-``field`` normalization: the LLM often reaches for the snake_case *filter*-key
# spelling it just saw (e.g. ``overall_status``) where the aggregation ``field`` slot wants the
# camelCase field alias the Plan Checker whitelists (``overallStatus``). Normalize it
# deterministically here so a predictable casing mismatch doesn't burn a re-plan; a genuinely
# unknown field still falls through unchanged and the Checker rejects it (the anti-hallucination
# gate is untouched). Verified against a LIVE compare query that emitted ``overall_status``.
_FIELD_ALIAS_MAP: dict[str, str] = {
    "overall_status": "overallStatus",
    "study_type": "studyType",
    "intervention_type": "interventionType",
    "sponsor_class": "sponsorClass",
}


def _normalize_field(field: str | None) -> str | None:
    """Map a snake_case aggregation-field spelling to its checker-legal alias; pass
    everything else (``phase`` / ``country`` / ``study_duration`` / already-camelCase) through."""
    if field is None:
        return None
    return _FIELD_ALIAS_MAP.get(field, field)

# --- The LLM-facing structured output (strict-schema-safe, CLOSED) ---------


class PlannerEntities(BaseModel):
    """The dimensions the query / structured fields name (all optional).

    Keys are exactly the internal entity-dimension names the Plan Checker whitelists
    (``term``/``condition``/``drug``/``sponsor``/``country`` → ClinicalTrials.gov query
    areas). Closed on purpose: the model cannot name a dimension the tools don't expose.
    """

    condition: str | None = None
    drug: str | None = None
    sponsor: str | None = None
    country: str | None = None
    term: str | None = None


class PlannerFilters(BaseModel):
    """The closed filter vocabulary → maps 1:1 to the checker's ``KNOWN_FILTER_KEYS``.

    Each field carries only real wire tokens (see the token sets enumerated into the
    system prompt); ``to_plan`` re-spells the snake_case keys here to the exact camelCase
    the Plan Checker expects (``overall_status`` → ``overallStatus``, etc.).
    """

    phase: list[str] | None = None  # -> Plan.filters["phase"]  (PHASE1..4/EARLY_PHASE1/NA)
    overall_status: str | None = None  # -> "overallStatus"
    study_type: str | None = None  # -> "studyType"
    sponsor_class: str | None = None  # -> "sponsorClass"
    intervention_type: str | None = None  # -> "interventionType"
    start_year: int | None = None  # -> "start_year"
    end_year: int | None = None  # -> "end_year"
    country: str | None = None  # -> "country" (a location is both a dimension and a filter)


class PlannerSeries(BaseModel):
    """One arm of a ``compare`` plan — a labelled, independently-scoped query set (G-24)."""

    label: str
    entities: PlannerEntities
    filters: PlannerFilters | None = None


class PlannerNetwork(BaseModel):
    """The graph spec for a ``network`` plan (bipartite sponsor↔drug or drug↔drug)."""

    kind: Literal["sponsor_drug", "drug_drug"]
    entity_a: str | None = None
    entity_b: str | None = None


class PlannerOutput(BaseModel):
    """The single typed object the planner LLM emits (ARCHITECTURE_SPEC §3.2).

    Covers all six query classes. Closed keys only — no ``dict[str, Any]`` — so the
    strict-mode JSON schema the real adapters send makes a hallucinated filter key
    unrepresentable at the schema layer. ``to_plan`` lowers it to the internal ``Plan``.
    """

    query_class: Literal[
        "distribution", "timeseries", "compare", "geographic", "network", "single_value"
    ]
    entities: PlannerEntities = PlannerEntities()
    filters: PlannerFilters = PlannerFilters()
    field: str | None = None  # aggregation field, e.g. "phase" / "country" / "study_duration"
    date_field: DateField | None = None
    grain: Literal["year", "month"] | None = None
    chart_type: ChartType
    alternates: list[ChartType] = []
    series: list[PlannerSeries] | None = None
    network: PlannerNetwork | None = None
    interventional_only: bool = False
    # single_value only: a scalar count renders as a stat card ("visualization");
    # a yes/no question renders as text ("answer"). Ignored for the other classes.
    answer_kind: Literal["visualization", "answer"] | None = None
    notes: list[str] = []  # planner interpretation notes (CC-1 override echoes land here)

    @staticmethod
    def _entities_to_dict(entities: PlannerEntities) -> dict[str, str]:
        """Set entity fields → a plain ``{dimension: value}`` dict (checker entity keys)."""
        out: dict[str, str] = {}
        for key in ("term", "condition", "drug", "sponsor", "country"):
            value = getattr(entities, key)
            if value is not None:
                out[key] = value
        return out

    @staticmethod
    def _filters_to_dict(filters: PlannerFilters | None) -> dict[str, Any]:
        """Set filter fields → a plain dict using the EXACT ``KNOWN_FILTER_KEYS`` spelling."""
        out: dict[str, Any] = {}
        if filters is None:
            return out
        for planner_key, checker_key in _FILTER_KEY_MAP.items():
            value = getattr(filters, planner_key)
            if value is not None:
                out[checker_key] = value
        return out

    def to_plan(self) -> Plan:
        """Lower this closed LLM output to the internal :class:`~app.plan.models.Plan`.

        Entities/filters become plain dicts keyed by the exact names the Plan Checker
        validates against; ``series``/``network`` become their internal counterparts;
        ``interventional_only`` rides on the Plan itself (NOT in ``filters`` — the executor
        applies it). Every other slot passes through unchanged.
        """
        series: list[Series] | None = None
        if self.series is not None:
            series = [
                Series(
                    label=arm.label,
                    entities=self._entities_to_dict(arm.entities),
                    filters=self._filters_to_dict(arm.filters),
                )
                for arm in self.series
            ]

        network: NetworkSpec | None = None
        if self.network is not None:
            network = NetworkSpec(
                kind=self.network.kind,
                entity_a=self.network.entity_a,
                entity_b=self.network.entity_b,
            )

        return Plan(
            query_class=self.query_class,
            entities=self._entities_to_dict(self.entities),
            filters=self._filters_to_dict(self.filters),
            field=_normalize_field(self.field),
            date_field=self.date_field,
            grain=self.grain,
            chart_type=self.chart_type,
            alternates=list(self.alternates),
            series=series,
            network=network,
            interventional_only=self.interventional_only,
            answer_kind=self.answer_kind,
            notes=list(self.notes),
        )


# --- The system prompt: 6 classes + recipe menu + real token vocab + CC-1 ---

# Enumerated from app.ctgov.enums so the model can only pick REAL tokens. Sorted for a
# stable prompt (frozensets have no order); this is read-only use, we touch nothing there.
_PHASE = ", ".join(sorted(PHASE_TOKENS))
_STATUS = ", ".join(sorted(OVERALL_STATUS_TOKENS))
_STUDY_TYPE = ", ".join(sorted(STUDY_TYPE_TOKENS))
_SPONSOR_CLASS = ", ".join(sorted(SPONSOR_CLASS_TOKENS))
_INTERVENTION_TYPE = ", ".join(sorted(INTERVENTION_TYPE_TOKENS))
_DATE_FIELD_TOKENS = ", ".join(sorted(DATE_FIELDS))

PLANNER_SYSTEM_PROMPT = f"""\
You are the ClinicalTrials.gov query planner. Read the user's natural-language question plus \
any structured fields and produce ONE typed plan. You make exactly two decisions: (1) CLASSIFY \
the question into one query class, then (2) FILL that class's slots.

HARD INVARIANT: You NEVER count, page, aggregate, or state a number. A deterministic tool \
computes every number afterward. Your job is only to choose the class, the entities/filters, \
the aggregation field, the date field, and the chart. If you emit a number, you are wrong.

THE SIX QUERY CLASSES (each maps to one recipe = one tool + a default chart):
- distribution  — categorical counts over one field (phase, study type, intervention type, \
sponsor class). Tool: aggregate_by. Chart: bar (alt: histogram, table). Requires `field`.
- timeseries    — a count binned over time. Tool: timeseries. Chart: time_series (alt: bar). \
Requires `date_field` AND `grain` ("year" or "month").
- compare       — two or more independently-scoped arms bucketed on the same field. \
Tool: compare. Chart: grouped_bar (alt: bar). Requires `series` (>=2 arms) AND `field`.
- geographic    — counts by country. Tool: aggregate_by. Chart: bar (alt: table). \
Requires `field` == "country".
- network       — a co-occurrence graph (sponsor<->drug or drug<->drug). Tool: build_network. \
Chart: network_graph (alt: bar). Requires `network` with a `kind`.
- single_value  — a single scalar the user asked for ("how many X", a yes/no). \
Tool: count_trials. Chart: single_value (alt: table). No `field`, no series/network — just \
the entities/filters that scope the count. Set `answer_kind`: "visualization" for a scalar \
count ("how many X trials" -> a stat card) or "answer" for a yes/no ("are there any X trials").

REAL TOKEN VOCABULARY — use ONLY these exact tokens (a combined phase is a LIST, e.g. \
["PHASE1","PHASE2"], never the string "PHASE1/PHASE2"):
- phase:              {_PHASE}
- overall_status:     {_STATUS}
- study_type:         {_STUDY_TYPE}
- sponsor_class:      {_SPONSOR_CLASS}
- intervention_type:  {_INTERVENTION_TYPE}
- date_field (pick per intent — started->startDate, registered->studyFirstPostDate, \
completed->primaryCompletionDate/completionDate, updated->lastUpdatePostDate): \
{_DATE_FIELD_TOKENS}
- start_year / end_year are inclusive integer year bounds.

ENTITIES name a dimension, not a filter: condition (a disease), drug (an intervention name), \
sponsor (an organization name), country (a location), term (free-text catch-all). Put the \
subject of the question in the right one; do not invent a filter for something that is a \
dimension.

CC-1 FIELD PRECEDENCE: fill `entities` with the value THE QUERY TEXT literally names on each \
dimension — report what the question says; do NOT substitute a structured field's value into an \
entity yourself, and do not drop a dimension the query names just because a structured field \
disagrees. Deterministic code runs AFTER you and applies precedence (a structured field wins on \
its dimension) and records the override note, so you never reconcile a conflict yourself: if the \
query says "Keytruda" and a drug_name field says "nivolumab", set drug="Keytruda" (the query's \
word) — the code overrides to nivolumab and discloses the override. The structured fields are \
still authoritative (the code enforces that); just report the question faithfully.

Return the typed plan object and nothing else."""


# --- CC-1 precedence + the offline stub's canned answer --------------------


def _apply_field_precedence(plan: PlannerOutput, merged_inputs: dict[str, Any]) -> None:
    """Re-assert CC-1: typed structured fields win on the dimension they name (in place).

    For each authoritative typed field present in ``merged_inputs``, overwrite the mapped
    entity/filter on ``plan`` and — when the planner had emitted a *different* value for that
    same dimension — append an override echo to ``plan.notes``. Correctness guard: a typed
    value is only written into a token-constrained filter (``phase``/``study_type``) when it is
    already a legal wire token; a raw human phase string ("Phase 1") is left for the LLM's
    already-tokenized choice rather than injected verbatim (the Plan Checker rejects non-tokens).
    """

    def _note_override(dimension: str, field_key: str, new: object, old: object) -> None:
        plan.notes.append(
            f"Override: used field {field_key}={new!r} over query {dimension} {old!r}."
        )

    # (a) entity dimensions — free text, so the typed value is always a legal plan value.
    for field_key, attr in (
        ("condition", "condition"),
        ("drug_name", "drug"),
        ("sponsor", "sponsor"),
        ("country", "country"),
    ):
        typed = merged_inputs.get(field_key)
        if typed in (None, ""):
            continue
        current = getattr(plan.entities, attr)
        if current is not None and str(current) != str(typed):
            _note_override(attr, field_key, typed, current)
        setattr(plan.entities, attr, typed)

    # (b) numeric year bounds — directly legal filter values.
    for field_key in ("start_year", "end_year"):
        typed = merged_inputs.get(field_key)
        if typed is None:
            continue
        current = getattr(plan.filters, field_key)
        if current is not None and current != typed:
            _note_override(field_key, field_key, typed, current)
        setattr(plan.filters, field_key, typed)

    # (c) interventional_only rides on the Plan itself, never in filters.
    interventional_only = merged_inputs.get("interventional_only")
    if interventional_only is not None:
        plan.interventional_only = bool(interventional_only)

    # (d) token-constrained filters — write ONLY if the typed value is already a legal token,
    #     else defer to the LLM's tokenized choice (never inject a checker-rejecting raw string).
    trial_phase = merged_inputs.get("trial_phase")
    if trial_phase not in (None, "", []):
        # The structured trial_phase is authoritative (CC-1): normalize the human
        # string to wire tokens via the SAME normalizer the request layer validated
        # with (so "Phase 1/2" → ["PHASE1","PHASE2"]). The request layer already
        # rejected an un-normalizable value (422), so this won't raise in practice;
        # stay defensive (skip on any residual failure) and keep the checker's
        # anti-hallucination gate the final word (only real tokens are applied).
        try:
            if isinstance(trial_phase, list):
                tokens = [str(p) for p in trial_phase]
            else:
                tokens = normalize_trial_phase(trial_phase)
        except ValueError:
            tokens = []
        if tokens and all(tok in PHASE_TOKENS for tok in tokens):
            current = plan.filters.phase
            if current is not None and list(current) != tokens:
                _note_override("phase", "trial_phase", tokens, current)
            plan.filters.phase = tokens

    study_type = merged_inputs.get("study_type")
    if study_type not in (None, ""):
        token = str(study_type).strip().upper()  # a study-type hint is just an uppercased word
        if token in STUDY_TYPE_TOKENS:
            current = plan.filters.study_type
            if current is not None and current != token:
                _note_override("study_type", "study_type", token, current)
            plan.filters.study_type = token


def _canned_planner_output(merged_inputs: dict[str, Any]) -> dict[str, Any]:
    """The offline StubAdapter's deterministic answer: a legal distribution-by-phase plan.

    Real OpenAI/Anthropic adapters IGNORE this and make a real call; the stub validates it into
    :class:`PlannerOutput`. Kept simple (distribution over the given/So-fallback condition) — it
    only has to be a checker-legal plan; CC-1 then layers the typed fields on top.
    """
    condition = merged_inputs.get("condition") or "pancreatic cancer"
    return {
        "query_class": "distribution",
        "entities": {"condition": condition},
        "filters": {},
        "field": "phase",
        "date_field": None,
        "grain": None,
        "chart_type": ChartType.BAR,
        "alternates": [ChartType.HISTOGRAM, ChartType.TABLE],
        "series": None,
        "network": None,
        "interventional_only": bool(merged_inputs.get("interventional_only", False)),
        "answer_kind": None,
        "notes": ["Offline stub plan: distribution-by-phase over the condition."],
    }


def _build_user_message(merged_inputs: dict[str, Any], feedback: str | None) -> str:
    """The single user message: the NL query + the present structured fields + any feedback."""
    lines: list[str] = []
    query = merged_inputs.get("query")
    lines.append(f"User question: {query}" if query else "User question: (none provided)")

    structured = [
        f"  - {key}: {merged_inputs[key]!r}"
        for key in (
            "drug_name",
            "condition",
            "sponsor",
            "country",
            "trial_phase",
            "study_type",
            "interventional_only",
            "start_year",
            "end_year",
        )
        if merged_inputs.get(key) not in (None, "")
    ]
    if structured:
        lines.append("Structured fields (authoritative on their dimension — CC-1):")
        lines.extend(structured)

    if feedback:
        lines.append(f"A previous attempt was rejected: {feedback}. Fix it.")
    return "\n".join(lines)


def plan_request(
    adapter: LLMAdapter,
    merged_inputs: dict[str, Any],
    feedback: str | None = None,
) -> Plan:
    """Classify the request into a typed :class:`~app.plan.models.Plan` (ARCHITECTURE_SPEC §3.2).

    Builds the system prompt (six classes + recipe menu + real token vocab + CC-1) and one user
    message (NL query + present structured fields + an optional ``feedback`` re-plan line), then
    asks the adapter to ``propose`` a :class:`PlannerOutput`. Offline the StubAdapter returns the
    deterministic ``canned`` distribution plan; live, a real adapter ignores ``canned`` and calls
    the model. CC-1 field precedence is re-asserted in code before lowering to a ``Plan``.

    ``feedback`` (when set) threads the prior rejection reason into the prompt — the reason→act→
    observe re-plan. The function stays importable and network-free; the graph passes a
    StubAdapter offline and a real adapter live.
    """
    user_message = _build_user_message(merged_inputs, feedback)
    proposed = adapter.propose(
        system=PLANNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        response_model=PlannerOutput,
        tools=None,
        model=None,
        canned=_canned_planner_output(merged_inputs),
    )
    # propose always returns a validated instance of response_model (adapter contract).
    assert isinstance(proposed, PlannerOutput)
    _apply_field_precedence(proposed, merged_inputs)
    return proposed.to_plan()
