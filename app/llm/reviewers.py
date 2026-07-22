"""The two LLM reviewers — gates, not generators (ARCHITECTURE_SPEC §3.4, §3.8).

Both reviewers emit a bounded verdict and are called *through* the adapter
(C-99), never a provider SDK directly. Their verdict models carry a ``decision``
plus an optional free-text ``reason``: there is no numeric FIELD for a count to
land in, but ``reason`` is prose and prose can carry digits, so the schema alone
does NOT enforce the governing invariant ("the model never emits a number") —
that enforcement is deterministic and lives one module away, on the single path
where reviewer prose reaches the wire. An Output-Reviewer ``flag`` reason is
appended to ``meta.notes`` only if it passes ``note_number_safe``
(``app.viz.review``, applied in ``app.graph.nodes.review_output``): every
digit-run in the note must already appear in the computed envelope, or the whole
reason is dropped for a fixed code-owned caveat. The Intent Reviewer's ``reason``
never reaches the wire at all — it is threaded into the bounded re-plan prompt
and otherwise stays in graph state. (That digit post-check was a found defect,
not the original design: the verdict models were schema-safe but not digit-safe.)

* ``review_intent`` — the Intent Reviewer (§3.4): judges whether a mechanically
  valid ``Plan`` actually captures the user's intent. The Plan Checker already
  proved every token/field/range is *legal*; this reviewer works the semantic
  layer the checker cannot see (right metric, right dimension, right date sense,
  apt chart, faithful filters) and returns ``approve`` or ``revise{field,reason}``
  so a single bounded re-plan has a precise target.
* ``should_skip_intent_review`` — the §3.4 skip gate: the intent review exists to
  catch a *misread NL parse*, so when there was no NL parse to misread it is safe
  (and cheaper) to skip it. Conservative by construction — see its docstring.
* ``review_output_llm`` — the Output Reviewer's LLM half (§3.8): a secondary,
  non-generative check over already-computed output — does the spec faithfully
  answer the question, is the encoding apt? It never re-aggregates and is never
  used to rebuild the spec; a ``flag`` verdict only annotates ``meta.notes``, and
  only through the ``note_number_safe`` digit gate above.

Each real call passes ``canned`` so the offline ``StubAdapter`` path stays
deterministic; the real OpenAI/Anthropic adapters ignore ``canned`` and make a
real bounded ``verify`` call (Interface Contract v4 §0).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.api.schemas import VisualizeResponse
from app.llm.adapter import LLMAdapter
from app.plan.models import Plan


class IntentVerdict(BaseModel):
    """The Intent Reviewer's verdict (§3.4). ``revise`` names the offending
    ``field`` (when known) so the bounded re-plan has a precise target."""

    decision: Literal["approve", "revise"]
    reason: str | None = None
    field: str | None = None


class OutputVerdict(BaseModel):
    """The Output Reviewer's LLM-half verdict (§3.8). ``flag`` annotates
    ``meta.notes``; it never triggers a rebuild of the already-computed spec."""

    decision: Literal["approve", "flag"]
    reason: str | None = None


# The §B.3 semantic checklist, encoded as the Intent Reviewer's system prompt.
# Reused verbatim by the real Phase 4 reviewer call (the StubAdapter ignores it).
INTENT_REVIEWER_SYSTEM_PROMPT = (
    "You are the Intent Reviewer for a ClinicalTrials.gov query planner. The Plan you "
    "receive has ALREADY passed mechanical validation — every token, field, and range "
    "in it is legal. Your job is the semantic layer the checker cannot see: does this "
    "Plan capture what the user actually asked? Work this checklist:\n"
    "1. Metric — does the chosen query_class / field compute the quantity the question "
    "asks for (a distribution vs a raw count vs a trend over time vs a comparison)?\n"
    "2. Dimension — is each entity mapped to the RIGHT dimension (a drug under `drug`, "
    "a condition under `condition`, a sponsor under `sponsor`, a country under "
    "`country`), never swapped or mislabelled?\n"
    "3. Date intent — if the question implies a date sense (when a trial started, "
    "completed, was first posted, or last updated), does `date_field` match that "
    "sense?\n"
    "4. Chart — is `chart_type` an APT rendering of that answer, not merely a legal "
    "one?\n"
    "5. Filters — is every filter faithful to the question, and were NONE invented "
    "that the user never asked for?\n"
    "Output `approve` when all five hold. Otherwise output `revise`, naming the single "
    "most offending `field` and a precise `reason` the planner can act on in one "
    "re-plan. You do NOT edit the Plan yourself, and you NEVER compute, count, or "
    "state a number."
)

# The §B.3 output checklist, encoded as the Output Reviewer's system prompt.
OUTPUT_REVIEWER_SYSTEM_PROMPT = (
    "You are the Output Reviewer (LLM half) for a ClinicalTrials.gov visualization "
    "pipeline. Every count and every citation excerpt in the spec you receive has "
    "ALREADY passed deterministic provenance and reconciliation checks in code — you "
    "do NOT re-check arithmetic or provenance. Your job is narrower, non-generative, "
    "and reduces to two questions:\n"
    "1. Faithfulness — does this spec actually answer the user's question (right "
    "metric, right framing, no drift from what was asked)? EXCEPTION: if meta.notes "
    "discloses a field-precedence override (an 'Override: used field ...' note stating "
    "that a structured request field took precedence over a value in the query text), "
    "that entity substitution is the system's DESIGNED input-precedence behavior — treat "
    "the response as faithful and do NOT flag it as drift.\n"
    "2. Encoding — is the chart type (and its title) an apt rendering of the computed "
    "data?\n"
    "Output `approve`, or `flag` with a short `reason`. A flag is advisory only: it is "
    "appended to meta.notes and the spec ships as-is. You NEVER rebuild or "
    "re-aggregate the spec, and because you inspect already-computed output only, you "
    "NEVER introduce a number."
)


# Which typed structured input field (a ``merged_inputs`` key) can authoritatively
# source each Plan.entities dimension. A dimension NOT in this map (e.g. a
# free-text ``term``) can only have come from the NL parse — never from a typed
# field — so its presence means the intent review is NOT skippable.
_ENTITY_DIMENSION_TO_INPUT_FIELD = {
    "drug": "drug_name",
    "condition": "condition",
    "sponsor": "sponsor",
    "country": "country",
}


def review_intent(adapter: LLMAdapter, question: str, plan: Plan) -> IntentVerdict:
    """The Intent Reviewer (§3.4): approve a mechanically-valid Plan, or ask for
    one bounded revision.

    Runs after the Plan Checker, on legal plans only, and is a GATE — it emits a
    verdict and never rewrites the Plan. It applies the semantic §B.3 checklist
    (metric matches the ask · entity → right dimension · date_field matches the
    date-intent · chart apt, not merely legal · filters faithful, none invented)
    via one bounded ``verify`` call and returns ``approve`` or ``revise`` with the
    offending ``field`` + a precise ``reason``. That ``reason`` is free prose but
    never reaches the response: it is threaded into the re-plan prompt
    (``plan_feedback``) and, if the re-plan budget is exhausted, the caller ships a
    fixed code-owned caveat instead of the reviewer's words
    (``app.graph.nodes.review_output``).

    ``canned={"decision": "approve"}`` is the offline StubAdapter's deterministic
    answer; real adapters ignore it and make the real call.
    """
    return adapter.verify(
        system=INTENT_REVIEWER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Review this Plan against the user's question using the "
                    "checklist.\n"
                    f"Question: {question}\n"
                    f"Plan (JSON): {plan.model_dump_json()}"
                ),
            }
        ],
        response_model=IntentVerdict,
        canned={"decision": "approve"},
        model=None,
    )


def should_skip_intent_review(merged_inputs: dict, plan: Plan) -> bool:
    """Return True only when it is clearly safe to skip the intent review (§3.4).

    The intent review guards against a *misread NL parse* — the planner reading a
    free-text question and inferring the wrong dimension for an entity. When there
    was no such parse to misread, the review has nothing to catch and can be
    skipped (a latency/cost saving on the common structured-request path). Two
    conservative cases return True:

    1. The ``query`` is empty/whitespace — the request is driven entirely by typed
       structured fields, so nothing was parsed from natural language at all.
    2. Every dimension present in ``plan.entities`` is backed by a matching typed
       structured field (``drug`` ← ``drug_name``, ``condition`` ← ``condition``,
       ``sponsor`` ← ``sponsor``, ``country`` ← ``country``). No dimension was
       NL-inferred, so there is no entity→dimension misread to catch.

    Everything else returns False (review). In particular: a non-empty query with
    NO entities still ran the NL parse to pick the class/field/date, so it is not
    skipped; and any entity dimension without a typed-field source (e.g. a
    free-text ``term``, or a dimension the query inferred while the typed field was
    absent) forces a review.

    What the skip gives up (be precise about this): entities are the ONLY checklist
    item this gate proves. Skipping drops the intent review whole, so items 1-5 of
    ``INTENT_REVIEWER_SYSTEM_PROMPT`` — metric, date sense, chart aptness and
    FILTERS — go unreviewed on that request. The Plan Checker still re-checks those
    slots mechanically, but mechanically only: it proves a filter token is *legal*,
    never that the user asked for it, so an invented-but-legal filter (checklist
    item 5) is exactly what a skip can let through. That is the residual risk the
    saved call buys. Keep it conservative: only skip when clearly safe.
    """
    query = merged_inputs.get("query") or ""
    if not query.strip():
        return True

    entities = plan.entities or {}
    # A non-empty query with no resolved dimensions means the NL parse alone drove
    # the plan (class/field/date) with nothing typed to anchor it — not safe to skip.
    if not entities:
        return False

    for dimension in entities:
        input_field = _ENTITY_DIMENSION_TO_INPUT_FIELD.get(dimension)
        if input_field is None:
            # No typed field can source this dimension (e.g. free-text `term`) →
            # it was NL-inferred → review.
            return False
        value = merged_inputs.get(input_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            # The typed field that could have sourced this dimension is absent →
            # the dimension was NL-inferred → review.
            return False

    return True


def review_output_llm(
    adapter: LLMAdapter, question: str, spec: VisualizeResponse
) -> OutputVerdict:
    """The Output Reviewer's LLM half (§3.8): one bounded ``approve|flag`` call.

    Non-generative — it inspects an already-built, already-computed spec (whose
    counts/citations already passed deterministic checks) and asks only whether
    the spec faithfully answers the question and whether the encoding is apt. A
    ``flag`` verdict annotates ``meta.notes`` and the spec ships as-is; this
    reviewer never re-runs aggregation and never rebuilds the spec. Its ``reason``
    is free prose and CAN contain digits — what keeps a fabricated number off the
    wire is the caller's deterministic ``note_number_safe`` check
    (``app.graph.nodes.review_output``), not this model or this schema.

    ``canned={"decision": "approve"}`` is the offline StubAdapter's deterministic
    answer; real adapters ignore it and make the real call.
    """
    return adapter.verify(
        system=OUTPUT_REVIEWER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Review whether this computed response faithfully answers the "
                    "question with an apt encoding.\n"
                    f"Question: {question}\n"
                    f"Response spec (JSON): {spec.model_dump_json()}"
                ),
            }
        ],
        response_model=OutputVerdict,
        canned={"decision": "approve"},
        model=None,
    )
