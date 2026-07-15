"""The LangGraph state schema (ARCHITECTURE_SPEC §3.12 / §B.5).

A single ``TypedDict`` shared by every node. Three fields are **append-only**
(``Annotated[list, operator.add]``): ``tool_results``, ``verifications``, and
``events``. Every node that contributes to one of these returns a small list
and LangGraph concatenates it onto the running total -- this is what lets
``events`` double as a derived execution trace (a node's name is appended the
moment it runs) without any node needing to know what ran before it. Every
other field is last-write-wins: a node returning a partial update simply
overwrites that key in the merged state.

Writer -> reader matrix (§B.5), restated field-by-field:

* ``question`` / ``raw_fields`` -- written once by ``merge_inputs``; read by
  ``plan``, ``review_intent``, ``review_output``.
* ``merged_inputs`` -- written by ``merge_inputs``; read by ``plan``.
* ``plan`` -- written by ``plan``; read by ``check``, ``review_intent``,
  ``execute``, ``build_spec``.
* ``escalation_count`` -- incremented by ``plan`` on a back-edge re-entry;
  read by ``check``/``review_intent``/``execute``'s routers (the shared,
  <=1 escalation budget across all three re-plan triggers).
* ``validation`` -- written by ``check``; read by ``review_intent``'s router.
* ``tool_results`` (append-only) -- written by ``execute``; read by
  ``build_spec``.
* ``iter_count`` / ``scratch`` -- read/write internal to ``plan`` (the ReAct
  loop lives there) and, for ``scratch``, also stashed-into by
  ``review_intent`` for its own router to read.
* ``partial`` -- written by ``plan``/``execute`` when a genuine truncation
  occurred; read by ``build_spec``/``respond``.
* ``spec`` -- written by ``build_spec`` (and ``error``); read by
  ``review_output``, ``respond``.
* ``verifications`` (append-only) -- written by ``review_intent`` and
  ``review_output``; read by ``respond``.
* ``error`` -- written by any node that hits a hard failure; read by
  ``respond``/``error``.
* ``status`` -- written by ``execute``/``build_spec``/``respond``.
* ``count_total`` / ``bucket_mode`` -- written by ``execute`` (the API's exact
  ``countTotal`` oracle + the aggregation ``combine``/``explode`` mode); read by
  ``review_output`` for the deterministic reconciliation pre-check.
* ``retrieved_at`` / ``query_provenance`` -- written by ``execute``; read by
  ``build_spec``/``respond``.
* ``fetched_records`` -- the bounded per-nctId excerpt index ``execute`` stashes
  from the records it actually paged (Phase 4); read by ``review_output`` for the
  record-grounded citation re-verify (each excerpt re-checked as a real substring
  at its ``field_path`` in the actual fetched record via ``is_substring_at`` --
  giving the load-bearing primitive a RUNTIME caller, LESSON M3). ``None`` when the
  path didn't page (too_large / offline sentinels).
* ``plan_feedback`` -- the machine reason from the last rejected attempt
  (checker reject / intent revise / zero-results), written by the router-adjacent
  nodes and read by ``plan`` on a re-entry so the bounded re-plan is a real
  ``reason -> act -> observe`` step, not a blind retry (§3.12 escalation back-edge).
* ``deadline_at`` -- the ABSOLUTE ``time.monotonic()`` wall-clock deadline
  (Phase 5, SEC-36): seeded by the entry point (60s sync / 90s SSE, §B.4), read at
  each expensive node so an over-deadline run short-circuits to ``error`` instead
  of hanging. ``None`` when unset (structural offline tests).
* ``tool_call_count`` -- total tool fan-out this request (Phase 5, ENG-27/§B.4);
  incremented by ``execute``, capped at ``MAX_TOOL_CALLS`` (a forward-compat
  backstop: v1's single-shot planner runs ``execute`` once, so it cannot fire in
  normal operation, but a future multi-tool planner is bounded by construction).
* ``seen_signatures`` -- the SET (as a list) of ``tool_name+canonical-args`` plan
  signatures already produced (Phase 5, ENG-57/G-41g); ``plan`` compares each new
  plan's signature against it to abort an oscillation / no-progress re-plan
  (a repeat means the bounded re-plan produced the SAME plan → stalled → abort,
  rather than redundantly re-executing an identical plan).
* ``clarification`` -- a code-templated disambiguation question when the planner
  cannot resolve an NL referent (Phase 5, E-13); read by ``build_spec`` to emit a
  ``kind:"clarification"`` envelope. ``None`` in the normal path.
* ``events`` (append-only) -- written by every node; a derived stream via
  node transitions, read by tests/observability (also the ``MAX_GRAPH_STEPS``
  node-visit backstop reads ``len(events)``).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.api.schemas import VisualizeResponse
from app.plan.models import CheckResult, Plan


class GraphState(TypedDict, total=False):
    question: str
    raw_fields: dict
    merged_inputs: dict
    plan: Plan | None
    escalation_count: int
    validation: CheckResult | None
    tool_results: Annotated[list, operator.add]
    iter_count: int
    scratch: dict
    partial: dict | None
    spec: VisualizeResponse | None
    verifications: Annotated[list, operator.add]
    error: dict | None
    status: str
    count_total: int | None
    bucket_mode: str | None
    retrieved_at: str | None
    query_provenance: dict | None
    fetched_records: list | None
    plan_feedback: str | None
    deadline_at: float | None  # absolute time.monotonic() wall-clock deadline (Phase 5, SEC-36)
    tool_call_count: int  # total tool fan-out, capped at MAX_TOOL_CALLS (Phase 5, ENG-27)
    seen_signatures: list  # SET of plan signatures already produced — stall detector (Phase 5, ENG-57)
    clarification: str | None  # code-templated disambiguation question (Phase 5, E-13)
    cache_hit: bool  # True when `execute` served the envelope from the response cache (Phase 5, C-74)
    events: Annotated[list, operator.add]
