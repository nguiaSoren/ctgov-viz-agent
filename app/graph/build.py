"""Graph assembly + the two request-facing entry points (ARCHITECTURE_SPEC
§3.12 / §B.5).

Wires the 9 nodes into the spine the spec describes:
``merge_inputs -> plan -> check -> review_intent -> execute -> build_spec ->
review_output -> respond``, plus one bounded escalation back-edge into
``plan`` (shared, <=1, across the check/review_intent/execute re-plan
triggers) and an ``error -> respond`` edge so every path terminates at the
same single node. Compiled with **no checkpointer** (ENG-47): stateless
per request, horizontally scalable; conversational memory (a stretch) is
exactly "flip a checkpointer on" later, not a redesign.

**Topology -- cyclic, not a DAG.** The spine above is acyclic, but the
escalation edge points *backward* (``check``/``review_intent``/``execute`` ->
``plan``), and a backward edge means cycles -- so this graph is cyclic, not a
DAG. It is bounded, not open-ended: the escalation budget is shared and <=1, so
the cycle fires at most once and every execution trace is finite (no infinite
loop) -- structurally cyclic, runtime-bounded. That combination is exactly why
LangGraph is used here instead of a plain DAG runner: the "Adaptive"
control x autonomy classification (ARCHITECTURE_SPEC §2) rests on runtime
tool-choice + retry + early-stop over a *cyclic* graph, which a pure DAG cannot
express.

**There is exactly ONE cycle -- this back-edge.** Earlier drafts of this
docstring described a second, node-internal "ReAct self-loop" inside ``plan``.
That does not exist: ``plan`` makes a single ``plan_request`` call, which makes a
single ``adapter.propose`` call with ``tools=None``
(``app.llm.planner.plan_request``). The reason -> act -> observe cycle IS the
back-edge drawn here, and nothing else. (The adapter can re-ask once when the model
returns unparseable structured output, but that is a parse repair inside one call,
not a reasoning loop.)

Deviations from the literal §B.5 routing table, flagged (not silently
diverged, per the build brief). The list is exhaustive as of Phase 5 -- items 3-5
are the branches Phase 4/5 added that §B.5 never had:

1. §B.5's table sends ``merge_inputs``'s invalid case straight to
   ``respond``(error 422); this build routes it through the dedicated
   ``error`` node instead (which itself edges to ``respond``), so every
   error path builds a real error envelope via the same ``build_envelope``
   call. Functionally equivalent, structurally more uniform. Unreachable in this
   build (in every phase, not just Phase 0): ``VisualizeRequest`` already guarantees
   a non-empty query before the graph runs.
2. **RESOLVED in Phase 4 (P4-ROUTING).** §B.5's table has ``review_intent``'s
   ``revise ∧ esc>=1`` case fall through to ``execute`` (best-effort, not a hard
   stop). The Phase-0 skeleton deviated to a hard stop → ``error``; now that the
   real Intent Reviewer can genuinely ``revise``, ``route_after_intent`` follows
   the spec: an exhausted re-plan budget ships the *legal* plan best-effort to
   ``execute`` (the Checker already proved it legal; only aptness is unconfirmed),
   with a disclosed ``meta.notes`` caveat added in ``review_output``. The
   ``check`` ``reject ∧ esc>=1`` case still routes to ``error`` — a mechanically
   *illegal* plan cannot ship. Code and spec now agree; no §B.5 amendment needed.
   Consequence worth stating: ``route_after_intent`` can now return only ``execute``
   or ``plan``, so ``review_intent`` no longer has an edge to ``error`` at all.
3. **``plan -> error`` (Phase 5).** §B.5 has no guard layer. The runtime-harness
   guards (deadline / iteration cap / node-visit backstop / stall) are checked at the
   top of ``plan``, and a trip short-circuits to the ``error`` node with a REDACTED
   code. §B.5's sketch for the iteration cap was "best-effort finalize with
   ``partial``"; this build aborts instead. Deliberate: a plan-node trip means no
   validated plan exists yet, so there is no partial result to finalize — shipping a
   chart with no proven plan behind it would be worse than refusing.
4. **``plan -> build_spec`` (Phase 5, E-13).** A dangling demonstrative reference
   ("this drug") produces a code-templated clarification question and short-circuits
   straight to ``build_spec``, skipping ``check`` and ``execute`` — there is nothing
   to validate or compute yet. §B.5 predates the ``kind:"clarification"`` envelope.
5. **``execute -> respond`` (Phase 5, C-74).** A response-cache hit already holds the
   final envelope, so it skips ``build_spec`` AND ``review_output``. §B.5 assumed
   every ``execute`` fed the builder.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app import config
from app.api.schemas import VisualizeRequest, VisualizeResponse
from app.graph.nodes import (
    build_spec,
    check,
    error,
    execute,
    merge_inputs,
    respond,
    review_intent,
    review_output,
    route_after_check,
    route_after_execute,
    route_after_intent,
    route_after_merge,
    route_after_output,
    route_after_plan,
)
from app.graph.nodes import plan as plan_node
from app.graph.state import GraphState

_compiled_graph: CompiledStateGraph | None = None


def build_graph() -> CompiledStateGraph:
    """Construct + compile the graph. Cached in a module global -- built
    once, reused across requests (safe because the graph carries no
    checkpointer / cross-request state)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(GraphState)
    graph.add_node("merge_inputs", merge_inputs)
    graph.add_node("plan", plan_node)
    graph.add_node("check", check)
    graph.add_node("review_intent", review_intent)
    graph.add_node("execute", execute)
    graph.add_node("build_spec", build_spec)
    graph.add_node("review_output", review_output)
    graph.add_node("respond", respond)
    graph.add_node("error", error)

    graph.add_edge(START, "merge_inputs")
    graph.add_conditional_edges(
        "merge_inputs", route_after_merge, {"plan": "plan", "error": "error"}
    )
    graph.add_conditional_edges(
        # "build_spec" = a dangling-reference clarification (E-13) short-circuits
        # check/execute; "error" = a tripped runtime guard.
        "plan", route_after_plan, {"check": "check", "error": "error", "build_spec": "build_spec"}
    )
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {"review_intent": "review_intent", "plan": "plan", "error": "error"},
    )
    graph.add_conditional_edges(
        "review_intent",
        route_after_intent,
        # No "error" target: since P4-ROUTING (deviation 2 above) ``route_after_intent``
        # returns only "execute" or "plan" — an advisory reviewer can no longer hard-stop
        # a plan the Checker already proved legal.
        {"execute": "execute", "plan": "plan"},
    )
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        # "respond" = a response-cache hit (execute already put the final envelope
        # in state, so build_spec + review_output are skipped).
        {"build_spec": "build_spec", "plan": "plan", "error": "error", "respond": "respond"},
    )
    graph.add_edge("build_spec", "review_output")
    graph.add_conditional_edges("review_output", route_after_output, {"respond": "respond"})
    graph.add_edge("error", "respond")
    graph.add_edge("respond", END)

    _compiled_graph = graph.compile()
    return _compiled_graph


def initial_state(
    request: VisualizeRequest,
    overrides: dict | None = None,
    *,
    deadline_seconds: float | None = None,
) -> GraphState:
    """Build the initial ``GraphState`` from a validated ``VisualizeRequest``.

    Every ``GraphState`` key is seeded explicitly. ``GraphState`` is ``total=False`` and
    every reader uses ``.get`` with a default, so this is belt-and-braces rather than a
    requirement -- but an exhaustive block is auditable at a glance, which a partial one
    is not.

    ``overrides`` seeds ``merged_inputs`` up front so tests can inject the offline
    sentinels (``_force_error`` / ``_force_empty`` / ``_force_too_large`` /
    ``_force_canned`` / ``_force_plan`` / ``_force_reject`` / ``_force_revise``) without
    adding non-contract fields to ``VisualizeRequest`` itself -- ``merge_inputs`` updates,
    never replaces, the incoming ``merged_inputs``, so these survive into ``execute``.

    ``deadline_seconds`` (Phase 5, SEC-36) seeds the ABSOLUTE wall-clock
    ``deadline_at`` (``time.monotonic() + deadline_seconds``); the entry points
    pass 60s (sync) / 90s (SSE). Left ``None`` — the default — the deadline guard
    is a no-op, so the structural offline suite stays deterministic (no test
    is at the mercy of a clock).
    """
    return GraphState(
        question=request.query,
        raw_fields=request.model_dump(exclude={"query"}),
        merged_inputs=dict(overrides or {}),
        plan=None,
        escalation_count=0,
        validation=None,
        tool_results=[],
        iter_count=0,
        scratch={},
        partial=None,
        spec=None,
        verifications=[],
        error=None,
        status="ok",
        count_total=None,
        bucket_mode=None,
        retrieved_at=None,
        query_provenance=None,
        fetched_records=None,
        plan_feedback=None,
        deadline_at=(time.monotonic() + deadline_seconds) if deadline_seconds is not None else None,
        tool_call_count=0,
        seen_signatures=[],
        clarification=None,
        cache_hit=False,
        events=[],
    )


def run_sync(request: VisualizeRequest, *, overrides: dict | None = None) -> VisualizeResponse:
    """Run the graph synchronously, end-to-end, for one request; return the
    final ``VisualizeResponse``. The sync ``POST /visualize`` transport is a thin
    wrapper over this. Seeds the 60s sync wall-clock deadline (§B.4) — unless a
    test passes ``overrides`` (the offline sentinel paths), which leaves the
    deadline unset so structural tests stay clock-independent."""
    graph = build_graph()
    deadline = None if overrides else config.WALL_CLOCK_SYNC_SECONDS
    final_state = graph.invoke(initial_state(request, overrides, deadline_seconds=deadline))
    return final_state["spec"]
