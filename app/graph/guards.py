"""Runtime-harness execution guards (ARCHITECTURE_SPEC §4 · §B.4 · §B.7).

Active, code-owned backstops the graph runner enforces on EVERY request: a
per-request **wall-clock deadline**, an **iteration cap**, a **total-node-visit
cap**, a **tool-call cap**, and a **stall (no-progress) detector**.

Honest framing (the §B.4 headroom, stated not hidden). **Exactly ONE of the five can
fire in normal v1 operation: the wall-clock deadline** — a genuinely slow request (a
long paging walk against a slow upstream) blows it and gets a clean redacted error
instead of a hang. The other four cannot, because the v1 single-shot classify→fill
planner plus the shared escalation budget ``<=1`` bound the traversal: ``plan`` is
entered at most twice and ``execute`` at most twice, so ``iter_count`` never reaches
``MAX_REACT_ITERATIONS`` (8), ``tool_call_count`` never reaches ``MAX_TOOL_CALLS``
(12), and ``len(events)`` peaks around a dozen against ``MAX_GRAPH_STEPS`` (40).

That unreachable set includes the stall detector, and the reason is worth being exact
about: :func:`is_stalled` is a live predicate, but its caller gates it on
``iter_count >= 2`` (``app.graph.nodes.plan``) — a THIRD plan entry, which the ``<=1``
budget never reaches. The gate is deliberate, not an oversight: the ONE sanctioned
re-plan is allowed to reproduce the same plan, because that run must still be able to
settle a clean ``empty`` or ship best-effort (§B.5); aborting it as a "stall" would be
wrong.

All five are built as *active* protections anyway — **defense-in-depth** against a
routing / implementation defect or a future multi-iteration planner — and each is
unit-tested to abort the pathological state when it is injected
(:mod:`tests.test_guards`). Plan signatures are recorded on every plan entry
regardless, so the SET is already complete the day a multi-iteration loop exists.

Every tripped guard funnels to a REDACTED error (the machine ``code`` carries the
specifics; the wire message is a fixed generic string — no internals leak,
LESSON B4). These are code-owned and never LLM-reachable.

Totality scope: the predicates are total against the state SHAPES that actually
occur — a missing key or a ``None`` optional (all handled with ``.get``/defaults).
They ASSUME the code-owned state TYPES the nodes always write (``iter_count``/
``tool_call_count`` are ints, ``events``/``seen_signatures`` are lists,
``deadline_at`` is a float-or-None, ``plan`` is a ``Plan``); these fields are never
wire-derived, so a wrong-TYPE value cannot reach them. If the state ever becomes
externally seedable, add ``isinstance`` coercion here.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app import config
from app.cache import plan_cache_key

if TYPE_CHECKING:
    from app.graph.state import GraphState
    from app.plan.models import Plan

# --- machine error codes (redacted; the generic wire message is below) -------
DEADLINE_EXCEEDED = "deadline_exceeded"
MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
MAX_STEPS_EXCEEDED = "max_steps_exceeded"
MAX_TOOL_CALLS_EXCEEDED = "max_tool_calls_exceeded"
STALLED_NO_PROGRESS = "stalled_no_progress"

# One fixed, generic wire message for every guard trip (API-22: the message never
# leaks internals; the machine ``code`` is the actionable part).
GUARD_MESSAGE = "the request exceeded a runtime safety limit and was stopped"


def over_deadline(deadline_at: float | None, *, now: float | None = None) -> bool:
    """Is the per-request wall-clock deadline blown? (SEC-36).

    ``deadline_at`` is an absolute ``time.monotonic()`` stamp (or ``None`` when
    unset — the structural offline path, where the guard is a no-op so the tests
    stay deterministic). ``now`` is injectable for testing without sleeping.
    """
    if deadline_at is None:
        return False
    current = time.monotonic() if now is None else now
    return current > deadline_at


def plan_signature(plan: Plan) -> str:
    """The stall detector's identity for a plan — DELIBERATELY the SAME function
    the response cache keys on (:func:`app.cache.plan_cache_key`), so "the same
    plan" means one thing to both the cache and the stall detector (one
    canonicalizer, no drift). Two plans that would compute the same result share
    a signature; a re-plan that changed nothing repeats it."""
    return plan_cache_key(plan)


def guard_error(code: str) -> dict:
    """The redacted ``error`` update for a tripped guard (machine ``code`` +
    fixed generic message)."""
    return {"code": code, "message": GUARD_MESSAGE}


def check_pre_plan_guards(state: GraphState) -> str | None:
    """Return a machine error ``code`` if a PRE-planning guard trips (checked at
    the top of the ``plan`` node, *before* an LLM call is spent), else ``None``.

    Order: wall-clock deadline → iteration cap → node-visit backstop. All three
    are read-only on ``state`` (pure), so they are trivially unit-testable by
    seeding a past ``deadline_at`` / a high ``iter_count`` / a long ``events``.
    """
    if over_deadline(state.get("deadline_at")):
        return DEADLINE_EXCEEDED
    if state.get("iter_count", 0) >= config.MAX_REACT_ITERATIONS:
        return MAX_ITERATIONS_EXCEEDED
    if len(state.get("events") or []) >= config.MAX_GRAPH_STEPS:
        return MAX_STEPS_EXCEEDED
    return None


def check_tool_budget(tool_call_count: int) -> str | None:
    """Return :data:`MAX_TOOL_CALLS_EXCEEDED` if the total tool fan-out is over
    budget (ENG-27), else ``None``.

    The counter is per ``execute`` ENTRY, not per upstream HTTP call: v1 enters
    ``execute`` at most twice (the zero-results re-plan is the only way back), so the
    live value never exceeds 2 against a cap of ``MAX_TOOL_CALLS`` (12). Headroom for a
    future multi-tool planner — enforced + tested all the same. The per-request HTTP
    volume is bounded separately, by the page budget and the per-call timeout."""
    if tool_call_count > config.MAX_TOOL_CALLS:
        return MAX_TOOL_CALLS_EXCEEDED
    return None


def is_stalled(plan: Plan, seen_signatures: list | None) -> bool:
    """``True`` iff this plan's signature was ALREADY produced this request — a
    repeat means the re-plan yielded the SAME plan (no progress, or an A→B→A
    oscillation compared as a SET, G-41g). Aborting on that avoids re-executing an
    identical plan for a second, wasted API round-trip (ENG-59's "no new data").

    The caller only consults this from the THIRD plan entry onward (``iter_count >= 2``
    in ``app.graph.nodes.plan``), which v1's ``<=1`` escalation budget never reaches;
    the second entry is permitted to repeat a signature by design."""
    return plan_signature(plan) in set(seen_signatures or [])
