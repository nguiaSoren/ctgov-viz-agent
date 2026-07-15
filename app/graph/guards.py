"""Runtime-harness execution guards (ARCHITECTURE_SPEC §4 · §B.4 · §B.7).

Active, code-owned backstops the graph runner enforces on EVERY request: a
per-request **wall-clock deadline**, an **iteration cap**, a **total-node-visit
cap**, a **tool-call cap**, and a **stall (no-progress) detector**.

Honest framing (the §B.4 headroom, stated not hidden): under the v1 single-shot
classify→fill planner + shared escalation budget ``<=1``, the iteration /
tool-call / node-visit caps **cannot fire in normal operation** — the plan node
is entered at most twice and ``execute`` runs once. They are built as *active*
protections anyway, as **defense-in-depth** against a routing / implementation
defect or a future multi-tool planner, and each is unit-tested to abort a
pathological loop (:mod:`tests.test_guards`). The **stall detector CAN fire under
v1**: a bounded re-plan that produces the *identical* plan is a genuine
no-progress stall, aborted here instead of redundantly re-executed.

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
    budget (ENG-27), else ``None``. Under v1 ``execute`` runs once per traversal
    so this is headroom, but it is enforced + tested all the same."""
    if tool_call_count > config.MAX_TOOL_CALLS:
        return MAX_TOOL_CALLS_EXCEEDED
    return None


def is_stalled(plan: Plan, seen_signatures: list | None) -> bool:
    """``True`` iff this plan's signature was ALREADY produced this request — a
    repeat means the bounded re-plan yielded the SAME plan (no progress, or an
    A→B→A oscillation compared as a SET, G-41g). Aborting here avoids
    re-executing an identical plan for a second, wasted API round-trip (ENG-59's
    "no new data" — we never re-page an already-seen plan)."""
    return plan_signature(plan) in set(seen_signatures or [])
