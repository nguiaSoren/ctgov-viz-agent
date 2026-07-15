"""Central, env-overridable configuration for every guard, cap, budget, and
timeout in the system (ARCHITECTURE_SPEC §B.4 "guard-values" + §A(f) DoS guards).

Before Phase 5, these numbers lived as scattered module constants
(``client._MAX_PAGE_SIZE``, ``nodes._TOO_LARGE_THRESHOLD``,
``tools._RECORD_INDEX_K``, ``network.build_graph``'s ``max_nodes`` default …).
Phase 5 pulls the canonical values into ONE place so a reviewer can read the
whole safety envelope at a glance and an operator can retune any of them with an
env var — the §B.4 footer rule: *"the cap is deploy-time config, never
agent-tunable."* Nothing here is reachable by the LLM; these are code-owned.

Every value has a spec citation. The ``_env_*`` helpers read an override once at
import (a restart re-reads), typed + range-guarded, so a malformed env var fails
loud at boot rather than silently corrupting a guard.
"""

from __future__ import annotations

import math
import os


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read ``name`` as an int (≥ ``minimum``), falling back to ``default``.

    A present-but-unparseable / out-of-range value raises at import — a
    misconfigured guard must fail loud at boot, never silently disable itself.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"config: {name}={raw!r} is not an integer") from exc
    if value < minimum:
        raise ValueError(f"config: {name}={value} is below the minimum {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read ``name`` as a float (≥ ``minimum``), falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"config: {name}={raw!r} is not a number") from exc
    # Reject NaN/inf explicitly: ``nan < minimum`` and ``inf < minimum`` are BOTH
    # False, so a non-finite value would sail past the range guard below and (for
    # WALL_CLOCK_*) make ``deadline_at = monotonic() + nan = nan``, and ``now > nan``
    # is always False → the wall-clock guard, the ONE that fires under v1, silently
    # never trips. Fail loud at boot instead (the whole point of this module).
    if not math.isfinite(value):
        raise ValueError(f"config: {name}={raw!r} must be a finite number")
    if value < minimum:
        raise ValueError(f"config: {name}={value} is below the minimum {minimum}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Read ``name`` as a bool (``1/true/yes/on`` → True), falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Runtime-harness guards (§B.4 · C-83/C-84/C-86 · SEC-34/SEC-36) ----------
# The v1 planner is single-shot classify→fill with a shared escalation budget ≤1,
# so the iteration / tool-call caps below CANNOT fire in normal operation. They
# are built as ACTIVE graph-runner backstops anyway — defense-in-depth against a
# routing/implementation defect or a future multi-tool planner — and each is
# tested to abort a pathological loop. The headroom is stated, not hidden.
MAX_REACT_ITERATIONS = _env_int("MAX_REACT_ITERATIONS", 8)  # plan re-entries; §B.4 "8 = 2× headroom"
MAX_TOOL_CALLS = _env_int("MAX_TOOL_CALLS", 12)  # total tool fan-out, independent of iterations (§B.4)
MAX_GRAPH_STEPS = _env_int("MAX_GRAPH_STEPS", 40)  # hard node-visit backstop (a bounded spine + one
# back-edge visits ≪ this; only a routing bug could approach it → abort-to-error, never hang)
WALL_CLOCK_SYNC_SECONDS = _env_float("WALL_CLOCK_SYNC_SECONDS", 60.0)  # POST /visualize deadline (§B.4)
WALL_CLOCK_SSE_SECONDS = _env_float("WALL_CLOCK_SSE_SECONDS", 90.0)  # /visualize/stream deadline (§B.4)

# --- Paging budget + over-budget refuse (§B.4/§B.7 · CC-6 · ENG-29 · SEC-31/32) ---
PAGE_BUDGET_PAGES = _env_int("PAGE_BUDGET_PAGES", 20)  # 20 pages × 1000 = 20,000 trials (§B.4)
PAGE_SIZE = _env_int("PAGE_SIZE", 1000, minimum=1)  # API hard cap ≤1000 (§A(f)/SEC-32/R-3, [LIVE])
TOO_LARGE_THRESHOLD = _env_int("TOO_LARGE_THRESHOLD", 20_000)  # >this → status:"too_large" (§B.7/ENG-60)

# --- Response-size caps (§B.4 "Response-size cap" · ENG-30/31/32 · SEC-33 · C-87) ---
TOP_N_CATEGORIES = _env_int("TOP_N_CATEGORIES", 50)  # ranked bar top-N + "Other" fold (§B.4; was 15 for
# country pre-P5 — reconciled UP to the spec's 50, P5-TOPN). The "Other" fold + *_truncated note disclose.
NETWORK_MAX_NODES = _env_int("NETWORK_MAX_NODES", 60)  # network top-N nodes by degree (§B.4/ENG-31)
NETWORK_MIN_EDGE_WEIGHT_DRUG_DRUG = _env_int("NETWORK_MIN_EDGE_WEIGHT_DRUG_DRUG", 2)  # CC-12 (k, tunable)
NETWORK_MIN_EDGE_WEIGHT_SPONSOR_DRUG = _env_int("NETWORK_MIN_EDGE_WEIGHT_SPONSOR_DRUG", 1)  # CC-12
NETWORK_MAX_DRUGS_PER_TRIAL = _env_int("NETWORK_MAX_DRUGS_PER_TRIAL", 25)  # G-41c: skip basket trials
# (>M drugs → C(N,2) edge blowup before the top-N prune)
CITATION_SAMPLE_K = _env_int("CITATION_SAMPLE_K", 20)  # ≤K cited nctIds per datum (§B.4/ENG-32/CC-9)
RECORD_INDEX_CAP = _env_int("RECORD_INDEX_CAP", 500)  # bounded per-request record index for re-verify

# --- Egress client (§A(f) · SEC-29/SEC-30 · ENG-55) --------------------------
PER_CALL_TIMEOUT_SECONDS = _env_float("PER_CALL_TIMEOUT_SECONDS", 30.0)  # single-call timeout (SEC-29)
RATE_LIMIT_RPS = _env_float("CTGOV_RATE_LIMIT_RPS", 3.0)  # shared politeness limiter (~3 req/s, SEC-30)
MAX_RETRIES = _env_int("MAX_RETRIES", 3, minimum=0)  # 429/5xx/transient, backoff+jitter (ENG-55)

# --- Input caps (§A(f) · SEC-35 · E-25 · G-41b · E-21) -----------------------
MAX_QUERY_CHARS = _env_int("MAX_QUERY_CHARS", 500)  # query length ≤ ~500 (SEC-35/E-25/API-4)
MAX_STRUCTURED_FIELD_CHARS = _env_int("MAX_STRUCTURED_FIELD_CHARS", 200)  # drug/condition/sponsor/country
# ≤ ~200 — the G-41b DoS/injection-surface cap (only query was bounded before)
MAX_COMPARE_ENTITIES = _env_int("MAX_COMPARE_ENTITIES", 5)  # cap on compare arms / network / multi-drug
# entities (E-21) → first-N + a dropped_entities note, never silent truncation

# --- Response cache (§3.10 · SEC-15/SEC-48 · C-73/C-74 · P5-CACHE) -----------
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)  # a bypass switch (tests + operators)
CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 300, minimum=0)  # short-TTL; non-authoritative (§3.10)
CACHE_MAX_ENTRIES = _env_int("CACHE_MAX_ENTRIES", 128)  # LRU bound (keeps the in-process cache small)
