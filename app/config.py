"""Central, env-overridable configuration for the guards, caps, budgets, and
timeouts of the system (ARCHITECTURE_SPEC §B.4 "guard-values" + §A(f) DoS guards).

Before Phase 5, these numbers lived as scattered module constants
(``client._MAX_PAGE_SIZE``, ``nodes._TOO_LARGE_THRESHOLD``,
``tools._RECORD_INDEX_K``, ``network.build_graph``'s ``max_nodes`` default …).
Phase 5 pulled most of them into this one place so a reviewer can read the safety
envelope at a glance and an operator can retune it with an env var — the §B.4
footer rule: *"the cap is deploy-time config, never agent-tunable."* Nothing here
is reachable by the LLM; these are code-owned.

The consolidation is NOT total, and each exception is named inline at the constant
it affects rather than glossed over: ``MAX_QUERY_CHARS`` /
``MAX_STRUCTURED_FIELD_CHARS`` are documentation-only (the live caps are Pydantic
literals), ``PAGE_BUDGET_PAGES`` is bypassed on the aggregate/timeseries paths,
``CITATION_SAMPLE_K`` is honoured on the ``tools.py`` paths only, and ``PAGE_SIZE``
is a default rather than the enforced API cap its name suggests. Read the comment
on a knob before quoting it as operator-tunable.

Fail-loud contract: the ``_env_*`` helpers read an override once at import (a
restart re-reads), typed + range-guarded, and every one of them RAISES on a value
it cannot parse — including :func:`_env_bool`, which accepts only the documented
true/false spellings so a typo'd ``CACHE_ENABLED=maybe`` can never silently
resolve to "off". A malformed env var fails at boot rather than silently
corrupting a guard.
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


# The only accepted boolean spellings (case-insensitive). A closed vocabulary is
# what lets ``_env_bool`` reject a typo instead of silently reading it as False.
_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _env_bool(name: str, default: bool) -> bool:
    """Read ``name`` as a bool (``1/true/yes/on`` / ``0/false/no/off``), else ``default``.

    Unset or empty → ``default``. Any other value raises at import, exactly as
    ``_env_int``/``_env_float`` do: a typo'd ``CACHE_ENABLED=maybe`` resolving
    silently to "off" is precisely the quiet misconfiguration this module exists
    to prevent, so the boolean reader fails loud too.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"config: {name}={raw!r} is not a boolean "
        f"(expected one of {sorted(_TRUE_VALUES + _FALSE_VALUES)})"
    )


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
PAGE_BUDGET_PAGES = _env_int("PAGE_BUDGET_PAGES", 20)  # 20 pages × 1000 = 20,000 trials (§B.4).
# PARTIAL REACH: read by ``client.iter_studies``'s default and passed explicitly by the histogram
# and network tools — but ``aggregate.page_and_group`` carries its OWN ``budget_pages=20`` default
# and ``tools.aggregate_by``/``tools.timeseries`` never override it, so those two paths page to a
# hardcoded 20 whatever this says. Same number today; not the same knob.
PAGE_SIZE = _env_int("PAGE_SIZE", 1000, minimum=1)  # the client's page size AND its clamp ceiling
# (``client._MAX_PAGE_SIZE``). The registry's documented maximum is 1000 (§A(f)/SEC-32/R-3, [LIVE]),
# but nothing here range-checks against it: setting PAGE_SIZE=5000 raises the clamp to 5000 and the
# request goes out at pageSize=5000. It is a deploy-time default, not an enforced API cap.
TOO_LARGE_THRESHOLD = _env_int("TOO_LARGE_THRESHOLD", 20_000)  # >this → status:"too_large" (§B.7/ENG-60)

# --- Response-size caps (§B.4 "Response-size cap" · ENG-30/31/32 · SEC-33 · C-87) ---
TOP_N_CATEGORIES = _env_int("TOP_N_CATEGORIES", 50)  # ranked bar top-N + "Other" fold (§B.4; was 15 for
# country pre-P5 — reconciled UP to the spec's 50, P5-TOPN). The "Other" fold + *_truncated note disclose.
NETWORK_MAX_NODES = _env_int("NETWORK_MAX_NODES", 60)  # network top-N nodes by degree (§B.4/ENG-31)
NETWORK_MIN_EDGE_WEIGHT_DRUG_DRUG = _env_int("NETWORK_MIN_EDGE_WEIGHT_DRUG_DRUG", 2)  # CC-12 (k, tunable)
NETWORK_MIN_EDGE_WEIGHT_SPONSOR_DRUG = _env_int("NETWORK_MIN_EDGE_WEIGHT_SPONSOR_DRUG", 1)  # CC-12
NETWORK_MAX_DRUGS_PER_TRIAL = _env_int("NETWORK_MAX_DRUGS_PER_TRIAL", 25)  # G-41c: skip basket trials
# (>M drugs → C(N,2) edge blowup before the top-N prune)
CITATION_SAMPLE_K = _env_int("CITATION_SAMPLE_K", 20)  # ≤K cited nctIds per datum (§B.4/ENG-32/CC-9).
# PARTIAL REACH: passed explicitly on the ``tools.py`` distribution/timeseries paths (:267, :483) and
# into the record index (:173); ``citations.build_bucket_citations``, ``network`` and ``histogram``
# keep their own literal ``k=20`` defaults, so those three sample 20 regardless of this value, and
# the exact-count path's per-bucket sample (``tools._COUNT_SAMPLE_K`` = 10) never moved here at all.
RECORD_INDEX_CAP = _env_int("RECORD_INDEX_CAP", 500)  # bounded per-request record index for re-verify

# --- Egress client (§A(f) · SEC-29/SEC-30 · ENG-55) --------------------------
# SCOPE: the ClinicalTrials.gov client ONLY. The LLM adapters set no request timeout of their
# own (a disclosed gap, app/llm/adapter.py), so nothing here bounds a provider call.
PER_CALL_TIMEOUT_SECONDS = _env_float("PER_CALL_TIMEOUT_SECONDS", 30.0)  # single-call timeout (SEC-29)
RATE_LIMIT_RPS = _env_float("CTGOV_RATE_LIMIT_RPS", 3.0)  # politeness limiter (~3 req/s, SEC-30).
# SCOPE: a PER-INSTANCE min-interval gate, not a process-global one. Every tool builds a fresh
# ``CTGovClient()``, so the throttle holds inside one tool's paging loop and resets between tool
# calls. Enough for the serial v1 pipeline; a truly global limiter needs a shared token bucket.
MAX_RETRIES = _env_int("MAX_RETRIES", 3, minimum=0)  # 429/5xx/transient, backoff+jitter (ENG-55)

# --- Input caps (§A(f) · SEC-35 · E-25 · G-41b · E-21) -----------------------
# ⚠ NOT WIRED: the next two are DOCUMENTATION ONLY. The caps that actually run are
# ``max_length`` literals on ``VisualizeRequest`` (app/api/schemas.py: query 500,
# drug/condition/sponsor/country 200) — Pydantic needs them at class-definition time and the
# schema module does not import this one. Setting either env var changes nothing today; they are
# kept here so the declared envelope is complete and one import away from being live.
MAX_QUERY_CHARS = _env_int("MAX_QUERY_CHARS", 500)  # query length ≤ ~500 (SEC-35/E-25/API-4)
MAX_STRUCTURED_FIELD_CHARS = _env_int("MAX_STRUCTURED_FIELD_CHARS", 200)  # drug/condition/sponsor/country
# ≤ ~200 — the G-41b DoS/injection-surface cap (only query was bounded before)
MAX_COMPARE_ENTITIES = _env_int("MAX_COMPARE_ENTITIES", 5)  # cap on compare arms / network / multi-drug
# entities (E-21) → first-N + a dropped_entities note, never silent truncation

# --- Response cache (§3.10 · SEC-15/SEC-48 · C-73/C-74 · P5-CACHE) -----------
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)  # a bypass switch (tests + operators)
CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 300, minimum=0)  # short-TTL; non-authoritative (§3.10)
CACHE_MAX_ENTRIES = _env_int("CACHE_MAX_ENTRIES", 128)  # LRU bound (keeps the in-process cache small)
