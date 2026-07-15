"""In-process TTL + LRU response cache, keyed on the normalized plan.

This is a **non-authoritative performance layer** (ARCHITECTURE_SPEC §3.10 ·
P5-CACHE). It holds only fully-computed :class:`VisualizeResponse` envelopes that
the deterministic aggregation core already produced — it can *replay* a prior
answer, but it can NEVER originate or override a live count. Every number in a
cached envelope was code-computed on the miss path that populated it; a cache hit
hands back an independent DEEP COPY of the same envelope (so a caller mutating its
result can never poison the shared entry). The moment a value would be
authoritative, it does not belong here.

Two further properties by design:

* **Bypassable.** ``CACHE_ENABLED`` (env / operator switch, and the per-instance
  ``enabled`` override) turns the whole layer off: ``get`` always misses and
  ``set`` is a no-op, so the system degrades to always recomputing. A short TTL
  (``CACHE_TTL_SECONDS``) plus ``TTL == 0`` ("always-expired") give the same
  escape hatch at finer grain.
* **Non-authoritative.** Nothing here is reachable by the LLM, and nothing here
  can invent a count. Staleness fails safe toward recomputation, never toward a
  wrong-but-cached number.

Scope / non-goals:

* **Single-process, not thread-safe.** The store is a plain ``OrderedDict`` with
  no lock. The v1 backend serves one request at a time per worker; if that ever
  changes, wrap the mutating methods — do not silently rely on GIL atomicity.
* Dependency-light: stdlib + pydantic + ``app.config`` + the wire schema only.

Time is read through an injected ``now`` clock (default ``time.monotonic``) so TTL
behaviour is testable without sleeping and immune to wall-clock jumps.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.api.schemas import VisualizeResponse
from app.config import CACHE_ENABLED, CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS
from app.plan.models import Plan


def _json_default(obj: Any) -> Any:
    """Canonicalize the few non-JSON-native values a plan field can hold.

    Nested Pydantic sub-models (``Series`` / ``NetworkSpec``) → ``model_dump``;
    enums (``ChartType`` and friends) → their raw ``.value``; anything else falls
    back to ``str`` so the key builder can never raise on an exotic filter value.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _significant_fields(plan: Plan) -> dict[str, Any]:
    """The plan's *identity* — the fields that determine what gets computed.

    Deliberately EXCLUDES ``notes`` and ``alternates``: those are interpretation
    (how the answer is framed / which other marks the frontend may offer), not
    identity — two plans that differ only there compute the same result and must
    share a key. Nested sub-models are dumped to plain dicts so the whole thing is
    canonical-JSON serializable.
    """
    return {
        "query_class": plan.query_class,
        "entities": plan.entities,
        "filters": plan.filters,
        "field": plan.field,
        "date_field": plan.date_field,
        "grain": plan.grain,
        "chart_type": plan.chart_type.value
        if isinstance(plan.chart_type, Enum)
        else plan.chart_type,
        "interventional_only": plan.interventional_only,
        # Order is semantically significant for a compare (arm order), so preserve it.
        "series": [s.model_dump() for s in plan.series] if plan.series else None,
        "network": plan.network.model_dump() if plan.network else None,
        "answer_kind": plan.answer_kind,
    }


def plan_cache_key(plan: Plan) -> str:
    """A pure, deterministic, collision-safe cache key for ``plan``.

    Canonical JSON (recursively key-sorted) over the plan's semantically-significant
    fields, hashed with SHA-256. Two plans that compute the same result produce the
    same key; two that differ in any significant field produce different keys.
    """
    canonical = json.dumps(
        _significant_fields(plan),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A bounded TTL + LRU cache of computed :class:`VisualizeResponse` envelopes.

    * TTL via the injected monotonic clock: an entry older than ``ttl`` seconds is
      a miss and is evicted on read. ``ttl <= 0`` means "always-expired" (a stored
      entry is stale the instant it is written) — the finest-grain bypass.
    * LRU: a hit refreshes recency; ``set`` evicts the least-recently-used entry
      once the store exceeds ``max_entries``.
    * ``enabled=False`` (or ``CACHE_ENABLED=False``) makes ``get`` always miss and
      ``set`` a no-op.

    Constructor overrides default to the ``app.config`` values, so a plain
    ``ResponseCache()`` follows deploy-time config; tests inject their own.
    """

    def __init__(
        self,
        *,
        ttl: int | None = None,
        max_entries: int | None = None,
        enabled: bool | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = CACHE_TTL_SECONDS if ttl is None else ttl
        self._max_entries = CACHE_MAX_ENTRIES if max_entries is None else max_entries
        self._enabled = CACHE_ENABLED if enabled is None else enabled
        self._now = now
        # key -> (stored_at_monotonic, envelope); insertion order == LRU order.
        self._store: OrderedDict[str, tuple[float, VisualizeResponse]] = OrderedDict()

    def _is_expired(self, stored_at: float) -> bool:
        """True if an entry stored at ``stored_at`` is past its TTL.

        ``ttl <= 0`` short-circuits to always-expired; otherwise strictly *older
        than* the TTL ("older than N seconds is a MISS").
        """
        if self._ttl <= 0:
            return True
        return (self._now() - stored_at) > self._ttl

    def get(self, key: str) -> VisualizeResponse | None:
        """Return an INDEPENDENT DEEP COPY of the cached envelope for ``key``, or
        ``None`` on miss.

        Miss cases: cache disabled, key absent, or the entry has expired (in which
        case it is evicted). A hit refreshes the entry's LRU recency.

        The returned envelope is a ``model_copy(deep=True)`` of the stored master —
        NOT the master object — so a caller that later mutates its result (e.g.
        appends a per-request caveat to ``meta.notes``) cannot poison the shared
        cache entry for the next identical-plan request. (Paired with the deep copy
        on :meth:`set`, this is what makes "the cache never hands back a mutable
        alias of a shared object" literally true.)
        """
        if not self._enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._is_expired(stored_at):
            del self._store[key]
            return None
        self._store.move_to_end(key)  # most-recently used
        return value.model_copy(deep=True)

    def set(self, key: str, value: VisualizeResponse) -> None:
        """Store a DEEP COPY of ``value`` under ``key`` (no-op when disabled).

        A ``model_copy(deep=True)`` is stored, NOT the caller's object, so a caller
        that keeps mutating the envelope it also passed to us cannot reach back into
        the cache and corrupt the stored master. Records the current clock reading
        as the entry's birth time and marks it most-recently-used, then LRU-evicts
        from the front until the store is within ``max_entries``.
        """
        if not self._enabled:
            return
        self._store[key] = (self._now(), value.model_copy(deep=True))
        self._store.move_to_end(key)  # freshly written == most-recently used
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)  # drop least-recently used

    def clear(self) -> None:
        """Empty the cache."""
        self._store.clear()

    def __len__(self) -> int:
        """Number of stored entries (may include not-yet-evicted stale ones)."""
        return len(self._store)


# Module-level singleton — the shared cache the graph wires in (build.py, kept
# separate per the layering: this module is the standalone mechanism).
RESPONSE_CACHE = ResponseCache()
