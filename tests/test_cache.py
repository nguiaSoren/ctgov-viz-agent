"""Tests for the in-process TTL + LRU response cache (app/cache.py).

Time is driven through an injected clock — no test sleeps.
"""

from __future__ import annotations

from app.api.schemas import ChartType, Meta, VisualizeResponse
from app.cache import RESPONSE_CACHE, ResponseCache, plan_cache_key
from app.plan.models import Plan

# --- fixtures / builders --------------------------------------------------


def make_plan(**overrides: object) -> Plan:
    """A minimal valid Plan; ``overrides`` tweak individual significant fields."""
    base: dict[str, object] = {
        "query_class": "distribution",
        "chart_type": ChartType.BAR,
    }
    base.update(overrides)
    return Plan(**base)  # type: ignore[arg-type]


def make_response(answer: str = "42") -> VisualizeResponse:
    """A minimal valid answer-kind envelope."""
    return VisualizeResponse(status="ok", kind="answer", answer=answer, meta=Meta())


class Clock:
    """A hand-cranked monotonic clock for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# --- key: determinism, sensitivity, and identity vs interpretation --------


def test_same_plan_same_key() -> None:
    assert plan_cache_key(make_plan()) == plan_cache_key(make_plan())


def test_key_is_hex_digest() -> None:
    key = plan_cache_key(make_plan())
    assert len(key) == 64
    int(key, 16)  # raises if not hex


def test_differing_entity_changes_key() -> None:
    a = plan_cache_key(make_plan(entities={}))
    b = plan_cache_key(make_plan(entities={"condition": "melanoma"}))
    assert a != b


def test_differing_filter_changes_key() -> None:
    a = plan_cache_key(make_plan(filters={"status": "RECRUITING"}))
    b = plan_cache_key(make_plan(filters={"status": "COMPLETED"}))
    assert a != b


def test_differing_field_changes_key() -> None:
    a = plan_cache_key(make_plan(field="phase"))
    b = plan_cache_key(make_plan(field="country"))
    assert a != b


def test_differing_chart_type_changes_key() -> None:
    a = plan_cache_key(make_plan(chart_type=ChartType.BAR))
    b = plan_cache_key(make_plan(chart_type=ChartType.TABLE))
    assert a != b


def test_key_handles_nested_series_submodels() -> None:
    """A compare plan with Series sub-models keys cleanly and stays sensitive."""
    p1 = make_plan(
        query_class="compare",
        chart_type=ChartType.GROUPED_BAR,
        series=[{"label": "A", "entities": {"drug": "a"}}],
    )
    p2 = make_plan(
        query_class="compare",
        chart_type=ChartType.GROUPED_BAR,
        series=[{"label": "B", "entities": {"drug": "b"}}],
    )
    assert plan_cache_key(p1) == plan_cache_key(p1)
    assert plan_cache_key(p1) != plan_cache_key(p2)


def test_key_ignores_notes_and_alternates() -> None:
    """notes/alternates are interpretation, not identity — must not change the key."""
    plain = make_plan()
    annotated = make_plan(
        notes=["CC-1 override echoed"],
        alternates=[ChartType.TABLE],
    )
    assert plan_cache_key(plain) == plan_cache_key(annotated)


# --- TTL behaviour --------------------------------------------------------


def test_hit_within_ttl() -> None:
    clock = Clock()
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True, now=clock)
    cache.set("k", make_response())
    clock.advance(100)  # still inside the 300s window
    assert cache.get("k") is not None


def test_miss_after_ttl_expiry_and_evicts() -> None:
    clock = Clock()
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True, now=clock)
    cache.set("k", make_response())
    clock.advance(301)  # past the window
    assert cache.get("k") is None
    assert len(cache) == 0  # the stale entry was evicted on read


def test_ttl_zero_always_miss() -> None:
    clock = Clock()
    cache = ResponseCache(ttl=0, max_entries=8, enabled=True, now=clock)
    cache.set("k", make_response())
    assert cache.get("k") is None  # 0 TTL == always-expired


# --- LRU behaviour --------------------------------------------------------


def test_lru_evicts_least_recently_used() -> None:
    clock = Clock()
    cache = ResponseCache(ttl=10_000, max_entries=2, enabled=True, now=clock)
    cache.set("a", make_response("a"))
    cache.set("b", make_response("b"))
    assert len(cache) == 2
    cache.set("c", make_response("c"))  # over capacity → evict LRU ("a")
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_get_refreshes_recency() -> None:
    clock = Clock()
    cache = ResponseCache(ttl=10_000, max_entries=2, enabled=True, now=clock)
    cache.set("a", make_response("a"))
    cache.set("b", make_response("b"))
    assert cache.get("a") is not None  # touch "a" → now "b" is LRU
    cache.set("c", make_response("c"))  # evicts "b", not the just-touched "a"
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


# --- bypass + lifecycle ---------------------------------------------------


def test_disabled_cache_always_misses() -> None:
    cache = ResponseCache(ttl=300, max_entries=8, enabled=False)
    cache.set("k", make_response())  # no-op
    assert cache.get("k") is None
    assert len(cache) == 0


def test_clear_empties() -> None:
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True)
    cache.set("a", make_response("a"))
    cache.set("b", make_response("b"))
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_roundtrip_returns_equal_but_independent_copy() -> None:
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True)
    resp = make_response("roundtrip")
    cache.set("k", resp)
    got = cache.get("k")
    assert got is not resp  # a DEEP COPY, not the same object (no shared alias)
    assert got.model_dump() == resp.model_dump()  # but value-equal


def test_cache_entry_is_isolated_from_caller_mutation() -> None:
    # The poisoning vector the Phase-5 review found: mutating a served envelope must
    # NOT corrupt the shared cache entry for the next caller.
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True)
    resp = make_response("iso")
    cache.set("k", resp)
    first = cache.get("k")
    first.meta.notes.append("POISON")  # caller mutates its own copy
    resp.meta.notes.append("ALSO_POISON")  # setter mutates its object post-store
    second = cache.get("k")
    assert "POISON" not in second.meta.notes
    assert "ALSO_POISON" not in second.meta.notes


def test_key_drives_a_real_hit() -> None:
    """End-to-end: the same plan's key retrieves what was stored under it."""
    cache = ResponseCache(ttl=300, max_entries=8, enabled=True)
    plan = make_plan(entities={"condition": "melanoma"}, field="phase")
    key = plan_cache_key(plan)
    cache.set(key, make_response("melanoma-phase"))
    hit = cache.get(plan_cache_key(make_plan(entities={"condition": "melanoma"}, field="phase")))
    assert hit is not None
    assert hit.answer == "melanoma-phase"


def test_module_singleton_is_a_response_cache() -> None:
    assert isinstance(RESPONSE_CACHE, ResponseCache)
