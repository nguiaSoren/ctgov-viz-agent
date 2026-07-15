"""Shared pytest fixtures.

The response cache (``app.cache.RESPONSE_CACHE``) is a module-global singleton.
Left alone it would leak a computed envelope from one test into the next (two
tests that resolve to the same normalized plan would see the first's result on
the second), making the suite order-dependent. This autouse fixture clears it
before every test so the cache still WORKS within a test (a single test issuing
two identical calls gets a hit) while staying isolated ACROSS tests.
"""

from __future__ import annotations

import pytest

from app.cache import RESPONSE_CACHE


@pytest.fixture(autouse=True)
def _clear_response_cache():
    RESPONSE_CACHE.clear()
    yield
    RESPONSE_CACHE.clear()
