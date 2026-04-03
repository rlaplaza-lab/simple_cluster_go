"""Tests for the unified cache system.

These tests cover basic semantics, eviction, and concurrent get_or_compute
to ensure the double-check logic prevents redundant computations under contention.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from scgo.database.cache import UnifiedCache, reset_global_cache


def test_unified_cache_max_size_validation():
    with pytest.raises(ValueError):
        UnifiedCache(max_size=0)


def test_unified_cache_eviction_and_len():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=2)
    namespace = "test_ns"

    c.set(namespace, "a", 1)
    c.set(namespace, "b", 2)
    c.set(namespace, "c", 3)

    assert c.get(namespace, "a") is None  # Evicted
    assert c.get(namespace, "b") == 2
    assert c.get(namespace, "c") == 3


def test_unified_cache_get_or_compute_concurrent():
    reset_global_cache()  # Clean slate
    calls = 0
    lock = threading.Lock()
    start_event = threading.Event()

    def compute():
        nonlocal calls
        # Wait for explicit signal to simulate expensive computation (deterministic)
        start_event.wait(timeout=1)
        with lock:
            calls += 1
        return "computed_value"

    cache = UnifiedCache(max_size=10)
    namespace = "test_ns"

    def worker():
        return cache.get_or_compute(namespace, "key", compute)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker) for _ in range(8)]
        # Allow compute to proceed now that workers are ready
        start_event.set()
        results = [f.result() for f in as_completed(futures)]

    assert all(r == "computed_value" for r in results)
    assert calls == 1, f"Compute function called {calls} times (expected 1)"


def test_unified_cache_namespace_isolation():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=20)

    # Set values in different namespaces
    c.set("ns1", "key", "value1")
    c.set("ns2", "key", "value2")

    # Should be isolated
    assert c.get("ns1", "key") == "value1"
    assert c.get("ns2", "key") == "value2"


def test_unified_cache_concurrent_get_or_compute():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=10)

    calls = 0
    lock = threading.Lock()
    start_event = threading.Event()

    def compute():
        nonlocal calls
        # Wait for an explicit signal to simulate expensive computation (deterministic)
        start_event.wait(timeout=1)
        with lock:
            calls += 1
        return "V"

    def worker():
        return c.get_or_compute("test_ns", "x", compute)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(worker) for _ in range(6)]
        # Allow compute to proceed now that workers are ready
        start_event.set()
        results = [f.result() for f in as_completed(futures)]

    assert all(r == "V" for r in results)
    assert calls == 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])
