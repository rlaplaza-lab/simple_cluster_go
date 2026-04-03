"""Unified cache system for database operations.

This module provides a unified caching layer that consolidates multiple cache
implementations across the SCGO codebase. It provides thread-safe LRU caching
with namespace support and automatic eviction.

Key features:
- Thread-safe operation with locks
- LRU eviction when max size reached
- Namespace support for logical separation
- Global singleton for easy access
- Metrics tracking (hits, misses, evictions)

Example:
    >>> from scgo.database.cache import get_global_cache
    >>> cache = get_global_cache()
    >>> cache.set("db_results", "key1", [1, 2, 3])
    >>> value = cache.get("db_results", "key1")
    >>> cache.clear_namespace("db_results")
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class UnifiedCache:
    """Unified cache with namespace support and LRU eviction.

    This cache provides a centralized caching layer for the entire database
    system. It supports namespaces for logical separation of different types
    of cached data (e.g., "db_results", "metadata", "structures").

    The cache is thread-safe and uses LRU eviction when the maximum size
    is exceeded. Cache metrics are tracked for monitoring performance.

    Example:
        >>> cache = UnifiedCache(max_size=1000)
        >>> cache.set("namespace1", "key1", {"data": 42})
        >>> value = cache.get("namespace1", "key1")
        >>> print(value)
        {'data': 42}
        >>> cache.clear_namespace("namespace1")
        >>> print(cache.get_metrics())
        {'hits': 1, 'misses': 0, 'evictions': 0, 'size': 0}
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the unified cache.

        Args:
            max_size: Maximum number of total entries across all namespaces
                     before LRU eviction begins
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        self._cache: OrderedDict[tuple[str, Any], Any] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

        # Metrics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Track keys currently being computed to avoid duplicate work
        self._inflight: dict[tuple[str, Any], threading.Event] = {}

    def get(self, namespace: str, key: Any, default: Any = None) -> Any:
        """Get a value from the cache.

        Args:
            namespace: Logical namespace for the cache entry
            key: Cache key within the namespace
            default: Value to return if key not found

        Returns:
            Cached value if found, default otherwise
        """
        cache_key = (namespace, key)
        with self._lock:
            if cache_key in self._cache:
                # Move to end (mark as recently used)
                self._cache.move_to_end(cache_key)
                self._hits += 1
                return self._cache[cache_key]
            self._misses += 1
            return default

    def set(self, namespace: str, key: Any, value: Any) -> None:
        """Set a value in the cache.

        Args:
            namespace: Logical namespace for the cache entry
            key: Cache key within the namespace
            value: Value to cache
        """
        cache_key = (namespace, key)
        with self._lock:
            # Update or insert
            self._cache[cache_key] = value
            self._cache.move_to_end(cache_key)

            # Evict oldest if over limit
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def get_or_compute(
        self, namespace: str, key: Any, compute_func: Callable[[], Any]
    ) -> Any:
        """Get value from cache or compute and cache it.

        This method avoids duplicate computation under concurrency by tracking
        in-flight computations per (namespace, key) using threading.Event objects.

        Args:
            namespace: Logical namespace for the cache entry
            key: Cache key within the namespace
            compute_func: Function to call if key is not in cache

        Returns:
            Cached or computed value
        """
        cache_key = (namespace, key)

        while True:
            # Fast-path: try to read existing value
            with self._lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return self._cache[cache_key]

                inflight_ev = self._inflight.get(cache_key)

            if inflight_ev is not None:
                inflight_ev.wait()
                # Loop to re-check cache after waiter is signaled
                continue

            # Try to become the computing thread
            with self._lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return self._cache[cache_key]
                if cache_key in self._inflight:
                    # Someone else raced in; loop and wait
                    continue
                ev = threading.Event()
                self._inflight[cache_key] = ev

            # We are responsible for computing
            try:
                value = compute_func()
                with self._lock:
                    self._cache[cache_key] = value
                    self._cache.move_to_end(cache_key)
                    if len(self._cache) > self._max_size:
                        self._cache.popitem(last=False)
                        self._evictions += 1
                    self._misses += 1  # Track as miss since we had to compute
            finally:
                # Signal waiters and clean up
                ev.set()
                with self._lock:
                    # Remove the inflight marker if still present
                    if cache_key in self._inflight:
                        del self._inflight[cache_key]

            return value

    def clear_namespace(self, namespace: str) -> None:
        """Clear all entries in a specific namespace.

        Args:
            namespace: Namespace to clear
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache if key[0] == namespace]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            # Reset metrics
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_metrics(self) -> dict[str, int]:
        """Get cache performance metrics.

        Returns:
            Dictionary containing:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - evictions: Number of LRU evictions
                - size: Current number of cached entries
                - hit_rate: Hit rate as a percentage (0-100)
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (
                int(100 * self._hits / total_requests) if total_requests > 0 else 0
            )
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, item: tuple[str, Any]) -> bool:
        """Check if (namespace, key) is in the cache.

        Args:
            item: Tuple of (namespace, key)
        """
        with self._lock:
            return item in self._cache


# Global cache instance
_global_cache: UnifiedCache | None = None
_global_cache_lock = threading.Lock()


def get_global_cache(max_size: int = 1000) -> UnifiedCache:
    """Return the global cache singleton."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = UnifiedCache(max_size=max_size)
    return _global_cache


def reset_global_cache() -> None:
    """Reset the global cache singleton (for testing)."""
    global _global_cache
    with _global_cache_lock:
        _global_cache = None
