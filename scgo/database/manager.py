"""Unified database manager for SCGO.

Provides a high-level interface for all database operations with built-in
caching and consistent error handling.
"""

from __future__ import annotations

import time
from pathlib import Path

from ase import Atoms

from scgo.database.cache import get_global_cache
from scgo.database.discovery import DatabaseDiscovery
from scgo.database.helpers import (
    load_previous_run_results,
    load_reference_structures,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


class SCGODatabaseManager:
    """Lightweight database manager for SCGO operations.

    Provides a high-level cached interface for loading previous run results
    and reference structures for diversity calculations.

    Example:
        >>> with SCGODatabaseManager(base_dir="output") as manager:
        ...     refs = manager.load_diversity_references("**/*.db")
    """

    def __init__(
        self,
        base_dir: str | Path,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize database manager.

        Args:
            base_dir: Base directory for database operations
            enable_caching: Whether to cache loaded results (default True)
            cache_ttl_seconds: Cache time-to-live in seconds (default 300)
        """
        self.base_dir = Path(base_dir)
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds

        # Initialize discovery service
        self._discovery = DatabaseDiscovery(base_dir)

        # Initialize unified cache
        self._cache = get_global_cache()
        self._cache_namespace = "db_manager"
        self._cache_timestamps: dict[tuple, float] = {}

        logger.debug(
            f"Initialized SCGODatabaseManager: base_dir={base_dir}, "
            f"caching={'enabled' if enable_caching else 'disabled'}"
        )

    def _is_cache_valid(self, cache_key: tuple) -> bool:
        """Check if cache entry is still valid based on TTL.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache is valid, False if expired
        """
        if not self.enable_caching:
            return False

        if cache_key not in self._cache_timestamps:
            return False

        age = time.time() - self._cache_timestamps[cache_key]
        return age < self.cache_ttl_seconds

    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear_namespace(self._cache_namespace)
        self._cache_timestamps.clear()
        logger.info("Cleared all caches")

    def load_previous_results(
        self,
        composition: list[str],
        current_run_id: str | None = None,
        db_filename: str | None = None,
        force_reload: bool = False,
        prefer_final_unique: bool = True,
    ) -> list[tuple[float, Atoms]]:
        """Load all minima from previous runs for this composition.

        Results are cached for improved performance on repeated calls.

        Args:
            composition: List of atomic symbols to filter by
            current_run_id: Current run ID to exclude from loading
            db_filename: Specific database filename to look for (optional)
            force_reload: Force reload from disk, bypassing cache
            prefer_final_unique: If True (default), only ``final_unique_minimum``
                rows are loaded. Set False to include all relaxed structures.

        Returns:
            List of (energy, Atoms) tuples from all previous runs
        """
        formula = get_cluster_formula(composition)
        cache_key = (
            "prev_results",
            tuple(composition),
            current_run_id,
            db_filename,
            prefer_final_unique,
        )

        # Check cache
        if not force_reload and self._is_cache_valid(cache_key):
            logger.debug("Using cached previous results for %s", formula)
            return self._cache.get(self._cache_namespace, cache_key)

        logger.info("Loading previous results for %s", formula)

        minima = load_previous_run_results(
            base_output_dir=str(self.base_dir),
            db_filename=db_filename,
            composition=composition,
            current_run_id=current_run_id,
            prefer_final_unique=prefer_final_unique,
        )

        # Cache results
        if self.enable_caching:
            self._cache.set(self._cache_namespace, cache_key, minima)
            self._cache_timestamps[cache_key] = time.time()

        logger.info("Loaded %s minima from previous runs", len(minima))
        return minima

    def load_reference_structures(
        self,
        db_glob_pattern: str,
        composition: list[str] | None = None,
        max_structures: int = 100,
        force_reload: bool = False,
    ) -> list[Atoms]:
        """Load reference structures for diversity calculation.

        Results are cached for improved performance.

        Args:
            db_glob_pattern: Glob pattern to find database files
            composition: Optional composition filter
            max_structures: Maximum number of structures to load
            force_reload: Force reload from disk, bypassing cache

        Returns:
            List of Atoms objects sorted by energy (lowest first)
        """
        cache_key = (
            "ref_structs",
            db_glob_pattern,
            tuple(composition) if composition else None,
            max_structures,
        )

        # Check cache
        if not force_reload and self._is_cache_valid(cache_key):
            if composition:
                formula = get_cluster_formula(composition)
                logger.debug("Using cached reference structures for %s", formula)
            else:
                logger.debug("Using cached reference structures")
            return self._cache.get(self._cache_namespace, cache_key)

        if composition:
            formula = get_cluster_formula(composition)
            logger.info("Loading reference structures for %s", formula)
        else:
            logger.info("Loading reference structures (all compositions)")

        # Make glob pattern relative to base_dir
        full_pattern = str(self.base_dir / db_glob_pattern)

        structures = load_reference_structures(
            db_glob_pattern=full_pattern,
            composition=composition,
            max_structures=max_structures,
            base_dir=self.base_dir,
        )

        # Cache results
        if self.enable_caching:
            self._cache.set(self._cache_namespace, cache_key, structures)
            self._cache_timestamps[cache_key] = time.time()

        logger.info("Loaded %s reference structures", len(structures))
        return structures

    def load_diversity_references(
        self,
        glob_pattern: str,
        composition: list[str] | None = None,
        max_structures: int = 100,
        use_cache: bool = True,
    ) -> list[Atoms]:
        """Alias for :meth:`load_reference_structures`."""
        return self.load_reference_structures(
            db_glob_pattern=glob_pattern,
            composition=composition,
            max_structures=max_structures,
            force_reload=not use_cache,
        )

    def close(self):
        """Release resources held by manager caches."""
        self.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
