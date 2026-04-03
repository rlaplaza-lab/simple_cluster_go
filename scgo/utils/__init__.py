"""Utility functions and helpers for cluster optimization.

This package contains various utility modules:

- helpers: Core utility functions for optimization and validation
- comparators: Structure comparison tools for duplicate detection
- mutation_weights: Adaptive mutation probability configuration
- atoms_helpers: ASE Atoms object manipulation utilities
- logging: Logging configuration and utilities
- rng_helpers: Random number generator utilities for reproducibility
- validation: Input validation utilities
"""

from __future__ import annotations

from .atoms_helpers import parse_energy_from_xyz_comment
from .comparators import PureInteratomicDistanceComparator
from .helpers import (
    auto_niter,
    auto_niter_local_relaxation,
    auto_population_size,
    get_cluster_formula,
    get_composition_counts,
    is_true_minimum,
    perform_local_relaxation,
)
from .logging import configure_logging, get_logger
from .mutation_weights import get_adaptive_mutation_config
from .rng_helpers import create_child_rng, ensure_rng

__all__ = [
    # atoms_helpers
    "parse_energy_from_xyz_comment",
    # comparators
    "PureInteratomicDistanceComparator",
    # helpers
    "auto_niter",
    "auto_niter_local_relaxation",
    "auto_population_size",
    "get_cluster_formula",
    "get_composition_counts",
    "is_true_minimum",
    "perform_local_relaxation",
    # logging
    "configure_logging",
    "get_logger",
    # mutation_weights
    "get_adaptive_mutation_config",
    # rng_helpers
    "create_child_rng",
    "ensure_rng",
]
