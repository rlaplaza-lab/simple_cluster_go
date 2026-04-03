"""Constants used across SCGO."""

from __future__ import annotations

PENALTY_ENERGY: float = 1.0e6
"""Penalty energy (eV) for failed optimizations."""

MIN_ATOMIC_DISTANCE_WARNING: float = 0.5
"""Minimum atomic distance (Å) for warnings."""

BOLTZMANN_K_EV_PER_K: float = 8.617e-5
"""Boltzmann constant (eV/K)."""

DEFAULT_ENERGY_TOLERANCE: float = 0.02
"""Default energy tolerance (eV)."""

DEFAULT_COMPARATOR_TOL: float = 0.015
"""Distance comparator tolerance (Å)."""

DEFAULT_PAIR_COR_MAX: float = 0.7
"""Max pairwise correlation coefficient."""

DEFAULT_PAIR_COR_CUM_DIFF: float = DEFAULT_COMPARATOR_TOL
"""Cumulative pair-correlation difference tolerance (same as DEFAULT_COMPARATOR_TOL)."""
