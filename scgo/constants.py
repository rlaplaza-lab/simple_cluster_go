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
"""Cumulative structure-comparator difference tolerance (normalized, unitless)."""

DEFAULT_PAIR_COR_MAX: float = 0.7
"""Max single interatomic-distance difference (Å) for GO uniqueness comparison."""

DEFAULT_TS_PAIR_COR_MAX: float = 0.1
"""Max single interatomic-distance difference (Å) for TS pair near-dupe gating.

Tighter than :data:`DEFAULT_PAIR_COR_MAX` because TS pairing must reject near-
duplicates before NEB, while GO uniqueness tolerates more structural variation.
"""

DEFAULT_FMAX_THRESHOLD: float = 0.05
"""Default local-relaxation / Hessian-validation force threshold (eV/Å)."""

DEFAULT_NEB_TANGENT_METHOD: str = "improvedtangent"
"""ASE :class:`ase.mep.neb.NEB` tangent method used by default."""

SURFACE_GA_MIN_LOCAL_RELAX_STEPS: int = 400
"""Minimum local-relaxation steps for GA with ``surface_config`` (slab adsorption)."""
