"""Global optimization algorithms for atomic clusters.

This package contains implementations of various global optimization algorithms
adapted for atomic cluster structure search:

- Simple: Single optimization for 1-2 atom clusters
- Basin Hopping: Random perturbations with Metropolis acceptance
- Genetic Algorithm: Population-based evolution with crossover and mutations
- TorchSim-enhanced GA: GPU-accelerated genetic algorithm using TorchSim

.. warning::
    These functions are primarily for internal use. Most users should use the
    high-level API in :mod:`scgo.run_minima` (e.g., :func:`run_scgo_trials`) instead
    of calling these algorithm functions directly.
"""

from __future__ import annotations

from .basinhopping_go import bh_go
from .geneticalgorithm_go import ga_go
from .geneticalgorithm_go_torchsim import ga_go_torchsim
from .simple_go import simple_go

__all__ = [
    "bh_go",
    "ga_go",
    "ga_go_torchsim",
    "simple_go",
]
