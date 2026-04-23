"""SCGO minima search: which global optimizer is used for a given composition."""

from __future__ import annotations

from typing import Literal

from scgo.system_types import SystemType, get_system_policy

ScgoMinimaAlgorithm = Literal["simple", "bh", "ga"]


def select_scgo_minima_algorithm(
    n_atoms: int, system_type: SystemType
) -> ScgoMinimaAlgorithm:
    """Same rules as :func:`scgo.run_minima.run_scgo_trials` (no logging)."""
    policy = get_system_policy(system_type)
    simple_allowed = not policy.uses_surface and not policy.has_adsorbate
    if n_atoms <= 2 and simple_allowed:
        return "simple"
    if n_atoms == 3:
        return "bh"
    return "ga"
