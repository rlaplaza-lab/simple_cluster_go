"""Deprecated graphene-specific helpers — use ``scgo.runner_surface`` instead.

This module is kept only for backward compatibility and will be removed in a
future release.  All slab-agnostic helpers now live in
:mod:`scgo.runner_surface`; the graphene slab builder
(:func:`make_graphene_slab`) is retained here as a convenience but callers
are encouraged to construct their own slab and pass it directly.
"""

from __future__ import annotations

import warnings

from ase import Atoms
from ase.build import graphene

from scgo.runner_surface import (
    make_surface_config,
    read_full_composition_from_first_xyz,
)
from scgo.surface.config import SurfaceSystemConfig

warnings.warn(
    "scgo.runner_graphene is deprecated; use scgo.runner_surface instead. "
    "Build your slab with ASE and call make_surface_config(slab). "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "default_surface_config",
    "make_graphene_slab",
    "make_surface_config",
    "read_full_composition_from_first_xyz",
]


def make_graphene_slab() -> Atoms:
    """Build a small periodic graphene supercell with vacuum along z.

    Uses a 3x3 supercell (18 C atoms) to keep TorchSim+MACE batch relaxations
    within typical single-GPU memory; a 4x4 cell is ~32 substrate atoms and can OOM.
    """
    slab = graphene(size=(3, 3, 1), vacuum=12.0)
    slab.pbc = True
    return slab


def default_surface_config(slab: Atoms | None = None) -> SurfaceSystemConfig:
    """Surface GA config — **deprecated**, use :func:`make_surface_config` instead."""
    if slab is None:
        slab = make_graphene_slab()
    return make_surface_config(slab)
