"""Reusable slab and surface presets for runners and benchmarks."""

from __future__ import annotations

from ase import Atoms
from ase.build import graphene

from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig

DEFAULT_GRAPHITE_SLAB_LAYERS = 5
DEFAULT_GRAPHITE_SLAB_REPEAT_XY = 4
DEFAULT_GRAPHITE_SLAB_VACUUM = 12.0


def build_graphite_slab(
    *,
    layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
) -> Atoms:
    """Build a graphite slab with periodic in-plane boundary conditions."""
    slab = graphene(formula="C2", vacuum=vacuum)
    slab = slab.repeat((repeat_xy, repeat_xy, max(1, layers)))
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)
    return slab


def make_graphite_surface_config(
    *,
    slab_layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    slab_repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> SurfaceSystemConfig:
    """Graphite slab preset (top layer relaxes with adsorbate during GO/NEB)."""
    slab = build_graphite_slab(
        layers=slab_layers, vacuum=vacuum, repeat_xy=slab_repeat_xy
    )
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=0.5,
        adsorption_height_max=1.0,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=1000,
        structure_connectivity_factor=structure_connectivity_factor,
    )


__all__ = [
    "DEFAULT_GRAPHITE_SLAB_LAYERS",
    "DEFAULT_GRAPHITE_SLAB_REPEAT_XY",
    "DEFAULT_GRAPHITE_SLAB_VACUUM",
    "build_graphite_slab",
    "make_graphite_surface_config",
]
