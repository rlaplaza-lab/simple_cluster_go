"""Full-system composition for slab + adsorbate (matches GA template ordering)."""

from __future__ import annotations

from scgo.surface.config import SurfaceSystemConfig


def full_adsorbate_slab_composition(
    adsorbate: list[str], surface_config: SurfaceSystemConfig
) -> list[str]:
    """Slab chemical symbols (first) + adsorbate, same as :func:`scgo.algorithms.geneticalgorithm_go.ga_go` surface template."""
    return list(surface_config.slab.get_chemical_symbols()) + list(adsorbate)
