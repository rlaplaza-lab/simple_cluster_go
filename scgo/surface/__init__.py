"""Cluster-on-surface (adsorbate + slab) support for SCGO."""

from __future__ import annotations

from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.surface.deposition import (
    combine_slab_adsorbate,
    create_deposited_cluster,
    create_deposited_cluster_batch,
    slab_surface_extreme,
)
from scgo.surface.objectives import adsorption_energy
from scgo.surface.validation import validate_supported_cluster_deposit

__all__ = [
    "SurfaceSystemConfig",
    "adsorption_energy",
    "attach_slab_constraints",
    "combine_slab_adsorbate",
    "create_deposited_cluster",
    "create_deposited_cluster_batch",
    "slab_surface_extreme",
    "validate_supported_cluster_deposit",
]
