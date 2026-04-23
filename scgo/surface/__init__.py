"""Cluster-on-surface (adsorbate + slab) support for SCGO."""

from __future__ import annotations

from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import (
    attach_slab_constraints,
    attach_slab_constraints_from_surface_config,
    surface_slab_constraint_summary,
)
from scgo.surface.deposition import (
    combine_slab_adsorbate,
    create_deposited_cluster,
    create_deposited_cluster_batch,
    slab_surface_extreme,
)
from scgo.surface.objectives import adsorption_energy
from scgo.surface.presets import (
    DEFAULT_GRAPHITE_SLAB_LAYERS,
    DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    DEFAULT_GRAPHITE_SLAB_VACUUM,
    build_graphite_slab,
    make_graphite_surface_config,
)
from scgo.surface.validation import (
    validate_stored_slab_adsorbate_metadata,
    validate_supported_cluster_deposit,
    validate_surface_config_slab_prefix,
)

__all__ = [
    "SurfaceSystemConfig",
    "full_adsorbate_slab_composition",
    "adsorption_energy",
    "attach_slab_constraints",
    "attach_slab_constraints_from_surface_config",
    "surface_slab_constraint_summary",
    "combine_slab_adsorbate",
    "create_deposited_cluster",
    "create_deposited_cluster_batch",
    "slab_surface_extreme",
    "validate_stored_slab_adsorbate_metadata",
    "validate_supported_cluster_deposit",
    "validate_surface_config_slab_prefix",
    "DEFAULT_GRAPHITE_SLAB_LAYERS",
    "DEFAULT_GRAPHITE_SLAB_REPEAT_XY",
    "DEFAULT_GRAPHITE_SLAB_VACUUM",
    "build_graphite_slab",
    "make_graphite_surface_config",
]
