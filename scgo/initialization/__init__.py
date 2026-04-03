"""Cluster initialization package.

Main entry points:
- create_initial_cluster: Main initialization function (smart mode by default)
- random_spherical: Random spherical placement
- combine_and_grow: Seed combination and growth
- generate_template_structure: Generate icosahedral/decahedral/octahedral templates
"""

from __future__ import annotations

from .geometry_helpers import (
    StructureDiagnostics,
    get_covalent_radius,
    get_structure_diagnostics,
    is_cluster_connected,
    validate_cluster,
    validate_cluster_structure,
)
from .initializers import (
    compute_cell_side,
    create_initial_cluster,
    create_initial_cluster_batch,
    generate_initial_population,
)
from .random_spherical import (
    grow_from_seed,
    random_spherical,
)
from .seed_combiners import combine_and_grow, combine_seeds
from .templates import (
    generate_cube,
    generate_cuboctahedron,
    generate_decahedron,
    generate_icosahedron,
    generate_octahedron,
    generate_template_structure,
    generate_tetrahedron,
    generate_truncated_octahedron,
    get_nearest_magic_number,
    is_near_magic_number,
)

__all__ = [
    # Main functions
    "create_initial_cluster",
    "create_initial_cluster_batch",
    "generate_initial_population",
    "random_spherical",
    "grow_from_seed",
    "combine_seeds",
    "combine_and_grow",
    "compute_cell_side",
    "is_cluster_connected",
    "validate_cluster",
    "validate_cluster_structure",
    # Diagnostics and utilities
    "StructureDiagnostics",
    "get_covalent_radius",
    "get_structure_diagnostics",
    # Template functions
    "generate_icosahedron",
    "generate_decahedron",
    "generate_octahedron",
    "generate_tetrahedron",
    "generate_cube",
    "generate_cuboctahedron",
    "generate_truncated_octahedron",
    "generate_template_structure",
    "get_nearest_magic_number",
    "is_near_magic_number",
]
