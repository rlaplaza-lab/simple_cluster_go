"""Centralized configuration constants for cluster initialization.

This module provides named constants for magic numbers scattered throughout
the initialization system, improving maintainability and documentation.
"""

from __future__ import annotations

from typing import Any

# Placement and retry parameters
MAX_PLACEMENT_ATTEMPTS_PER_ATOM = 500  # Reduced from 1000 for better performance
MAX_CONNECTIVITY_RETRIES = (
    10  # Increased from 5 to improve success rate for strict connectivity
)
MAX_CONSECUTIVE_FAILURES = (
    50  # Early termination: if this many consecutive attempts fail, abort early
)

# ============================================================================
# Distance and Connectivity Parameters
# ============================================================================
# Connectivity: atoms connected if distance <= (r_i + r_j) * CONNECTIVITY_FACTOR
CONNECTIVITY_FACTOR = 1.4  # Connectivity threshold used consistently throughout
MIN_DISTANCE_FACTOR_DEFAULT = 0.5
PLACEMENT_RADIUS_SCALING_DEFAULT = 1.2
SEED_CLASH_FACTOR = MIN_DISTANCE_FACTOR_DEFAULT  # Use same factor as random placement

# ============================================================================
# Cell and Vacuum Parameters
# ============================================================================
VACUUM_DEFAULT = 10.0
MAX_REASONABLE_CELL_SIDE = 1000.0  # Maximum reasonable cell side (Å)

# ============================================================================
# Boltzmann Sampling Parameters
# ============================================================================
BOLTZMANN_TEMPERATURE_MIN = 0.05
BOLTZMANN_TEMPERATURE_MAX = 0.5
ENERGY_SPREAD_TOLERANCE = 1e-6  # Tolerance for energy spread comparison
ENERGY_SPREAD_DIVISOR = 10.0  # Divisor for adaptive temperature calculation

# ============================================================================
# Convex Hull and Geometric Parameters
# ============================================================================
CONVEX_HULL_PERTURBATION_SCALE = 0.1
CONVEX_HULL_CACHE_SIZE = 100
# Cache sizes for initialization caches (tunable via configuration)
CANDIDATE_CACHE_SIZE = 100
COMPOSITION_CACHE_SIZE = 100
CONVEX_HULL_VOLUME_TOLERANCE = 1e-6  # Tolerance for degenerate convex hulls

# ============================================================================
# Magic Numbers and Template Parameters
# ============================================================================
# Magic number detection tolerance (atoms)
# Clusters within this many atoms of a magic number are considered "near"
MAGIC_NUMBER_TOLERANCE = 2

# Magic numbers for different structure types
# Based on Doye group study: https://doye.chem.ox.ac.uk/jon/structures/Morse/paper/node5.html
ICOSAHEDRAL_MAGIC_NUMBERS = [
    13,
    19,
    23,
    26,
    29,
    34,
    39,
    45,
    46,
    49,
    55,
    58,
    61,
    64,
    71,
    78,
    127,
    147,
    309,
]
DECAHEDRAL_MAGIC_NUMBERS = [7, 13, 23, 39, 55, 54, 85, 116, 147, 105, 156, 207]

# Regular polyhedra (Platonic solids) - vertices only
PLATONIC_SOLID_MAGIC_NUMBERS = [
    4,
    6,
    8,
    12,
    20,
]  # tetrahedron, octahedron, cube, icosahedron, dodecahedron

# Octahedral cluster magic numbers (from ASE Octahedron generator)
OCTAHEDRAL_MAGIC_NUMBERS = [6, 19, 44, 85]  # length=2,3,4,5 with cutoff=0

# Cubic cluster magic numbers (n×n×n cubes)
CUBIC_MAGIC_NUMBERS = [8, 27, 64, 125]  # 2³, 3³, 4³, 5³

# Tetrahedral cluster magic numbers (layered tetrahedra)
TETRAHEDRAL_MAGIC_NUMBERS = [4, 10, 20, 35]  # vertices, +1 shell, +2 shells, +3 shells

# Archimedean solids and related structures
ARCHIMEDEAN_MAGIC_NUMBERS = [
    12,
    13,
    24,
]  # cuboctahedron (12 vertices), cuboctahedron+center (13), truncated octahedron (24)

# Combined list of all magic numbers
MAGIC_NUMBERS = sorted(
    set(
        ICOSAHEDRAL_MAGIC_NUMBERS
        + DECAHEDRAL_MAGIC_NUMBERS
        + PLATONIC_SOLID_MAGIC_NUMBERS
        + OCTAHEDRAL_MAGIC_NUMBERS
        + CUBIC_MAGIC_NUMBERS
        + TETRAHEDRAL_MAGIC_NUMBERS
        + ARCHIMEDEAN_MAGIC_NUMBERS
    )
)

# ============================================================================
# Strategy and Diversity Parameters
# ============================================================================
# Number of seed combination strategies available
SEED_COMBINATION_STRATEGY_COUNT = (
    5  # 0=Boltzmann, 1=low-energy, 2=high-energy, 3=mid-energy, 4=random
)

# Number of growth order strategies available
GROWTH_ORDER_STRATEGY_COUNT = 6  # 0=random, 1=by_element, 2=alternating, 3=size_based, 4=element_clustering, 5=composition_balance

# Metropolis allocation scaling parameters
# These control logarithmic scaling for strategy allocation
TEMPLATE_BASE_PCT = 0.10  # Base percentage for template allocation
TEMPLATE_PREFACTOR = 2.0  # Scaling prefactor for templates (higher = more allocation with more templates)
SEED_BASE_PCT = 0.10  # Base percentage for seed+growth allocation
SEED_PREFACTOR = (
    1.5  # Scaling prefactor for seeds (higher = more allocation with more combinations)
)

# Template diversity enhancement
TEMPLATE_ROTATION_CANDIDATES = (
    3  # Number of rotation variants to generate per template for diversity
)

# Template weight configuration (base weights and thresholds)
TEMPLATE_BASE_WEIGHTS: dict[str, dict[str, Any]] = {
    "icosahedron": {"base": 1.5, "large_threshold": 20, "large_weight": 2.0},
    "decahedron": {"base": 1.3, "large_threshold": 20, "large_weight": 1.8},
    "cuboctahedron": {"base": 1.0, "size_range": (10, 30), "range_weight": 1.5},
    "octahedron": {"base": 1.0, "size_range": (10, 30), "range_weight": 1.4},
    "truncated_octahedron": {"base": 1.0, "size_range": (20, 30), "range_weight": 1.6},
    "cube": {"base": 0.8, "magic_sizes": [8, 27, 64, 125], "magic_weight": 1.3},
    "tetrahedron": {"base": 0.8, "small_threshold": 10, "small_weight": 1.2},
}

# Multi-element composition penalty factor
# Applied to icosahedron and decahedron for multi-element clusters
# (these structures are less favorable for mixed compositions)
MULTI_ELEMENT_TEMPLATE_PENALTY = 0.9

# Diversity boost factor for underrepresented template types
# Promotes exploration of less common template types
TEMPLATE_DIVERSITY_BOOST_FACTOR = 0.15

# ============================================================================
# Tolerance and Threshold Values
# ============================================================================
LINEAR_GEOMETRY_TOLERANCE = 1e-4  # Tolerance for linear geometry detection
ROTATION_AXIS_TOLERANCE = 1e-10  # Tolerance for rotation axis normalization
CLASH_TOLERANCE = (
    0.02  # Tolerance for clash detection (accounts for placement relaxation)
)
POSITION_COMPARISON_TOLERANCE_FACTOR = 0.05  # 5% of bond length for position comparison
SMART_FILTERING_PERTURBATION_SCALE = (
    0.3  # Reduced perturbation scale for smart facet filtering
)

# ============================================================================
# Relaxation and Scaling Factors
# ============================================================================
PLACEMENT_RELAXATION_FACTOR = 0.25  # Relaxation factor for placement attempts
MIN_DISTANCE_THRESHOLD_LOW = 0.4  # Lower threshold for min_distance_factor
MIN_DISTANCE_THRESHOLD_HIGH = 0.8  # Upper threshold for min_distance_factor
BOND_DISTANCE_MULTIPLIER_2ATOM = 1.2  # Multiplier for 2-atom bond distances
BOND_DISTANCE_MULTIPLIER_3ATOM = 3.0  # Multiplier for 3-atom bond distances
CONNECTIVITY_SUGGESTION_BUFFER = 1.05  # Buffer for connectivity factor suggestions

# ============================================================================
# Physical and Computational Parameters
# ============================================================================
PACKING_EFFICIENCY_FCC_HCP = 0.74  # FCC/HCP packing efficiency
KDTREE_THRESHOLD = 50  # Use KDTree for clusters with >= 50 atoms
