"""Shared test constants and configurations.

This module centralizes test configuration values to avoid duplication
across test files and make it easier to adjust test parameters globally.
"""

# Random seed configurations for reproducibility testing
REPRODUCIBILITY_SEEDS = [42, 123, 456, 789, 1001, 2022, 3033, 4044, 5055, 6066]
QUICK_SEEDS = [42, 123]  # For fast tests

# Cluster size configurations
SMALL_SIZES = [4, 6, 8, 10]
MEDIUM_SIZES = [15, 20, 25, 30]
LARGE_SIZES = [40, 50, 60]
ALL_SIZES = SMALL_SIZES + MEDIUM_SIZES + LARGE_SIZES

# Composition configurations
SINGLE_ELEMENT_COMPOSITIONS = [
    ["Pt"],
    ["Pt", "Pt"],
    ["Pt", "Pt", "Pt"],
    ["Pt"] * 5,
]

BIMETALLIC_COMPOSITIONS = [
    ["Pt", "Au"],
    ["Pt", "Pd"],
    ["Au", "Pd"],
]

MIXED_COMPOSITIONS = {
    "PtAu": lambda n: ["Pt", "Au"] * (n // 2) + ["Pt"] * (n % 2),
    "PtPd": lambda n: ["Pt", "Pd"] * (n // 2) + ["Pt"] * (n % 2),
    "AuPdPt": lambda n: (["Au", "Pd", "Pt"] * ((n // 3) + 1))[:n],
}

# Initialization modes
INITIALIZATION_MODES = ["random_spherical", "seed+growth", "template", "smart"]
FAST_MODES = ["random_spherical", "smart"]  # Faster modes for quick tests
SLOW_MODES = ["seed+growth", "template"]  # Slower modes, use sparingly

# Batch testing configurations
BATCH_TEST_SAMPLES = 100  # Number of samples for batch tests
BATCH_TEST_SAMPLES_SLOW = 15  # For slow batch tests
UNIQUENESS_THRESHOLD = 0.8  # Minimum uniqueness ratio (80%)

# Diversity testing thresholds
DIVERSITY_THRESHOLD_MIN = 0.6  # Minimum diversity threshold
DIVERSITY_THRESHOLD_DEFAULT = 0.7  # Default diversity threshold
DIVERSITY_TEST_SAMPLES_SMALL = 10
DIVERSITY_TEST_SAMPLES_MEDIUM = 15
DIVERSITY_TEST_SAMPLES_LARGE = 20

# RNG seed range for random sampling
RNG_SEED_RANGE = (0, 100000)

# Default parameters for algorithms
DEFAULT_BH_PARAMS = {
    "niter": 3,
    "dr": 0.3,
    "niter_local_relaxation": 3,
    "temperature": 0.01,
}

DEFAULT_GA_PARAMS = {
    "niter": 2,
    "population_size": 3,
    "niter_local_relaxation": 3,
}

# Geometry parameters
CONNECTIVITY_FACTOR_DEFAULT = 1.4
MIN_DISTANCE_FACTOR_DEFAULT = 0.5

# Magic number cluster sizes (known stable structures)
MAGIC_NUMBER_SIZES = {
    "icosahedron": [13, 55, 147, 309],
    "decahedron": [7, 13, 23, 39, 55],
    "octahedron": [6, 19, 44, 85, 146],
}
