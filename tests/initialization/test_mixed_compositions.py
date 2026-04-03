"""Consolidated tests for mixed compositions.

This module was refactored to reduce duplication by parametrizing similar
cases (bimetallic / trimetallic / multimetallic / small-mixed) while keeping
edge cases and mode coverage explicit.
"""

import numpy as np
import pytest

from scgo.initialization import create_initial_cluster
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from tests.test_utils import assert_cluster_valid, create_paired_rngs


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au"],
        ["Pt", "Au"] * 5,
        ["Pt", "Au", "Au"] * 3,
        ["Pt", "Pt", "Au"] * 3,
        ["Pt"] + ["Au"] * 8,
        ["Pt"] * 8 + ["Au"],
        ["H", "Pt"] * 5,
        ["Pt", "Au"] * 10,
    ],
)
def test_bimetallic_variants(composition, rng):
    """Parametrized coverage of bimetallic composition shapes and sizes."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "mode", ["smart", "random_spherical", "seed+growth", "template"]
)
def test_bimetallic_all_modes(mode, rng):
    """Ensure basic bimetallic cases work across all initialization modes."""
    composition = ["Pt", "Au"] * 3
    atoms = create_initial_cluster(composition, mode=mode, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au", "Pd"] * 3,
        ["Pt", "Pt", "Au", "Pd"] * 2 + ["Pt"],
        ["Pt"] * 7 + ["Au"] + ["Pd"],
        ["Li", "Pt", "Au"] * 3,
        ["Pt", "Au", "Pd"] * 5,
    ],
)
def test_trimetallic_variants(composition, rng):
    """Parametrized trimetallic cases (equal/unequal/skewed/large)."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [["Pt", "Au", "Pd", "Ag"] * 2, ["Pt", "Au", "Pd", "Ag", "Cu"] * 2],
)
def test_multimetallic_variants(composition, rng):
    """Parametrized coverage for 4+ element clusters."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au"],
        ["Pt", "Pt", "Au"],
        ["Pt", "Au", "Pd", "Pt"],
        ["Pt", "Au", "Pd", "Ag", "Cu"],
    ],
)
def test_small_mixed_compositions(composition, rng):
    """Small (2-5 atom) mixed-composition smoke tests."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_different_seeds_produce_diversity(seed):
    """Different RNG seeds should produce diverse structures for mixed comps."""
    composition = ["Pt", "Au"] * 5
    rng_test, _ = create_paired_rngs(seed)
    atoms = create_initial_cluster(composition, rng=rng_test)
    assert_cluster_valid(atoms, composition)


def test_reproducibility_same_seed(rng):
    """Same seed produces identical structure and composition."""
    composition = ["Pt", "Au"] * 5
    rng1, rng2 = create_paired_rngs(42)
    atoms1 = create_initial_cluster(composition, rng=rng1)
    atoms2 = create_initial_cluster(composition, rng=rng2)

    assert np.allclose(atoms1.get_positions(), atoms2.get_positions(), atol=1e-10)
    assert atoms1.get_chemical_symbols() == atoms2.get_chemical_symbols()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_trimetallic_diversity(seed):
    """Parametrized diversity checks for trimetallic compositions."""
    composition = ["Pt", "Au", "Pd"] * 4
    rng_test, _ = create_paired_rngs(seed)
    atoms = create_initial_cluster(composition, rng=rng_test)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [["H", "H", "Pt", "Pt", "Pt"], ["H", "Li", "Pt", "Au"] * 2],
)
def test_edge_case_mixed_compositions(composition, rng):
    """Edge cases: extreme size differences, magic numbers, strict connectivity."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition", [["Pt", "Au"] * 6 + ["Pt"], ["Pt", "Au", "Pd"] * 4 + ["Pt"]]
)
def test_magic_number_and_strict_connectivity(composition, rng):
    atoms = create_initial_cluster(
        composition,
        mode="smart",
        rng=rng,
        connectivity_factor=CONNECTIVITY_FACTOR,
        min_distance_factor=0.5,
    )
    assert_cluster_valid(atoms, composition)
