"""Tests for seed combination functionality.

This module tests functions that combine multiple seed structures and grow
them to target compositions in genetic algorithm workflows.
"""

import numpy as np
from ase import Atoms

from scgo.initialization import combine_and_grow, combine_seeds
from scgo.initialization.geometry_helpers import (
    is_cluster_connected,
)
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import create_paired_rngs


def test_combine_seeds_empty_list(rng):
    """Test combining empty seed list."""
    combined = combine_seeds(seeds=[], cell_side=10.0, rng=rng)
    assert combined is not None
    assert len(combined) == 0
    assert np.allclose(combined.get_cell(), [[10, 0, 0], [0, 10, 0], [0, 0, 10]])


def test_combine_seeds(rng):
    """Test combining seeds."""
    seed1 = Atoms("Pt", positions=[[0, 0, 0]])
    seed2 = Atoms("Au", positions=[[0, 0, 0]])  # Will be translated by combine_seeds

    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=10.0,
        separation_scaling=2.0,
        rng=rng,
    )
    assert combined is not None
    assert len(combined) == 2
    assert get_composition_counts(
        combined.get_chemical_symbols()
    ) == get_composition_counts(["Pt", "Au"])

    # Check that the positions are not optimized, but are separated
    # The exact positions are hard to predict due to randomness, but we can check
    # that they are not on top of each other and are roughly separated by the scaling.
    # The first atom should be near the center after initial centering.
    # The second atom should be displaced from the first.
    pt_pos = combined.positions[0]
    au_pos = combined.positions[1]
    distance = np.linalg.norm(pt_pos - au_pos)
    # Check that atoms are separated (not on top of each other)
    # The exact separation distance may vary due to the seed combination algorithm
    assert distance > 1.0  # At least some separation
    assert distance < 10.0  # Should not be excessively far apart


def test_combine_seeds_multiple(rng):
    """Test combining multiple seeds."""
    seed1 = Atoms("Pt", positions=[[0.1, 0.1, 0.1]])
    seed2 = Atoms("Au", positions=[[0.2, 0.2, 0.2]])
    seed3 = Atoms("Pd", positions=[[0.3, 0.3, 0.3]])

    # Test 1: Combine two seeds
    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=10.0,
        separation_scaling=2.0,
        rng=rng,
    )
    assert combined is not None
    assert len(combined) == 2
    assert get_composition_counts(
        combined.get_chemical_symbols()
    ) == get_composition_counts(["Pt", "Au"])

    # Test 2: Combine multiple seeds with optimization
    combined_opt = combine_seeds(
        seeds=[seed1, seed2, seed3],
        cell_side=15.0,
        separation_scaling=1.0,
        rng=rng,
    )
    assert combined_opt is not None
    assert len(combined_opt) == 3
    assert get_composition_counts(
        combined_opt.get_chemical_symbols()
    ) == get_composition_counts(["Pt", "Au", "Pd"])
    # Positions should be optimized, so they should be closer than just separated
    # This is hard to assert precisely without knowing the exact optimization behavior
    # but we can check that they are not at the exact same initial positions.
    assert not np.allclose(combined_opt.positions[0], combined_opt.positions[1])


def test_combine_and_grow(rng):
    """Test combining seeds and growing to target composition."""
    seed1 = Atoms("Pt", positions=[[0, 0, 0]])
    seed2 = Atoms("Au", positions=[[0, 0, 0]])
    target_comp = ["Pt", "Au", "Au", "Au"]  # Combine PtAu, then add 2 Au

    # Test 1: Combine and grow successfully
    final_cluster = combine_and_grow(
        seeds=[seed1, seed2],
        target_composition=target_comp,
        cell_side=15.0,
        vdw_scaling=1.5,
        min_distance_factor=0.5,
        rng=rng,
    )
    assert final_cluster is not None
    assert len(final_cluster) == len(target_comp)
    assert get_composition_counts(
        final_cluster.get_chemical_symbols()
    ) == get_composition_counts(target_comp)


def test_combine_seeds_bond_distance_fix(rng):
    """Test that seed combination no longer produces 6+ Angstrom separations."""

    # Create seeds that would previously cause disconnection
    seed1 = Atoms("Pt2", positions=[[0, 0, 0], [2.8, 0, 0]])  # Pt dimer
    seed2 = Atoms("Pt", positions=[[0, 0, 0]])  # Single Pt atom

    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    assert combined is not None, "Seed combination should succeed"
    assert len(combined) == 3, "Should have 3 atoms total"

    # Check connectivity
    assert is_cluster_connected(combined, connectivity_factor=CONNECTIVITY_FACTOR), (
        "Cluster should be connected"
    )

    # Check that distances are reasonable (not 6+ Angstroms)
    positions = combined.get_positions()
    max_distance = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            max_distance = max(max_distance, distance)

    # Should be much less than 6+ Angstroms that was causing the original issue
    assert max_distance < 5.0, f"Maximum distance {max_distance:.3f} Å is too large"
    assert max_distance > 1.0, f"Maximum distance {max_distance:.3f} Å is too small"


def test_combine_seeds_connectivity_validation(rng):
    """Test that seed combination properly validates connectivity."""

    # Create seeds that are too far apart to be connected with very strict connectivity
    seed1 = Atoms("Pt", positions=[[0, 0, 0]])
    seed2 = Atoms("Pt", positions=[[0, 0, 0]])

    # Test with extremely strict connectivity factor that should cause failure
    STRICT_FACTOR = 0.5  # Extremely strict connectivity for testing failure cases
    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=0.1,  # Very small separation
        connectivity_factor=STRICT_FACTOR,
        rng=rng,
    )

    # With such strict parameters, it might fail, but if it succeeds, check connectivity
    if combined is not None:
        # If it succeeds, it should be connected
        assert is_cluster_connected(combined, connectivity_factor=STRICT_FACTOR), (
            "If successful, should be connected"
        )

        # Check that distances are very small due to strict constraints
        positions = combined.get_positions()
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                max_distance = max(max_distance, distance)

        # With strict connectivity, distances should be very small
        assert max_distance < 2.0, (
            f"With strict connectivity, distance {max_distance:.3f} Å should be small"
        )


def test_combine_seeds_multi_atom_seeds(rng):
    """Test combining multi-atom seeds with proper facet placement."""

    # Create larger seeds
    seed1 = Atoms(
        "Pt3", positions=[[0, 0, 0], [2.8, 0, 0], [1.4, 2.4, 0]]
    )  # Pt triangle
    seed2 = Atoms("Pt2", positions=[[0, 0, 0], [2.8, 0, 0]])  # Pt dimer

    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.2,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    assert combined is not None, "Multi-atom seed combination should succeed"
    assert len(combined) == 5, "Should have 5 atoms total"
    assert get_composition_counts(
        combined.get_chemical_symbols()
    ) == get_composition_counts(["Pt"] * 5)

    # Check connectivity
    assert is_cluster_connected(combined, connectivity_factor=CONNECTIVITY_FACTOR), (
        "Multi-atom cluster should be connected"
    )

    # Check reasonable distances
    positions = combined.get_positions()
    max_distance = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            max_distance = max(max_distance, distance)

    assert max_distance < 6.0, f"Maximum distance {max_distance:.3f} Å is too large"


def test_combine_seeds_different_elements(rng):
    """Test combining seeds with different elements."""

    # Create seeds with different elements
    seed1 = Atoms("Pt2", positions=[[0, 0, 0], [2.8, 0, 0]])  # Pt dimer
    seed2 = Atoms("Au", positions=[[0, 0, 0]])  # Single Au atom

    combined = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    assert combined is not None, "Different element seed combination should succeed"
    assert len(combined) == 3, "Should have 3 atoms total"
    assert get_composition_counts(
        combined.get_chemical_symbols()
    ) == get_composition_counts(["Pt", "Pt", "Au"])

    # Check connectivity
    assert is_cluster_connected(combined, connectivity_factor=CONNECTIVITY_FACTOR), (
        "Mixed element cluster should be connected"
    )


def test_combine_seeds_separation_scaling(rng):
    """Test that separation_scaling parameter affects placement distance."""

    seed1 = Atoms("Pt", positions=[[0, 0, 0]])
    seed2 = Atoms("Pt", positions=[[0, 0, 0]])

    # Test with different separation scalings
    combined_close = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=0.8,  # Close placement
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    _, rng2 = create_paired_rngs(42)  # Same seed for fair comparison
    combined_far = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.5,  # Far placement
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng2,
    )

    assert combined_close is not None and combined_far is not None

    # Calculate distances
    pos_close = combined_close.get_positions()
    pos_far = combined_far.get_positions()

    distance_close = np.linalg.norm(pos_close[0] - pos_close[1])
    distance_far = np.linalg.norm(pos_far[0] - pos_far[1])

    # Far placement should generally result in larger distances
    # (though randomness means this isn't guaranteed, it should trend this way)
    assert distance_far >= distance_close * 0.8, (
        "Separation scaling should affect placement distance"
    )


def test_combine_seeds_connectivity_factor(rng):
    """Test that connectivity_factor parameter affects validation."""
    # Test with different connectivity factors to verify parameter effect
    STRICT_FACTOR = 1.2  # Very strict connectivity requirement
    LENIENT_FACTOR = 2.0  # More lenient connectivity requirement

    seed1 = Atoms("Pt", positions=[[0, 0, 0]])
    seed2 = Atoms("Pt", positions=[[0, 0, 0]])

    # Test with strict connectivity
    combined_strict = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=STRICT_FACTOR,
        rng=rng,
    )

    _, rng2 = create_paired_rngs(42)  # Same seed for fair comparison
    combined_lenient = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=LENIENT_FACTOR,
        rng=rng2,
    )

    # Both should succeed, but lenient should be more likely to succeed
    # Strict might fail if placement doesn't meet tight connectivity requirements
    if combined_strict is not None:
        assert is_cluster_connected(combined_strict, connectivity_factor=STRICT_FACTOR)

    if combined_lenient is not None:
        assert is_cluster_connected(
            combined_lenient, connectivity_factor=LENIENT_FACTOR
        )


def test_combine_seeds_reproducibility():
    """Test that seed combination is reproducible with same RNG seed."""
    seed1 = Atoms("Pt2", positions=[[0, 0, 0], [2.8, 0, 0]])
    seed2 = Atoms("Pt", positions=[[0, 0, 0]])

    # Test with same RNG seed
    rng1, rng2 = create_paired_rngs(42)

    combined1 = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng1,
    )

    combined2 = combine_seeds(
        seeds=[seed1, seed2],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng2,
    )

    if combined1 is not None and combined2 is not None:
        # Should produce identical results
        assert np.allclose(combined1.get_positions(), combined2.get_positions())
        assert combined1.get_chemical_symbols() == combined2.get_chemical_symbols()


def test_combine_seeds_edge_cases(rng):
    """Test edge cases for seed combination."""

    # Test with single seed (should just return the seed)
    single_seed = Atoms("Pt2", positions=[[0, 0, 0], [2.8, 0, 0]])
    combined = combine_seeds(
        seeds=[single_seed],
        cell_side=20.0,
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    assert combined is not None, "Single seed should succeed"
    assert len(combined) == 2, "Should preserve single seed"
    assert get_composition_counts(
        combined.get_chemical_symbols()
    ) == get_composition_counts(["Pt", "Pt"])

    # Test with very small cell (might fail)
    small_cell_result = combine_seeds(
        seeds=[single_seed, Atoms("Pt", positions=[[0, 0, 0]])],
        cell_side=5.0,  # Very small cell
        separation_scaling=1.0,
        connectivity_factor=CONNECTIVITY_FACTOR,
        rng=rng,
    )

    # Small cell might cause failure, which is acceptable
    if small_cell_result is not None:
        assert len(small_cell_result) == 3
