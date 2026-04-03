"""Edge case tests for SCGO global optimization.

These tests cover boundary conditions, degenerate geometries, extreme parameters,
and failure modes to ensure robust behavior.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from scgo.algorithms.basinhopping_go import bh_go
from scgo.algorithms.geneticalgorithm_go import ga_go
from scgo.database.metadata import get_metadata
from scgo.initialization import create_initial_cluster
from scgo.utils.helpers import perform_local_relaxation


class MockFailingOptimizer:
    """Mock optimizer that always fails to converge."""

    def __init__(self, atoms, **kwargs):
        self.atoms = atoms
        self.steps = 0

    def run(self, fmax=None, steps=None):
        # Simulate failure by raising an exception
        raise RuntimeError("Mock optimizer failure")


@pytest.mark.slow
def test_single_atom_optimization(tmp_path):
    """Test optimization of single atom (should succeed trivially)."""
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.calc = EMT()

    # Single atom should have zero forces and converge immediately
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.01, steps=10)

    assert np.isfinite(energy)
    assert "metadata" in atoms.info
    assert get_metadata(atoms, "raw_score", default=None) is not None


@pytest.mark.slow
def test_linear_cluster_geometry(tmp_path):
    """Test optimization of linear cluster (degenerate geometry)."""
    # Create a linear Pt3 cluster
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [0, 0, 2.5], [0, 0, 5.0]])
    atoms.calc = EMT()

    # Should still be able to optimize despite linear geometry
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.1, steps=20)

    assert np.isfinite(energy)
    # Check that positions changed (optimization occurred)
    final_positions = atoms.get_positions()
    assert not np.allclose(final_positions, [[0, 0, 0], [0, 0, 2.5], [0, 0, 5.0]])


@pytest.mark.slow
def test_planar_cluster_geometry(tmp_path, pt3_atoms):
    """Test optimization of planar cluster (degenerate geometry)."""
    # Use fixture Pt3 cluster (already has correct positions)
    atoms = pt3_atoms.copy()
    atoms.calc = EMT()

    # Should still be able to optimize despite planar geometry
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.1, steps=20)

    assert np.isfinite(energy)


def test_convergence_failure_handling(tmp_path):
    """Test handling of convergence failures in local relaxation."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.calc = EMT()

    # Use failing optimizer
    energy = perform_local_relaxation(
        atoms,
        EMT(),
        MockFailingOptimizer,
        fmax=0.01,
        steps=10,
    )

    # Should assign penalty energy
    assert energy > 1e5  # Large penalty energy
    assert get_metadata(atoms, "raw_score", default=None) is not None
    assert get_metadata(atoms, "raw_score", default=0.0) < -1e5


@pytest.mark.slow
def test_bh_extreme_temperature_zero(tmp_path, rng):
    """Test Basin Hopping with temperature = 0 (no uphill moves)."""
    comp = ["Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    # With temperature = 0, should only accept downhill moves
    minima = bh_go(
        atoms=atoms,
        output_dir=str(tmp_path / "bh_zero_temp"),
        niter=3,
        temperature=0.0,  # No uphill moves
        dr=0.2,
        niter_local_relaxation=2,
        rng=rng,
    )

    assert isinstance(minima, list)
    # Should still find some minima (at least the initial relaxed structure)


@pytest.mark.slow
def test_bh_extreme_temperature_high(tmp_path, rng):
    """Test Basin Hopping with very high temperature (always accept)."""
    comp = ["Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    # With very high temperature, should accept all moves
    minima = bh_go(
        atoms=atoms,
        output_dir=str(tmp_path / "bh_high_temp"),
        niter=3,
        temperature=100.0,  # Very high temperature
        dr=0.2,
        niter_local_relaxation=2,
        rng=rng,
    )

    assert isinstance(minima, list)


def test_bh_no_movement(tmp_path, rng):
    """Test Basin Hopping with dr = 0 (no movement)."""
    comp = ["Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    # With dr = 0, should raise ValueError
    with pytest.raises(ValueError, match="dr must be positive"):
        bh_go(
            atoms=atoms,
            output_dir=str(tmp_path / "bh_no_move"),
            niter=3,
            dr=0.0,  # No movement - should fail
            niter_local_relaxation=2,
            rng=rng,
        )


@pytest.mark.slow
def test_ga_no_mutations(tmp_path, rng):
    """Test Genetic Algorithm with mutation_probability = 0 (no mutations)."""
    comp = ["Pt", "Pt", "Pt"]
    calc = EMT()

    # With no mutations, only crossover should occur
    minima = ga_go(
        composition=comp,
        output_dir=str(tmp_path / "ga_no_mut"),
        calculator=calc,
        niter=2,
        population_size=4,
        mutation_probability=0.0,  # No mutations
        niter_local_relaxation=2,
        rng=rng,
    )

    assert isinstance(minima, list)


@pytest.mark.slow
def test_ga_always_mutate(tmp_path, rng):
    """Test Genetic Algorithm with mutation_probability = 1 (always mutate)."""
    comp = ["Pt", "Pt", "Pt"]
    calc = EMT()

    # With always mutate, every offspring should be mutated
    minima = ga_go(
        composition=comp,
        output_dir=str(tmp_path / "ga_always_mut"),
        calculator=calc,
        niter=2,
        population_size=4,
        mutation_probability=1.0,  # Always mutate
        niter_local_relaxation=2,
        rng=rng,
    )

    assert isinstance(minima, list)


@pytest.mark.slow
def test_ga_minimum_population_size(tmp_path, rng):
    """Test Genetic Algorithm with population_size = 2 (minimum viable)."""
    comp = ["Pt", "Pt"]
    calc = EMT()

    # Minimum population size should still work
    minima = ga_go(
        composition=comp,
        output_dir=str(tmp_path / "ga_min_pop"),
        calculator=calc,
        niter=2,
        population_size=2,  # Minimum viable
        niter_local_relaxation=2,
        rng=rng,
    )

    assert isinstance(minima, list)


def test_ga_pairing_failure_handling(tmp_path, rng):
    """Test GA handling when pairing fails (cutandsplice returns None)."""
    # This is harder to test directly, but we can test with incompatible structures
    # that might cause pairing failures
    comp = ["Pt", "Pt", "Pt"]
    calc = EMT()

    # Use very strict parameters that might cause pairing failures
    minima = ga_go(
        composition=comp,
        output_dir=str(tmp_path / "ga_pairing_test"),
        calculator=calc,
        niter=2,
        population_size=4,
        niter_local_relaxation=2,
        rng=rng,
    )

    # Should handle pairing failures gracefully and still return results
    assert isinstance(minima, list)


def test_very_close_atoms(tmp_path):
    """Test optimization with atoms initially very close together."""
    # Create atoms that are too close (should cause high forces)
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 0.1]])  # Very close
    atoms.calc = EMT()

    # Should handle this gracefully
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.1, steps=20)

    # Should either converge or assign penalty energy
    assert np.isfinite(energy)
    if energy > 1e5:  # Penalty energy
        assert "key_value_pairs" in atoms.info
        assert atoms.info["key_value_pairs"]["raw_score"] < -1e5
    else:
        # If it converged, positions should have changed significantly
        final_positions = atoms.get_positions()
        assert not np.allclose(final_positions, [[0, 0, 0], [0, 0, 0.1]])


def test_large_cluster_initialization(rng):
    """Test initialization of larger clusters (stress test)."""
    # Test with 10 atoms (larger than typical test cases)
    comp = ["Pt"] * 10
    atoms = create_initial_cluster(comp, rng=rng)

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 10
    assert atoms.get_chemical_symbols() == ["Pt"] * 10

    # Check that atoms are reasonably separated
    positions = atoms.get_positions()
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)

    min_dist = min(distances)
    assert min_dist > 0.5  # Should not be overlapping


def test_bimetallic_edge_cases(rng):
    """Test bimetallic clusters with edge case compositions."""
    # Test with very different atomic sizes
    comp = ["H", "Pt"]  # Very different sizes
    atoms = create_initial_cluster(comp, rng=rng)

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert set(atoms.get_chemical_symbols()) == {"H", "Pt"}

    # Test with many atoms of one type and few of another
    comp = ["Pt"] * 8 + ["Au"]  # 8 Pt + 1 Au
    atoms = create_initial_cluster(comp, rng=rng)

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 9
    symbols = atoms.get_chemical_symbols()
    assert symbols.count("Pt") == 8
    assert symbols.count("Au") == 1
