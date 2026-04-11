"""Tests for ASE GA patches - custom implementations for genetic algorithm components.

These tests verify the custom implementations in scgo/ase_ga_patches/ including
CutAndSplicePairing and Population classes.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase_ga.data import DataConnection
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.ase_ga_patches.population import Population
from tests.test_utils import create_ga_comparator, create_paired_rngs, create_preparedb


@pytest.fixture
def ga_database(tmp_path, pt3_atoms, rng):
    """Create a GA database with some test structures."""
    db_path = tmp_path / "test_ga.db"

    # Create database
    db = create_preparedb(pt3_atoms, db_path)

    # Add some test structures
    for i in range(3):
        atoms = pt3_atoms.copy()
        atoms.positions += rng.random((3, 3)) * 0.1  # Small random displacement
        atoms.calc = EMT()
        from scgo.database.metadata import add_metadata

        add_metadata(atoms, raw_score=-10.0 - i)
        atoms.info["confid"] = i  # Add confid for GA compatibility
        db.add_unrelaxed_candidate(atoms, description=f"test_{i}")

    # Create DataConnection and mark candidates as relaxed
    da = DataConnection(str(db_path))
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        # Ensure the atoms have the required info for add_relaxed_step
        from scgo.database.metadata import get_all_metadata, get_metadata

        if "key_value_pairs" not in a.info:
            a.info["key_value_pairs"] = get_all_metadata(a).copy()

        # Ensure raw_score exists for ASE GA expectations
        if "raw_score" not in a.info["key_value_pairs"]:
            raw = get_metadata(a, "raw_score", default=None)
            if raw is not None:
                a.info["key_value_pairs"]["raw_score"] = raw
            else:
                # Fallback to a default value
                a.info["key_value_pairs"]["raw_score"] = -10.0

        da.add_relaxed_step(a)

    return str(db_path)


def test_cutandsplicepairing_successful_pairing(pt3_atoms, rng):
    """Test successful pairing of compatible structures."""
    n_top = len(pt3_atoms)
    all_atom_types = get_all_atom_types(pt3_atoms, range(n_top))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    # Create slab (empty for clusters)
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())

    # Create pairing object
    pairing = CutAndSplicePairing(slab, n_top, blmin, rng=rng)

    # Create two parent structures
    parent1 = pt3_atoms.copy()
    parent1.positions += np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    parent1.info["confid"] = 1

    parent2 = pt3_atoms.copy()
    parent2.positions += np.array([[0, 0.1, 0], [0, 0, 0.1], [0.1, 0, 0]])
    parent2.info["confid"] = 2

    # Perform pairing
    offspring, description = pairing.get_new_individual([parent1, parent2])

    # Verify results
    assert offspring is not None
    assert isinstance(offspring, Atoms)
    assert len(offspring) == len(pt3_atoms)
    assert offspring.get_chemical_symbols() == pt3_atoms.get_chemical_symbols()
    assert isinstance(description, str)
    assert len(description) > 0


def test_cutandsplicepairing_incompatible_compositions(pt3_atoms, au2pt2_atoms, rng):
    """Test pairing failure with incompatible compositions."""
    n_top = len(pt3_atoms)
    all_atom_types = get_all_atom_types(pt3_atoms, range(n_top))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    # Create slab
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())

    # Create pairing object
    pairing = CutAndSplicePairing(slab, n_top, blmin, rng=rng)

    # Try to pair incompatible structures (different compositions)
    # Should raise ValueError for incompatible compositions
    with pytest.raises(ValueError, match="same length"):
        pairing.get_new_individual([pt3_atoms, au2pt2_atoms])


def test_cutandsplicepairing_rng_reproducibility(pt3_atoms, rng):
    """Test RNG reproducibility in pairing operations."""
    n_top = len(pt3_atoms)
    all_atom_types = get_all_atom_types(pt3_atoms, range(n_top))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())

    # Create two identical RNGs
    rng1, rng2 = create_paired_rngs(42)

    pairing1 = CutAndSplicePairing(slab, n_top, blmin, rng=rng1)
    pairing2 = CutAndSplicePairing(slab, n_top, blmin, rng=rng2)

    # Create parent structures
    parent1 = pt3_atoms.copy()
    parent1.positions += np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    parent1.info["confid"] = 1
    parent2 = pt3_atoms.copy()
    parent2.positions += np.array([[0, 0.1, 0], [0, 0, 0.1], [0.1, 0, 0]])
    parent2.info["confid"] = 2

    # Perform pairing with both RNGs
    offspring1, desc1 = pairing1.get_new_individual([parent1, parent2])
    offspring2, desc2 = pairing2.get_new_individual([parent1, parent2])

    # Results should be identical
    assert offspring1 is not None
    assert offspring2 is not None
    assert np.allclose(offspring1.get_positions(), offspring2.get_positions())
    assert desc1 == desc2


def test_cutandsplicepairing_offspring_composition(pt3_atoms, rng):
    """Test that offspring inherits correct composition."""
    n_top = len(pt3_atoms)
    all_atom_types = get_all_atom_types(pt3_atoms, range(n_top))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())

    pairing = CutAndSplicePairing(slab, n_top, blmin, rng=rng)

    # Create parent structures
    parent1 = pt3_atoms.copy()
    parent1.info["confid"] = 1
    parent2 = pt3_atoms.copy()
    parent2.info["confid"] = 2

    offspring, _ = pairing.get_new_individual([parent1, parent2])

    # Offspring should have same composition as parents
    assert offspring is not None
    assert offspring.get_chemical_symbols() == pt3_atoms.get_chemical_symbols()
    assert len(offspring) == len(pt3_atoms)


def test_population_initialization(ga_database, pt3_atoms, rng):
    """Test Population initialization from database."""
    from ase_ga.data import DataConnection

    # Create comparator
    n_top = len(pt3_atoms)
    comp = create_ga_comparator(n_top)

    # Create population
    da = DataConnection(ga_database)
    population = Population(
        data_connection=da,
        population_size=3,
        comparator=comp,
        logfile=None,  # No log file for testing
        rng=rng,
    )

    # Verify population was created
    assert population is not None
    assert population.population_size == 3


def test_population_candidate_selection(ga_database, pt3_atoms, rng):
    """Test candidate selection from population."""
    from ase_ga.data import DataConnection

    # Create comparator
    n_top = len(pt3_atoms)
    comp = create_ga_comparator(n_top)

    # Create population
    da = DataConnection(ga_database)
    population = Population(
        data_connection=da,
        population_size=3,
        comparator=comp,
        logfile=None,
        rng=rng,
    )

    # Test candidate selection
    candidates = population.get_two_candidates()

    # Verify candidates are returned (should not be None if population has enough candidates)
    if candidates is not None:
        candidate1, candidate2 = candidates
        assert candidate1 is not None
        assert candidate2 is not None
        assert (
            candidate1.info["confid"] != candidate2.info["confid"]
        )  # Different candidates
        assert isinstance(candidate1, Atoms)
        assert isinstance(candidate2, Atoms)
        assert len(candidate1) == len(pt3_atoms)
        assert len(candidate2) == len(pt3_atoms)
    else:
        # If None is returned, it means population doesn't have enough candidates
        # This is valid behavior when population size is too small
        assert len(population.pop) < 2


def test_population_rng_reproducibility(ga_database, pt3_atoms):
    """Test RNG reproducibility in population operations."""
    from ase_ga.data import DataConnection

    # Create comparator
    n_top = len(pt3_atoms)
    comp = create_ga_comparator(n_top)

    # Create two identical RNGs
    rng1, rng2 = create_paired_rngs(42)

    # Create two populations with identical RNGs
    da1 = DataConnection(ga_database)
    da2 = DataConnection(ga_database)

    population1 = Population(
        data_connection=da1,
        population_size=3,
        comparator=comp,
        logfile=None,
        rng=rng1,
    )

    population2 = Population(
        data_connection=da2,
        population_size=3,
        comparator=comp,
        logfile=None,
        rng=rng2,
    )

    # Get candidates from both populations
    candidates1 = population1.get_two_candidates()
    candidates2 = population2.get_two_candidates()

    # Both should return the same result (or both None)
    if candidates1 is not None and candidates2 is not None:
        c1_1, c1_2 = candidates1
        c2_1, c2_2 = candidates2
    else:
        # If either returns None, both should return None
        assert candidates1 is None
        assert candidates2 is None
        return  # Skip the rest of the test

    # Results should be identical (same selection order)
    assert np.allclose(c1_1.get_positions(), c2_1.get_positions())
    assert np.allclose(c1_2.get_positions(), c2_2.get_positions())


def test_population_with_bimetallic_clusters(au2pt2_atoms, rng, tmp_path):
    """Test population with bimetallic clusters."""
    # Create database for bimetallic clusters
    db_path = tmp_path / "test_bi_ga.db"
    db = create_preparedb(au2pt2_atoms, db_path)

    # Add some bimetallic structures
    for i in range(3):
        atoms = au2pt2_atoms.copy()
        atoms.positions += rng.random((4, 3)) * 0.1
        atoms.calc = EMT()
        atoms.info["key_value_pairs"] = {"raw_score": -20.0 - i}
        atoms.info["confid"] = i
        db.add_unrelaxed_candidate(atoms, description=f"test_bi_{i}")
        db.c.write(atoms, data=atoms.info)

    # Create population
    from ase_ga.data import DataConnection

    n_top = len(au2pt2_atoms)
    comp = create_ga_comparator(n_top)

    da = DataConnection(str(db_path))
    population = Population(
        data_connection=da,
        population_size=3,
        comparator=comp,
        logfile=None,
        rng=rng,
    )

    # Test candidate selection
    candidates = population.get_two_candidates()

    # Verify candidates are returned (should not be None if population has enough candidates)
    if candidates is not None:
        candidate1, candidate2 = candidates
        assert candidate1 is not None
        assert candidate2 is not None
        assert len(candidate1) == len(au2pt2_atoms)
        assert len(candidate2) == len(au2pt2_atoms)
        assert set(candidate1.get_chemical_symbols()) == {"Au", "Pt"}
        assert set(candidate2.get_chemical_symbols()) == {"Au", "Pt"}
    else:
        # If None is returned, it means population doesn't have enough candidates
        # This is valid behavior when population size is too small
        assert len(population.pop) < 2


def test_cutandsplicepairing_with_bimetallic_clusters(au2pt2_atoms, rng):
    """Test CutAndSplicePairing with bimetallic clusters."""
    n_top = len(au2pt2_atoms)
    all_atom_types = get_all_atom_types(au2pt2_atoms, range(n_top))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    slab = Atoms(cell=au2pt2_atoms.get_cell(), pbc=au2pt2_atoms.get_pbc())

    pairing = CutAndSplicePairing(slab, n_top, blmin, rng=rng)

    # Create parent structures
    parent1 = au2pt2_atoms.copy()
    parent1.positions += np.array(
        [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0.1, 0.1, 0]],
    )
    parent1.info["confid"] = 1

    parent2 = au2pt2_atoms.copy()
    parent2.positions += np.array([[0, 0.1, 0], [0, 0, 0.1], [0.1, 0, 0], [0, 0, 0.1]])
    parent2.info["confid"] = 2

    # Perform pairing
    offspring, description = pairing.get_new_individual([parent1, parent2])

    # Verify results
    assert offspring is not None
    assert isinstance(offspring, Atoms)
    assert len(offspring) == len(au2pt2_atoms)

    # Check composition is preserved
    offspring_symbols = offspring.get_chemical_symbols()
    parent_symbols = au2pt2_atoms.get_chemical_symbols()
    assert sorted(offspring_symbols) == sorted(parent_symbols)
    assert offspring_symbols.count("Au") == 2
    assert offspring_symbols.count("Pt") == 2


# Tests merged from test_mutations.py
def get_blmin(atoms):
    """Helper to create blmin for mutation testing."""
    all_atom_types = get_all_atom_types(atoms, range(len(atoms)))
    return closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)


def test_rattle_mutation(pt4_tetrahedron, rng):
    """Test rattle mutation operator."""
    from scgo.ase_ga_patches.standardmutations import RattleMutation

    atoms = pt4_tetrahedron.copy()
    initial_positions = atoms.get_positions().copy()
    blmin = get_blmin(atoms)
    n_top = len(atoms)

    rattle_mut = RattleMutation(
        blmin,
        n_top,
        rattle_strength=0.5,
        rattle_prop=1.0,
        rng=rng,
    )
    mutated_atoms = rattle_mut.mutate(atoms)

    assert mutated_atoms is not None
    assert not np.allclose(initial_positions, mutated_atoms.get_positions())
    assert len(mutated_atoms) == len(atoms)
    assert mutated_atoms.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_custom_permutation_mutation_bimetallic(au2pt2_atoms, rng):
    """Test permutation mutation on bimetallic cluster."""
    from scgo.ase_ga_patches.standardmutations import CustomPermutationMutation

    atoms = au2pt2_atoms.copy()
    initial_positions = atoms.get_positions().copy()
    initial_symbols = atoms.get_chemical_symbols()
    n_top = len(atoms)

    perm_mut = CustomPermutationMutation(n_top, probability=1.0, rng=rng)
    mutated_atoms = perm_mut.mutate(atoms)

    assert mutated_atoms is not None
    assert len(mutated_atoms) == len(atoms)
    # Permutation mutation swaps positions of atoms with different types
    # Symbols order stays the same, but positions change
    assert mutated_atoms.get_chemical_symbols() == initial_symbols
    assert not np.allclose(mutated_atoms.get_positions(), initial_positions)


def test_custom_permutation_mutation_monometallic(pt4_tetrahedron, rng):
    """Test permutation mutation on monometallic cluster."""
    from scgo.ase_ga_patches.standardmutations import CustomPermutationMutation

    atoms = pt4_tetrahedron.copy()
    n_top = len(atoms)

    # Permutation mutation should return None for monometallic clusters (not applicable)
    perm_mut = CustomPermutationMutation(n_top, probability=1.0, rng=rng)
    mutated_atoms = perm_mut.mutate(atoms)

    # For monometallic clusters, mutation is not applicable and returns None
    assert mutated_atoms is None


def test_flattening_mutation(pt4_tetrahedron, rng):
    """Test flattening mutation operator."""
    from scgo.ase_ga_patches.standardmutations import FlatteningMutation

    atoms = pt4_tetrahedron.copy()
    initial_positions = atoms.get_positions().copy()
    blmin = get_blmin(atoms)
    n_top = len(atoms)

    flatten_mut = FlatteningMutation(blmin, n_top, thickness_factor=1.0, rng=rng)
    mutated_atoms = flatten_mut.mutate(atoms)

    assert mutated_atoms is not None
    assert not np.allclose(initial_positions, mutated_atoms.get_positions())
    assert len(mutated_atoms) == len(atoms)
    assert mutated_atoms.get_chemical_symbols() == atoms.get_chemical_symbols()

    # Quantify flattening: Check spread along principal axes
    def get_spread(atoms_obj):
        positions = atoms_obj.get_positions()
        # Center the positions
        centered_positions = positions - np.mean(positions, axis=0)
        # Calculate covariance matrix
        covariance_matrix = np.cov(centered_positions, rowvar=False)
        # Eigenvalues are proportional to the spread along principal axes
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        return np.sort(eigenvalues)  # Sort from smallest to largest

    initial_spread = get_spread(atoms)
    mutated_spread = get_spread(mutated_atoms)

    # The smallest eigenvalue (spread along the thinnest dimension) should be reduced
    # This is a heuristic check, exact reduction depends on random plane
    # We expect the smallest spread to be significantly smaller, but not zero due to thickness
    # Adjusted threshold to 0.55 (55%) to account for randomness in plane selection
    assert (
        mutated_spread[0] < initial_spread[0] * 0.55
    )  # Expect at least 45% reduction in thinnest dimension
    assert mutated_spread[0] > 0  # Should not be perfectly flat due to thickness
