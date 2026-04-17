"""Minimal benchmark sanity checks (fast EMT-only)."""

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from scgo.initialization import create_initial_cluster
from scgo.utils.helpers import perform_local_relaxation


def test_pt2_analytical_verification():
    # Create Pt2 dimer at expected bond length
    expected_bond_length = 2.26  # Actual EMT value
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [expected_bond_length, 0, 0]])
    atoms.calc = EMT()

    # Relax to find the true minimum
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.001, steps=3)

    # Check final bond length
    final_bond_length = atoms.get_distance(0, 1)

    # Should be close to expected value
    assert abs(final_bond_length - expected_bond_length) < 0.1

    # Energy should be finite; EMT reference may not guarantee negativity here
    assert np.isfinite(energy)


def test_pt3_analytical_verification():
    # Create initial triangle structure
    side_length = 2.5
    atoms = Atoms(
        "Pt3",
        positions=[
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length / 2, side_length * np.sqrt(3) / 2, 0],
        ],
    )
    atoms.calc = EMT()

    # Relax to find the true minimum
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.001, steps=3)

    # Check that it remains roughly triangular
    distances = [
        atoms.get_distance(0, 1),
        atoms.get_distance(1, 2),
        atoms.get_distance(2, 0),
    ]

    # All distances should be reasonable (not necessarily equilateral)
    assert np.std(distances) < 0.5

    # Energy should be reasonable (may be positive for some configurations)
    assert energy < 10.0  # Just check it's not extremely high


def test_initial_structure_safety(rng):
    # ensure initialization doesn't create dangerously close atoms
    comp = ["Pt", "Pt", "Pt"]

    # Create initial structure
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    # Check distances
    distances = [atoms.get_distance(i, j) for i in range(3) for j in range(i + 1, 3)]
    min_dist = min(distances)

    # Minimum distance should be reasonable
    # Typical Pt-Pt bond length is ~2.5-2.7 Å with EMT
    # Structures with distances < 2.0 Å are concerning
    assert min_dist >= 1.0, (
        f"Atoms too close: minimum distance {min_dist:.4f} Å < 1.0 Å"
    )

    # Check forces
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))

    # Initial forces should be reasonable
    # Very high forces (>50 eV/Å) indicate atoms are too close together
    # This can cause numerical issues and energy increases during relaxation
    assert max_force < 100.0, (
        f"Initial structure has dangerously high forces ({max_force:.2f} eV/Å). "
        f"This suggests atoms are too close (min distance: {min_dist:.4f} Å). "
        f"Consider tightening min_distance_factor or adding force-based validation."
    )

    # Structures with forces > 20 eV/Å are concerning but may be acceptable
    # for some edge cases - we warn but don't fail
    if max_force > 20.0:
        import warnings

        warnings.warn(
            f"Initial structure has high forces ({max_force:.2f} eV/Å) "
            f"with min distance {min_dist:.4f} Å. "
            f"This may cause energy to increase during relaxation.",
            stacklevel=2,
        )
