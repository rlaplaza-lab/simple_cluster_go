"""Tests for true minimum detection functionality.

This module tests the is_true_minimum function which verifies that optimized
structures are true local minima by checking for imaginary frequencies.
"""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.vibrations import Vibrations

from scgo.utils.helpers import is_true_minimum, perform_local_relaxation


class MockVibrationsSaddle:
    def __init__(self, atoms, name=None):
        self.atoms = atoms

    def run(self):
        pass

    def get_frequencies(self):
        # Simulate a saddle point with one significant imaginary frequency
        return np.array([-100.0, 0.0, 0.0, 50.0, 100.0, 150.0])

    def clean(self):
        pass


class MockVibrationsTrueMinimum:
    def __init__(self, atoms, name=None):
        self.atoms = atoms

    def run(self):
        pass

    def get_frequencies(self):
        # Simulate a true minimum with only near-zero frequencies (translational/rotational)
        return np.array([-0.1, 0.0, 0.1, 50.0, 100.0, 150.0])

    def clean(self):
        pass


def test_is_true_minimum_high_forces(tmp_path):
    # Create an unrelaxed structure with high forces
    atoms = Atoms(["Pt", "Pt"], positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.calc = EMT()

    # Forces will be high for this close distance
    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.01,  # Very strict threshold
        check_hessian=False,  # Skip expensive Hessian check for this test
    )
    assert is_min is False


def test_is_true_minimum_true_minimum(tmp_path, monkeypatch):
    # Create a relaxed structure (e.g., H2 molecule)
    atoms = Atoms(["H", "H"], positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = EMT()

    # Perform a quick relaxation to ensure forces are low
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.01, steps=10)

    # Mock Vibrations to ensure a controlled outcome for Hessian check
    monkeypatch.setattr("scgo.utils.helpers.Vibrations", MockVibrationsTrueMinimum)

    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=True,
        imag_freq_threshold=1.0,  # Strict threshold for imaginary frequencies
    )
    assert is_min is True


def test_is_true_minimum_saddle_point(tmp_path, monkeypatch):
    # Create a structure that should be a saddle point (e.g., linear H3)
    atoms = Atoms(["H", "H", "H"], positions=[[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]])
    atoms.calc = EMT()

    # Perform a quick relaxation (forces might not be zero for a saddle point)
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.01, steps=10)

    # Mock Vibrations to simulate a saddle point
    monkeypatch.setattr("scgo.utils.helpers.Vibrations", MockVibrationsSaddle)

    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=True,
        imag_freq_threshold=50.0,  # Threshold to catch the -100.0 freq
    )
    assert is_min is False


def test_is_true_minimum_hessian_skipped(tmp_path):
    # Test case where Hessian check is explicitly skipped
    atoms = Atoms(["Pt"], positions=[[0, 0, 0]])
    atoms.calc = EMT()

    # Forces are low for a single atom
    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=False,  # Hessian check skipped
    )
    assert is_min is True


@pytest.mark.slow
def test_is_true_minimum_real_hessian(tmp_path):
    # Run in a temporary directory so Vibrations writes files there and cleans up
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Small dimer which EMT can relax quickly
        atoms = Atoms(["Pt", "Pt"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        calc = EMT()

        # Local relaxation: attach calculator and run a few LBFGS steps
        perform_local_relaxation(
            atoms,
            calculator=calc,
            optimizer=LBFGS,
            fmax=0.05,
            steps=50,
        )

        # After relaxation, check Hessian for true minimum
        is_min = is_true_minimum(
            atoms,
            calculator=calc,
            fmax_threshold=0.1,
            check_hessian=True,
            imag_freq_threshold=50.0,
        )

        assert is_min is True
        # Additionally, run Vibrations directly and assert the number of
        # near-zero modes (translational + rotational) is as expected for a
        # linear dimer (expected 5 near-zero modes).
        # Re-attach calculator since perform_local_relaxation replaces it with SinglePointCalculator
        atoms.calc = calc
        vib = Vibrations(atoms, name="vib_modes")
        try:
            vib.run()
            freqs = vib.get_frequencies()
        finally:
            vib.clean()

        # Determine if the molecule is linear via moments of inertia
        moi = atoms.get_moments_of_inertia(vectors=False)
        is_linear = any(np.isclose(moi, 0, atol=1e-5))
        expected_zero_modes = 5 if is_linear else 6

        # Count near-zero modes (absolute frequency smaller than 1 cm^-1)
        near_zero_count = int(np.sum(np.abs(freqs) < 1.0))
        assert near_zero_count >= expected_zero_modes
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_is_true_minimum_fmax_threshold_variations(tmp_path):
    """Test various fmax_threshold values."""
    # Create a structure with moderate forces
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.0]])
    atoms.calc = EMT()

    # Don't relax completely to keep some forces
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.1, steps=5)

    # Test with very strict threshold
    is_min_strict = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.001,  # Very strict
        check_hessian=False,
    )

    # Test with moderate threshold
    is_min_moderate = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.1,  # Moderate
        check_hessian=False,
    )

    # Test with loose threshold
    is_min_loose = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=1.0,  # Loose
        check_hessian=False,
    )

    # Strict should be False, loose should be True
    assert is_min_strict is False
    assert is_min_loose is True
    # Moderate result should be between strict and loose
    assert isinstance(is_min_moderate, bool)


def test_is_true_minimum_imag_freq_threshold_variations(tmp_path, monkeypatch):
    """Test various imag_freq_threshold values with mocked vibrations."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms.calc = EMT()
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.01, steps=10)

    # Mock vibrations with small imaginary frequency
    class MockVibrationsSmallImag:
        def __init__(self, atoms, name=None):
            self.atoms = atoms

        def run(self):
            pass

        def get_frequencies(self):
            # Small imaginary frequency
            return np.array([-5.0, 0.0, 0.0, 50.0, 100.0, 150.0])

        def clean(self):
            pass

    monkeypatch.setattr("scgo.utils.helpers.Vibrations", MockVibrationsSmallImag)

    # Test with threshold that catches small imaginary frequency
    is_min_strict = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=True,
        imag_freq_threshold=1.0,  # Catches -5.0
    )
    assert is_min_strict is False

    # Test with threshold that ignores small imaginary frequency
    is_min_loose = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=True,
        imag_freq_threshold=10.0,  # Ignores -5.0
    )
    assert is_min_loose is True


def test_is_true_minimum_single_atom_edge_case(tmp_path):
    """Test single atom - edge case with no vibrations."""
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.calc = EMT()

    # Single atom should pass force check (forces are zero)
    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.01,
        check_hessian=False,  # Skip Hessian for single atom
    )
    assert is_min is True


def test_is_true_minimum_pathological_geometry_close_atoms(tmp_path):
    """Test pathological geometry with very close atoms."""
    # Very close atoms (should cause high forces)
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 0.1]])
    atoms.calc = EMT()

    # Should fail force check
    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.01,
        check_hessian=False,
    )
    assert is_min is False


def test_is_true_minimum_pathological_geometry_overlapping_atoms(tmp_path):
    """Test pathological geometry with overlapping atoms."""
    # Overlapping atoms (should cause very high forces)
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 0.01]])
    atoms.calc = EMT()

    # Should fail force check
    is_min = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.1,  # Even with loose threshold
        check_hessian=False,
    )
    assert is_min is False


def test_is_true_minimum_force_threshold_boundary(tmp_path):
    """Test force threshold boundary conditions."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.0]])
    atoms.calc = EMT()

    # Relax partially to get specific force magnitude
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.05, steps=3)

    # Get actual forces
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))

    # Test with threshold just below actual force
    is_min_below = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=max_force * 0.9,  # Just below
        check_hessian=False,
    )
    assert is_min_below is False

    # Test with threshold just above actual force
    is_min_above = is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=max_force * 1.1,  # Just above
        check_hessian=False,
    )
    assert is_min_above is True
