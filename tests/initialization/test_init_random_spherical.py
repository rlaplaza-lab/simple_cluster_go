"""Tests for random_spherical initialization mode.

This module consolidates all tests for random_spherical initialization including:
- Basic functionality and edge cases
- Boundary value testing
- Retry logic and diversity
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
"""

import numpy as np
import pytest

from scgo.initialization import (
    is_cluster_connected,
    random_spherical,
)
from tests.test_utils import assert_cluster_valid


class TestRandomSphericalInitialization:
    """Tests for random spherical cluster initialization.

    Note: Basic smoke tests (empty, single atom, two atoms) have been consolidated
    into parametrized tests in test_initialization_modes.py to reduce redundancy.
    This class now focuses on mode-specific edge cases and stress tests.
    """

    def test_random_spherical_count_and_bounds(self, rng):
        """Test random spherical produces correct number of atoms in bounds and satisfies invariants."""
        comp = ["Pt"] * 5
        side = 30.0
        atoms = random_spherical(
            comp,
            placement_radius_scaling=1.2,
            cell_side=side,
            rng=rng,
        )
        # Verify atom count
        assert len(atoms) == len(comp)
        # Verify all invariants using helper
        assert_cluster_valid(atoms, comp)
        # cell is cubic and set
        c = atoms.get_cell()
        assert np.allclose(c[0, 0], side)
        assert np.allclose(c[1, 1], side)
        assert np.allclose(c[2, 2], side)
        # ensure the cluster is centered in the cell (COM near origin).
        com = atoms.get_center_of_mass()
        half = side / 2.0
        assert np.allclose(com, [half, half, half], atol=0.5)

    def test_random_spherical_placement_failure(self, rng):
        """Test that placement failure raises appropriate error."""
        # Try to place many atoms in a very small space, should result in a ValueError.
        comp = ["H"] * 20  # Many small atoms
        side = 5.0  # Very small cell
        with pytest.raises(ValueError):
            random_spherical(
                comp, placement_radius_scaling=0.1, cell_side=side, rng=rng
            )


class TestBoundaryValues:
    """Tests for boundary value parameters."""

    def test_very_small_placement_radius(self, rng):
        """Test with very small placement_radius_scaling."""
        # Very small radius may fail, which is acceptable
        try:
            atoms = random_spherical(
                ["Pt", "Pt"], cell_side=20.0, placement_radius_scaling=0.01, rng=rng
            )
            # If it succeeds, should have 2 atoms
            assert len(atoms) == 2
        except ValueError:
            # Failure is acceptable with extreme parameters
            pass

    def test_very_large_placement_radius(self, rng):
        """Test with very large placement_radius_scaling."""
        # Very large radius may cause connectivity issues, which is acceptable
        try:
            atoms = random_spherical(
                ["Pt", "Pt"], cell_side=20.0, placement_radius_scaling=100.0, rng=rng
            )
            assert len(atoms) == 2
        except ValueError:
            # Failure is acceptable with extreme parameters
            pass

    def test_min_distance_factor_zero(self, rng):
        """Test with min_distance_factor = 0."""
        atoms = random_spherical(
            ["Pt", "Pt"], cell_side=20.0, min_distance_factor=0.0, rng=rng
        )
        # Should work (allows overlap)
        assert len(atoms) == 2

    def test_min_distance_factor_very_large(self, rng):
        """Test with very large min_distance_factor."""
        # This might cause placement failures
        try:
            atoms = random_spherical(
                ["Pt", "Pt"], cell_side=10.0, min_distance_factor=10.0, rng=rng
            )
            # If it succeeds, should be valid
            assert atoms is None or len(atoms) == 2
        except ValueError:
            # Placement failure is acceptable with very large factor
            pass

    def test_connectivity_factor_very_small(self, rng):
        """Test with very small connectivity_factor."""
        # Very small connectivity may fail, which is acceptable
        VERY_STRICT_FACTOR = 0.1  # Very strict for testing boundary conditions
        try:
            atoms = random_spherical(
                ["Pt", "Pt"],
                cell_side=20.0,
                connectivity_factor=VERY_STRICT_FACTOR,
                rng=rng,
            )
            # If it succeeds, should have 2 atoms
            assert len(atoms) == 2
        except ValueError:
            # Failure is acceptable with extreme parameters
            pass

    def test_connectivity_factor_very_large(self, rng):
        """Test with very large connectivity_factor."""
        VERY_LARGE_FACTOR = 100.0  # Very large for testing boundary conditions
        atoms = random_spherical(
            ["Pt", "Pt"], cell_side=20.0, connectivity_factor=VERY_LARGE_FACTOR, rng=rng
        )
        # Should always work
        assert len(atoms) == 2


class TestRetryDiversity:
    """Tests for retry logic diversity."""

    def test_retry_logic_maintains_diversity(self, rng):
        """Verify that retry attempts produce diverse structures."""
        # Test random_spherical which uses retry logic
        comp = ["Pt"] * 8
        structures = []

        # Stress test: generate multiple clusters to verify diversity
        for _ in range(10):
            atoms = random_spherical(
                comp,
                placement_radius_scaling=1.2,
                cell_side=20.0,
                rng=rng,
            )
            structures.append(atoms)

        # Check diversity
        def get_signature(atoms):
            pos = atoms.get_positions()
            dists = [
                np.linalg.norm(pos[i] - pos[j])
                for i in range(len(pos))
                for j in range(i + 1, len(pos))
            ]
            return tuple(np.round(np.sort(dists), 4))

        signatures = [get_signature(s) for s in structures]
        unique = set(signatures)
        # Should have diversity
        assert len(unique) >= 3, "Retry logic should maintain diversity"

    def test_connectivity_retries_diverse(self, rng):
        """Verify connectivity retries don't always produce identical structures."""
        comp = ["Pt"] * 6
        structures = []

        # Stress test: generate multiple clusters to verify diversity
        for _ in range(8):
            atoms = random_spherical(
                comp,
                placement_radius_scaling=1.2,
                cell_side=20.0,
                rng=rng,
            )
            structures.append(atoms)

        # Check diversity
        def get_signature(atoms):
            pos = atoms.get_positions()
            dists = [
                np.linalg.norm(pos[i] - pos[j])
                for i in range(len(pos))
                for j in range(i + 1, len(pos))
            ]
            return tuple(np.round(np.sort(dists), 4))

        signatures = [get_signature(s) for s in structures]
        unique = set(signatures)
        # Should have diversity
        assert len(unique) >= 2, "Connectivity retries should maintain diversity"


# Reliability tests have been consolidated into TestReliabilityAllModes in test_init_common.py
# Large cluster connectivity tests have been consolidated into TestLargeClusterConnectivityAllModes in test_init_common.py


class TestRandomSphericalStressAndPerformance:
    """Stress and performance tests for random_spherical mode."""

    def test_retry_exhaustion_error_message(self, rng):
        """Test that retry exhaustion provides helpful error message."""
        # Try to place many atoms in very small space
        comp = ["H"] * 50  # Many small atoms
        with pytest.raises(ValueError) as exc_info:
            random_spherical(
                comp, cell_side=5.0, placement_radius_scaling=0.01, rng=rng
            )
        # Error message should include suggestions
        error_msg = str(exc_info.value)
        assert "placement_radius_scaling" in error_msg or "cell_side" in error_msg

    def test_connectivity_retry_exhaustion(self, rng):
        """Test connectivity retry exhaustion."""
        # Use very strict connectivity that might cause failures
        STRICT_FACTOR = 0.5  # Very strict for testing retry exhaustion
        comp = ["Pt"] * 10
        try:
            atoms = random_spherical(
                comp,
                cell_side=10.0,
                placement_radius_scaling=0.5,
                connectivity_factor=STRICT_FACTOR,
                rng=rng,
            )
            # If it succeeds, should be valid
            if atoms is not None:
                assert is_cluster_connected(atoms, connectivity_factor=STRICT_FACTOR)
        except ValueError as e:
            # Error should mention connectivity, clashes, or validation failure
            # (strict parameters can cause either type of failure)
            error_msg = str(e).lower()
            assert (
                "connectivity" in error_msg
                or "connected" in error_msg
                or "clash" in error_msg
                or "validation failed" in error_msg
            )
