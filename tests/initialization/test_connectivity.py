"""Tests for connectivity validation functionality.

This module tests the connectivity checking functions to ensure that
disjointed clusters are properly detected and handled.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from scgo.initialization import is_cluster_connected
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR

# Lenient connectivity factor for testing edge cases and threshold behavior
LENIENT_CONNECTIVITY_FACTOR = 2.0


class TestConnectivityValidation:
    """Test suite for connectivity validation functions."""

    def test_empty_cluster_connected(self, empty_atoms):
        """Test that empty cluster is considered connected."""
        assert is_cluster_connected(empty_atoms) is True

    def test_single_atom_connected(self, single_atom):
        """Test that single atom is considered connected."""
        assert is_cluster_connected(single_atom) is True

    def test_two_atoms_connected(self, pt2_atoms):
        """Test that two atoms within bonding distance are connected."""
        # Pt-Pt distance of ~2.8 Å (within 2 * 1.4 Å = 2.8 Å threshold)
        assert is_cluster_connected(pt2_atoms) is True

    def test_two_atoms_disconnected(self):
        """Test that two atoms beyond bonding distance are disconnected."""
        # Pt-Pt distance of ~10 Å (far beyond 2 * 1.4 Å = 2.8 Å threshold)
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [10, 0, 0]])
        assert is_cluster_connected(atoms) is False

    def test_three_atoms_connected_chain(self):
        """Test that three atoms in a chain are connected."""
        # Pt-Pt distances of ~2.5 Å each
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [5, 0, 0]])
        assert is_cluster_connected(atoms) is True

    def test_three_atoms_disconnected(self):
        """Test that three atoms with one isolated are disconnected."""
        # Two atoms close, one far away
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [10, 0, 0]])
        assert is_cluster_connected(atoms) is False

    def test_triangle_connected(self, pt3_atoms):
        """Test that three atoms in a triangle are connected."""
        # Equilateral triangle with ~2.5 Å sides
        assert is_cluster_connected(pt3_atoms) is True

    def test_tetrahedron_connected(self, pt4_tetrahedron):
        """Test that four atoms in a tetrahedron are connected."""
        # Tetrahedron with ~2.5 Å sides
        assert is_cluster_connected(pt4_tetrahedron) is True

    def test_mixed_elements_connected(self):
        """Test connectivity with mixed element types."""
        # Pt-Au distance ~2.7 Å (within threshold)
        atoms = Atoms("PtAu", positions=[[0, 0, 0], [2.7, 0, 0]])
        assert is_cluster_connected(atoms) is True

    def test_mixed_elements_disconnected(self):
        """Test disconnected mixed element types."""
        # Pt-Au distance ~10 Å (far beyond threshold)
        atoms = Atoms("PtAu", positions=[[0, 0, 0], [10, 0, 0]])
        assert is_cluster_connected(atoms) is False

    def test_connectivity_factor_parameter(self):
        """Test that connectivity factor parameter works correctly."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [6, 0, 0]])

        # With lenient factor=2.0, should be disconnected (threshold ~5.44 Å)
        assert (
            is_cluster_connected(atoms, connectivity_factor=LENIENT_CONNECTIVITY_FACTOR)
            is False
        )

        # With factor=2.5, should be connected (threshold ~6.8 Å)
        assert is_cluster_connected(atoms, connectivity_factor=2.5) is True

        # With factor=3.0, should be connected
        assert is_cluster_connected(atoms, connectivity_factor=3.0) is True

    def test_large_cluster_connected(self):
        """Test connectivity of a larger connected cluster."""
        # Create a simple cubic-like structure
        positions = [
            [i * 2.5, j * 2.5, k * 2.5]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ]

        atoms = Atoms("Pt27", positions=positions)
        assert is_cluster_connected(atoms) is True

    def test_large_cluster_disconnected(self):
        """Test connectivity of a larger cluster with isolated atoms."""
        # Create two separate clusters
        # First cluster (3x3x3)
        positions = [
            [i * 2.5, j * 2.5, k * 2.5]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        ]
        # Second isolated cluster
        positions.extend(
            [
                [i * 2.5 + 20, j * 2.5 + 20, k * 2.5 + 20]
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )

        atoms = Atoms("Pt35", positions=positions)
        assert is_cluster_connected(atoms) is False

    def test_edge_case_exactly_at_threshold(self):
        """Test connectivity when atoms are exactly at the threshold distance."""
        # Pt covalent radius is ~1.36 Å, so threshold at factor=2.0 is ~5.44 Å
        # Using lenient factor to test exact threshold behavior
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [5.44, 0, 0]])
        assert (
            is_cluster_connected(atoms, connectivity_factor=LENIENT_CONNECTIVITY_FACTOR)
            is True
        )

    def test_edge_case_just_beyond_threshold(self):
        """Test connectivity when atoms are just beyond the threshold distance."""
        # Pt covalent radius is ~1.36 Å, so threshold at factor=2.0 is ~5.44 Å
        # Using lenient factor to test threshold boundary behavior
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [5.5, 0, 0]])
        assert (
            is_cluster_connected(atoms, connectivity_factor=LENIENT_CONNECTIVITY_FACTOR)
            is False
        )

        # Test with factor=2.5 (default), should be connected
        assert is_cluster_connected(atoms, connectivity_factor=2.5) is True

    def test_very_small_connectivity_factor(self):
        """Test with very small connectivity factor."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.0, 0, 0]])
        # With factor=0.5, threshold is ~1.4 Å, so atoms at 1.0 Å should be connected
        assert is_cluster_connected(atoms, connectivity_factor=0.5) is True

    def test_very_large_connectivity_factor(self):
        """Test with very large connectivity factor."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [20, 0, 0]])
        # With factor=10.0, threshold is ~28 Å, so atoms at 20 Å should be connected
        assert is_cluster_connected(atoms, connectivity_factor=10.0) is True

    def test_3d_cluster_connectivity(self):
        """Test connectivity in 3D cluster structure."""
        # Create a 3D cluster with atoms at various distances
        atoms = Atoms(
            "Pt8",
            positions=[
                [0, 0, 0],  # Center
                [2.5, 0, 0],  # Connected to center
                [0, 2.5, 0],  # Connected to center
                [0, 0, 2.5],  # Connected to center
                [2.5, 2.5, 0],  # Connected to (2.5,0,0) and (0,2.5,0)
                [2.5, 0, 2.5],  # Connected to (2.5,0,0) and (0,0,2.5)
                [0, 2.5, 2.5],  # Connected to (0,2.5,0) and (0,0,2.5)
                [2.5, 2.5, 2.5],  # Connected to multiple atoms
            ],
        )
        assert is_cluster_connected(atoms) is True

    def test_chain_vs_ring_connectivity(self):
        """Test that both chain and ring structures are connected."""
        # Linear chain
        chain_atoms = Atoms(
            "Pt4",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [5, 0, 0],
                [7.5, 0, 0],
            ],
        )
        assert is_cluster_connected(chain_atoms) is True

        # Ring structure (square)
        ring_atoms = Atoms(
            "Pt4",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [2.5, 2.5, 0],
                [0, 2.5, 0],
            ],
        )
        assert is_cluster_connected(ring_atoms) is True

    def test_performance_large_cluster(self):
        """Test performance with a reasonably large cluster."""
        # Create a 5x5x5 cluster (125 atoms)
        positions = [
            [i * 2.5, j * 2.5, k * 2.5]
            for i in range(5)
            for j in range(5)
            for k in range(5)
        ]

        atoms = Atoms("Pt125", positions=positions)

        # This should complete quickly even with O(n²) algorithm
        import time

        start_time = time.time()
        result = is_cluster_connected(atoms)
        end_time = time.time()

        assert result is True
        assert end_time - start_time < 1.0  # Should complete in less than 1 second


class TestConnectivityOptimization:
    """Tests for optimized connectivity check with KDTree."""

    def test_connectivity_large_cluster_connected(self):
        """Test connectivity for large cluster (uses KDTree path)."""
        # Create a large FCC-like Pt cluster (> 50 atoms)
        # FCC has 1 atom per conventional cell, so (4,4,4) gives 64 atoms
        atoms = bulk("Pt", "fcc", a=4.0).repeat((4, 4, 4))
        # Remove vacuum and center
        atoms.center()
        assert len(atoms) > 50, (
            f"Test requires >50 atoms to trigger KDTree path, got {len(atoms)}"
        )

        # FCC structure should be fully connected
        assert (
            is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR) is True
        )

    def test_connectivity_large_cluster_disconnected(self):
        """Test disconnected large cluster (uses KDTree path)."""
        # Create two separate FCC clusters (4x4x4 = 64 atoms each)
        cluster1 = bulk("Pt", "fcc", a=4.0).repeat((4, 4, 4))
        cluster2 = bulk("Pt", "fcc", a=4.0).repeat((4, 4, 4))

        # Move cluster2 far away
        cluster2.translate([50, 0, 0])

        # Combine them
        combined = cluster1 + cluster2
        combined.center()

        assert len(combined) > 50, (
            f"Test requires >50 atoms to trigger KDTree path, got {len(combined)}"
        )
        assert (
            is_cluster_connected(combined, connectivity_factor=CONNECTIVITY_FACTOR)
            is False
        )

    @pytest.mark.parametrize("n", [48, 49, 50, 51, 52])
    def test_connectivity_boundary_small_large(self, n):
        """Test connectivity at boundary between small and large cluster paths."""
        # Test clusters around the 50-atom threshold
        # Create a compact cluster
        atoms = bulk("Pt", "fcc", a=4.0)
        # Adjust repeat to get approximately n atoms
        repeat = int(np.ceil(n ** (1 / 3)))
        atoms = atoms.repeat((repeat, repeat, repeat))

        # Trim to exact size if needed
        if len(atoms) > n:
            atoms = atoms[:n]

        atoms.center()

        # All should be connected (compact FCC structure)
        result = is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
        assert result is True, f"Failed for {n} atoms"

    @pytest.mark.parametrize(
        "positions,expected",
        [
            ([[0, 0, 0], [2.5, 0, 0]], True),  # Connected
            ([[0, 0, 0], [20, 0, 0]], False),  # Disconnected
            ([[0, 0, 0], [2.5, 0, 0], [5, 0, 0]], True),  # Chain
            ([[0, 0, 0], [2.5, 0, 0], [20, 0, 0]], False),  # Partial disconnect
        ],
    )
    def test_connectivity_regression_against_original(self, positions, expected):
        """Test that optimized version gives same results as original for various cases."""
        atoms = Atoms(f"Pt{len(positions)}", positions=positions)
        # Use stricter factor (1.2) to test factor sensitivity
        STRICT_FACTOR = 1.2
        result = is_cluster_connected(atoms, connectivity_factor=STRICT_FACTOR)
        assert result == expected, f"Failed for positions {positions}"

    @pytest.mark.slow
    def test_large_cluster_performance(self):
        """Test that large cluster connectivity check completes in reasonable time."""
        import time

        # Create a very large cluster (100+ atoms)
        # FCC: (5,5,5) = 125 atoms
        atoms = bulk("Pt", "fcc", a=4.0).repeat((5, 5, 5))
        atoms.center()

        assert len(atoms) >= 100, (
            f"Need large cluster for performance test, got {len(atoms)}"
        )

        start = time.time()
        result = is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
        elapsed = time.time() - start

        # Should complete reasonably quickly; allow more generous bound in CI
        assert elapsed < 5.0, f"Connectivity check took {elapsed:.3f}s, should be <5s"
        assert result is True, "Large FCC cluster should be connected"

        # Performance logging removed to keep test output concise
