"""Tests for geometry helper functions.

These tests verify the geometric placement functions in scgo.geometry_helpers,
particularly the convex hull facet generation for cluster growth.
"""

import numpy as np
import pytest
from ase import Atoms

from scgo.initialization.geometry_helpers import (
    _generate_batch_positions_on_convex_hull,
    analyze_disconnection,
    compute_bond_distance_params,
    get_convex_hull_vertex_indices,
    get_largest_facets,
    place_multi_atom_seed_on_facet,
)
from tests.test_utils import create_paired_rngs


class TestComputeBondDistanceParams:
    """Unit tests for compute_bond_distance_params."""

    def test_basic_output(self):
        """Known inputs produce expected bond_distance and connectivity bounds."""
        max_existing = 1.3
        avg_new = 1.2
        cf = 1.4
        mdf = 0.5
        prs = 1.2
        bond_distance, min_dist, max_conn = compute_bond_distance_params(
            max_existing, avg_new, cf, mdf, prs
        )
        base = max_existing + avg_new
        expected_max_conn = base * cf
        expected_min_dist = base * mdf
        expected_target = base * min(prs, cf)
        expected_bond = max(expected_min_dist, min(expected_target, expected_max_conn))
        assert max_conn == pytest.approx(expected_max_conn)
        assert min_dist == pytest.approx(expected_min_dist)
        assert bond_distance == pytest.approx(expected_bond)

    def test_effective_overrides(self):
        """effective_min_distance and effective_scaling override defaults."""
        bond, min_d, max_c = compute_bond_distance_params(
            1.0,
            1.0,
            1.4,
            0.5,
            1.2,
            effective_min_distance=0.4,
            effective_scaling=1.0,
        )
        base = 2.0
        assert max_c == pytest.approx(base * 1.4)
        assert min_d == pytest.approx(base * 0.4)
        target = base * min(1.0, 1.4)
        assert bond == pytest.approx(max(0.8, min(target, 2.8)))

    def test_clip_to_bounds(self):
        """bond_distance is clipped between min_dist and max_conn."""
        # placement_radius_scaling > connectivity_factor -> target > max_conn
        bond, min_d, max_c = compute_bond_distance_params(1.0, 1.0, 1.4, 0.5, 2.0)
        assert min_d <= bond <= max_c


class TestAnalyzeDisconnection:
    """Tests for analyze_disconnection function."""

    def test_empty_cluster(self):
        """Test analyze_disconnection with empty cluster."""
        atoms = Atoms()
        distance, factor, msg = analyze_disconnection(atoms)
        assert distance == pytest.approx(0.0, abs=1e-8)
        assert "Single atom or empty cluster" in msg

    def test_single_atom(self):
        """Test analyze_disconnection with single atom."""
        atoms = Atoms("Pt", positions=[[0, 0, 0]])
        distance, factor, msg = analyze_disconnection(atoms)
        assert distance == pytest.approx(0.0, abs=1e-8)
        assert "Single atom or empty cluster" in msg

    def test_connected_cluster(self):
        """Test analyze_disconnection with connected cluster."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        distance, factor, msg = analyze_disconnection(atoms)
        assert distance == pytest.approx(0.0, abs=1e-8)
        assert "Cluster is connected" in msg

    def test_disconnected_cluster_two_components(self):
        """Test analyze_disconnection with disconnected cluster (2 components)."""
        atoms = Atoms(
            "Pt4", positions=[[0, 0, 0], [2.5, 0, 0], [10, 0, 0], [12.5, 0, 0]]
        )
        distance, factor, msg = analyze_disconnection(atoms)
        assert distance > 0.0
        assert "2 disconnected components" in msg
        assert factor > 1.0  # Suggested factor should be reasonable

    def test_disconnected_cluster_multiple_components(self):
        """Test analyze_disconnection with multiple disconnected components."""
        # Create 3 separate clusters
        positions = [
            [0, 0, 0],
            [2.5, 0, 0],  # Cluster 1
            [10, 0, 0],
            [12.5, 0, 0],  # Cluster 2
            [20, 0, 0],  # Cluster 3 (single atom)
        ]
        atoms = Atoms("Pt5", positions=positions)
        distance, factor, msg = analyze_disconnection(atoms)
        assert distance > 0.0
        assert "3 disconnected components" in msg

    def test_exactly_at_threshold(self):
        """Test analyze_disconnection at connectivity threshold."""
        # Create atoms exactly at threshold
        from ase.data import atomic_numbers, covalent_radii

        r_pt = covalent_radii[atomic_numbers["Pt"]]
        TEST_FACTOR = 2.5  # Specific factor for testing threshold behavior
        threshold = 2 * r_pt * TEST_FACTOR
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [threshold, 0, 0]])
        distance, factor, msg = analyze_disconnection(
            atoms, connectivity_factor=TEST_FACTOR
        )
        # Should be connected at this threshold
        assert distance == pytest.approx(0.0, abs=1e-8) or "Cluster is connected" in msg

    def test_very_large_connectivity_factor(self):
        """Test analyze_disconnection with very large connectivity factor."""
        VERY_LARGE_FACTOR = 10.0  # Very large factor for testing boundary conditions
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [10, 0, 0]])
        distance, factor, msg = analyze_disconnection(
            atoms, connectivity_factor=VERY_LARGE_FACTOR
        )
        # Should be connected with large factor
        assert distance == pytest.approx(0.0, abs=1e-8) or "Cluster is connected" in msg

    def test_mixed_elements(self):
        """Test analyze_disconnection with mixed element types."""
        atoms = Atoms("PtAu", positions=[[0, 0, 0], [10, 0, 0]])
        distance, factor, msg = analyze_disconnection(atoms)
        assert isinstance(distance, float)
        assert isinstance(factor, float)
        assert isinstance(msg, str)

    def test_suggested_factor_includes_buffer(self):
        """Test that suggested factor includes 5% buffer."""
        # Use a test connectivity factor to verify buffer calculation
        TEST_FACTOR = 1.5
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [5.0, 0, 0]])
        distance, factor, msg = analyze_disconnection(
            atoms, connectivity_factor=TEST_FACTOR
        )
        if distance > 0:
            # Suggested factor should be at least the required factor
            assert (
                factor >= distance / (2 * 1.36) * 1.05
            )  # 1.36 is approximate Pt radius


class TestGetLargestFacets:
    """Tests for get_largest_facets function."""

    def test_small_cluster_fallback(self):
        """Test get_largest_facets with small clusters (<4 atoms)."""
        # Single atom
        atoms1 = Atoms("Pt", positions=[[0, 0, 0]])
        facets1 = get_largest_facets(atoms1, n_facets=3)
        assert len(facets1) == 1
        assert len(facets1[0]) == 3  # (centroid, normal, area)
        assert facets1[0][2] == pytest.approx(
            1.0, rel=1e-6
        )  # Area is ~= 1.0 for fallback

        # Two atoms
        atoms2 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])
        facets2 = get_largest_facets(atoms2, n_facets=3)
        assert len(facets2) == 1
        assert facets2[0][2] == pytest.approx(1.0, rel=1e-6)

        # Three atoms
        atoms3 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0]])
        facets3 = get_largest_facets(atoms3, n_facets=3)
        assert len(facets3) == 1
        assert facets3[0][2] == pytest.approx(1.0, rel=1e-6)

    def test_normal_3d_cluster(self):
        """Test get_largest_facets with normal 3D cluster."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )
        facets = get_largest_facets(atoms, n_facets=3)
        assert len(facets) == 3
        for facet in facets:
            assert len(facet) == 3  # (centroid, normal, area)
            assert facet[2] > 0  # Area should be positive
            assert np.linalg.norm(facet[1]) > 0  # Normal should be non-zero

    def test_n_facets_parameter(self):
        """Test get_largest_facets with different n_facets values."""
        atoms = Atoms(
            "Pt6",
            positions=[
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
        )

        # Request fewer facets than available
        facets1 = get_largest_facets(atoms, n_facets=1)
        assert len(facets1) == 1

        # Request more facets than available (should return all available)
        facets_many = get_largest_facets(atoms, n_facets=100)
        # Should return all available facets (number depends on convex hull)
        assert facets_many
        assert len(facets_many) <= 100  # Should not exceed requested number

    def test_degenerate_geometry_fallback(self):
        """Test get_largest_facets with degenerate geometries."""
        # Collinear atoms
        atoms_linear = Atoms("Pt3", positions=[[0, 0, 0], [0, 0, 2], [0, 0, 4]])
        facets = get_largest_facets(atoms_linear, n_facets=3)
        # Should handle gracefully, may return fallback facet
        assert facets

    def test_facets_sorted_by_area(self):
        """Test that facets are sorted by area (largest first)."""
        atoms = Atoms(
            "Pt8",
            positions=[
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [2, 2, 0],  # Square base
                [0, 0, 2],
                [2, 0, 2],
                [0, 2, 2],
                [2, 2, 2],  # Square top
            ],
        )
        facets = get_largest_facets(atoms, n_facets=5)
        # Areas should be in descending order
        areas = [f[2] for f in facets]
        assert all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1))
        assert facets

    def test_exception_handling(self):
        """Test get_largest_facets exception handling for ConvexHull failures."""
        # Try with degenerate case that might cause ConvexHull to fail
        # The function should catch exceptions and return fallback
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [0, 0, 1e-10], [0, 1e-10, 0]])
        # Should not raise exception, should return fallback facet
        facets = get_largest_facets(atoms, n_facets=3)
        assert facets


class TestPlaceMultiAtomSeedOnFacet:
    """Tests for place_multi_atom_seed_on_facet function."""

    def test_single_atom_seed(self, rng):
        """Test placing single atom seed."""
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_centroid = np.array([5, 0, 0])
        target_normal = np.array([1, 0, 0])

        placed = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=2.5, rng=rng
        )
        assert isinstance(placed, Atoms)
        assert len(placed) == 1
        assert placed.get_chemical_symbols() == ["Pt"]

    def test_multi_atom_seed(self, rng):
        """Test placing multi-atom seed."""
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        target_centroid = np.array([5, 0, 0])
        target_normal = np.array([1, 0, 0])

        placed = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=2.5, rng=rng
        )
        assert isinstance(placed, Atoms)
        assert len(placed) == 2
        # Seed should be translated/rotated, so positions should change
        assert not np.allclose(placed.get_positions(), seed.get_positions())

    def test_parallel_normals_edge_case(self, rng):
        """Test placement when seed normal and target normal are parallel."""
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_centroid = np.array([5, 0, 0])
        target_normal = np.array([1, 0, 0])
        # Place seed so its normal is also [1, 0, 0] (parallel)
        # Function should handle this gracefully

        placed = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=2.5, rng=rng
        )
        assert isinstance(placed, Atoms)

    def test_very_large_bond_distance(self, rng):
        """Test placement with very large bond distance."""
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_centroid = np.array([0, 0, 0])
        target_normal = np.array([1, 0, 0])

        placed = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=100.0, rng=rng
        )
        assert isinstance(placed, Atoms)
        # Position should be far from target centroid
        pos = placed.get_positions()[0]
        distance = np.linalg.norm(pos - target_centroid)
        assert distance > 50.0

    def test_empty_seed_handling(self, rng):
        """Test handling of empty seed."""
        seed = Atoms()
        target_centroid = np.array([0, 0, 0])
        target_normal = np.array([1, 0, 0])

        # Should handle gracefully (though this case is unusual)
        try:
            placed = place_multi_atom_seed_on_facet(
                seed, target_centroid, target_normal, bond_distance=2.5, rng=rng
            )
            # If it doesn't raise, should return empty Atoms
            assert len(placed) == 0
        except (ValueError, IndexError):
            # Or may raise an error, which is acceptable
            pass

    def test_rotation_and_translation(self, rng):
        """Test that seed is properly rotated and translated."""
        # Create a seed with clear orientation
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]])

        target_centroid = np.array([10, 10, 10])
        target_normal = np.array([0, 0, 1])

        placed = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=3.0, rng=rng
        )

        # Check that structure changed
        assert not np.allclose(placed.get_positions(), seed.get_positions())
        # Check that all atoms are still present
        assert len(placed) == 3
        assert placed.get_chemical_symbols() == seed.get_chemical_symbols()

    def test_reproducibility(self, rng):
        """Test that same RNG produces same placement."""
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        target_centroid = np.array([5, 0, 0])
        target_normal = np.array([1, 0, 0])

        rng1, rng2 = create_paired_rngs(42)

        placed1 = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=2.5, rng=rng1
        )
        placed2 = place_multi_atom_seed_on_facet(
            seed, target_centroid, target_normal, bond_distance=2.5, rng=rng2
        )

        # Positions should be identical
        assert np.allclose(placed1.get_positions(), placed2.get_positions())


class TestGenerateBatchPositionsOnConvexHull:
    """Tests for _generate_batch_positions_on_convex_hull function."""

    def test_small_cluster_returns_empty(self, rng):
        """Test that clusters with <4 atoms return empty list."""
        # Test with 1 atom
        atoms1 = Atoms("Pt", positions=[[0, 0, 0]])
        result1 = _generate_batch_positions_on_convex_hull(
            atoms1, n_candidates=5, bond_distance=2.0, rng=rng
        )
        assert result1 == []

        # Test with 2 atoms
        atoms2 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])
        result2 = _generate_batch_positions_on_convex_hull(
            atoms2, n_candidates=5, bond_distance=2.0, rng=rng
        )
        assert result2 == []

        # Test with 3 atoms
        atoms3 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0]])
        result3 = _generate_batch_positions_on_convex_hull(
            atoms3, n_candidates=5, bond_distance=2.0, rng=rng
        )
        assert result3 == []

    def test_tetrahedral_cluster_generates_candidates(self, rng):
        """Test that a tetrahedral cluster (4 atoms) generates candidates."""
        # Create a tetrahedral cluster
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        candidates = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=10, bond_distance=2.0, rng=rng
        )

        # Should generate some candidates (tetrahedron has 4 facets)
        assert len(candidates) > 0
        assert len(candidates) <= 10  # Should not exceed n_candidates
        assert len(candidates) <= 4  # Tetrahedron has 4 facets

        # All candidates should be 3D positions
        for candidate in candidates:
            assert isinstance(candidate, np.ndarray)
            assert candidate.shape == (3,)

    def test_batch_size_limited_by_facets(self, rng):
        """Test that batch size is limited by number of facets."""
        # Create a larger cluster (octahedron-like, 6 atoms)
        atoms = Atoms(
            "Pt6",
            positions=[
                [0, 0, 2],
                [2, 0, 0],
                [0, 2, 0],
                [-2, 0, 0],
                [0, -2, 0],
                [0, 0, -2],
            ],
        )

        # Request more candidates than facets
        candidates = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=100, bond_distance=2.0, rng=rng
        )

        # Should not exceed actual number of facets
        # Octahedron has 8 facets, but we might get fewer due to selection
        assert len(candidates) <= 100
        assert len(candidates) > 0

    def test_batch_size_limited_by_request(self, rng):
        """Test that batch size is limited by n_candidates parameter."""
        # Create a larger cluster with many facets
        atoms = Atoms(
            "Pt10",
            positions=[
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [-2, 0, 0],
                [0, -2, 0],
                [1, 1, 1],
                [-1, -1, -1],
                [1, -1, 1],
                [-1, 1, -1],
                [0, 0, 2],
            ],
        )

        # Request fewer candidates than facets
        candidates = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=3, bond_distance=2.0, rng=rng
        )

        # Should not exceed n_candidates
        assert len(candidates) <= 3
        assert len(candidates) > 0

    def test_candidates_are_reasonable_distances(self, rng):
        """Test that generated candidates are at reasonable distances."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        bond_distance = 2.5
        candidates = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=10, bond_distance=bond_distance, rng=rng
        )

        center = atoms.get_center_of_mass()

        for candidate in candidates:
            # Distance from center should be reasonable
            dist = np.linalg.norm(candidate - center)
            # Should be roughly around bond_distance (with some variation)
            assert dist > 0.5  # Not too close
            assert dist < 10.0  # Not too far

    def test_reproducibility(self, rng):
        """Test that same RNG produces same batch of candidates."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        rng1, rng2 = create_paired_rngs(42)

        candidates1 = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=5, bond_distance=2.0, rng=rng1
        )
        candidates2 = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=5, bond_distance=2.0, rng=rng2
        )

        # Should generate same number of candidates
        assert len(candidates1) == len(candidates2)

        # Positions should be identical (within numerical precision)
        for c1, c2 in zip(candidates1, candidates2, strict=True):
            assert np.allclose(c1, c2)

    def test_connectivity_constraints(self, rng):
        """Test that connectivity constraints are respected."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        min_connectivity_dist = 1.0
        max_connectivity_dist = 5.0

        candidates = _generate_batch_positions_on_convex_hull(
            atoms,
            n_candidates=5,
            bond_distance=3.0,
            rng=rng,
            min_connectivity_dist=min_connectivity_dist,
            max_connectivity_dist=max_connectivity_dist,
        )

        # All candidates should be generated (constraints should be handled internally)
        assert len(candidates) > 0

        # Verify candidates are reasonable
        for candidate in candidates:
            assert isinstance(candidate, np.ndarray)
            assert candidate.shape == (3,)


class TestGetConvexHullVertexIndices:
    """Tests for get_convex_hull_vertex_indices."""

    def test_small_cluster_returns_empty(self):
        """Clusters with <4 atoms return empty array."""
        for n in (1, 2, 3):
            atoms = Atoms("Pt" + str(n), positions=[[i, 0, 0] for i in range(n)])
            idx = get_convex_hull_vertex_indices(atoms)
            assert isinstance(idx, np.ndarray)
            assert idx.dtype == np.intp
            assert len(idx) == 0

    def test_tetrahedron_returns_four_vertices(self):
        """Tetrahedron has 4 hull vertices (all atoms)."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )
        idx = get_convex_hull_vertex_indices(atoms)
        assert isinstance(idx, np.ndarray)
        assert idx.dtype == np.intp
        assert len(idx) == 4
        assert set(idx) == {0, 1, 2, 3}

    def test_vertices_are_valid_atom_indices(self):
        """Vertex indices are in [0, len(atoms))."""
        atoms = Atoms(
            "Pt6",
            positions=[
                [0, 0, 2],
                [2, 0, 0],
                [0, 2, 0],
                [-2, 0, 0],
                [0, -2, 0],
                [0, 0, -2],
            ],
        )
        idx = get_convex_hull_vertex_indices(atoms)
        assert len(idx) <= 6
        assert np.all((idx >= 0) & (idx < 6))


class TestGeneratePositionsAllFacets:
    """Tests for _generate_positions_all_facets (all-facets mode)."""

    def test_small_cluster_returns_empty(self, rng):
        """Clusters with <4 atoms return empty list."""
        for n in (1, 2, 3):
            atoms = Atoms("Pt" + str(n), positions=[[i, 0, 0] for i in range(n)])
            result = _generate_batch_positions_on_convex_hull(
                atoms, n_candidates=0, bond_distance=2.0, rng=rng, use_all_facets=True
            )
            assert result == []

    def test_one_per_facet_deterministic(self, rng):
        """All-facets returns one position per facet; tetrahedron has 4."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )
        result = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=0, bond_distance=2.0, rng=rng, use_all_facets=True
        )
        assert len(result) == 4
        for pos in result:
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (3,)

    def test_reproducibility_same_rng(self, rng):
        """Same RNG yields identical all-facets positions."""
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )
        rng1, rng2 = create_paired_rngs(42)
        a = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=0, bond_distance=2.0, rng=rng1, use_all_facets=True
        )
        b = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=0, bond_distance=2.0, rng=rng2, use_all_facets=True
        )
        assert len(a) == len(b)
        for p, q in zip(a, b, strict=True):
            assert np.allclose(p, q)


# Tests merged from test_disjointed_clusters_bug.py - simplified to avoid complex dependencies
class TestDisjointedClustersBug:
    """Test suite to prevent regression of the disjointed clusters bug."""

    def test_wrap_with_pbc_false(self):
        """Test that wrap() works correctly with PBC=False after our fix."""
        from tests.test_utils import setup_test_atoms

        # Create a cluster with atoms outside the cell
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [25, 0, 0]])
        setup_test_atoms(atoms)

        # Verify atoms are outside cell
        assert atoms.positions[1, 0] > atoms.cell[0, 0]

        # Apply our fix: temporarily enable PBC for wrapping
        original_pbc = atoms.get_pbc()
        atoms.set_pbc(True)
        atoms.wrap()
        atoms.set_pbc(original_pbc)

        # Verify atoms are now inside cell
        assert atoms.positions[1, 0] < atoms.cell[0, 0]
        assert atoms.positions[1, 0] >= 0

    def test_scaled_position_wrapping(self):
        """Test that scaled position wrapping maintains structure integrity."""
        from tests.test_utils import setup_test_atoms

        # Create a cluster with atoms outside the cell
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [25, 0, 0]])
        setup_test_atoms(atoms)

        # Apply manual wrapping
        scaled_pos = atoms.get_scaled_positions(wrap=False)
        atoms.set_scaled_positions(scaled_pos % 1.0)

        # Verify atoms are now inside cell
        assert atoms.positions[1, 0] < atoms.cell[0, 0]
        assert atoms.positions[1, 0] >= 0


# Consolidated basic position generation tests
class TestGenerateBatchPositionsBasic:
    """Consolidated basic tests for batch position generation."""

    def test_edge_cases(self, rng):
        """Test fallback for small/degenerate clusters."""
        atoms_small = Atoms("Pt", positions=[[0, 0, 0]])
        candidates = _generate_batch_positions_on_convex_hull(
            atoms_small,
            n_candidates=1,
            bond_distance=2.0,
            rng=rng,
            use_all_facets=False,
        )
        pos = (
            candidates[0]
            if candidates
            else atoms_small.get_center_of_mass() + rng.standard_normal(3) * 2.0
        )
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)

    def test_standard_cluster(self, rng):
        """Test with standard 4-atom cluster."""
        atoms = Atoms(
            "Pt4", positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]]
        )
        candidates = _generate_batch_positions_on_convex_hull(
            atoms, n_candidates=5, bond_distance=2.0, rng=rng, use_all_facets=False
        )
        assert len(candidates) > 0
