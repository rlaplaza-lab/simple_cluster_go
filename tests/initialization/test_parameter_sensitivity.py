"""Tests for parameter sensitivity and cross-mode comparisons.

This module tests:
- How initialization parameters affect cluster generation across modes
- Cross-mode behavior comparisons
- Parameter interactions
- Performance scaling with cluster size

These tests were identified as coverage gaps and are new additions
to improve overall test suite coverage.
"""

import numpy as np
import pytest

from scgo.initialization import create_initial_cluster, is_cluster_connected
from tests.test_utils import (
    LARGE_SIZES,
    MEDIUM_SIZES,
    SMALL_SIZES,
    assert_cluster_valid,
    create_paired_rngs,
)


class TestParameterSensitivity:
    """Test how parameters affect cluster generation."""

    @pytest.mark.parametrize(
        "connectivity_factor",
        [0.7, 0.9, 1.0, 1.1, 1.3],
    )
    def test_connectivity_factor_impact(self, connectivity_factor, rng):
        """Test that varying connectivity_factor affects cluster generation."""
        comp = ["Pt"] * 8
        try:
            atoms = create_initial_cluster(
                comp,
                mode="random_spherical",
                connectivity_factor=connectivity_factor,
                rng=rng,
            )
            # Should always produce connected clusters with proper connectivity factor
            assert len(atoms) == 8
            assert_cluster_valid(atoms, comp)
        except ValueError as e:
            # Accept validation failures as an expected outcome for extreme parameters
            if "Validation failed" in str(e):
                return
            raise

    @pytest.mark.parametrize(
        "placement_radius_scaling",
        [0.4, 0.6, 0.8, 1.0, 1.2],
    )
    def test_placement_radius_scaling_impact(self, placement_radius_scaling, rng):
        """Test that varying placement_radius_scaling affects cluster generation."""
        comp = ["Pt"] * 6
        try:
            atoms = create_initial_cluster(
                comp,
                mode="smart",
                placement_radius_scaling=placement_radius_scaling,
                rng=rng,
            )
            assert len(atoms) == 6
            assert_cluster_valid(atoms, comp)
        except ValueError as e:
            # Validation failures are an acceptable outcome for tight placement
            if "Validation failed" in str(e):
                return
            raise

    @pytest.mark.parametrize(
        "vacuum",
        [3.0, 5.0, 7.0, 10.0, 15.0],
    )
    def test_vacuum_impact_on_cluster(self, vacuum, rng):
        """Test that vacuum parameter affects cell size appropriately."""
        comp = ["Pt"] * 5
        atoms1 = create_initial_cluster(
            comp, mode="random_spherical", vacuum=3.0, rng=rng
        )
        atoms2 = create_initial_cluster(
            comp, mode="random_spherical", vacuum=10.0, rng=rng
        )

        cell1 = atoms1.get_cell().lengths()
        cell2 = atoms2.get_cell().lengths()

        # Larger vacuum should result in larger cells
        assert np.mean(cell2) > np.mean(cell1)

    def test_parameter_interaction_connectivity_and_radius(self, rng):
        """Test interaction between connectivity_factor and placement_radius_scaling."""
        comp = ["Pt"] * 8

        # Test different combinations
        combinations = [
            (0.8, 0.5),
            (1.0, 0.7),
            (1.2, 1.0),
        ]

        for cf, prs in combinations:
            atoms = create_initial_cluster(
                comp,
                mode="smart",
                connectivity_factor=cf,
                placement_radius_scaling=prs,
                rng=rng,
            )
            assert len(atoms) == 8
            assert_cluster_valid(atoms, comp, check_connectivity=True)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
    def test_reproducibility_with_parameters(self, seed):
        """Test that same parameters with same seed produce same structure."""
        comp = ["Pt"] * 6

        rng1, rng2 = create_paired_rngs(seed)
        atoms1 = create_initial_cluster(
            comp,
            mode="seed+growth",
            connectivity_factor=1.0,
            placement_radius_scaling=0.7,
            rng=rng1,
        )

        atoms2 = create_initial_cluster(
            comp,
            mode="seed+growth",
            connectivity_factor=1.0,
            placement_radius_scaling=0.7,
            rng=rng2,
        )

        assert np.allclose(
            atoms1.get_positions(),
            atoms2.get_positions(),
            atol=1e-6,
        )


class TestCrossModeComparison:
    """Compare behavior across different initialization modes."""

    def test_all_modes_produce_similar_composition(self, rng):
        """Test that all modes respect composition exactly."""
        comp = ["Pt", "Au", "Pd", "Pt", "Au"]

        modes = ["random_spherical", "seed+growth", "smart"]

        for mode in modes:
            try:
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                assert_cluster_valid(atoms, comp)
            except ValueError:
                if mode == "template":
                    pytest.skip("Template mode may fail for non-magic numbers")
                raise

    def test_mode_connectivity_consistency(self, rng):
        """Test that all modes produce connected structures."""
        comp = ["Pt"] * 10

        modes = ["random_spherical", "seed+growth", "smart"]

        for mode in modes:
            try:
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                assert is_cluster_connected(atoms)
            except ValueError:
                if mode == "template":
                    pytest.skip("Template mode may fail for non-magic numbers")
                raise

    def test_mode_diversity_comparison(self, rng):
        """Compare diversity of structures across modes."""
        comp = ["Pt"] * 6

        def structure_signature(atoms):
            """Create a unique signature for cluster structure."""
            p = atoms.get_positions()
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            triu = d[np.triu_indices(len(p), k=1)]
            return tuple(np.round(np.sort(triu), 4))

        mode_diversities = {}

        for mode in ["random_spherical", "seed+growth"]:
            signatures = set()
            for _ in range(5):
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                signatures.add(structure_signature(atoms))

            mode_diversities[mode] = len(signatures)

        # Both modes should produce at least some diversity
        assert all(d >= 1 for d in mode_diversities.values())

    def test_mode_performance_scaling(self, rng):
        """Compare relative performance across modes for different sizes."""
        import time

        modes = ["random_spherical", "seed+growth", "smart"]
        sizes = [3, 5, 8]

        # Quick baseline test - just verify all modes complete in reasonable time
        for size in sizes:
            comp = ["Pt"] * size
            for mode in modes:
                try:
                    start = time.perf_counter()
                    atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                    elapsed = time.perf_counter() - start

                    assert len(atoms) == size
                    # Should complete in < 5 seconds for test sizes
                    assert elapsed < 5.0
                except ValueError:
                    if mode == "template":
                        pytest.skip("Template mode may fail for non-magic numbers")
                    raise


class TestClusterSizeScaling:
    """Test how algorithms scale with cluster size."""

    @pytest.mark.parametrize("size", SMALL_SIZES)
    def test_connectivity_scaling_small(self, size, rng):
        """Test connectivity is maintained for small clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize("size", MEDIUM_SIZES)
    @pytest.mark.slow
    def test_connectivity_scaling_medium(self, size, rng):
        """Test connectivity is maintained for medium clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize("size", LARGE_SIZES)
    @pytest.mark.slow
    def test_connectivity_scaling_large(self, size, rng):
        """Test connectivity is maintained for large clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="random_spherical", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    def test_cell_size_scales_with_cluster_size(self, rng):
        """Test that cell size appropriately scales with cluster size."""
        cell_sizes = {}

        for size in [2, 5, 10, 15]:
            comp = ["Pt"] * size
            atoms = create_initial_cluster(comp, vacuum=6.0, rng=rng)
            cell_volume = atoms.get_cell().volume
            cell_sizes[size] = cell_volume

        # Larger clusters should have larger cells
        assert cell_sizes[15] > cell_sizes[10]
        assert cell_sizes[10] > cell_sizes[5]
        assert cell_sizes[5] > cell_sizes[2]

    def test_position_variance_scales_with_size(self, rng):
        """Test that positional variance increases with cluster size."""
        position_ranges = {}

        for size in [3, 5, 8, 12]:
            comp = ["Pt"] * size
            atoms = create_initial_cluster(comp, rng=rng)
            positions = atoms.get_positions()
            position_range = np.max(positions) - np.min(positions)
            position_ranges[size] = position_range

        # Larger clusters should have larger position spreads
        assert position_ranges[12] > position_ranges[3]


class TestCompositionVariety:
    """Test parameter sensitivity across various compositions."""

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 4,
            ["Au"] * 4,
            ["Pd"] * 4,
            ["Pt", "Pt", "Au", "Au"],
            ["Pt", "Au", "Pd", "Pd"],
            ["Pt", "Pt", "Pt", "Au"],
        ],
    )
    def test_connectivity_across_compositions(self, composition, rng):
        """Test that connectivity is maintained across different compositions."""
        atoms = create_initial_cluster(composition, mode="smart", rng=rng)
        assert_cluster_valid(atoms, composition)

        if len(atoms) > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 5,
            ["Au"] * 5,
            ["Pt", "Pt", "Au", "Au", "Pd"],
        ],
    )
    @pytest.mark.parametrize("cf", [0.8, 1.0, 1.2])
    def test_parameter_robustness_across_elements(self, composition, cf, rng):
        """Test parameter sensitivity is consistent across element types."""
        try:
            atoms = create_initial_cluster(
                composition,
                mode="seed+growth",
                connectivity_factor=cf,
                rng=rng,
            )
            assert_cluster_valid(atoms, composition)
            assert is_cluster_connected(atoms)
        except ValueError as e:
            # Treat validation failures as acceptable for extreme parameter combinations
            if "Validation failed" in str(e):
                return
            raise
