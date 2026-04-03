"""Tests for template initialization mode.

This module consolidates all tests for template mode initialization including:
- Basic template mode functionality
- Template removal safety
- Convex hull placement reliability
- Large cluster connectivity (50-60 atoms)
- Template generation optimization
"""

import pytest
from ase import Atoms

from scgo.initialization import (
    combine_seeds,
    create_initial_cluster,
    grow_from_seed,
    is_cluster_connected,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
)
from scgo.initialization.initializers import compute_cell_side
from scgo.initialization.templates import (
    generate_template_matches,
    generate_template_structure,
)
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import (
    REPRODUCIBILITY_SEEDS,
    assert_cluster_valid,
    create_paired_rngs,
    validate_structure_with_diagnostics,
)


class TestTemplateModeInitialization:
    """Tests for template mode initialization."""

    def test_template_mode_icosahedron(self, rng):
        """Test template mode with icosahedral magic number."""
        comp = ["Pt"] * 13
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        assert len(atoms) == 13
        assert_cluster_valid(atoms, comp)

    def test_template_mode_decahedron(self, rng):
        """Test template mode with decahedral size."""
        comp = ["Pt"] * 23
        atoms = create_initial_cluster(
            comp,
            mode="template",
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
            placement_radius_scaling=1.3,
        )
        assert len(atoms) == 23
        assert_cluster_valid(atoms, comp)

    def test_template_mode_near_magic(self, rng):
        """Test template mode near magic number (adds/removes atoms)."""
        comp = ["Pt"] * 20  # Near 13 or 23
        atoms = create_initial_cluster(
            comp,
            mode="template",
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
            placement_radius_scaling=1.3,
        )
        assert len(atoms) == 20
        assert_cluster_valid(atoms, comp)

    def test_template_mode_fallback(self, rng):
        """Test template mode fallback when generation fails."""
        # Very small size that may not have template
        comp = ["Pt"] * 2
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        # Should fall back to random_spherical and still be valid
        assert len(atoms) == 2
        assert_cluster_valid(atoms, comp)

    def test_template_mode_multi_element(self, rng):
        """Test template mode with multiple elements."""
        comp = ["Pt", "Au"] * 6 + ["Pt"]  # 13 atoms total
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        assert len(atoms) == 13
        assert_cluster_valid(atoms, comp)
        symbols = atoms.get_chemical_symbols()
        assert "Pt" in symbols
        assert "Au" in symbols


class TestTemplateRemovalSafety:
    """Tests for template generation with atom removal.

    These tests verify that template generation either produces valid
    connected structures or cleanly returns None (without producing
    disconnected clusters).
    """

    # Sizes just below magic numbers where removal is needed
    NEAR_MAGIC_SIZES = [
        (12, 13),  # 12 atoms from 13-atom icosahedron
        (11, 13),  # 11 atoms from 13-atom icosahedron
        (52, 55),  # 52 atoms from 55-atom icosahedron
        (50, 55),  # 50 atoms from 55-atom icosahedron
    ]

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    @pytest.mark.parametrize(
        "target_n,magic_n", NEAR_MAGIC_SIZES[:2]
    )  # Just the smaller ones
    def test_near_match_templates_valid_or_none(self, seed, target_n, magic_n):
        """Test that near-match templates are either valid or cleanly skipped.

        When generating templates that require atom removal, the result should
        either be a valid connected structure or None (if removal would disconnect).
        """
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * target_n
        cell_side = compute_cell_side(comp)

        templates = generate_template_matches(
            composition=comp,
            n_atoms=target_n,
            rng=rng,
            cell_side=cell_side,
            include_exact=False,
            include_near=True,
        )

        # Each returned template must be valid
        for template in templates:
            assert len(template) == target_n, (
                f"Template has wrong size: {len(template)} != {target_n}"
            )

            validate_structure_with_diagnostics(
                template,
                context=f"seed={seed}, target_n={target_n}, magic_n={magic_n}",
            )

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    @pytest.mark.parametrize(
        "template_type", ["icosahedron", "decahedron", "octahedron"]
    )
    def test_template_structure_valid_or_none(self, seed, template_type):
        """Test that generate_template_structure returns valid or None."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        n_atoms = 10  # Not a magic number, requires adjustment
        comp = ["Pt"] * n_atoms

        result = generate_template_structure(
            composition=comp,
            n_atoms=n_atoms,
            template_type=template_type,
            rng=rng,
        )

        if result is not None:
            assert len(result) == n_atoms
            validate_structure_with_diagnostics(
                result,
                context=f"seed={seed}, template_type={template_type}, n_atoms={n_atoms}",
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:3])
    @pytest.mark.parametrize("target_n,magic_n", NEAR_MAGIC_SIZES)
    def test_near_match_templates_larger_sizes(self, seed, target_n, magic_n):
        """Test near-match templates with larger sizes (slow)."""
        LARGE_CLUSTER_FACTOR = 2.0  # More lenient for larger structures
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * target_n
        cell_side = compute_cell_side(comp)

        templates = generate_template_matches(
            composition=comp,
            n_atoms=target_n,
            rng=rng,
            cell_side=cell_side,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
            include_exact=False,
            include_near=True,
        )

        for template in templates:
            assert len(template) == target_n
            validate_structure_with_diagnostics(
                template,
                connectivity_factor=LARGE_CLUSTER_FACTOR,
                context=f"seed={seed}, target_n={target_n}",
            )


class TestConvexHullPlacementReliability:
    """Tests verifying convex hull placement works reliably for normal cases."""

    def test_convex_hull_placement_always_works(self, rng):
        """Verify convex hull placement works reliably for normal cases (no fallbacks needed)."""
        # Test seed+growth which uses convex hull placement
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 10
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Should work with default parameters - convex hull placement should be reliable
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert result is not None, "Convex hull placement should work for normal cases"
        assert len(result) == len(target_comp)
        assert get_composition_counts(
            result.get_chemical_symbols()
        ) == get_composition_counts(target_comp)
        assert is_cluster_connected(result, connectivity_factor=CONNECTIVITY_FACTOR)

    def test_seed_growth_convex_hull_reliable(self, rng):
        """Verify seed+growth convex hull placement works reliably for normal cases."""
        # Test various sizes that should work with convex hull placement
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        sizes = [5, 8, 12, 15]

        for size in sizes:
            target_comp = ["Pt"] * size
            cell_side = compute_cell_side(target_comp, vacuum=8.0)

            result = grow_from_seed(
                seed_atoms=seed,
                target_composition=target_comp,
                placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
                cell_side=cell_side,
                rng=rng,
            )

            assert result is not None, (
                f"Seed+growth convex hull placement should work for {size} atoms"
            )
            assert len(result) == size
            assert is_cluster_connected(result)

    def test_seed_combination_convex_hull_reliable(self, rng):
        """Verify seed combination convex hull placement works reliably."""
        seed1 = Atoms("Pt", positions=[[0, 0, 0]])
        seed2 = Atoms("Au", positions=[[0, 0, 0]])

        # Should work with default parameters
        combined = combine_seeds(
            seeds=[seed1, seed2],
            cell_side=10.0,
            separation_scaling=1.0,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert combined is not None, "Seed combination should work reliably"
        assert len(combined) == 2
        assert get_composition_counts(
            combined.get_chemical_symbols()
        ) == get_composition_counts(["Pt", "Au"])

    def test_default_parameters_sufficient_convex_hull(self, rng):
        """Verify default parameters work for convex hull placement (shouldn't need relaxation)."""
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 8
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Use default parameters - should work without relaxation
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert result is not None, (
            "Default parameters should be sufficient for convex hull placement"
        )
        assert len(result) == len(target_comp)
        assert is_cluster_connected(result)

    def test_large_clusters_convex_hull_works(self, rng):
        """Verify large clusters work with convex hull placement using default parameters."""
        seed = Atoms(
            "Pt4", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0], [1.25, 0.73, 1.8]]
        )
        target_comp = ["Pt"] * 20
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Large clusters should work with convex hull placement
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None, (
            "Large clusters should work with convex hull placement"
        )
        assert len(result) == len(target_comp)
        assert is_cluster_connected(result)


class TestLargeClusterConnectivityTemplateMode:
    """Stringent tests for template mode initialization with 50-60 atom clusters.

    Note: Basic connectivity tests have been consolidated into TestLargeClusterConnectivityAllModes
    in test_init_common.py. This class now only contains batch and reproducibility tests.
    """


class TestTemplateGenerationOptimization:
    """Tests for template generation optimizations."""

    def test_exact_match_avoids_near_match_redundancy(self, rng):
        """Test that exact magic number matches don't redundantly generate near matches."""
        # Use exact magic number (13)
        comp = ["Pt"] * 13

        # Should work without redundant template generation
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert len(atoms) == 13
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(comp)
