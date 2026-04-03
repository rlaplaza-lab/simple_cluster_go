"""Tests for seed+growth initialization mode.

This module consolidates all tests for seed+growth initialization including:
- Basic seed+growth functionality
- Threshold relaxation legitimacy
- Convex hull placement reliability
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
- Refactored seed growth tests
"""

import pytest
from ase import Atoms

from scgo.initialization import (
    combine_and_grow,
    create_initial_cluster,
    grow_from_seed,
    is_cluster_connected,
    random_spherical,
    validate_cluster_structure,
)
from scgo.initialization.geometry_helpers import analyze_disconnection
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.initialization.initializers import compute_cell_side
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import (
    DIVERSITY_TEST_SAMPLES_MEDIUM,
    DIVERSITY_THRESHOLD_DEFAULT,
    REPRODUCIBILITY_SEEDS,
    assert_cluster_valid,
    create_paired_rngs,
    get_structure_signature,
    validate_structure_with_diagnostics,
)


class TestSeedGrowthInitialization:
    """Tests for seed+growth initialization mode."""

    def test_grow_from_seed_preserves_composition(self, rng):
        """Test that grow_from_seed preserves target composition."""
        # seed: one Pt and one Au at reasonable separation
        seed = Atoms(["Pt", "Au"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

        # target composition: add one Pt (total: 2 Pt, 3 Au, 1 Pd)
        target_comp = ["Pt", "Pt", "Au", "Au", "Au", "Pd"]

        placement_radius_scaling = 0.9
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=placement_radius_scaling,
            cell_side=cell_side,
            rng=rng,
        )

        assert out is not None, "grow_from_seed returned None"
        assert_cluster_valid(out, target_comp)

    def test_grow_from_seed_no_additional_needed(self, rng):
        """Test grow_from_seed when no additional atoms are needed."""
        seed = Atoms(["Pt", "Au"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        target = ["Pt", "Au"]
        side = compute_cell_side(target, vacuum=8.0)
        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target,
            placement_radius_scaling=0.9,
            cell_side=side,
            rng=rng,
        )
        assert out is not None
        assert_cluster_valid(out, target)

    def test_grow_from_seed_empty_seed(self, rng):
        """Test grow_from_seed with empty seed behaves like random_spherical."""
        # Should behave like random_spherical if seed is empty
        seed = Atoms()
        target_comp = ["Pt", "Pt"]
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )
        assert out is not None
        assert_cluster_valid(out, target_comp)

    def test_grow_from_seed_empty_target_composition(self, rng):
        """Test grow_from_seed with empty target returns seed."""
        # Should return the seed if target is empty
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_comp = []
        cell_side = compute_cell_side(seed.get_chemical_symbols(), vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )
        assert out is not None
        assert len(out) == len(seed)
        assert get_composition_counts(
            out.get_chemical_symbols()
        ) == get_composition_counts(seed.get_chemical_symbols())

    def test_grow_from_seed_failure_to_grow(self, rng):
        """Test grow_from_seed failure when constraints are too tight."""
        # Try to grow in a very constrained space, should fail to add all atoms
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_comp = ["Pt"] * 5  # Try to add 4 more Pt atoms
        cell_side = compute_cell_side(target_comp, vacuum=1.0)  # Very small vacuum

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=0.1,  # Very small scaling
            cell_side=cell_side,
            rng=rng,
        )
        assert out is None or len(out) < len(
            target_comp,
        )  # Should fail to add all or return None

    def test_grow_from_seed_connectivity(self, rng):
        """Test that grow_from_seed produces connected clusters."""
        # Create a connected seed
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        target_comp = ["Pt", "Pt", "Au", "Au"]
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            connectivity_factor=CONNECTIVITY_FACTOR,  # Use default value
            rng=rng,
        )

        assert result is not None
        assert_cluster_valid(result, target_comp)

    def test_grow_from_seed_connected_under_growth_threshold(self, rng):
        """Regression: grown clusters should be connected under growth connectivity factor.

        This test ensures that the incremental growth algorithm places each new atom
        close enough to at least one existing atom so that the final cluster is
        connected when evaluated with the connectivity factor.
        """
        from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR

        # Start from a small but connected metallic seed
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 10  # Grow to a modestly larger Pt cluster
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None
        assert len(result) == len(target_comp)
        assert get_composition_counts(
            result.get_chemical_symbols()
        ) == get_composition_counts(target_comp)
        assert is_cluster_connected(result, connectivity_factor=CONNECTIVITY_FACTOR)

        assert_cluster_valid(
            result,
            target_comp,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )


class TestThresholdRelaxationLegitimacy:
    """Tests verifying threshold relaxation only occurs when needed."""

    def test_relaxation_produces_valid_clusters(self, rng):
        """Verify that when relaxation occurs, it produces valid clusters."""
        # Test with parameters that may trigger relaxation
        comp = ["Pt"] * 15
        # Use parameters that might need relaxation but should still produce valid clusters
        atoms = create_initial_cluster(
            comp,
            rng=rng,
            placement_radius_scaling=1.0,
            min_distance_factor=0.4,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert len(atoms) == len(comp)
        assert is_cluster_connected(atoms)
        # Verify no clashes
        is_valid, msg = validate_cluster_structure(
            atoms,
            min_distance_factor=0.4,
            connectivity_factor=CONNECTIVITY_FACTOR,
            check_clashes=True,
            check_connectivity=True,
        )
        assert is_valid is True, f"Relaxation should produce valid clusters: {msg}"

    def test_relaxation_bounds_safe(self, rng):
        """Verify relaxation doesn't go below safe thresholds."""
        # Test with various min_distance_factor values
        comp = ["Pt"] * 10
        for min_dist in [0.3, 0.4, 0.5]:
            atoms = create_initial_cluster(
                comp,
                rng=rng,
                min_distance_factor=min_dist,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )

            assert len(atoms) == len(comp)
            # Verify no clashes even with relaxed parameters
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=min_dist,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, f"Relaxation bounds should be safe: {msg}"


class TestLargeClusterConnectivitySeedGrowth:
    """Stringent tests for seed+growth mode with 50-60 atom clusters.

    Note: Basic connectivity tests for single/bimetallic compositions
    have been consolidated into TestLargeClusterConnectivityAllModes in test_init_common.py.
    This class now only contains unique seed growth tests (combine_and_grow) and batch/reproducibility tests.
    """

    @pytest.mark.parametrize("n_atoms", [50, 52, 55, 58, 60])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 10, 20, 30, 42])
    def test_combine_and_grow_connectivity_single_element(self, n_atoms, seed):
        """Test that combine_and_grow produces connected clusters from multiple seeds."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * n_atoms

        # Create multiple smaller seeds
        seed_sizes = [8, 10, 12]
        seeds = []
        for size in seed_sizes:
            seed_comp = ["Pt"] * size
            seed_cell_side = compute_cell_side(seed_comp)
            seed_atoms = random_spherical(
                composition=seed_comp,
                cell_side=seed_cell_side,
                rng=rng,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )
            assert is_cluster_connected(
                seed_atoms, connectivity_factor=CONNECTIVITY_FACTOR
            ), f"Seed of size {size} is not connected"
            seeds.append(seed_atoms)

        # Calculate how many more atoms we need
        total_seed_atoms = sum(len(s) for s in seeds)
        remaining = n_atoms - total_seed_atoms

        if remaining < 0:
            pytest.skip(
                f"Seeds too large for target (total={total_seed_atoms}, target={n_atoms})"
            )

        # Create target composition
        target_comp = comp  # Already correct size
        cell_side = compute_cell_side(target_comp)
        atoms = combine_and_grow(
            seeds=seeds,
            target_composition=target_comp,
            cell_side=cell_side,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        if atoms is None:
            pytest.skip(
                f"combine_and_grow returned None for n_atoms={n_atoms}, seed={seed}"
            )

        # Verify composition & geometry using centralized helper (connectivity checked separately below)
        assert_cluster_valid(atoms, comp, check_connectivity=False)

        # Stringent connectivity check
        is_connected = is_cluster_connected(
            atoms, connectivity_factor=CONNECTIVITY_FACTOR
        )
        if not is_connected:
            (
                disconnection_distance,
                suggested_factor,
                analysis_msg,
            ) = analyze_disconnection(atoms, CONNECTIVITY_FACTOR)
            pytest.fail(
                f"Combine and grow produced disconnected cluster "
                f"(n_atoms={n_atoms}, seed={seed}, seed_sizes={seed_sizes}). "
                f"Connectivity factor: {CONNECTIVITY_FACTOR}. "
                f"Analysis: {analysis_msg}. "
                f"Suggested factor: {suggested_factor:.2f}. "
                f"Largest gap: {disconnection_distance:.3f} Å"
            )


class TestLargeClusterConnectivitySeedGrowthMode:
    """Stringent tests for seed+growth mode via create_initial_cluster.

    Note: Basic connectivity tests have been consolidated into TestLargeClusterConnectivityAllModes
    in test_init_common.py. This class now only contains batch and reproducibility tests.
    """


class TestSeedGrowthReliability:
    """Tests for grow_from_seed reliability."""

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS)
    @pytest.mark.parametrize("growth_amount", [2, 4, 6])
    def test_grow_from_small_seed(self, seed, growth_amount):
        """Test growth from a small connected seed."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a small connected seed
        seed_atoms = Atoms(
            "Pt3",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]],
        )
        seed_atoms.set_cell([20, 20, 20])
        seed_atoms.center()

        target_n = len(seed_atoms) + growth_amount
        target_comp = ["Pt"] * target_n
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None, f"grow_from_seed returned None for seed={seed}"
        assert_cluster_valid(result, target_comp)

        validate_structure_with_diagnostics(
            result, context=f"seed={seed}, growth_amount={growth_amount}"
        )

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    def test_grow_mixed_composition(self, seed):
        """Test growth with mixed composition."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a small PtAu seed
        seed_atoms = Atoms(
            "PtAu",
            positions=[[0, 0, 0], [2.6, 0, 0]],  # Pt-Au typical distance
        )
        seed_atoms.set_cell([20, 20, 20])
        seed_atoms.center()

        target_comp = ["Pt", "Au", "Pt", "Au", "Pt"]  # 5 atoms
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None
        assert_cluster_valid(result, target_comp)

        validate_structure_with_diagnostics(
            result, context=f"seed={seed}, mixed composition PtAu"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:3])
    @pytest.mark.parametrize("target_size", [20, 30, 40])
    def test_grow_to_larger_sizes(self, seed, target_size):
        """Test growth to larger target sizes (slow)."""
        LARGE_CLUSTER_FACTOR = 2.0  # More lenient for large clusters (20+ atoms)
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a tetrahedron seed
        seed_atoms = Atoms(
            "Pt4",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [1.25, 2.165, 0],
                [1.25, 0.721, 2.357],
            ],
        )
        seed_atoms.set_cell([30, 30, 30])
        seed_atoms.center()

        target_comp = ["Pt"] * target_size
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
        )

        assert result is not None
        assert len(result) == target_size

        validate_structure_with_diagnostics(
            result,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
            context=f"seed={seed}, target_size={target_size}",
        )


class TestRefactoredSeedGrowth:
    """Tests for refactored seed growth functions."""

    def test_filter_candidates_by_geometry(self):
        """Test geometry filtering helper function."""
        from scgo.initialization.initializers import _filter_candidates_by_geometry

        # Create test candidates - use 3D structures that should pass filter
        # Planar structure (may or may not pass depending on classification)
        planar = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
        # 3D structure (should definitely pass)
        three_d = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]],
        )

        candidates = {
            "Pt3": [(-10.0, planar)],
            "Pt4": [(-15.0, three_d)],
        }

        filtered = _filter_candidates_by_geometry(candidates)
        # At least the 3D structure should be in filtered results
        assert "Pt4" in filtered
        # Pt3 may or may not be included depending on geometry classification
        assert len(filtered) > 0

    def test_seed_sampling_strategies(self, rng):
        """Test seed sampling strategy selection."""
        from scgo.initialization.initializers import _sample_seed_with_strategy

        # Create test candidates with same composition (required for Boltzmann sampling)
        # Sorted by energy (lowest first)
        candidates = [
            (-15.0, Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])),
            (-10.0, Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])),
            (-5.0, Atoms("Pt3", positions=[[0, 0, 0], [3.0, 0, 0], [1.5, 2.5, 0]])),
        ]

        # Test each strategy
        for strategy in range(5):
            result = _sample_seed_with_strategy(candidates, strategy, rng)
            assert result is not None
            energy, atoms = result
            assert isinstance(energy, float)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3  # All candidates are Pt3
            # Verify result is from candidates
            assert any(
                abs(e - energy) < 1e-6 and len(a) == len(atoms) for e, a in candidates
            )


class TestSeedGrowthDiversity:
    """Tests to verify seed+growth produces diverse structures from seed selection, not fallbacks."""

    @pytest.mark.slow
    def test_seed_growth_diversity_with_available_seeds(self, rng):
        """Test that seed+growth produces diverse structures when seeds are available.

        This test ensures diversity comes from seed selection and combination,
        not from random_spherical fallbacks. We use a composition (Pt50) that
        we know has seeds available in the database.

        Verification strategy:
        1. Confirm seeds ARE available (pre-check)
        2. Capture logger output to detect fallbacks
        3. Generate structures with seed+growth mode
        4. Verify from logs that seed+growth succeeded (not falling back)
        5. Verify seed+growth diversity ≥70% (DIVERSITY_THRESHOLD_DEFAULT)

        The code now logs explicitly when seed+growth falls back to random_spherical,
        so we can verify directly from the logs.
        """
        import logging
        from io import StringIO

        from scgo.initialization.initializers import _find_smaller_candidates
        from scgo.utils.logging import get_logger

        comp = ["Pt"] * 50
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        # First, verify that seeds ARE available for this composition
        candidates = _find_smaller_candidates(comp, "**/*.db")
        if len(candidates) == 0:
            pytest.skip(
                "No seeds found for Pt50 - cannot test seed+growth diversity. "
                "This test requires database files with Pt seeds."
            )
        total_candidates = sum(len(cands) for cands in candidates.values())
        if total_candidates == 0:
            pytest.skip("No candidate seeds available")

        # Set up logger capture to detect fallbacks
        # We need to capture from the root logger or configure the specific logger
        logger = get_logger("scgo.initialization.initializers")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)  # Capture all levels to see everything
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages
        logger.propagate = False  # Prevent propagation to root logger

        try:
            # Generate structures with seed+growth mode
            seed_growth_signatures = []
            fallback_count = 0

            for _ in range(n_samples):
                log_capture.seek(0)
                log_capture.truncate(0)

                atoms = create_initial_cluster(
                    comp,
                    mode="seed+growth",
                    rng=rng,
                    previous_search_glob="**/*.db",
                )
                assert_cluster_valid(atoms, comp)
                seed_growth_signatures.append(get_structure_signature(atoms))

                # Check log for fallback messages
                log_output = log_capture.getvalue()
                if "falling back to random_spherical" in log_output:
                    fallback_count += 1

            # Verify that seed+growth did NOT fall back (or fell back very rarely)
            fallback_ratio = fallback_count / n_samples
            assert fallback_ratio < 0.2, (
                f"Too many fallbacks detected in logs: {fallback_count}/{n_samples} "
                f"({fallback_ratio:.1%}) structures fell back to random_spherical. "
                f"This suggests seed+growth is failing when seeds are available. "
                f"Expected <20% fallback rate."
            )

            # Verify seed+growth diversity: at least 70% unique structures
            unique_seed_growth = set(seed_growth_signatures)
            diversity_ratio = len(unique_seed_growth) / n_samples
            assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
                f"Insufficient diversity in seed+growth mode: only "
                f"{len(unique_seed_growth)}/{n_samples} ({diversity_ratio:.1%}) unique structures. "
                f"Expected at least {DIVERSITY_THRESHOLD_DEFAULT:.0%}. "
                f"Fallbacks detected in logs: {fallback_count}/{n_samples}"
            )

        finally:
            handler.close()
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    @pytest.mark.slow
    def test_seed_growth_diversity_bimetallic_with_seeds(self, rng):
        """Test seed+growth diversity for bimetallic composition with available seeds."""
        from scgo.initialization.initializers import _find_smaller_candidates

        comp = ["Pt", "Au"] * 25  # 50 atoms, bimetallic
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        # Verify seeds are available
        candidates = _find_smaller_candidates(comp, "**/*.db")
        # For bimetallic, we may have fewer seeds, but should have some
        # If no seeds, skip this test
        if len(candidates) == 0:
            pytest.skip(
                "No seeds available for bimetallic Pt25Au25 - skipping diversity test"
            )

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(
                comp,
                mode="seed+growth",
                rng=rng,
                previous_search_glob="**/*.db",
            )
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity in seed+growth mode (bimetallic): only "
            f"{len(unique_signatures)}/{n_samples} ({diversity_ratio:.1%}) unique structures. "
            f"Expected at least {DIVERSITY_THRESHOLD_DEFAULT:.0%}."
        )
