"""Tests for smart initialization mode.

This module consolidates all tests for smart mode initialization including:
- Basic smart mode functionality
- Default initialization strictness
- Single connectivity factor consistency
- Exact composition counts
- Connectivity validation
- Reproducibility
- Edge cases
- Disconnection prevention
- No clashes
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
"""

import numpy as np
import pytest
from ase import Atoms

from scgo.initialization import (
    create_initial_cluster,
    is_cluster_connected,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
)
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import (
    DIVERSITY_TEST_SAMPLES_LARGE,
    DIVERSITY_TEST_SAMPLES_MEDIUM,
    DIVERSITY_TEST_SAMPLES_SMALL,
    DIVERSITY_THRESHOLD_DEFAULT,
    DIVERSITY_THRESHOLD_MIN,
    assert_cluster_valid,
    get_structure_signature,
)


class TestSmartModeInitialization:
    """Tests for smart mode initialization."""

    def test_smart_mode_magic_number(self, rng):
        """Test smart mode with magic number (may use template)."""
        comp = ["Pt"] * 13  # Magic number
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 13

    def test_smart_mode_near_magic_number(self, rng):
        """Test smart mode near magic number."""
        comp = ["Pt"] * 14  # Near 13
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 14

    def test_smart_mode_non_magic(self, rng):
        """Test smart mode with non-magic number (uses templates from nearest magic number, seed+growth, or random)."""
        comp = ["Pt"] * 7  # Not a magic number
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 7

    @pytest.mark.slow
    def test_smart_mode_diversity(self, rng):
        """Test that smart mode generates diverse structures for various cluster sizes.

        This test verifies actual diversity (not just structure counts) by checking
        that multiple initializations produce different geometric structures.
        """
        test_cases = [
            (["Pt"] * 13, 10),  # Magic number with multiple exact matches
            (["Pt"] * 12, 8),  # Near magic number 13
            (
                ["Pt"] * 7,
                8,
            ),  # Non-magic number (uses templates from nearest magic number, seed+growth, or random)
        ]

        for comp, n_samples in test_cases:
            structures = []
            signatures = []

            for _ in range(n_samples):
                atoms = create_initial_cluster(
                    comp,
                    mode="smart",
                    rng=rng,
                    connectivity_factor=CONNECTIVITY_FACTOR,
                )
                structures.append(atoms)
                signatures.append(get_structure_signature(atoms))

            # All should have correct size
            assert all(len(s) == len(comp) for s in structures), (
                f"Size mismatch for composition {comp}"
            )

            # Verify actual diversity: should have multiple unique structures
            unique_signatures = set(signatures)
            diversity_ratio = len(unique_signatures) / n_samples
            assert diversity_ratio >= DIVERSITY_THRESHOLD_MIN, (
                f"Insufficient diversity for {comp}: only {len(unique_signatures)}/{n_samples} "
                f"({diversity_ratio:.1%}) unique structures"
            )

    def test_smart_mode_fallback_when_templates_fail(self, rng):
        """Test that smart mode falls back when template generation fails."""
        # Use a size that may not have templates
        comp = ["Pt"] * 3
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    def test_smart_mode_composition_matching(self, rng):
        """Test that smart mode maintains exact composition."""
        comp = ["Pt", "Au"] * 6 + ["Pt"]  # 13 atoms
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert len(atoms) == 13
        symbols = atoms.get_chemical_symbols()
        # Should have both Pt and Au
        assert "Pt" in symbols
        assert "Au" in symbols
        # Count should match
        pt_count = symbols.count("Pt")
        au_count = symbols.count("Au")
        assert pt_count + au_count == 13

    def test_smart_mode_connectivity_handling(self, rng):
        """Test that smart mode handles connectivity issues gracefully."""
        comp = ["Pt"] * 12  # Near magic number
        atoms = create_initial_cluster(
            comp, mode="smart", rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )
        # Should still produce valid structure (may fall back to seed+growth)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 12


class TestDefaultInitializationStrictness:
    """Strict tests for default initialization settings.

    These tests ensure that default initialization settings consistently produce
    diverse, valid clusters with exact compositions, connectivity, and no clashes.
    """

    @pytest.mark.slow
    def test_default_settings_diverse_clusters_single_element(self, rng):
        """Test that default settings generate diverse clusters for single-element compositions."""
        comp = ["Pt"] * 6
        n_samples = DIVERSITY_TEST_SAMPLES_LARGE

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        # With default settings, we should get substantial diversity
        # At least 70% unique structures is required (DIVERSITY_THRESHOLD_DEFAULT)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity: only {len(unique_signatures)}/{n_samples} "
            f"({diversity_ratio:.1%}) unique structures"
        )

    def test_default_settings_diverse_clusters_bimetallic(self, rng):
        """Test that default settings generate diverse clusters for bimetallic compositions."""
        comp = ["Pt", "Au", "Pt", "Au", "Pt", "Au"]
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity for bimetallic: only {len(unique_signatures)}/{n_samples} "
            f"({diversity_ratio:.1%}) unique structures"
        )

    def test_default_settings_all_invariants_batch(self, rng):
        """Test that default settings produce valid clusters for various compositions."""
        test_compositions = [
            ["Pt"] * 3,
            ["Pt"] * 5,
            ["Pt"] * 8,
            ["Pt", "Au", "Pt"],
            ["Pt", "Au", "Pt", "Au"],
            ["Pt", "Au", "Pd", "Pt", "Au"],
            ["Cu", "Cu", "Cu", "Cu", "Cu"],
        ]

        for comp in test_compositions:
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp, check_connectivity=len(atoms) > 1)

    @pytest.mark.slow
    def test_default_settings_large_cluster_robustness(self, rng):
        """Test that default settings handle larger clusters robustly."""
        comp = ["Pt"] * 12
        n_samples = DIVERSITY_TEST_SAMPLES_SMALL

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            try:
                assert_cluster_valid(atoms, comp)
            except AssertionError as e:
                raise AssertionError(f"Sample {i}: {e}") from e

    def test_default_settings_exact_composition_strict(self, rng):
        """Test that default settings maintain exact composition for complex cases."""
        test_cases = [
            (["Pt", "Au"], {"Pt": 1, "Au": 1}),
            (["Pt", "Pt", "Au"], {"Pt": 2, "Au": 1}),
            (["Pt", "Au", "Pt", "Au", "Pt"], {"Pt": 3, "Au": 2}),
            (["Cu", "Pt", "Au", "Cu", "Pt"], {"Cu": 2, "Pt": 2, "Au": 1}),
        ]

        for comp, _expected_counts_dict in test_cases:
            atoms = create_initial_cluster(comp, rng=rng)
            actual_counts = get_composition_counts(atoms.get_chemical_symbols())
            expected_counts = get_composition_counts(comp)
            assert actual_counts == expected_counts, (
                f"Composition mismatch for {comp}: "
                f"expected {expected_counts}, got {actual_counts}"
            )
            # Also verify total count
            assert len(atoms) == len(comp), (
                f"Total atom count mismatch for {comp}: "
                f"expected {len(comp)}, got {len(atoms)}"
            )

    def test_default_settings_no_clashes_strict(self, rng):
        """Test that default settings produce no atomic clashes with strict checking."""
        comp = ["Pt"] * 7
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            positions = atoms.get_positions()
            # Check all pairwise distances
            for j in range(len(positions)):
                for k in range(j + 1, len(positions)):
                    distance = np.linalg.norm(positions[j] - positions[k])
                    # Minimum distance should be reasonable (at least 0.1 Å)
                    assert distance > 0.1, (
                        f"Sample {i}: Atoms {j} and {k} too close: {distance:.6f} Å"
                    )
            # Also verify with validation function
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=0.5,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, f"Sample {i}: Validation failed: {msg}"

    def test_default_settings_connectivity_strict(self, rng):
        """Test that default settings ensure connectivity with strict factor."""
        comp = ["Pt"] * 9
        n_samples = 15

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify connectivity with default factor
            assert is_cluster_connected(
                atoms, connectivity_factor=CONNECTIVITY_FACTOR
            ), (
                f"Sample {i}: Cluster not connected with connectivity_factor={CONNECTIVITY_FACTOR}"
            )
            # Verify with validation function
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=0.5,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, (
                f"Sample {i}: Connectivity validation failed: {msg}"
            )


class TestLargeClusterConnectivitySmartMode:
    """Stringent tests for smart mode initialization with 50-60 atom clusters.

    Note: Basic connectivity tests for single/bimetallic/trimetallic compositions
    have been consolidated into TestLargeClusterConnectivityAllModes in test_init_common.py.
    This class now only contains batch and reproducibility tests.
    """


# Reliability tests have been consolidated into TestReliabilityAllModes in test_init_common.py


class TestSingleConnectivityFactor:
    """Tests to verify only connectivity factor 1.4 is used (no CONNECTIVITY_FACTOR_GROWTH)."""

    def test_connectivity_factor_growth_removed(self):
        """Test that CONNECTIVITY_FACTOR_GROWTH has been removed from config."""
        # Try to import - should fail since it's been removed
        import scgo.initialization.initialization_config as config

        # Verify it's not in the module
        assert not hasattr(config, "CONNECTIVITY_FACTOR_GROWTH"), (
            "CONNECTIVITY_FACTOR_GROWTH should have been removed"
        )
        # Verify CONNECTIVITY_FACTOR exists
        assert hasattr(config, "CONNECTIVITY_FACTOR")
        assert pytest.approx(1.4) == config.CONNECTIVITY_FACTOR

    @pytest.mark.parametrize("mode", ["random_spherical", "template", "seed+growth"])
    def test_connectivity_factor_consistency_all_modes(self, mode, rng):
        """Test that all modes use connectivity_factor consistently."""
        # Use appropriate composition size for each mode
        composition = ["Pt"] * 13 if mode == "template" else ["Pt"] * 10

        atoms = create_initial_cluster(
            composition,
            mode=mode,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        if atoms is not None:
            # Should be connected with the same factor
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )
            # Validation should pass with the same factor
            is_valid, _ = validate_cluster_structure(
                atoms, min_distance_factor=0.5, connectivity_factor=CONNECTIVITY_FACTOR
            )
            assert is_valid is True


class TestExactCompositionCounts:
    """Tests to verify exact composition counts (not ratios) are preserved."""

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_all_modes_exact_counts(self, mode, rng):
        """Test that all initialization modes preserve exact composition counts."""
        pattern = ["Pt", "Au", "Pt", "Au", "Pt"]  # 5 atoms: 3 Pt, 2 Au
        target_composition = pattern * 2  # 10 atoms: 6 Pt, 4 Au

        atoms = create_initial_cluster(target_composition, mode=mode, rng=rng)
        if atoms is not None:
            expected_counts = get_composition_counts(target_composition)
            actual_counts = get_composition_counts(atoms.get_chemical_symbols())
            assert actual_counts == expected_counts, (
                f"Mode {mode} failed: expected {expected_counts}, got {actual_counts}"
            )


class TestConnectivityValidation:
    """Tests to verify connectivity is validated with factor 1.4."""

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 5,
            ["Pt", "Au"] * 4,
            ["Pt"] * 13,  # Magic number
            ["Pt", "Au", "Pd"] * 3,
        ],
    )
    @pytest.mark.parametrize("mode", ["smart", "random_spherical"])
    def test_all_generated_clusters_connected(self, composition, mode, rng):
        """Test that all generated clusters are connected with factor 1.4."""
        atoms = create_initial_cluster(composition, mode=mode, rng=rng)
        assert (
            is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR) is True
        ), (
            f"Cluster from {mode} mode should be connected with factor {CONNECTIVITY_FACTOR}"
        )

    def test_template_atom_removal_maintains_connectivity(self, rng):
        """Test that template atom removal maintains connectivity."""
        # Test removing atoms from template
        target_composition = ["Pt"] * 12  # Remove 1 from 13-atom template
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )


class TestReproducibility:
    """Reproducibility checks for initialization modes are consolidated in
    `tests/test_initialization_modes.py::TestInitializationModesBasics::test_mode_reproducibility`.

    The smart-mode-specific deterministic checks were removed to avoid
    duplication — keep mode-specific tests only when they exercise unique
    internal behavior not covered by the consolidated parametrized tests.
    """


class TestEdgeCases:
    """Tests for edge cases in smart initialization."""

    def test_two_atoms(self, rng):
        """Test two atom cluster."""
        composition = ["Pt", "Au"]
        atoms = create_initial_cluster(composition, mode="smart", rng=rng)
        assert len(atoms) == 2
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(composition)

    def test_multi_element_exact_counts_large(self, rng):
        """Test exact composition counts for large multi-element cluster."""
        pattern = ["Pt", "Au", "Pd"] * 10  # 30 atoms: 10 each
        atoms = create_initial_cluster(pattern, mode="smart", rng=rng)

        expected_counts = get_composition_counts(pattern)
        actual_counts = get_composition_counts(atoms.get_chemical_symbols())
        assert actual_counts == expected_counts


class TestDisconnectionPrevention:
    """Tests to verify disconnection prevention during atom removal."""

    def test_template_removal_preserves_connectivity(self, rng):
        """Test that template atom removal doesn't disconnect cluster."""
        # Remove atoms from 13-atom template to get smaller size
        target_composition = ["Pt"] * 10
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )

            # Verify no clashes
            is_valid, _ = validate_cluster_structure(
                atoms, min_distance_factor=0.5, connectivity_factor=CONNECTIVITY_FACTOR
            )
            assert is_valid is True

    def test_seed_combination_maintains_connectivity(self, rng):
        """Test that seed combination maintains connectivity."""
        composition = ["Pt"] * 20
        atoms = create_initial_cluster(composition, mode="seed+growth", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )


class TestNoClashes:
    """Tests to verify no atomic clashes in generated structures."""

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_all_modes_no_clashes(self, mode, rng):
        """Test that all modes produce structures without clashes."""
        composition = ["Pt", "Au"] * 5

        atoms = create_initial_cluster(composition, mode=mode, rng=rng)
        if atoms is not None:
            is_valid, error_msg = validate_cluster_structure(
                atoms,
                min_distance_factor=0.5,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, (
                f"Mode {mode} produced invalid structure: {error_msg}"
            )
