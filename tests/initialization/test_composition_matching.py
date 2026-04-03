"""Tests to ensure exact composition matching in generated structures."""

from scgo.initialization import create_initial_cluster
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import assert_cluster_valid


class TestExactCompositionMatching:
    """Tests to verify that generated structures always match exact composition."""

    def test_single_element_exact_match(self, rng):
        """Test that single element composition matches exactly."""
        composition = ["Pt"] * 13
        atoms = create_initial_cluster(composition, rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_multi_element_exact_match(self, rng):
        """Test that multi-element composition matches exactly."""
        composition = ["Pt", "Au", "Pt", "Au", "Pt"]
        atoms = create_initial_cluster(composition, rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_template_mode_exact_match(self, rng):
        """Test that template mode produces exact composition."""
        composition = ["Pt"] * 13
        atoms = create_initial_cluster(composition, mode="template", rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_smart_mode_exact_match(self, rng):
        """Test that smart mode produces exact composition."""
        composition = ["Pt", "Au"] * 5 + ["Pt"]  # 11 atoms: 6 Pt, 5 Au
        atoms = create_initial_cluster(composition, mode="smart", rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_random_spherical_exact_match(self, rng):
        """Test that random_spherical mode produces exact composition."""
        composition = ["Pt", "Au", "Pt"]
        atoms = create_initial_cluster(composition, mode="random_spherical", rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_seed_growth_exact_match(self, rng):
        """Test that seed+growth mode produces exact composition."""
        composition = ["Pt"] * 10
        atoms = create_initial_cluster(composition, mode="seed+growth", rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_complex_composition_exact_match(self, rng):
        """Test complex multi-element composition."""
        composition = ["Pt", "Au", "Pd", "Pt", "Au", "Pd", "Pt"]
        atoms = create_initial_cluster(composition, rng=rng)
        assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_all_magic_numbers_exact_match(self, rng):
        """Test that all magic numbers produce exact composition."""
        from scgo.initialization.initialization_config import MAGIC_NUMBERS

        for magic in MAGIC_NUMBERS[:10]:  # Test first 10 to avoid long test
            if magic > 50:  # Skip very large ones for speed
                continue
            composition = ["Pt"] * magic
            atoms = create_initial_cluster(
                composition,
                mode="template",
                rng=rng,
                connectivity_factor=CONNECTIVITY_FACTOR,
                placement_radius_scaling=1.3,
            )
            if atoms is not None:
                assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_multi_element_magic_number(self, rng):
        """Test multi-element composition at magic number."""
        composition = ["Pt", "Au"] * 6 + ["Pt"]  # 13 atoms: 7 Pt, 6 Au
        atoms = create_initial_cluster(composition, mode="template", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, composition, check_connectivity=False)

    def test_template_composition_after_atom_removal(self, rng):
        """Test that templates preserve exact composition counts after atom removal."""
        target_composition = ["Pt", "Au"] * 6  # 12 atoms: 6 Pt, 6 Au
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, target_composition, check_connectivity=False)

    def test_template_composition_after_atom_addition(self, rng):
        """Test that templates preserve exact composition counts after atom addition."""
        target_composition = ["Pt", "Au"] * 7 + ["Pt"]  # 15 atoms: 8 Pt, 7 Au
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, target_composition, check_connectivity=False)

    def test_seed_growth_composition_after_cycling(self, rng):
        """Test that seed+growth preserves exact composition when composition is cycled."""
        composition_pattern = ["Pt", "Au", "Pt"]
        target_composition = composition_pattern * 4  # 12 atoms: 8 Pt, 4 Au
        atoms = create_initial_cluster(target_composition, mode="seed+growth", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, target_composition, check_connectivity=False)


class TestCompositionCountAccuracy:
    """Tests to verify exact composition counts (not ratios) are preserved."""

    def test_cycling_preserves_exact_counts(self, rng):
        """Test that composition cycling produces exact counts, not approximate ratios."""
        pattern = ["Pt", "Pt", "Au"]
        target_composition = pattern * 3  # 9 atoms: 6 Pt, 3 Au
        atoms = create_initial_cluster(target_composition, rng=rng)
        assert_cluster_valid(atoms, target_composition, check_connectivity=False)
        actual_counts = get_composition_counts(atoms.get_chemical_symbols())
        assert actual_counts["Pt"] == 6
        assert actual_counts["Au"] == 3

    def test_template_removal_preserves_exact_counts(self, rng):
        """Test that template atom removal preserves exact composition counts."""
        target_pattern = ["Pt", "Au"] * 6  # 12 atoms: 6 Pt, 6 Au
        atoms = create_initial_cluster(target_pattern, mode="template", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, target_pattern, check_connectivity=False)

    def test_template_addition_preserves_exact_counts(self, rng):
        """Test that template atom addition preserves exact composition counts."""
        target_composition = ["Pt", "Au", "Pt"] * 5  # 15 atoms: 10 Pt, 5 Au
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert_cluster_valid(atoms, target_composition, check_connectivity=False)
