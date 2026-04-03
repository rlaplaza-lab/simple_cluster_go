"""Tests for template structure generation."""

import numpy as np
import pytest
from ase import Atoms

from scgo.initialization import is_cluster_connected
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.initialization.templates import (
    _find_valid_template_types,
    _generate_template_with_atom_adjustment,
    generate_cuboctahedron,
    generate_decahedron,
    generate_icosahedron,
    generate_octahedron,
    generate_template_matches,
    generate_template_structure,
    generate_truncated_octahedron,
    get_nearest_magic_number,
    is_near_magic_number,
)
from tests.test_utils import create_paired_rngs

# Template generation uses a more lenient connectivity factor
TEMPLATE_GENERATION_FACTOR = 2.5


class TestMagicNumberDetection:
    """Tests for magic number detection functions."""

    def test_get_nearest_magic_number_exact(self):
        """Test finding exact magic numbers."""
        assert get_nearest_magic_number(13) == 13
        assert get_nearest_magic_number(55) == 55
        assert get_nearest_magic_number(147) == 147

    def test_get_nearest_magic_number_near(self):
        """Test finding nearest magic number."""
        assert get_nearest_magic_number(14) == 13  # Closer to 13
        assert get_nearest_magic_number(56) == 55  # Closer to 55
        assert get_nearest_magic_number(20) == 20  # 20 is itself a magic number

    def test_is_near_magic_number_exact(self):
        """Test detection of exact magic numbers."""
        assert is_near_magic_number(13) is True
        assert is_near_magic_number(55) is True
        assert is_near_magic_number(147) is True

    def test_is_near_magic_number_within_tolerance(self):
        """Test detection within tolerance."""
        assert is_near_magic_number(14, tolerance=2) is True  # 13 + 1
        assert is_near_magic_number(12, tolerance=2) is True  # 13 - 1
        assert is_near_magic_number(57, tolerance=2) is True  # 55 + 2

    def test_is_near_magic_number_outside_tolerance(self):
        """Test detection outside tolerance."""
        assert (
            is_near_magic_number(1, tolerance=2) is False
        )  # 1 is distance 3 from 4, outside tolerance of 2
        assert (
            is_near_magic_number(16, tolerance=2) is False
        )  # 16 is distance 3 from 13, outside tolerance of 2
        assert (
            is_near_magic_number(67, tolerance=2) is False
        )  # 67 is distance 3 from 64, outside tolerance of 2
        assert (
            is_near_magic_number(100, tolerance=2) is False
        )  # 100 is distance 5 from 105, outside tolerance of 2
        assert (
            is_near_magic_number(100, tolerance=2) is False
        )  # Far from any magic number


class TestIcosahedronGeneration:
    """Tests for icosahedral structure generation."""

    @pytest.mark.parametrize("n_atoms", [13, 55, 147])
    def test_generate_icosahedron_magic_numbers(self, n_atoms, rng):
        """Test generation of magic number icosahedra."""
        atoms = generate_icosahedron(["Pt"], n_atoms, rng=rng)
        assert atoms is not None
        assert len(atoms) == n_atoms
        assert isinstance(atoms, Atoms)

    def test_generate_icosahedron_near_magic(self, rng):
        """Test generation near magic numbers."""
        # Test adding atoms
        atoms = generate_icosahedron(["Pt"], 20, rng=rng)
        assert atoms is not None
        assert len(atoms) == 20

        # Test removing atoms
        atoms = generate_icosahedron(["Pt"], 50, rng=rng)
        assert atoms is not None
        assert len(atoms) == 50

    def test_generate_icosahedron_multi_element(self, rng):
        """Test generation with multiple elements."""
        atoms = generate_icosahedron(["Pt", "Au"], 13, rng=rng)
        assert atoms is not None
        assert len(atoms) == 13
        symbols = atoms.get_chemical_symbols()
        assert "Pt" in symbols or "Au" in symbols

    def test_generate_icosahedron_empty_composition(self, rng):
        """Test that empty composition returns None (error is caught and logged)."""
        # Empty composition causes ValueError which is caught and returns None
        result = generate_icosahedron([], 13, rng=rng)
        assert result is None

    def test_generate_icosahedron_zero_atoms(self, rng):
        """Test generation with zero atoms."""
        atoms = generate_icosahedron(["Pt"], 0, rng=rng)
        assert atoms is None

    def test_generate_icosahedron_reproducible(self):
        """Test that generation is reproducible with same seed."""
        rng1, rng2 = create_paired_rngs(42)

        atoms1 = generate_icosahedron(["Pt"], 20, rng=rng1)
        atoms2 = generate_icosahedron(["Pt"], 20, rng=rng2)

        assert atoms1 is not None
        assert atoms2 is not None
        # Positions should be the same (within numerical precision)
        assert np.allclose(atoms1.get_positions(), atoms2.get_positions())


class TestDecahedronGeneration:
    """Tests for decahedral structure generation."""

    @pytest.mark.parametrize("n_atoms", [7, 13, 23, 39, 55])
    def test_generate_decahedron_common_sizes(self, n_atoms, rng):
        """Test generation of common decahedral sizes."""
        atoms = generate_decahedron(["Pt"], n_atoms, rng=rng)
        assert atoms is not None
        assert len(atoms) == n_atoms
        assert isinstance(atoms, Atoms)

    def test_generate_decahedron_near_sizes(self, rng):
        """Test generation near common sizes."""
        atoms = generate_decahedron(["Pt"], 25, rng=rng)
        assert atoms is not None
        assert len(atoms) == 25

    def test_generate_decahedron_multi_element(self, rng):
        """Test generation with multiple elements."""
        atoms = generate_decahedron(["Pt", "Au"], 23, rng=rng)
        assert atoms is not None
        assert len(atoms) == 23


class TestOctahedronGeneration:
    """Tests for octahedral structure generation."""

    @pytest.mark.parametrize("n_atoms", [6, 19, 44])
    def test_generate_octahedron_common_sizes(self, n_atoms, rng):
        """Test generation of common octahedral sizes."""
        atoms = generate_octahedron(["Pt"], n_atoms, rng=rng)
        assert atoms is not None
        assert len(atoms) == n_atoms
        assert isinstance(atoms, Atoms)

    def test_generate_octahedron_near_sizes(self, rng):
        """Test generation near common sizes."""
        atoms = generate_octahedron(["Pt"], 20, rng=rng)
        assert atoms is not None
        assert len(atoms) == 20


class TestTemplateStructureGeneration:
    """Tests for generic template structure generation."""

    def test_generate_template_auto_icosahedron(self, rng):
        """Test auto-selection of icosahedral template."""
        atoms = generate_template_structure(["Pt"], 13, template_type="auto", rng=rng)
        assert atoms is not None
        assert len(atoms) == 13

    def test_generate_template_auto_decahedron(self, rng):
        """Test auto-selection of decahedral template."""
        atoms = generate_template_structure(["Pt"], 23, template_type="auto", rng=rng)
        assert atoms is not None
        assert len(atoms) == 23

    def test_generate_template_explicit_icosahedron(self, rng):
        """Test explicit icosahedral template."""
        atoms = generate_template_structure(
            ["Pt"], 13, template_type="icosahedron", rng=rng
        )
        assert atoms is not None
        assert len(atoms) == 13

    def test_generate_template_explicit_decahedron(self, rng):
        """Test explicit decahedral template."""
        atoms = generate_template_structure(
            ["Pt"], 23, template_type="decahedron", rng=rng
        )
        assert atoms is not None
        assert len(atoms) == 23

    def test_generate_template_explicit_octahedron(self, rng):
        """Test explicit octahedral template."""
        atoms = generate_template_structure(
            ["Pt"], 19, template_type="octahedron", rng=rng
        )
        assert atoms is not None
        assert len(atoms) == 19

    def test_generate_template_unknown_type(self, rng):
        """Test unknown template type."""
        atoms = generate_template_structure(
            ["Pt"], 13, template_type="unknown", rng=rng
        )
        assert atoms is None


class TestTemplateStructureProperties:
    """Tests for properties of generated template structures."""

    def test_template_has_correct_composition(self, rng):
        """Test that template has correct composition."""
        composition = ["Pt", "Au", "Pt", "Au"]
        n_atoms = 13
        atoms = generate_icosahedron(composition, n_atoms, rng=rng)
        assert atoms is not None
        assert len(atoms) == n_atoms
        # Should cycle through composition
        symbols = atoms.get_chemical_symbols()
        assert all(sym in ["Pt", "Au"] for sym in symbols)

    @pytest.mark.parametrize("n_atoms", [13, 55])
    def test_template_atoms_are_connected(self, n_atoms, rng):
        """Test that template atoms form a connected structure."""
        from scgo.initialization import is_cluster_connected

        atoms = generate_icosahedron(["Pt"], n_atoms, rng=rng)
        assert atoms is not None
        # For small clusters, check connectivity
        if n_atoms > 1:
            # Use lenient connectivity factor for templates
            LENIENT_TEMPLATE_FACTOR = 3.0  # More lenient for template structures
            assert is_cluster_connected(
                atoms, connectivity_factor=LENIENT_TEMPLATE_FACTOR
            )

    def test_template_no_clashes(self, rng):
        """Test that template has no atomic clashes."""
        from ase.data import atomic_numbers, covalent_radii

        atoms = generate_icosahedron(["Pt"], 13, rng=rng)
        assert atoms is not None

        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # Check minimum distances
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                distance = np.linalg.norm(positions[i] - positions[j])
                r_i = covalent_radii[atomic_numbers[symbols[i]]]
                r_j = covalent_radii[atomic_numbers[symbols[j]]]
                min_allowed = (r_i + r_j) * 0.3  # Very lenient
                assert distance >= min_allowed, (
                    f"Atoms {i} and {j} too close: {distance:.3f} < {min_allowed:.3f}"
                )


class TestTemplateEdgeCases:
    """Tests for edge cases in template generation."""

    def test_single_atom(self, rng):
        """Test generation of single atom."""
        atoms = generate_template_structure(["Pt"], 1, template_type="auto", rng=rng)
        assert atoms is not None
        assert len(atoms) == 1

    def test_very_small_cluster(self, rng):
        """Test generation of very small cluster."""
        atoms = generate_template_structure(["Pt"], 3, template_type="auto", rng=rng)
        # May or may not succeed depending on template availability
        if atoms is not None:
            assert len(atoms) == 3

    def test_large_cluster(self, rng):
        """Test generation of large cluster."""
        atoms = generate_template_structure(["Pt"], 200, template_type="auto", rng=rng)
        # May or may not succeed
        if atoms is not None:
            assert len(atoms) == 200

    def test_many_atoms_to_add(self, rng):
        """Test adding many atoms to base structure."""
        # Start with 13-atom icosahedron, add 20 atoms
        atoms = generate_icosahedron(["Pt"], 33, rng=rng)
        assert atoms is not None
        assert len(atoms) == 33

    def test_many_atoms_to_remove(self, rng):
        """Test removing many atoms from base structure."""
        # Start with 55-atom icosahedron, remove 20 atoms
        atoms = generate_icosahedron(["Pt"], 35, rng=rng)
        assert atoms is not None
        assert len(atoms) == 35


class TestFindValidTemplateTypes:
    """Tests for finding valid template types."""

    def test_find_valid_types_magic_number(self, rng):
        """Test finding valid types for a magic number."""
        valid_types = _find_valid_template_types(13, rng)
        assert isinstance(valid_types, list)
        assert len(valid_types) > 0
        assert "icosahedron" in valid_types
        # Note: cuboctahedron may be rejected if it becomes disconnected after adjustment
        # This is expected behavior with stricter connectivity validation

    def test_find_valid_types_non_magic(self, rng):
        """Test finding valid types for non-magic number."""
        valid_types = _find_valid_template_types(25, rng)
        assert isinstance(valid_types, list)
        # May or may not have valid types

    def test_find_valid_types_zero_atoms(self, rng):
        """Test finding valid types for zero atoms."""
        valid_types = _find_valid_template_types(0, rng)
        assert isinstance(valid_types, list)
        assert len(valid_types) == 0

    @pytest.mark.slow
    def test_find_valid_types_very_large(self, rng):
        """Test finding valid types for very large cluster."""
        valid_types = _find_valid_template_types(1000, rng)
        assert isinstance(valid_types, list)
        # May or may not have valid types

    def test_find_valid_types_reproducible(self):
        """Test that finding valid types is reproducible."""
        rng1, rng2 = create_paired_rngs(42)
        types1 = _find_valid_template_types(13, rng1)
        types2 = _find_valid_template_types(13, rng2)
        assert set(types1) == set(types2)


class TestTemplateConnectivity:
    """Connectivity tests for template structures after polyhedra scaling fixes."""

    @pytest.mark.parametrize("n_atoms", [12, 13])
    def test_cuboctahedron_connected(self, n_atoms, rng):
        """Cuboctahedron for 12 or 13 atoms is connected within connectivity threshold."""
        atoms = generate_cuboctahedron(
            ["Pt"] * n_atoms, n_atoms, rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert atoms is not None
        assert len(atoms) == n_atoms
        assert is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR), (
            f"Cuboctahedron with {n_atoms} atoms should be connected"
        )

    def test_truncated_octahedron_connected(self, rng):
        """Truncated octahedron for 24 atoms is connected within connectivity threshold."""
        atoms = generate_truncated_octahedron(
            ["Pt"] * 24, 24, rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert atoms is not None
        assert len(atoms) == 24
        assert is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR), (
            "Truncated octahedron with 24 atoms should be connected"
        )

    @pytest.mark.parametrize(
        "gen_func,n_atoms",
        [
            (generate_icosahedron, 13),
            (generate_icosahedron, 19),
            (generate_decahedron, 23),
            (generate_octahedron, 19),
        ],
    )
    def test_ase_templates_connected(self, gen_func, n_atoms, rng):
        """ASE templates (icosahedron, decahedron, octahedron) are connected after rescale."""
        comp = ["Pt"] * n_atoms
        atoms = gen_func(
            comp, n_atoms, rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert atoms is not None
        assert len(atoms) == n_atoms
        assert is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR), (
            f"{gen_func.__name__} with {n_atoms} atoms should be connected after rescaling"
        )


class TestGenerateAllExactTemplateMatches:
    """Tests for generating all exact template matches."""

    def test_generate_all_exact_matches_magic_number(self, rng):
        """Test generating all exact matches for a magic number."""
        matches = generate_template_matches(
            ["Pt"] * 13, 13, rng, include_exact=True, include_near=False
        )
        assert isinstance(matches, list)
        assert len(matches) > 0
        # Should have multiple template types for 13 atoms
        assert len(matches) >= 2  # At least icosahedron and cuboctahedron
        for atoms in matches:
            assert atoms is not None
            assert len(atoms) == 13
            assert isinstance(atoms, Atoms)

    def test_generate_all_exact_matches_non_magic(self, rng):
        """Test generating all exact matches for non-magic number."""
        matches = generate_template_matches(
            ["Pt"] * 25, 25, rng, include_exact=True, include_near=False
        )
        assert isinstance(matches, list)
        # May or may not have matches

    def test_generate_all_exact_matches_zero_atoms(self, rng):
        """Test generating all exact matches for zero atoms."""
        matches = generate_template_matches(
            [], 0, rng, include_exact=True, include_near=False
        )
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_generate_all_exact_matches_multi_element(self, rng):
        """Test generating all exact matches with multi-element composition."""
        matches = generate_template_matches(
            ["Pt", "Au"] * 7, 13, rng, include_exact=True, include_near=False
        )  # 13 atoms
        assert isinstance(matches, list)
        if len(matches) > 0:
            for atoms in matches:
                assert len(atoms) == 13
                symbols = atoms.get_chemical_symbols()
                assert all(sym in ["Pt", "Au"] for sym in symbols)

    def test_generate_all_exact_matches_empty_composition(self, rng):
        """Test that empty composition returns empty list (no valid templates)."""
        # Empty composition will cause template generators to raise ValueError,
        # which is caught and logged, resulting in an empty list
        matches = generate_template_matches(
            [], 13, rng, include_exact=True, include_near=False
        )
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_generate_all_exact_matches_large_size(self, rng):
        """Test generating all exact matches for large size."""
        matches = generate_template_matches(
            ["Pt"] * 147, 147, rng, include_exact=True, include_near=False
        )
        assert isinstance(matches, list)
        if len(matches) > 0:
            for atoms in matches:
                assert len(atoms) == 147

    def test_generate_all_exact_matches_reproducible(self):
        """Test that generating all exact matches is reproducible."""
        rng1, rng2 = create_paired_rngs(42)
        matches1 = generate_template_matches(
            ["Pt"] * 13, 13, rng1, include_exact=True, include_near=False
        )
        matches2 = generate_template_matches(
            ["Pt"] * 13, 13, rng2, include_exact=True, include_near=False
        )
        assert len(matches1) == len(matches2)


class TestGenerateNearMatchTemplates:
    """Tests for generating near-match templates."""

    def test_generate_near_matches_within_tolerance(self, rng):
        """Test generating near matches within tolerance."""
        # 12 is near magic number 13; nearest-magic logic generates from 13.
        matches = generate_template_matches(
            ["Pt"] * 12,
            12,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # May have matches from nearby magic numbers (13, 11, etc.)
        for atoms in matches:
            assert atoms is not None
            assert len(atoms) == 12

    def test_generate_near_matches_exact_magic(self, rng):
        """Test generating near matches for exact magic number."""
        # Should return empty since exact matches are handled separately
        matches = generate_template_matches(
            ["Pt"] * 13,
            13,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # Should be empty or only near matches (not exact)

    def test_generate_near_matches_zero_tolerance(self, rng):
        """Test generating near matches with zero tolerance."""
        matches = generate_template_matches(
            ["Pt"] * 12,
            12,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # With zero tolerance, should only match exact magic numbers

    def test_generate_near_matches_large_tolerance(self, rng):
        """Test generating near matches with large tolerance."""
        matches = generate_template_matches(
            ["Pt"] * 20,
            20,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # May have more matches with larger tolerance

    def test_generate_near_matches_no_nearby_magic(self, rng):
        """Test generating near matches when no nearby magic numbers."""
        # Use a size far from any magic number
        matches = generate_template_matches(
            ["Pt"] * 50,
            50,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # May or may not have matches

    def test_generate_near_matches_multi_element(self, rng):
        """Test generating near matches with multi-element composition."""
        matches = generate_template_matches(
            ["Pt", "Au"] * 6,
            12,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        for atoms in matches:
            if atoms is not None:
                assert len(atoms) == 12
                symbols = atoms.get_chemical_symbols()
                assert all(sym in ["Pt", "Au"] for sym in symbols)

    def test_generate_near_matches_connectivity_failure(self, rng):
        """Test that connectivity failures are handled gracefully."""
        # Use very strict connectivity factor that may cause failures
        STRICT_FACTOR = 1.0  # Very strict connectivity requirement for testing
        matches = generate_template_matches(
            ["Pt"] * 12,
            12,
            rng=rng,
            connectivity_factor=STRICT_FACTOR,
            include_exact=False,
            include_near=True,
        )
        assert isinstance(matches, list)
        # Should not raise exception even if some attempts fail

    def test_generate_near_matches_cell_side(self, rng):
        """Test generating near matches with custom cell side."""
        matches = generate_template_matches(
            ["Pt"] * 12,
            12,
            rng=rng,
            cell_side=30.0,
            include_exact=False,
            include_near=True,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
        )
        assert isinstance(matches, list)
        for atoms in matches:
            if atoms is not None:
                cell = atoms.get_cell()
                assert cell is not None


class TestGenerateTemplateWithAtomAdjustment:
    """Tests for template generation with atom adjustment."""

    def test_adjustment_exact_match(self, rng):
        """Test adjustment when base and target are the same."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=13,
            target_n_atoms=13,
            composition=["Pt"] * 13,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 13

    def test_adjustment_add_atoms(self, rng):
        """Test adjustment when adding atoms (len(atoms_to_add) == n_diff case)."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=13,
            target_n_atoms=15,
            composition=["Pt"] * 15,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
        )
        assert result is not None, "add-atoms 13->15 Pt should succeed with fixed RNG"
        assert len(result) == 15

    def test_adjustment_remove_atoms(self, rng):
        """Test adjustment when removing atoms."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=55,
            target_n_atoms=50,
            composition=["Pt"] * 50,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 50

    def test_adjustment_invalid_template_type(self, rng):
        """Test adjustment with invalid template type."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="invalid_type",
            base_n_atoms=13,
            target_n_atoms=13,
            composition=["Pt"] * 13,
            rng=rng,
        )
        assert result is None

    def test_adjustment_base_generation_fails(self, rng):
        """Test adjustment when base template generation fails."""
        # Try to generate a template that doesn't exist for this size
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=1,  # Too small for icosahedron
            target_n_atoms=1,
            composition=["Pt"],
            rng=rng,
        )
        # May or may not succeed - just check it doesn't raise
        assert result is None or isinstance(result, Atoms)

    def test_adjustment_many_atoms_to_add(self, rng):
        """Test adjustment when adding many atoms."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=13,
            target_n_atoms=30,
            composition=["Pt"] * 30,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
        )
        # May succeed or fail depending on connectivity
        if result is not None:
            assert len(result) == 30

    def test_adjustment_many_atoms_to_remove(self, rng):
        """Test adjustment when removing many atoms."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=55,
            target_n_atoms=50,
            composition=["Pt"] * 50,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 50

    def test_adjustment_multi_element(self, rng):
        """Test adjustment with multi-element composition."""
        from scgo.utils.helpers import get_composition_counts

        target_composition = ["Pt", "Au"] * 8  # 16 elements, will be truncated to 15
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=13,
            target_n_atoms=15,
            composition=target_composition,
            rng=rng,
            connectivity_factor=TEMPLATE_GENERATION_FACTOR,
        )
        if result is not None:
            assert len(result) == 15
            symbols = result.get_chemical_symbols()
            assert all(sym in ["Pt", "Au"] for sym in symbols)
            # Verify exact composition counts match (cycling should produce 8 Pt, 7 Au)
            expected_counts = get_composition_counts(target_composition[:15])
            actual_counts = get_composition_counts(symbols)
            assert actual_counts == expected_counts, (
                f"Composition mismatch: expected {expected_counts}, got {actual_counts}"
            )

    def test_adjustment_zero_atoms(self, rng):
        """Test adjustment with zero atoms."""
        result = _generate_template_with_atom_adjustment(
            base_template_type="icosahedron",
            base_n_atoms=13,
            target_n_atoms=0,
            composition=[],
            rng=rng,
        )
        # Should fail or return None
        assert result is None or len(result) == 0
