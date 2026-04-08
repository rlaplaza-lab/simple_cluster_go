"""Tests for helper utility functions.

This module tests various utility functions used throughout the SCGO package.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from scgo.utils.helpers import (
    auto_niter,
    auto_niter_local_relaxation,
    auto_niter_ts,
    auto_population_size,
    canonicalize_storage_frame,
    ensure_float64_forces,
    filter_unique_minima,
    get_cluster_formula,
)


class TestFilterUniqueMinima:
    """Tests for filter_unique_minima function."""

    def test_filter_unique_minima_empty(self):
        """Test filtering of empty minima list."""
        result = filter_unique_minima([])
        assert len(result) == 0

    def test_filter_unique_minima_single(self):
        """Test filtering of single minimum."""
        atoms = Atoms("Pt", positions=[[0, 0, 0]])
        atoms.info = {"key_value_pairs": {"raw_score": 1.0}, "provenance": {"trial": 1}}

        result = filter_unique_minima([(1.0, atoms)])
        assert len(result) == 1

    def test_filter_unique_minima_basic(self):
        """Test basic filtering functionality."""
        # Create atoms with required metadata
        atoms1 = Atoms("Pt", positions=[[0, 0, 0]])
        atoms1.info = {
            "key_value_pairs": {"raw_score": 1.0},
            "provenance": {"trial": 1},
        }

        atoms2 = Atoms("Pt", positions=[[1, 1, 1]])  # Different position
        atoms2.info = {
            "key_value_pairs": {"raw_score": 2.0},
            "provenance": {"trial": 1},
        }

        # Should return both since they have different positions
        result = filter_unique_minima([(1.0, atoms1), (2.0, atoms2)])
        assert len(result) == 2

    def test_filter_unique_minima_ignores_fixed_slab_atom_differences(self):
        """Minima dedup should ignore changes confined to fixed slab atoms."""
        base = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.6, 1.0, 0.0],
                [0.6, 0.4, 1.8],
            ],
        )
        base.set_constraint(FixAtoms(indices=[0, 1, 2]))
        slab_shifted = base.copy()
        slab_shifted.set_constraint(FixAtoms(indices=[0, 1, 2]))
        p = slab_shifted.get_positions()
        p[:3, 2] += 0.3
        slab_shifted.set_positions(p)

        result = filter_unique_minima([(0.0, base), (0.001, slab_shifted)])
        assert len(result) == 1


class TestAutoNiter:
    """Tests for auto_niter function."""

    @pytest.mark.parametrize(
        "composition,expected_range",
        [
            (["Pt"], (3, 1000)),
            (["Pt", "Pt"], (3, 1000)),
            (["Pt"] * 5, (3, 1000)),
            (["Pt"] * 10, (3, 1000)),
            (["Pt"] * 20, (3, 1000)),
            (["Pt"] * 50, (100, 200)),
        ],
    )
    def test_auto_niter_scaling(self, composition, expected_range):
        """Test that auto_niter scales appropriately with composition size."""
        result = auto_niter(composition)
        assert expected_range[0] <= result <= expected_range[1]

    def test_auto_niter_reproducibility(self):
        """Test that auto_niter is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_niter(composition)
        result2 = auto_niter(composition)
        assert result1 == result2

    def test_auto_niter_different_elements(self):
        """Test auto_niter with different element types."""
        # Should work with any composition
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter(comp)
            assert isinstance(result, int)
            assert result > 0


class TestAutoPopulationSize:
    """Tests for auto_population_size function."""

    @pytest.mark.parametrize(
        "composition,expected_range",
        [
            (["Pt"], (3, 1000)),
            (["Pt", "Pt"], (3, 1000)),
            (["Pt"] * 5, (3, 1000)),
            (["Pt"] * 10, (3, 1000)),
            (["Pt"] * 20, (3, 1000)),
            (["Pt"] * 50, (100, 200)),
        ],
    )
    def test_auto_population_size_scaling(self, composition, expected_range):
        """Test that auto_population_size scales appropriately with composition size."""
        result = auto_population_size(composition)
        assert expected_range[0] <= result <= expected_range[1]

    def test_auto_population_size_reproducibility(self):
        """Test that auto_population_size is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_population_size(composition)
        result2 = auto_population_size(composition)
        assert result1 == result2

    def test_auto_population_size_different_elements(self):
        """Test auto_population_size with different element types."""
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_population_size(comp)
            assert isinstance(result, int)
            assert result > 0


class TestAutoNiterLocalRelaxation:
    """Tests for auto_niter_local_relaxation function."""

    @pytest.mark.parametrize(
        "composition,expected_range",
        [
            (["Pt"], (50, 2000)),
            (["Pt", "Pt"], (50, 2000)),
            (["Pt"] * 5, (100, 200)),
            (["Pt"] * 10, (150, 250)),
            (["Pt"] * 20, (200, 300)),
            (["Pt"] * 50, (240, 300)),
        ],
    )
    def test_auto_niter_local_relaxation_scaling(self, composition, expected_range):
        """Test that auto_niter_local_relaxation scales appropriately with composition size."""
        result = auto_niter_local_relaxation(composition)
        assert expected_range[0] <= result <= expected_range[1]

    def test_auto_niter_local_relaxation_reproducibility(self):
        """Test that auto_niter_local_relaxation is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_niter_local_relaxation(composition)
        result2 = auto_niter_local_relaxation(composition)
        assert result1 == result2

    def test_auto_niter_local_relaxation_different_elements(self):
        """Test auto_niter_local_relaxation with different element types."""
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter_local_relaxation(comp)
            assert isinstance(result, int)
            assert result >= 50  # Minimum is 50

    def test_auto_niter_local_relaxation_increases_with_size(self):
        """Test that relaxation steps increase with cluster size."""
        sizes = [2, 5, 10, 20, 50]
        results = [auto_niter_local_relaxation(["Pt"] * n) for n in sizes]
        # Should generally increase (allowing for some non-monotonicity due to rounding)
        assert results[-1] >= results[0]  # Largest should be >= smallest


class TestAutoNiterTS:
    """Tests for `auto_niter_ts` (TS/NEB auto-scaling helper)."""

    @pytest.mark.parametrize(
        "composition,expected_range",
        [
            (["Pt"], (150, 220)),
            (["Pt", "Pt"], (220, 280)),
            (["Pt"] * 5, (350, 400)),
            (["Pt"] * 6, (380, 420)),
            (["Pt"] * 10, (430, 520)),
            (["Pt"] * 20, (520, 650)),
            (["Pt"] * 50, (700, 900)),
        ],
    )
    def test_auto_niter_ts_scaling(self, composition, expected_range):
        result = auto_niter_ts(composition)
        assert expected_range[0] <= result <= expected_range[1]

    def test_auto_niter_ts_reproducibility(self):
        composition = ["Pt"] * 6
        result1 = auto_niter_ts(composition)
        result2 = auto_niter_ts(composition)
        assert result1 == result2

    def test_auto_niter_ts_different_elements(self):
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter_ts(comp)
            assert isinstance(result, int)
            assert result >= 150  # minimum is 150

    def test_auto_niter_ts_increases_with_size(self):
        sizes = [1, 6, 10, 20]
        results = [auto_niter_ts(["Pt"] * n) for n in sizes]
        assert results[-1] >= results[0]


class TestGetClusterFormula:
    """Tests for get_cluster_formula function."""

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], "Pt"),
            (["Pt", "Pt"], "Pt2"),
            (["Pt", "Au"], "AuPt"),
            (["Pt", "Pt", "Au"], "AuPt2"),
            (["Pt", "Au", "Pd"], "AuPdPt"),
        ],
    )
    def test_get_cluster_formula(self, composition, expected):
        """Test get_cluster_formula with various compositions."""
        result = get_cluster_formula(composition)
        assert result == expected

    def test_get_cluster_formula_empty(self):
        """Test get_cluster_formula with empty composition."""
        result = get_cluster_formula([])
        assert result == ""

    def test_get_cluster_formula_performance(self):
        """Test get_cluster_formula performance with large clusters."""
        # Create a large composition
        composition = ["Pt"] * 50

        # Should complete quickly
        result = get_cluster_formula(composition)
        assert result == "Pt50"


class TestEnsureFloat64Forces:
    """Tests for ensure_float64_forces utility function."""

    def test_ensure_float64_forces_converts_float32(self):
        """Test that float32 forces are converted to float64."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Simulate float32 forces
        forces_f32 = atoms.get_forces().astype(np.float32)
        atoms.arrays["forces"] = forces_f32

        # Apply conversion
        ensure_float64_forces(atoms)

        # Verify forces are now float64
        assert atoms.arrays["forces"].dtype == np.float64

    def test_ensure_float64_forces_updates_calc_results(self):
        """Test that calculator results are also updated."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Get forces and convert to float32
        forces_f32 = atoms.get_forces().astype(np.float32)
        atoms.arrays["forces"] = forces_f32
        atoms.calc.results["forces"] = forces_f32

        # Apply conversion
        ensure_float64_forces(atoms)

        # Verify both locations updated
        assert atoms.arrays["forces"].dtype == np.float64
        assert atoms.calc.results["forces"].dtype == np.float64

    def test_ensure_float64_forces_no_forces(self):
        """Test handling when no forces are available."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        # No calculator attached

        # Should raise RuntimeError when no calculator is attached
        with pytest.raises(RuntimeError, match="no calculator"):
            ensure_float64_forces(atoms)

    def test_ensure_float64_forces_preserves_values(self):
        """Test that force values are preserved during conversion."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Get original forces
        original_forces = atoms.get_forces().copy()

        # Convert to float32
        atoms.arrays["forces"] = original_forces.astype(np.float32)

        # Apply conversion
        ensure_float64_forces(atoms)

        # Values should be approximately equal (allowing for float32 precision loss)
        np.testing.assert_allclose(atoms.arrays["forces"], original_forces, rtol=1e-5)


class TestCanonicalizeStorageFrame:
    """Tests for translation-only canonicalization before persistence."""

    def test_non_pbc_recenter_to_cell_midpoint(self):
        atoms = Atoms(
            "Pt2",
            positions=[[1.0, 2.0, 3.0], [2.5, 2.0, 3.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        canonicalize_storage_frame(atoms)
        com = atoms.get_center_of_mass()
        np.testing.assert_allclose(com, [5.0, 5.0, 5.0], atol=1e-8)

    def test_pbc_wraps_before_centering(self):
        atoms = Atoms(
            "Pt2",
            positions=[[-0.2, 0.1, 0.1], [9.8, 0.1, 0.1]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        canonicalize_storage_frame(atoms)
        positions = atoms.get_positions()
        assert np.all(positions >= -1e-8)
        assert np.all(positions <= 10.0 + 1e-8)

    def test_can_skip_centering_for_surface_frames(self):
        atoms = Atoms(
            "Pt2",
            positions=[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        before = atoms.get_center_of_mass().copy()
        canonicalize_storage_frame(atoms, center=False)
        after = atoms.get_center_of_mass()
        np.testing.assert_allclose(after, before, atol=1e-8)


class TestCanonicalizeSlabAdsorbateFrame:
    """Rigid lattice shift for slab+adsorbate under in-plane PBC."""

    def test_adsorbate_com_fractional_in_unit_cell_after_canonicalize(self):
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 15.0]]
        pbc = [True, True, False]
        atoms = Atoms(
            ["Pt", "Pt"],
            positions=[[1.0, 5.0, 2.0], [12.0, 5.0, 3.0]],
            cell=cell,
            pbc=pbc,
        )
        canonicalize_storage_frame(atoms, pbc_aware=True, center=False, n_slab=1)
        scaled = atoms.get_scaled_positions(wrap=False)[1:]
        com = np.average(scaled, axis=0, weights=atoms.get_masses()[1:])
        assert 0 <= com[0] < 1
        assert 0 <= com[1] < 1

    def test_lattice_translated_pairs_match_after_canonicalize(self):
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 15.0]]
        pbc = [True, True, False]
        a = Atoms(
            ["Pt", "Pt"],
            positions=[[1.0, 5.0, 2.0], [12.0, 5.0, 3.0]],
            cell=cell,
            pbc=pbc,
        )
        b = a.copy()
        b.positions += np.array([-10.0, 0.0, 0.0])
        canonicalize_storage_frame(a, pbc_aware=True, center=False, n_slab=1)
        canonicalize_storage_frame(b, pbc_aware=True, center=False, n_slab=1)
        np.testing.assert_allclose(a.get_positions(), b.get_positions(), atol=1e-10)

    def test_n_slab_zero_still_wraps_and_centers_pbc_cluster(self):
        """Regression: gas-style PBC dimer path unchanged when n_slab is 0."""
        atoms = Atoms(
            "Pt2",
            positions=[[-0.2, 0.1, 0.1], [9.8, 0.1, 0.1]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        canonicalize_storage_frame(atoms, n_slab=0)
        positions = atoms.get_positions()
        assert np.all(positions >= -1e-8)
        assert np.all(positions <= 10.0 + 1e-8)
