"""Tests verifying that mutations and crossover are valid for both
gas-phase and surface-supported nanoparticles.

Key invariants checked:
- Slab atom positions are never modified by any operator.
- `.center()` is only applied to gas-phase (empty-slab) clusters.
- `atoms_too_close_two_sets` is checked for surface systems.
- Operators return valid slab+top composites with correct atom count.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.ase_ga_patches.standardmutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    FlatteningMutation,
    InPlaneSlideMutation,
    MirrorMutation,
    RattleMutation,
    RotationalMutation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_system(n_adsorbate=3, seed=0):
    """Create a small slab + adsorbate system for testing.

    Returns (combined, slab, n_slab, n_top, blmin).
    """
    rng = np.random.default_rng(seed)
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = True
    n_slab = len(slab)

    # Place Cu adsorbate atoms well above the slab
    slab_zmax = slab.positions[:, 2].max()
    ads_positions = []
    for i in range(n_adsorbate):
        x = rng.uniform(1.0, slab.cell[0, 0] - 1.0)
        y = rng.uniform(1.0, slab.cell[1, 1] - 1.0)
        z = slab_zmax + 2.0 + i * 2.5
        ads_positions.append([x, y, z])

    adsorbate = Atoms(
        "Cu" * n_adsorbate,
        positions=ads_positions,
        cell=slab.cell,
        pbc=slab.pbc,
    )
    combined = slab + adsorbate
    n_top = n_adsorbate

    all_types = get_all_atom_types(combined, range(n_slab, len(combined)))
    blmin = closest_distances_generator(all_types, ratio_of_covalent_radii=0.5)
    return combined, slab, n_slab, n_top, blmin


def _make_gas_phase_cluster(seed=0):
    """Create a small gas-phase cluster for testing.

    Returns (atoms, blmin).
    """
    atoms = Atoms(
        "Pt4",
        positions=[
            [0, 0, 0],
            [2.5, 0, 0],
            [1.25, 2.165, 0],
            [1.25, 0.721, 2.357],
        ],
    )
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(False)
    all_types = get_all_atom_types(atoms, range(len(atoms)))
    blmin = closest_distances_generator(all_types, ratio_of_covalent_radii=0.7)
    return atoms, blmin


# ---------------------------------------------------------------------------
# 1. Slab positions are never modified by any mutation
# ---------------------------------------------------------------------------
class TestSlabPositionsPreserved:
    """Every mutation must return slab + top where slab positions are
    identical to the input slab positions."""

    @pytest.fixture
    def surface_system(self):
        return _make_surface_system(n_adsorbate=3, seed=42)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_rattle_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = RattleMutation(
            blmin,
            n_top,
            rattle_strength=0.8,
            rattle_prop=0.6,
            test_dist_to_slab=True,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        if result is not None:
            assert len(result) == len(combined)
            assert np.allclose(result.positions[:n_slab], slab.positions)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_anisotropic_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = AnisotropicRattleMutation(
            blmin,
            n_top,
            test_dist_to_slab=True,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        if result is not None:
            assert len(result) == len(combined)
            assert np.allclose(result.positions[:n_slab], slab.positions)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_mirror_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = MirrorMutation(
            blmin,
            n_top,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        if result is not None:
            assert len(result) == len(combined)
            assert np.allclose(result.positions[:n_slab], slab.positions)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_rotational_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        # Tag each adsorbate atom separately as a single-atom moiety
        tags = [0] * n_slab + list(range(n_top))
        combined.set_tags(tags)
        # RotationalMutation only acts on multi-atom moieties.
        # Use 2 adsorbate atoms sharing a tag to trigger rotation.
        combined2, slab2, n_slab2, n_top2, blmin2 = _make_surface_system(
            n_adsorbate=4, seed=seed + 100
        )
        tags2 = [0] * n_slab2 + [1, 1, 2, 2]
        combined2.set_tags(tags2)
        mut = RotationalMutation(
            blmin2,
            system_type="surface_cluster",
            n_top=n_top2,
            fraction=1.0,
            min_angle=0.5,
            test_dist_to_slab=True,
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined2)
        if result is not None:
            assert len(result) == len(combined2)
            assert np.allclose(result.positions[:n_slab2], slab2.positions)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_flattening_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = FlatteningMutation(
            blmin,
            n_top,
            thickness_factor=1.0,
            test_dist_to_slab=True,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        assert result is not None
        assert len(result) == len(combined)
        assert np.allclose(result.positions[:n_slab], slab.positions)
        assert mut.last_attempt_count <= 2

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_breathing_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = BreathingMutation(
            blmin,
            n_top,
            scale_min=0.85,
            scale_max=1.15,
            test_dist_to_slab=True,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        assert result is not None
        assert len(result) == len(combined)
        assert np.allclose(result.positions[:n_slab], slab.positions)
        assert mut.last_attempt_count <= 5

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_plane_slide_preserves_slab(self, surface_system, seed):
        combined, slab, n_slab, n_top, blmin = surface_system
        mut = InPlaneSlideMutation(
            blmin,
            n_top,
            surface_normal_axis=2,
            max_displacement=1.0,
            system_type="surface_cluster",
            rng=np.random.default_rng(seed),
        )
        result = mut.mutate(combined)
        assert result is not None
        assert len(result) == len(combined)
        assert np.allclose(result.positions[:n_slab], slab.positions)
        assert mut.last_attempt_count <= 12


# ---------------------------------------------------------------------------
# 2. `.center()` is NOT applied to surface-system adsorbates
# ---------------------------------------------------------------------------
class TestNoCenteringOnSurfaceSystems:
    """For surface systems, `.center()` must NOT be called on the top
    fragment, as it would destroy positions relative to the slab."""

    def test_rotational_no_centering_on_surface(self):
        """RotationalMutation with a slab must not center the adsorbate."""
        combined, slab, n_slab, n_top, blmin = _make_surface_system(
            n_adsorbate=4, seed=77
        )
        tags = [0] * n_slab + [1, 1, 2, 2]
        combined.set_tags(tags)

        # Run many mutations and verify none center the adsorbate in the cell
        cell_center = np.diag(combined.cell) / 2.0
        centered_count = 0
        trials = 0
        for seed in range(30):
            mut = RotationalMutation(
                blmin,
                system_type="surface_cluster",
                n_top=n_top,
                fraction=1.0,
                min_angle=0.5,
                test_dist_to_slab=True,
                rng=np.random.default_rng(seed),
            )
            result = mut.mutate(combined)
            if result is not None:
                trials += 1
                top_cm = np.mean(result.positions[n_slab:], axis=0)
                # If centered, the top CM would be near the cell center.
                # The original top CM is above the slab, not at cell center.
                if np.linalg.norm(top_cm - cell_center) < 1.0:
                    centered_count += 1

        # With the fix, none should be accidentally centered
        assert trials > 0, "No successful mutations to test"
        assert centered_count == 0, (
            f"{centered_count}/{trials} mutations centered the adsorbate "
            f"(should be 0 for surface systems)"
        )

    def test_flattening_no_centering_on_surface(self):
        """FlatteningMutation with a slab must not center the adsorbate."""
        combined, slab, n_slab, n_top, blmin = _make_surface_system(
            n_adsorbate=3, seed=88
        )

        cell_center = np.diag(combined.cell) / 2.0
        for seed in range(5):
            mut = FlatteningMutation(
                blmin,
                n_top,
                thickness_factor=1.0,
                test_dist_to_slab=True,
                system_type="surface_cluster",
                rng=np.random.default_rng(seed),
            )
            result = mut.mutate(combined)
            assert result is not None
            top_cm = np.mean(result.positions[n_slab:], axis=0)
            assert np.linalg.norm(top_cm - cell_center) >= 1.0
            assert mut.last_attempt_count <= 2


# ---------------------------------------------------------------------------
# 3. Gas-phase clusters still get centered (regression guard)
# ---------------------------------------------------------------------------
class TestGasPhaseCenteringPreserved:
    """Gas-phase clusters (empty slab) should still be centered after
    mutation, as before."""

    def test_rotational_centers_gas_phase(self):
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0], [7.5, 0, 0]],
            tags=[0, 0, 1, 1],
        )
        atoms.set_cell([20, 20, 20])
        atoms.set_pbc(False)
        blmin = {(78, 78): 0.1}
        mut = RotationalMutation(
            blmin,
            n_top=4,
            fraction=1.0,
            min_angle=0.5,
            test_dist_to_slab=False,
            system_type="gas_cluster",
            rng=np.random.default_rng(42),
        )
        result = mut.mutate(atoms)
        assert result is not None
        # After centering, the center of mass should be near the cell center
        cm = np.mean(result.positions, axis=0)
        cell_center = np.array([10.0, 10.0, 10.0])
        assert np.linalg.norm(cm - cell_center) < 2.0

    def test_flattening_centers_gas_phase(self):
        atoms, blmin = _make_gas_phase_cluster()
        mut = FlatteningMutation(
            blmin,
            n_top=4,
            thickness_factor=1.0,
            test_dist_to_slab=False,
            system_type="gas_cluster",
            rng=np.random.default_rng(42),
        )
        result = mut.mutate(atoms)
        assert result is not None
        cm = np.mean(result.positions, axis=0)
        cell_center = np.array([5.0, 5.0, 5.0])
        assert np.linalg.norm(cm - cell_center) < 2.0


# ---------------------------------------------------------------------------
# 4. CutAndSplicePairing preserves slab for surface systems
# ---------------------------------------------------------------------------
class TestCutAndSpliceSurfaceCoherence:
    def test_pairing_preserves_slab_positions(self):
        combined, slab, n_slab, n_top, blmin = _make_surface_system(
            n_adsorbate=3, seed=0
        )
        combined2, _, _, _, _ = _make_surface_system(n_adsorbate=3, seed=1)

        pairing = CutAndSplicePairing(
            slab,
            n_top,
            blmin,
            test_dist_to_slab=True,
            system_type="surface_cluster",
            rng=np.random.default_rng(42),
        )

        for seed in range(5):
            pairing.rng = np.random.default_rng(seed)
            child = pairing.cross(combined, combined2)
            assert child is not None
            assert len(child) == len(combined)
            assert np.allclose(child.positions[:n_slab], slab.positions)
            assert pairing.last_attempt_count <= 2

    def test_pairing_gas_phase_centers(self):
        atoms1, blmin = _make_gas_phase_cluster(seed=0)
        atoms2, _ = _make_gas_phase_cluster(seed=1)
        # Gas-phase: empty slab
        empty_slab = Atoms(cell=atoms1.cell, pbc=atoms1.pbc)
        pairing = CutAndSplicePairing(
            empty_slab,
            len(atoms1),
            blmin,
            test_dist_to_slab=False,
            system_type="gas_cluster",
            rng=np.random.default_rng(42),
        )

        for seed in range(5):
            pairing.rng = np.random.default_rng(seed)
            child = pairing.cross(atoms1, atoms2)
            assert child is not None
            cm = np.mean(child.positions, axis=0)
            cell_center = np.diag(child.cell) / 2.0
            assert np.linalg.norm(cm - cell_center) < 3.0
            assert pairing.last_attempt_count <= 2


# ---------------------------------------------------------------------------
# 5. InPlaneSlideMutation only displaces in-plane axes
# ---------------------------------------------------------------------------
class TestInPlaneSlideAxes:
    """InPlaneSlideMutation displaces along axes 0 and 1 (for normal=2),
    leaving the normal-axis coordinate unchanged."""

    def test_normal_axis_unchanged(self):
        combined, slab, n_slab, n_top, blmin = _make_surface_system(
            n_adsorbate=2, seed=10
        )
        original_z = combined.positions[n_slab:, 2].copy()

        mut = InPlaneSlideMutation(
            blmin,
            n_top,
            surface_normal_axis=2,
            max_displacement=1.0,
            system_type="surface_cluster",
            rng=np.random.default_rng(42),
        )
        result = mut.mutate(combined)
        if result is not None:
            result_z = result.positions[n_slab:, 2]
            assert np.allclose(result_z, original_z), (
                "InPlaneSlideMutation moved atoms along the surface normal"
            )
