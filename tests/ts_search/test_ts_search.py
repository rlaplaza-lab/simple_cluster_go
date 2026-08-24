"""Tests for transition state search (NEB, TorchSim, MACE)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths

from scgo.exceptions import SCGOValidationError
from scgo.metadata.provenance import OUTPUT_JSON_SCHEMA_VERSION
from scgo.pair_selection_defaults import (
    DEFAULT_PAIR_CORE_RMS_MAX_GAS,
    DEFAULT_PAIR_CORE_RMS_MAX_SURFACE,
)
from scgo.param_presets import get_ts_defaults
from scgo.system_types import SYSTEM_TYPE_POLICIES, get_system_policy
from scgo.ts_search import transition_state_run as ts_run_mod
from scgo.ts_search.transition_state import (
    _overlay_product_core,
    calculate_structure_similarity,
    find_transition_state,
    interpolate_path,
    load_completed_neb_result,
    neb_max_atom_force,
    save_neb_result,
)
from scgo.ts_search.transition_state_io import (
    _adsorbate_max_displacement,
    _core_rms_displacement,
    adsorbate_pair_select_cap,
    resolve_ts_pair_select_cap,
    select_structure_pairs,
)
from scgo.utils.ts_runner_kwargs import NebRunConfig


def test_neb_max_atom_force_uses_per_atom_norm():
    """Component-wise abs max can falsely converge; ASE uses max atom ||f||."""
    # |f|_inf = 0.04 < 0.05, but ||f|| ≈ 0.069 > 0.05
    forces = np.array([[0.04, 0.04, 0.04], [0.01, 0.0, 0.0]])
    fmax = neb_max_atom_force(forces)
    assert fmax == pytest.approx(np.sqrt(3 * 0.04**2))
    assert fmax >= 0.05
    assert float(np.max(np.abs(forces))) < 0.05


def test_image_potential_energy_falls_back_to_metadata():
    """Finalize helper reads cached potential_energy when SinglePoint is stale."""
    from ase.calculators.singlepoint import SinglePointCalculator

    from scgo.metadata.atoms import set_tags
    from scgo.ts_search.transition_state import _image_potential_energy

    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.5, forces=np.zeros((1, 3)))
    set_tags(atoms, potential_energy=-1.5, raw_score=1.5)
    atoms.positions += 0.2  # invalidate SinglePoint
    assert _image_potential_energy(atoms) == pytest.approx(-1.5)


def test_torchsim_neb_forces_match_ase_neb_with_same_pes(cu3_triangle, cu3_linear):
    """TorchSimNEB must reuse ASE spring/climb/tangent; only PES eval is batched.

    With identical per-image forces from a fake relaxer, NEB band forces must
    match a plain ASE ``NEB`` on the same images.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.mep import NEB

    from scgo.ts_search.transition_state import TorchSimNEB

    images_ase = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    images_ts = [img.copy() for img in images_ase]

    rng = np.random.default_rng(0)
    pes_forces = [rng.normal(scale=0.05, size=(len(img), 3)) for img in images_ase]
    for img, forces in zip(images_ase, pes_forces, strict=True):
        img.calc = SinglePointCalculator(img, energy=0.0, forces=forces)

    class _PesRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            assert steps == 0
            out = []
            for atoms in atoms_list:
                idx = next(i for i, im in enumerate(images_ts) if im is atoms)
                ra = atoms.copy()
                ra.arrays["forces"] = np.asarray(pes_forces[idx], dtype=float)
                out.append((0.0, ra))
            return out

    ase_neb = NEB(images_ase, k=0.1, climb=True, method="improvedtangent")
    ts_neb = TorchSimNEB(
        images_ts,
        _PesRelaxer(),
        k=0.1,
        climb=True,
        method="improvedtangent",
    )
    ase_f = ase_neb.get_forces()
    ts_f = ts_neb.get_forces()
    assert ts_neb.get_force_calls() == 1
    np.testing.assert_allclose(ts_f, ase_f, atol=1e-12)


def test_interpolate_path_basic(h2_reactant, h2_product):
    """Test basic geodesic interpolation between two structures."""
    n_images = 5
    # Explicitly disable endpoint alignment for this test so endpoints remain
    # bitwise-identical to the provided Atoms objects (previous default).
    images = interpolate_path(
        h2_reactant,
        h2_product,
        n_images=n_images,
        method="linear",
        align_endpoints=False,
    )

    # Should return n_images + 2 (including endpoints)
    assert len(images) == n_images + 2
    assert images[0] == h2_reactant
    assert images[-1] == h2_product

    # Check that intermediate images are interpolated
    # Distance should increase monotonically
    for i in range(len(images) - 1):
        d1 = images[i].get_distance(0, 1)
        d2 = images[i + 1].get_distance(0, 1)
        assert d2 >= d1


def test_interpolate_path_idpp(li2_isomer1, li2_isomer2):
    """Test IDPP interpolation (default)."""
    n_images = 3
    images = interpolate_path(
        li2_isomer1, li2_isomer2, n_images=n_images, method="idpp"
    )

    assert len(images) == n_images + 2
    # IDPP should avoid atom overlaps better than linear
    # Check all images have reasonable distances
    for img in images:
        d = img.get_distance(0, 1)
        assert d > 1.0  # Atoms shouldn't overlap


def test_interpolate_path_align_reduces_max_displacement(pt4_tetrahedron):
    """Endpoint alignment should reduce per-atom displacements for permuted endpoints."""
    a = pt4_tetrahedron.copy()
    b = a.copy()
    perm = [2, 3, 0, 1]
    b.set_positions(a.get_positions()[perm])
    b.rotate(0.7, "z", rotate_cell=False)

    imgs_no_align = interpolate_path(
        a, b, n_images=3, method="idpp", align_endpoints=False
    )
    max_disp_no_align = float(
        np.max(
            np.linalg.norm(
                imgs_no_align[0].get_positions() - imgs_no_align[-1].get_positions(),
                axis=1,
            )
        )
    )

    imgs_align = interpolate_path(a, b, n_images=3, method="idpp", align_endpoints=True)
    max_disp_align = float(
        np.max(
            np.linalg.norm(
                imgs_align[0].get_positions() - imgs_align[-1].get_positions(), axis=1
            )
        )
    )

    assert max_disp_align < max_disp_no_align


def test_interpolate_path_perturb_keeps_endpoints_changes_interior(
    h2_reactant, h2_product
):
    """Perturbation should not change endpoints but should perturb interior images."""
    rng = np.random.default_rng(12345)
    imgs_clean = interpolate_path(h2_reactant, h2_product, n_images=3, method="idpp")
    imgs_pert = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng
    )

    # endpoints unchanged
    assert np.allclose(imgs_clean[0].get_positions(), imgs_pert[0].get_positions())
    assert np.allclose(imgs_clean[-1].get_positions(), imgs_pert[-1].get_positions())

    # interior image changed
    assert not np.allclose(imgs_clean[1].get_positions(), imgs_pert[1].get_positions())


def test_interpolate_path_perturb_deterministic_with_rng(h2_reactant, h2_product):
    """Seeded RNG should produce deterministic perturbations."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    imgs1 = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng1
    )
    imgs2 = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng2
    )

    assert np.allclose(imgs1[1].get_positions(), imgs2[1].get_positions())


@pytest.mark.slow
def test_find_transition_state_records_align_and_perturb(
    temp_output_dir, h2_reactant, h2_product
):
    """`find_transition_state` should record `align_endpoints` and `perturb_sigma` in the result metadata."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="meta_test",
        n_images=3,
        fmax=0.1,
        neb_steps=20,
        verbosity=0,
        align_endpoints=True,
        perturb_sigma=0.03,
        rng=np.random.default_rng(1),
    )

    assert result.get("align_endpoints") is True
    assert result.get("perturb_sigma") == pytest.approx(0.03)
    # returned structure should not retain a calculator instance
    ts = result.get("transition_state")
    if ts is not None:
        assert ts.calc is None

    # The same metadata must be recorded when the flat kwargs arrive bundled in
    # a single NebRunConfig (the serial runner's call path).
    from tests.ts_search.test_parallel_neb import _gas_neb_cfg

    cfg = _gas_neb_cfg(
        neb_n_images=3,
        neb_fmax=0.1,
        neb_steps=20,
        neb_align_endpoints=True,
        neb_perturb_sigma=0.03,
    )
    result_cfg = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="meta_cfg_test",
        rng=np.random.default_rng(1),
        neb_cfg=cfg,
    )
    assert result_cfg.get("align_endpoints") is True
    assert result_cfg.get("perturb_sigma") == pytest.approx(0.03)
    ts_cfg = result_cfg.get("transition_state")
    if ts_cfg is not None:
        assert ts_cfg.calc is None


def test_find_ts_endpoint_marked_not_converged(temp_output_dir, h2_reactant):
    """Identical endpoints raise a structured failure (no interior saddle)."""
    a = h2_reactant.copy()
    b = h2_reactant.copy()
    a.calc = EMT()
    b.calc = EMT()

    result = find_transition_state(
        a,
        b,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="endpoint_test",
        n_images=3,
        fmax=0.05,
        neb_steps=50,
        verbosity=0,
    )

    assert result["neb_converged"] is False
    assert result["status"] == "skipped"
    error = result.get("error")
    assert error is not None
    assert "endpoint" in error.lower() or "identical" in error.lower()


def test_interpolate_path_different_lengths_fails():
    """Test that interpolation fails with different atom counts."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)

    with pytest.raises((ValueError, SCGOValidationError), match="different lengths"):
        interpolate_path(atoms1, atoms2, n_images=3)


def test_calculate_similarity_basic():
    """Test similarity comparison wrapper (comparator logic tested elsewhere)."""
    # Identical structures
    atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2 = atoms1.copy()
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms2)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True

    # Permuted (swapped atoms) - should still be identical
    atoms3 = Atoms("Cu2", positions=[[2.5, 0, 0], [0, 0, 0]])
    atoms3.center(vacuum=5.0)
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms3)
    assert cum_diff < 0.001
    assert are_similar is True

    # Different structures
    atoms4 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])
    atoms4.center(vacuum=5.0)
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms4)
    assert cum_diff > 0.01
    # are_similar may be implementation-dependent; ensure the difference is large enough to indicate dissimilarity


def test_calculate_similarity_ignores_fixed_slab_atoms():
    """Fixed slab atoms should not affect similarity metrics."""
    slab_mobile = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0], [0.6, 0.4, 1.8]]
    atoms1 = Atoms("Pt4", positions=slab_mobile)
    atoms2 = atoms1.copy()
    atoms1.set_constraint(FixAtoms(indices=[0, 1, 2]))
    atoms2.set_constraint(FixAtoms(indices=[0, 1, 2]))
    # Move only frozen slab atoms in atoms2
    pos2 = atoms2.get_positions()
    pos2[:3, 0] += 0.4
    atoms2.set_positions(pos2)

    cum_diff, _max_diff, are_similar = calculate_structure_similarity(atoms1, atoms2)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True


def test_calculate_similarity_uses_mic_for_periodic_surfaces():
    """MIC-aware similarity should treat periodic translations as equivalent.

    ``PureInteratomicDistanceComparator`` honors ``mic`` literally even when the
    cell has PBC, so ``use_mic=False`` must not fold images across the boundary.
    """
    cell = [8.0, 8.0, 12.0]
    a1 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [7.90, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    a2 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    _, _, no_mic_flag_similar = calculate_structure_similarity(a1, a2, use_mic=False)
    _, _, mic_similar = calculate_structure_similarity(a1, a2, use_mic=True)
    assert bool(no_mic_flag_similar) is False
    assert bool(mic_similar) is True


def test_calculate_similarity_uses_adsorbate_slice_when_n_slab_from_surface_config():
    """Explicit n_slab (from SurfaceSystemConfig) scopes comparison without metadata."""
    atoms1 = Atoms(
        "Pt4OH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.6, 0.4, 2.0],
            [0.6, 0.4, 2.9],
        ],
    )
    atoms2 = atoms1.copy()
    pos2 = atoms2.get_positions()
    pos2[:4, 0] += 0.7  # slab-only displacement
    atoms2.set_positions(pos2)

    cum_diff, _, are_similar = calculate_structure_similarity(atoms1, atoms2, n_slab=4)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True


def test_calculate_similarity_uses_adsorbate_slice_when_surface_metadata_present():
    """When slab constraints are absent, n_slab metadata should scope comparison."""
    atoms1 = Atoms(
        "Pt4OH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.6, 0.4, 2.0],
            [0.6, 0.4, 2.9],
        ],
    )
    atoms2 = atoms1.copy()
    atoms1.info.setdefault("key_value_pairs", {})["n_slab_atoms"] = 4
    atoms2.info.setdefault("key_value_pairs", {})["n_slab_atoms"] = 4
    pos2 = atoms2.get_positions()
    pos2[:4, 0] += 0.7  # slab-only displacement
    atoms2.set_positions(pos2)

    cum_diff, _, are_similar = calculate_structure_similarity(atoms1, atoms2)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True


def test_select_structure_pairs_ignores_slab_when_n_slab_from_surface_config():
    """Pair selection uses n_slab from surface_config, not FixAtoms on loaded minima."""
    base = Atoms(
        "Pt5",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.6, 0.4, 1.8],
            [1.8, 0.4, 1.8],
        ],
    )
    slab_shifted = base.copy()
    shifted_pos = slab_shifted.get_positions()
    shifted_pos[:3, 1] += 0.35
    slab_shifted.set_positions(shifted_pos)
    mobile_changed = base.copy()
    mobile_pos = mobile_changed.get_positions()
    mobile_pos[4, 0] += 0.8
    mobile_changed.set_positions(mobile_pos)

    minima = [(-1.0, base), (-0.95, slab_shifted), (-0.90, mobile_changed)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=None,
        similarity_tolerance=0.01,
        similarity_pair_cor_max=0.2,
        use_mic=False,
        n_slab=3,
    )

    assert (0, 1) not in pairs
    assert (0, 2) in pairs


def test_select_structure_pairs_ignores_fixed_slab_atom_differences():
    """Pair filtering should reject endpoint pairs that differ only in frozen slab atoms."""
    base = Atoms(
        "Pt5",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.6, 0.4, 1.8],
            [1.8, 0.4, 1.8],
        ],
    )
    base.set_constraint(FixAtoms(indices=[0, 1, 2]))
    slab_shifted = base.copy()
    slab_shifted.set_constraint(FixAtoms(indices=[0, 1, 2]))
    shifted_pos = slab_shifted.get_positions()
    shifted_pos[:3, 1] += 0.35
    slab_shifted.set_positions(shifted_pos)
    mobile_changed = base.copy()
    mobile_pos = mobile_changed.get_positions()
    mobile_pos[4, 0] += 0.8
    mobile_changed.set_positions(mobile_pos)

    minima = [(-1.0, base), (-0.95, slab_shifted), (-0.90, mobile_changed)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=None,
        similarity_tolerance=0.01,
        similarity_pair_cor_max=0.2,
        use_mic=False,
    )

    assert (0, 1) not in pairs
    assert (0, 2) in pairs


@pytest.mark.slow
def test_find_ts_simple(h2_reactant, h2_product, temp_output_dir):
    """Test basic TS finding with EMT calculator."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="test_h2",
        n_images=3,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
        # Retry bumps n_images to >=5; keep a single attempt so result matches n_images=3.
    )

    # Accept either a successful NEB or a correctly-detected endpoint case
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert result["transition_state"] is not None
        assert result["ts_energy"] is not None
        assert result["barrier_height"] is not None
    else:
        # NEB may report the endpoint as the highest-energy image and be
        # intentionally rejected by the endpoint-detection safeguard.
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()

    assert result["pair_id"] == "test_h2"
    assert result["n_images"] == 3


@pytest.mark.slow
def test_find_ts_with_climb(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with climbing image NEB."""
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_climb",
        n_images=5,
        spring_constant=0.05,
        climb=True,
        fmax=0.1,
        neb_steps=200,
        verbosity=0,
    )

    assert result["climb"] is True
    # With the staged-retry defaults enabled, this should converge for Cu3
    assert result["status"] == "success"
    assert result["neb_converged"] is True


def test_find_transition_state_endpoint_failure_cu3(
    cu3_triangle, cu3_linear, temp_output_dir
):
    """NEB with conservative settings may report an endpoint TS; verify graceful failure."""
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_endpoint",
        n_images=3,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
        rng=np.random.default_rng(0),
    )

    # Result should be either success (interior TS) or failed (endpoint TS)
    assert result["status"] in ("success", "failed")
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert result.get("ts_image_index") not in (0, result.get("n_images") + 1)
    else:
        assert result.get("error") is not None


@pytest.mark.slow
def test_find_ts_linear_interpolation(h2_reactant, h2_product, temp_output_dir):
    """Test TS finding with linear interpolation instead of IDPP."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="linear_test",
        n_images=3,
        interpolation_method="linear",
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
    )

    # NEB may either succeed or be rejected because the highest-energy
    # image is an endpoint — accept both behaviors.
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert result["transition_state"] is not None
    else:
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()


@pytest.mark.slow
def test_find_ts_saves_trajectory(h2_reactant, h2_product, temp_output_dir):
    """Test that NEB trajectory is saved."""
    traj_path = os.path.join(temp_output_dir, "custom_neb.traj")

    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="traj_test",
        n_images=3,
        trajectory=traj_path,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
    )

    # Accept either success or a detected endpoint-TS failure; trajectory
    # should still be written by the optimizer run in either case.
    if result["status"] == "success":
        assert result["neb_converged"] is True
    else:
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()

    assert os.path.exists(traj_path)


def test_find_transition_state_defaults_reflect_promoted_retry(
    h2_reactant, h2_product, temp_output_dir
):
    """Defaults align with tuned TS presets (spring k=0.1, climb False, etc.)."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="defaults_test",
        n_images=3,
        fmax=0.1,
        neb_steps=20,
        verbosity=0,
        # Retry promotes climb=True on the second attempt; final `result` then
        # echoes retry params, which is environment-dependent and flaky in CI.
    )

    assert result.get("spring_constant") == pytest.approx(0.1)
    assert result.get("climb") is False
    assert result.get("perturb_sigma") == pytest.approx(0.0)


def test_find_ts_different_lengths_fails(temp_output_dir):
    """Test TS finding fails with different atom counts."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)
    atoms1.calc = EMT()
    atoms2.calc = EMT()

    with pytest.raises(SCGOValidationError, match="different lengths"):
        find_transition_state(
            atoms1,
            atoms2,
            calculator=EMT(),
            output_dir=temp_output_dir,
            pair_id="fail_test",
        )


def test_find_ts_no_calculator_fails(h2_reactant, h2_product, temp_output_dir):
    """Test that TS finding fails without calculator when use_torchsim=False."""
    # Remove calculators
    h2_reactant.calc = None
    h2_product.calc = None

    with pytest.raises(
        SCGOValidationError, match="must have a calculator attached"
    ):  # Should fail validation without calculator
        find_transition_state(
            h2_reactant,
            h2_product,
            calculator=None,
            output_dir=temp_output_dir,
            pair_id="no_calc",
            use_torchsim=False,
        )


def test_save_neb_result_success(temp_output_dir, default_rel_tol):
    """Test saving successful NEB result."""
    # Create mock TS result
    ts_atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
    ts_atoms.center(vacuum=5.0)

    r_atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    r_atoms.center(vacuum=4.0)
    p_atoms = Atoms("H2", positions=[[0, 0, 0], [1.4, 0, 0]])
    p_atoms.center(vacuum=4.0)

    result = {
        "status": "success",
        "pair_id": "0_1",
        "neb_converged": True,
        "n_images": 5,
        "spring_constant": 0.1,
        "reactant_energy": -1.0,
        "product_energy": -0.8,
        "ts_energy": -0.5,
        "barrier_height": 0.5,
        "transition_state": ts_atoms,
        "ts_image_index": 3,
        "error": None,
        "reactant_structure": r_atoms,
        "product_structure": p_atoms,
        "use_torchsim": False,
        "fmax": 0.05,
        "neb_steps": 100,
        "interpolation_method": "idpp",
    }

    save_neb_result(result, temp_output_dir, "0_1")

    # Check files created
    assert os.path.exists(os.path.join(temp_output_dir, "ts_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "reactant_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "product_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "neb_0_1_metadata.json"))

    with open(os.path.join(temp_output_dir, "neb_0_1_metadata.json")) as f:
        metadata = json.load(f)

    assert metadata["status"] == "success"
    assert metadata["schema_version"] == OUTPUT_JSON_SCHEMA_VERSION
    assert metadata["scgo_version"] != "unknown"
    assert "created_at" in metadata
    assert metadata["barrier_height"] == pytest.approx(0.5, rel=default_rel_tol)
    assert metadata["spring_constant"] == pytest.approx(0.1, rel=default_rel_tol)
    assert metadata["ts_image_index"] == 3
    assert "steps_taken" in metadata
    assert metadata.get("neb_backend") == "ase"
    loaded = load_completed_neb_result(temp_output_dir, "0_1")
    assert loaded is not None
    assert loaded["status"] == "success"
    assert loaded["resumed"] is True
    assert loaded["transition_state"] is not None
    assert not any(name.startswith(".tmp_neb_") for name in os.listdir(temp_output_dir))


def test_save_neb_result_failed(temp_output_dir):
    """Test saving failed NEB result."""
    result = {
        "status": "failed",
        "pair_id": "1_2",
        "neb_converged": False,
        "n_images": 5,
        "spring_constant": 0.1,
        "reactant_energy": -1.0,
        "product_energy": -0.8,
        "ts_energy": None,
        "barrier_height": None,
        "transition_state": None,
        "error": "Test error",
    }

    save_neb_result(result, temp_output_dir, "1_2")

    # Metadata should be saved even for failed runs
    meta_path = os.path.join(temp_output_dir, "neb_1_2_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        failed_meta = json.load(f)
    assert failed_meta["schema_version"] == OUTPUT_JSON_SCHEMA_VERSION
    assert "created_at" in failed_meta

    # TS structure should not be saved for failed runs
    assert not os.path.exists(os.path.join(temp_output_dir, "ts_1_2.xyz"))
    assert load_completed_neb_result(temp_output_dir, "1_2") is None

    corrupt_path = os.path.join(temp_output_dir, "neb_3_4_metadata.json")
    with open(corrupt_path, "w") as f:
        f.write('{"status": "success", "pair_id": "3_4"')
    assert load_completed_neb_result(temp_output_dir, "3_4") is None


def test_load_completed_neb_result_requires_converged_and_ts_xyz(temp_output_dir):
    """Resume rejects success-without-convergence or missing TS geometry."""
    ts_atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
    ts_atoms.center(vacuum=5.0)
    base = {
        "status": "success",
        "pair_id": "0_1",
        "neb_converged": False,
        "n_images": 5,
        "spring_constant": 0.1,
        "reactant_energy": -1.0,
        "product_energy": -0.8,
        "ts_energy": -0.5,
        "barrier_height": 0.5,
        "transition_state": ts_atoms,
        "ts_image_index": 3,
        "error": None,
        "use_torchsim": False,
        "fmax": 0.05,
        "neb_steps": 100,
        "interpolation_method": "idpp",
    }
    save_neb_result(base, temp_output_dir, "0_1")
    assert load_completed_neb_result(temp_output_dir, "0_1") is None

    converged = dict(base)
    converged["neb_converged"] = True
    converged["pair_id"] = "2_3"
    save_neb_result(converged, temp_output_dir, "2_3")
    os.remove(os.path.join(temp_output_dir, "ts_2_3.xyz"))
    assert load_completed_neb_result(temp_output_dir, "2_3") is None

    converged["pair_id"] = "5_6"
    save_neb_result(converged, temp_output_dir, "5_6")
    meta_path = os.path.join(temp_output_dir, "neb_5_6_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.pop("neb_converged", None)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    assert load_completed_neb_result(temp_output_dir, "5_6") is None


def test_serial_resume_skips_completed_pair(tmp_path, monkeypatch):
    """Completed success metadata under run_dir skips find_transition_state."""
    atoms_a = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms_a.center(vacuum=5.0)
    atoms_b = atoms_a.copy()
    atoms_b.positions[1, 0] += 0.3
    minima = [(-1.0, atoms_a), (-0.9, atoms_b)]
    run_dir = tmp_path / "run_resume"
    pair_dir = run_dir / "pair_0_1"
    pair_dir.mkdir(parents=True)
    save_neb_result(
        {
            "status": "success",
            "pair_id": "0_1",
            "neb_converged": True,
            "n_images": 5,
            "spring_constant": 0.1,
            "reactant_energy": -1.0,
            "product_energy": -0.9,
            "ts_energy": -0.5,
            "barrier_height": 0.5,
            "transition_state": atoms_a.copy(),
            "ts_image_index": 2,
            "error": None,
            "use_torchsim": False,
            "fmax": 0.05,
            "neb_steps": 10,
            "interpolation_method": "idpp",
            "climb": False,
            "align_endpoints": True,
            "perturb_sigma": 0.0,
            "neb_interpolation_mic": False,
            "neb_tangent_method": "improvedtangent",
        },
        str(pair_dir),
        "0_1",
    )

    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("find_transition_state should not run on resume")

    monkeypatch.setattr(ts_run_mod, "find_transition_state", _boom)
    neb_cfg = NebRunConfig(
        neb_n_images=5,
        neb_spring_constant=0.1,
        neb_fmax=0.05,
        neb_steps=10,
        neb_climb=False,
        neb_interpolation_method="idpp",
        neb_align_endpoints=True,
        neb_perturb_sigma=0.0,
        neb_interpolation_mic=False,
        neb_tangent_method="improvedtangent",
        neb_surface_cell_remap=True,
        neb_surface_lattice_rotation=True,
        neb_surface_max_lattice_shift=1,
        n_slab=0,
        n_core_mobile=None,
        n_adsorbate_mobile=None,
        adsorbate_fragment_lengths=None,
        max_endpoint_mismatch=None,
        neb_prescreen_clash_distance=1.0,
        min_saddle_prominence=0.10,
        neb_max_spurious_barrier=8.0,
        layer_cluster_threshold_ang=0.8,
        neb_interpolation_bond_tolerance_a=0.0,
        adsorbate_definition=None,
        connectivity_factor=None,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=False,
        enforce_adsorbate_subgraph_integrity=True,
        system_type="gas_cluster",
        surface_config=None,
        torchsim_params={},
    )
    results = ts_run_mod._run_serial_neb_search(
        [(0, 1)],
        minima,
        neb_cfg=neb_cfg,
        run_dir=run_dir,
        calculator_class=EMT,
        calculator_kwargs={},
        rng=None,
        use_torchsim=False,
        verbosity=0,
    )
    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0].get("resumed") is True
    assert calls["n"] == 0


def test_adsorbate_pair_select_cap_bounds_oversample() -> None:
    assert adsorbate_pair_select_cap(3) == 30
    assert adsorbate_pair_select_cap(10) == 50
    assert adsorbate_pair_select_cap(60) == 60


@pytest.mark.parametrize("system_type", sorted(SYSTEM_TYPE_POLICIES))
def test_resolve_ts_pair_select_cap_for_all_system_types(system_type) -> None:
    """Preset mismatch must not inflate the NEB budget on bare system types."""
    policy = get_system_policy(system_type)
    mismatch = get_ts_defaults(system_type)["max_endpoint_mismatch"]
    max_pairs = 6
    cap = resolve_ts_pair_select_cap(
        max_pairs,
        has_adsorbate=policy.has_adsorbate,
        max_endpoint_mismatch=mismatch,
    )
    if policy.has_adsorbate and mismatch is not None:
        assert cap == adsorbate_pair_select_cap(max_pairs)
    else:
        assert cap == max_pairs


def test_select_structure_pairs_basic():
    """Test basic pair selection."""
    # Create mock minima
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])  # Different geometry
    atoms3 = Atoms("H2", positions=[[0, 0, 0], [3, 0, 0]])

    minima = [
        (-1.0, atoms1),
        (-0.9, atoms2),
        (-0.7, atoms3),
    ]

    pairs = select_structure_pairs(
        minima, max_pairs=None, similarity_tolerance=0.1, use_mic=False
    )

    # Should get all unique pairs: (0,1), (0,2), (1,2)
    assert len(pairs) == 3
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_select_structure_pairs_max_limit():
    """Test pair selection with max_pairs limit."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    minima = [(float(-i), atoms.copy()) for i in range(10)]

    pairs = select_structure_pairs(
        minima, max_pairs=5, similarity_tolerance=0.01, use_mic=False
    )

    assert len(pairs) <= 5


def test_select_structure_pairs_energy_gap():
    """Test pair selection with energy gap threshold."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])
    atoms3 = Atoms("H2", positions=[[0, 0, 0], [3, 0, 0]])

    minima = [
        (-1.0, atoms1),
        (-0.95, atoms2),  # Small gap
        (-0.2, atoms3),  # Large gap from first two
    ]

    # Only allow pairs within 0.1 eV
    pairs = select_structure_pairs(
        minima, energy_gap_threshold=0.1, similarity_tolerance=0.01, use_mic=False
    )

    # Should only get (0,1) since gap to 2 is too large
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_select_structure_pairs_physics_ranking_when_capped(monkeypatch):
    """Capped pair lists use physics-guided ranking (score + stable tie-break)."""
    atoms0 = Atoms("H2", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    atoms1 = Atoms("H2", positions=[[1.0, 0, 0], [1.5, 0, 0]])
    atoms2 = Atoms("H2", positions=[[2.0, 0, 0], [2.5, 0, 0]])
    minima = [(-1.0, atoms0), (-0.95, atoms1), (-0.55, atoms2)]

    def _fake_similarity(
        a_i: Atoms,
        a_j: Atoms,
        tolerance: float = 0.1,
        pair_cor_max: float = 0.1,
        use_mic: bool = False,
        **kwargs: object,
    ) -> tuple[float, float, bool]:
        pair = tuple(
            sorted(
                (
                    int(round(a_i.get_positions()[0, 0])),
                    int(round(a_j.get_positions()[0, 0])),
                )
            )
        )
        table = {
            (0, 1): (0.02, 0.10, False),
            (0, 2): (0.20, 0.20, False),
            (1, 2): (0.25, 0.15, False),
        }
        return table[pair]

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_io.calculate_structure_similarity",
        _fake_similarity,
    )

    ranked = select_structure_pairs(minima, max_pairs=2, use_mic=False)
    assert ranked == [(1, 2), (0, 2)]


def test_select_structure_pairs_adsorbate_prefers_site_hop_over_slide():
    """Identical cores: prefer a real OH site hop over a near-duplicate slide."""
    core = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
    atoms0 = Atoms(
        "Pt2OH",
        positions=[*core, [1.2, 0.0, 1.5], [1.2, 0.0, 2.5]],
    )
    atoms_slide = Atoms(
        "Pt2OH",
        positions=[*core, [1.25, 0.0, 1.5], [1.25, 0.0, 2.5]],
    )
    atoms_hop = Atoms(
        "Pt2OH",
        positions=[*core, [1.8, 0.0, 1.6], [1.9, 0.0, 2.5]],
    )
    minima = [(-1.0, atoms0), (-0.55, atoms_slide), (-0.50, atoms_hop)]

    ranked = select_structure_pairs(
        minima,
        max_pairs=1,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=2,
        max_endpoint_mismatch=1.25,
    )
    assert ranked == [(0, 2)]


def test_select_structure_pairs_max_endpoint_mismatch_hard_gate(monkeypatch):
    """Pairs with comparator max_diff above the gate are dropped."""
    atoms0 = Atoms("H2", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    atoms1 = Atoms("H2", positions=[[0.2, 0, 0], [1.2, 0, 0]])
    atoms2 = Atoms("H2", positions=[[3.0, 0, 0], [4.0, 0, 0]])
    minima = [(-1.0, atoms0), (-0.95, atoms1), (-0.90, atoms2)]

    def _fake_similarity(
        a_i: Atoms,
        a_j: Atoms,
        tolerance: float = 0.1,
        pair_cor_max: float = 0.1,
        use_mic: bool = False,
        **kwargs: object,
    ) -> tuple[float, float, bool]:
        xi = float(a_i.get_positions()[0, 0])
        xj = float(a_j.get_positions()[0, 0])
        pair = tuple(sorted((xi, xj)))
        table = {
            (0.0, 0.2): (0.05, 0.4, False),
            (0.0, 3.0): (0.20, 2.5, False),
            (0.2, 3.0): (0.18, 2.2, False),
        }
        return table[pair]

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_io.calculate_structure_similarity",
        _fake_similarity,
    )
    pairs = select_structure_pairs(minima, max_endpoint_mismatch=1.25, use_mic=False)
    assert pairs == [(0, 1)]


def _cycle_axes_rotation() -> np.ndarray:
    """Proper rotation that cycles x→y→z (not a symmetry of the scalene Pt3)."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


def _asymmetric_pt3_core() -> Atoms:
    """Scalene Pt3 with an out-of-plane vertex (no rotational symmetry)."""
    core = Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.70, 0.0, 0.0],
            [0.85, 2.15, 0.45],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    core.positions -= core.positions.mean(axis=0)
    return core


def _pt5_tbp_core() -> Atoms:
    """Trigonal-bipyramid Pt5 with ~2.7 Å nearest-neighbor spacing."""
    r = 2.70
    z = r * np.sqrt(6.0) / 3.0
    core = Atoms(
        "Pt5",
        positions=[
            [0.0, 0.0, z],
            [0.0, 0.0, -z],
            [r, 0.0, 0.0],
            [-r / 2.0, r * np.sqrt(3.0) / 2.0, 0.0],
            [-r / 2.0, -r * np.sqrt(3.0) / 2.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    core.positions -= core.positions.mean(axis=0)
    return core


def _pt5_square_pyramid_core() -> Atoms:
    """Square-pyramid Pt5 isomer (distinct from the TBP core)."""
    r = 2.70
    core = Atoms(
        "Pt5",
        positions=[
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [r, r, 0.0],
            [0.0, r, 0.0],
            [r / 2.0, r / 2.0, r * 0.85],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    core.positions -= core.positions.mean(axis=0)
    return core


def _attach_oh(core: Atoms, o_pos: np.ndarray, oh_bond: float = 0.96) -> Atoms:
    """Concatenate OH with H along +z from O onto a core copy."""
    h_pos = np.asarray(o_pos, dtype=float) + np.array([0.0, 0.0, oh_bond])
    oh = Atoms("OH", positions=[o_pos, h_pos], cell=core.cell, pbc=False)
    return core + oh


def test_core_rms_displacement_is_permutation_invariant() -> None:
    """Same-element core reorder must not inflate core RMS after matching."""
    atoms_i = Atoms(
        "Pt2OH",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 0.0, 1.5], [1.2, 0.0, 2.5]],
    )
    atoms_j = Atoms(
        "Pt2OH",
        positions=[[2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.3, 0.0, 1.5], [1.3, 0.0, 2.5]],
    )
    rms = _core_rms_displacement(atoms_i, atoms_j, n_slab=0, n_core=2, use_mic=False)
    assert rms < 0.05


def test_core_rms_displacement_asymmetric_gas_core_is_rotation_invariant() -> None:
    """Low-symmetry cores overlay after fingerprint correspondence + Kabsch."""
    core = _asymmetric_pt3_core()
    atoms_i = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
    atoms_j = atoms_i.copy()
    atoms_j.positions = atoms_j.positions @ _cycle_axes_rotation().T
    rms = _core_rms_displacement(atoms_i, atoms_j, n_slab=0, n_core=3, use_mic=False)
    assert rms < 0.05


def test_core_rms_and_hop_after_translated_gas_pt_oh() -> None:
    """One-atom gas core overlays by translation; hop is adsorbate travel about Pt."""
    atoms_i = Atoms(
        "PtOH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.0, 2.76]],
    )
    atoms_j = atoms_i.copy()
    atoms_j.positions[1] = [1.8, 0.0, 0.0]
    atoms_j.positions[2] = [1.8, 0.0, 0.96]
    atoms_j.positions += np.array([3.0, 1.2, -0.5])
    rms = _core_rms_displacement(atoms_i, atoms_j, n_slab=0, n_core=1, use_mic=False)
    hop = _adsorbate_max_displacement(
        atoms_i, atoms_j, n_slab=0, n_core=1, use_mic=False
    )
    assert rms < 0.05
    assert hop == pytest.approx(np.linalg.norm([1.8, 0.0, -1.8]), abs=0.05)


def _block_rms(a: Atoms, b: Atoms, *, n_slab: int, n_core: int) -> float:
    i0, i1 = n_slab, n_slab + n_core
    dlt = b.get_positions()[i0:i1] - a.get_positions()[i0:i1]
    return float(np.sqrt(np.mean(np.sum(dlt * dlt, axis=1))))


def _with_reflected_tbp_labels(atoms: Atoms) -> Atoms:
    """Swap TBP equatorials (indices 3, 4) and apply a proper rotation."""
    out = atoms.copy()
    pos = out.positions.copy()
    pos[3], pos[4] = pos[4].copy(), pos[3].copy()
    out.positions = pos @ _cycle_axes_rotation().T
    return out


def _slab_rotated_core_oh_pair() -> tuple[Atoms, Atoms]:
    slab = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    core = _asymmetric_pt3_core()
    core.positions += np.array([0.0, 0.0, 2.0])
    atoms_i = slab + _attach_oh(core, core.positions[0] + np.array([0.0, 0.0, 1.8]))
    rot_x90 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    )
    atoms_j = atoms_i.copy()
    mobile = atoms_j.positions[2:]
    com = mobile[:3].mean(axis=0)
    atoms_j.positions[2:] = (mobile - com) @ rot_x90.T + com
    return atoms_i, atoms_j


def test_core_rms_displacement_slab_keeps_lab_frame_rotation() -> None:
    """Supported cores are not 3D-Kabsch'd; lab-frame rotation stays large."""
    atoms_i, atoms_j = _slab_rotated_core_oh_pair()
    rms = _core_rms_displacement(atoms_i, atoms_j, n_slab=2, n_core=3, use_mic=False)
    assert rms > 1.0


def test_select_structure_pairs_keeps_rotated_asymmetric_core_oh_site_hop() -> None:
    """Rotated scalene cores with an OH site hop must remain pairable."""
    core = _asymmetric_pt3_core()
    site_a = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
    site_b = _attach_oh(core.copy(), core.positions[1] + np.array([1.8, 0.0, 0.0]))
    site_b = site_b.copy()
    site_b.positions = site_b.positions @ _cycle_axes_rotation().T
    pairs = select_structure_pairs(
        [(-10.0, site_a), (-9.7, site_b)],
        max_pairs=1,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=3,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_keeps_permuted_core_adsorbate_hop() -> None:
    """Permuted identical cores should pass the core-RMS gate and remain selectable."""
    atoms0 = Atoms(
        "Pt2OH",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 0.0, 1.5], [1.2, 0.0, 2.5]],
    )
    atoms1 = Atoms(
        "Pt2OH",
        positions=[[2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.8, 0.0, 1.6], [1.9, 0.0, 2.5]],
    )
    minima = [(-1.0, atoms0), (-0.5, atoms1)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=1,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=2,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_keeps_same_core_oh_site_hop() -> None:
    """Identical Pt5 core with OH on axial vs equatorial sites must remain pairable.

    Whole-structure fingerprinting treated these as identical (O/H add no pairs)
    and emptied the adsorbate NEB pool; core-only fingerprinting keeps them.
    """
    core = _pt5_tbp_core()
    axial = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
    equatorial = _attach_oh(core.copy(), core.positions[2] + np.array([1.8, 0.0, 0.0]))
    minima = [(-260.0, axial), (-259.7, equatorial)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=1,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=5,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_keeps_rotated_same_core_oh_site_hop() -> None:
    """Rotated identical Pt5 cores with an OH site hop must remain pairable.

    Without Kabsch on the core-RMS gate, a rigid rotation inflates Cartesian RMS
    past ``pair_core_rms_max`` and empties the gas-adsorbate NEB pool.
    """
    core = _pt5_tbp_core()
    axial = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
    equatorial = _attach_oh(core.copy(), core.positions[2] + np.array([1.8, 0.0, 0.0]))
    equatorial = equatorial.copy()
    equatorial.positions = equatorial.positions @ _cycle_axes_rotation().T
    minima = [(-260.0, axial), (-259.7, equatorial)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=1,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=5,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_reflected_tbp_labels_pass_gas_pair_gate_and_match_neb_overlay() -> None:
    """Reflected TBP labels overlay below the gas RMS gate; pairing matches NEB.

    Fingerprint Hungarian can assign a reflected equatorial labeling; without
    spatial rematch, proper Kabsch leaves RMS ~2.8 Å and empties the pair pool.
    """
    core = _pt5_tbp_core()
    axial = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
    equatorial = _with_reflected_tbp_labels(
        _attach_oh(core.copy(), core.positions[2] + np.array([1.8, 0.0, 0.0]))
    )
    rms = _core_rms_displacement(axial, equatorial, n_slab=0, n_core=5, use_mic=False)
    assert rms < 0.05
    assert rms < DEFAULT_PAIR_CORE_RMS_MAX_GAS
    pairs = select_structure_pairs(
        [(-260.0, axial), (-259.7, equatorial)],
        max_pairs=1,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=5,
        max_endpoint_mismatch=1.25,
        pair_core_rms_max=DEFAULT_PAIR_CORE_RMS_MAX_GAS,
    )
    assert pairs == [(0, 1)]

    images = interpolate_path(
        axial,
        equatorial,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=5,
        n_adsorbate_mobile=2,
    )
    neb_rms = _block_rms(images[0], images[-1], n_slab=0, n_core=5)
    assert neb_rms < 0.05
    assert neb_rms == pytest.approx(rms, abs=1e-6)

    pos_overlay, _nums = _overlay_product_core(
        axial,
        equatorial.get_positions(),
        np.asarray(equatorial.numbers, dtype=int),
        n_slab=0,
        n_core=5,
        mic_cell=None,
        mic_pbc=None,
    )
    np.testing.assert_allclose(
        pos_overlay[:5], images[-1].get_positions()[:5], atol=1e-6
    )


def test_reflected_tbp_labels_overlay_for_bare_gas_cluster() -> None:
    """Bare gas_cluster NEB overlay rematches reflected TBP labels too."""
    react = _pt5_tbp_core()
    prod = _with_reflected_tbp_labels(react)
    images = interpolate_path(
        react,
        prod,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster",
        n_slab=0,
        n_core_mobile=5,
    )
    assert _block_rms(images[0], images[-1], n_slab=0, n_core=5) < 0.05


def test_overlay_product_core_is_noop_without_core() -> None:
    """surface / surface_adsorbate (n_core=0) must not permute or Kabsch."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    shifted = atoms.get_positions() + np.array([1.5, -0.4, 0.2])
    pos, nums = _overlay_product_core(
        atoms,
        shifted,
        np.asarray(atoms.numbers, dtype=int),
        n_slab=0,
        n_core=0,
        mic_cell=None,
        mic_pbc=None,
    )
    np.testing.assert_allclose(pos, shifted)
    np.testing.assert_array_equal(nums, atoms.numbers)


@pytest.mark.parametrize(
    ("surface", "scale", "limit", "expect_pair"),
    [
        (False, 1.4, DEFAULT_PAIR_CORE_RMS_MAX_GAS, True),
        (False, 1.8, DEFAULT_PAIR_CORE_RMS_MAX_GAS, False),
        (True, 1.4, DEFAULT_PAIR_CORE_RMS_MAX_SURFACE, True),
        (True, 1.8, DEFAULT_PAIR_CORE_RMS_MAX_SURFACE, False),
    ],
)
def test_pair_core_rms_max_defaults_are_hard_gates(
    surface: bool, scale: float, limit: float, expect_pair: bool
) -> None:
    """Uniform core breathing is kept or dropped by the gas 1.5 / surface 2.0 gates."""
    core = _pt5_tbp_core()
    if surface:
        slab = Atoms(
            "C2",
            positions=[[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=[True, True, False],
        )
        core = core.copy()
        core.positions += np.array([0.0, 0.0, 3.0])
        atoms_i = slab + _attach_oh(core, core.positions[0] + np.array([0.0, 0.0, 1.8]))
        n_slab, n_core = 2, 5
    else:
        atoms_i = _attach_oh(core.copy(), core.positions[0] + np.array([0.0, 0.0, 1.8]))
        n_slab, n_core = 0, 5
    atoms_j = atoms_i.copy()
    i0, i1 = n_slab, n_slab + n_core
    com = atoms_j.positions[i0:i1].mean(axis=0)
    atoms_j.positions[i0:i1] = com + (atoms_j.positions[i0:i1] - com) * scale
    rms = _core_rms_displacement(
        atoms_i, atoms_j, n_slab=n_slab, n_core=n_core, use_mic=False
    )
    if expect_pair:
        assert rms < limit
    else:
        assert rms > limit
    pairs = select_structure_pairs(
        [(-260.0, atoms_i), (-259.7, atoms_j)],
        max_pairs=1,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=n_core,
        n_slab=n_slab if n_slab else None,
        max_endpoint_mismatch=10.0,
        pair_core_rms_max=limit,
        surface_aware=surface,
    )
    assert (pairs == [(0, 1)]) is expect_pair


def test_interpolate_path_surface_cluster_adsorbate_keeps_lab_frame_core() -> None:
    """NEB must not 3D-Kabsch a supported core the way gas overlay does."""
    atoms_i, atoms_j = _slab_rotated_core_oh_pair()
    images = interpolate_path(
        atoms_i,
        atoms_j,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="surface_cluster_adsorbate",
        n_slab=2,
        n_core_mobile=3,
        n_adsorbate_mobile=2,
    )
    assert _block_rms(images[0], images[-1], n_slab=2, n_core=3) > 1.0


def test_select_structure_pairs_skips_distinct_core_isomers() -> None:
    """TBP vs square-pyramid cores still fail max_endpoint_mismatch and/or core RMS."""
    tbp = _attach_oh(
        _pt5_tbp_core(),
        _pt5_tbp_core().positions[0] + np.array([0.0, 0.0, 1.8]),
    )
    sq = _attach_oh(
        _pt5_square_pyramid_core(),
        _pt5_square_pyramid_core().positions[4] + np.array([0.0, 0.0, 1.8]),
    )
    minima = [(-260.0, tbp), (-259.8, sq)]
    pairs = select_structure_pairs(
        minima,
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=5,
        max_endpoint_mismatch=1.25,
        pair_core_rms_max=DEFAULT_PAIR_CORE_RMS_MAX_GAS,
    )
    assert pairs == []


def test_select_structure_pairs_core_only_ignores_oo_fingerprint() -> None:
    """Two OH on the same core must not be rejected via O–O fingerprint max_diff."""
    core = _pt5_tbp_core()
    # Same Pt5; two OH fragments at different relative sites (layout: Pt5 + OHOH).
    oh_a = Atoms(
        "OHOH",
        positions=[
            core.positions[0] + [0.0, 0.0, 1.8],
            core.positions[0] + [0.0, 0.0, 2.76],
            core.positions[2] + [1.8, 0.0, 0.0],
            core.positions[2] + [2.76, 0.0, 0.0],
        ],
        cell=core.cell,
        pbc=False,
    )
    oh_b = Atoms(
        "OHOH",
        positions=[
            core.positions[1] + [0.0, 0.0, -1.8],
            core.positions[1] + [0.0, 0.0, -2.76],
            core.positions[3] + [-1.8, 0.0, 0.0],
            core.positions[3] + [-2.76, 0.0, 0.0],
        ],
        cell=core.cell,
        pbc=False,
    )
    atoms0 = core.copy() + oh_a
    atoms1 = core.copy() + oh_b
    # Whole-structure O–O max_diff would be large; core-only must still keep the pair.
    cum_full, max_full, _ = calculate_structure_similarity(
        atoms0, atoms1, use_mic=False
    )
    assert max_full > 1.25 or cum_full > 0.015
    pairs = select_structure_pairs(
        [(-260.0, atoms0), (-259.6, atoms1)],
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=5,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_adsorbate_empty_core_energy_only() -> None:
    """n_core_mobile=0: gate/rank on adsorbate Cartesian hop, not vacuous OH FP."""
    a = Atoms("OH", positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.96]])
    b = Atoms("OH", positions=[[1.0, 0.0, 1.0], [1.0, 0.0, 1.96]])
    pairs = select_structure_pairs(
        [(-1.0, a), (-0.5, b)],
        energy_gap_threshold=0.75,
        use_mic=False,
        adsorbate_aware=True,
        n_core_mobile=0,
        max_endpoint_mismatch=1.25,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_surface_adsorbate_keeps_site_isomers() -> None:
    """surface_adsorbate: local OH site hop on a slab stays pairable."""
    slab = Atoms(
        "Pt4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[10.0, 10.0, 20.0],
        pbc=[True, True, False],
    )
    oh_a = Atoms("OH", positions=[[1.25, 1.25, 2.0], [1.25, 1.25, 2.96]])
    oh_b = Atoms("OH", positions=[[2.45, 1.25, 2.0], [2.45, 1.25, 2.96]])
    atoms0 = slab + oh_a
    atoms1 = slab + oh_b
    pairs = select_structure_pairs(
        [(-10.0, atoms0), (-9.6, atoms1)],
        energy_gap_threshold=0.75,
        surface_aware=True,
        use_mic=True,
        n_slab=4,
        adsorbate_aware=True,
        n_core_mobile=0,
        max_endpoint_mismatch=1.5,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_surface_cluster_adsorbate_same_core_hop() -> None:
    """surface_cluster_adsorbate: same Pt core on slab, OH site hop is kept."""
    slab = Atoms(
        "C4",
        positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [1.4, 1.4, 0]],
        cell=[12.0, 12.0, 20.0],
        pbc=[True, True, False],
    )
    core = Atoms(
        "Pt2",
        positions=[[3.0, 3.0, 2.0], [5.5, 3.0, 2.0]],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    oh_a = Atoms("OH", positions=[[4.25, 3.0, 3.5], [4.25, 3.0, 4.46]])
    oh_b = Atoms("OH", positions=[[5.5, 4.2, 3.5], [5.5, 4.2, 4.46]])
    atoms0 = slab + core + oh_a
    atoms1 = slab + core.copy() + oh_b
    pairs = select_structure_pairs(
        [(-50.0, atoms0), (-49.6, atoms1)],
        energy_gap_threshold=0.75,
        surface_aware=True,
        use_mic=True,
        n_slab=4,
        adsorbate_aware=True,
        n_core_mobile=2,
        max_endpoint_mismatch=1.5,
    )
    assert pairs == [(0, 1)]


def test_select_structure_pairs_surface_cluster_adsorbate_skips_core_isomer() -> None:
    """Different deposited cores on the same slab are gated out."""
    slab = Atoms(
        "C4",
        positions=[[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], [1.4, 1.4, 0]],
        cell=[12.0, 12.0, 20.0],
        pbc=[True, True, False],
    )
    core_a = Atoms("Pt2", positions=[[3.0, 3.0, 2.0], [5.5, 3.0, 2.0]])
    core_b = Atoms("Pt2", positions=[[3.0, 3.0, 2.0], [3.0, 5.8, 2.0]])
    oh = Atoms("OH", positions=[[4.25, 3.0, 3.5], [4.25, 3.0, 4.46]])
    atoms0 = slab + core_a + oh
    atoms1 = slab + core_b + oh.copy()
    pairs = select_structure_pairs(
        [(-50.0, atoms0), (-49.6, atoms1)],
        energy_gap_threshold=0.75,
        surface_aware=True,
        use_mic=True,
        n_slab=4,
        adsorbate_aware=True,
        n_core_mobile=2,
        max_endpoint_mismatch=1.5,
    )
    assert pairs == []


@pytest.mark.slow
def test_find_ts_emt_basic(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with EMT calculator on CPU.

    Verifies the standard ASE NEB path works with classical potentials.
    Uses Cu3 which has predictable EMT behavior.
    """
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),  # Override fixtures' EMT with fresh one
        output_dir=temp_output_dir,
        pair_id="emt_cu3",
        n_images=3,
        spring_constant=0.1,
        fmax=0.1,  # Relaxed convergence for testing
        neb_steps=50,
        verbosity=0,
    )

    # Verify EMT path was used (not TorchSim)
    assert result["use_torchsim"] is False
    assert result["n_images"] == 3
    # Either converged or failed gracefully with error
    assert result["status"] in ["success", "failed"]
    # If successful, check key outputs
    if result["status"] == "success":
        assert "transition_state" in result
        assert result["neb_converged"] is True


@pytest.mark.slow
@pytest.mark.requires_mace
def test_find_ts_mace_cpu(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with MACE on CPU (no TorchSim).

    Uses ML potential on CPU with standard ASE NEB optimization.
    This verifies MACE works without GPU acceleration.
    """
    from scgo.calculators.mace_helpers import MACE

    # Create fresh MACE calculators (not from fixtures)
    mace_calc_reactant = MACE(model="small", device="cpu")
    mace_calc_product = MACE(model="small", device="cpu")

    # Attach to new atoms copies
    reactant = cu3_triangle.copy()
    product = cu3_linear.copy()
    reactant.calc = mace_calc_reactant
    product.calc = mace_calc_product

    result = find_transition_state(
        reactant,
        product,
        calculator=MACE(model="small", device="cpu"),  # For NEB images
        output_dir=temp_output_dir,
        pair_id="mace_cpu_cu3",
        n_images=3,
        spring_constant=0.1,
        fmax=0.1,  # Relaxed convergence for testing
        neb_steps=30,
        use_torchsim=False,  # Explicitly disable TorchSim
        verbosity=0,
    )

    # Verify MACE CPU path was used
    assert result["use_torchsim"] is False
    assert result["n_images"] == 3
    assert result["status"] in ["success", "failed"]
    if result["status"] == "success":
        assert "transition_state" in result
        assert result["neb_converged"] is True


@pytest.mark.slow
@pytest.mark.requires_mace
@pytest.mark.requires_cuda
def test_find_ts_mace_gpu_torchsim(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with MACE on GPU using TorchSim batching.

    This is the primary production use case: GPU acceleration with batched NEB.
    Verifies that GPU batching via TorchSim works end-to-end.
    """
    device = "cuda"

    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=None,  # TorchSim provides forces via MACE
        output_dir=temp_output_dir,
        pair_id="mace_gpu_cu3",
        n_images=5,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=100,
        use_torchsim=True,
        torchsim_params={
            "device": device,
            "mace_model_name": "mace_matpes_0",
            "autobatcher": True,  # GPU batching enabled
            "max_steps": 100,
        },
        verbosity=0,
    )

    # Verify TorchSim path was used and completed
    assert result["use_torchsim"] is True
    assert result["n_images"] == 5
    # Result should either succeed or have an error (not hang)
    assert "error" in result or result["status"] == "success"
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert "transition_state" in result


@pytest.mark.requires_mace
class TestTorchSimNEB:
    """TorchSim NEB with MACE (small clusters for fast tests)."""

    def test_torchsim_neb_initialization_with_mace(self, cu3_triangle, cu3_linear):
        """TorchSimNEB + MACE relaxer wires up a batched NEB path."""
        from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
        from scgo.ts_search.transition_state import TorchSimNEB

        device = "cuda" if torch.cuda.is_available() else "cpu"

        relaxer = TorchSimBatchRelaxer(
            device=device,
            mace_model_name="mace_matpes_0",
            force_tol=0.05,
            max_steps=100,
        )

        # Create path for Cu3 (triangle -> linear)
        images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

        # Initialize TorchSimNEB - this is where GPU batching setup happens
        neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

        assert neb.relaxer is relaxer
        assert len(neb.images) == 5  # 3 intermediate + 2 endpoints
        assert neb.get_force_calls() == 0

    @pytest.mark.slow
    @pytest.mark.requires_cuda
    @pytest.mark.requires_mace
    def test_find_ts_with_torchsim_cu3(self, cu3_triangle, cu3_linear, temp_output_dir):
        """Cu3 triangle–linear TS search with TorchSim + MACE (GPU-only)."""
        device = "cuda"

        result = find_transition_state(
            cu3_triangle,
            cu3_linear,
            calculator=None,  # TorchSim provides the calculator
            output_dir=temp_output_dir,
            pair_id="cu3_torchsim",
            n_images=5,  # Use more images to better utilize GPU batching
            spring_constant=0.1,
            fmax=0.05,
            neb_steps=100,
            use_torchsim=True,
            torchsim_params={
                "device": device,
                "mace_model_name": "mace_matpes_0",
                "autobatcher": True,  # Use autobatching for GPU efficiency
                "max_steps": 100,
            },
            verbosity=0,
        )

        # Validate the result - verify TorchSim was used and ran
        assert result["n_images"] == 5
        assert result["use_torchsim"] is True
        # Note: Full convergence is not guaranteed in tests, but we verify the run was attempted
        assert "barrier_height" in result or result.get("error") is not None


def test_find_ts_allows_missing_endpoint_energies_when_use_torchsim(
    monkeypatch, temp_output_dir, cu3_triangle, cu3_linear
):
    """When `use_torchsim=True` missing endpoint energies on Atoms are allowed
    and the relaxer provides single-point endpoint energies instead.
    """
    # Ensure Atoms have no attached calculators (so extract_energy_from_atoms -> None)
    react = cu3_triangle.copy()
    prod = cu3_linear.copy()
    react.calc = None
    prod.calc = None

    class FakeRelaxer:
        def __init__(self, **kw):
            pass

        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                # return a deterministic single-point energy for endpoints
                results.append((-4.1234, ra))
            return results

    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer",
        FakeRelaxer,
    )

    # Run a very short NEB (neb_steps small) to exercise the endpoint-energy path
    result = find_transition_state(
        react,
        prod,
        calculator=None,
        output_dir=temp_output_dir,
        pair_id="ts_torchsim_endpoint",
        n_images=3,
        fmax=1.0,
        neb_steps=1,
        use_torchsim=True,
        torchsim_params={},
        verbosity=0,
    )

    # Relaxer-provided endpoint energies should be recorded
    assert result.get("reactant_energy") == pytest.approx(-4.1234)
    assert result.get("product_energy") == pytest.approx(-4.1234)


@pytest.mark.slow
def test_find_ts_high_spring_constant(
    h2_reactant, h2_product, temp_output_dir, default_rel_tol
):
    """Test with very high spring constant."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="high_spring",
        n_images=3,
        spring_constant=10.0,  # Very high
        fmax=0.5,
        neb_steps=10,
        verbosity=0,
    )

    assert result["spring_constant"] == pytest.approx(10.0, rel=default_rel_tol)


def test_interpolate_path_many_images(h2_reactant, h2_product):
    """Test interpolation with many images."""
    n_images = 50
    images = interpolate_path(
        h2_reactant, h2_product, n_images=n_images, method="linear"
    )

    assert len(images) == n_images + 2


@pytest.mark.slow
def test_full_neb_convergence(cu3_triangle, cu3_linear, temp_output_dir):
    """Full NEB convergence test with Cu3 (slow)."""
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_full",
        n_images=7,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=200,
        climb=False,
        verbosity=1,
    )

    # Should converge for Cu3 with EMT
    # (though TS might not be meaningful for EMT)
    assert "status" in result


# ---------------------------------------------------------------------------
# T1: interior NEB images must not share the reactant's key_value_pairs dict
# ---------------------------------------------------------------------------


def test_interpolate_path_interior_images_have_isolated_key_value_pairs(
    cu3_triangle, cu3_linear
):
    """Each band image owns its ``info['key_value_pairs']`` dict.

    ``Atoms.copy()`` shallow-copies ``info``, so the nested ``key_value_pairs``
    dict was shared across interior images: ``set_tags`` on one image (e.g.
    ``potential_energy``/``raw_score``) overwrote every other image. The source
    minimum must also stay untouched.
    """
    from scgo.metadata.atoms import get_tag, set_tags

    reactant = cu3_triangle.copy()
    reactant.info["key_value_pairs"] = {"raw_score": 1.23}
    product = cu3_linear.copy()
    product.info["key_value_pairs"] = {"raw_score": 4.56}

    images = interpolate_path(
        reactant, product, n_images=3, method="idpp", align_endpoints=True
    )
    assert len(images) == 5

    for i, img in enumerate(images):
        set_tags(img, potential_energy=float(i))

    # Each image reports its own potential_energy (no cross-image clobbering).
    for i, img in enumerate(images):
        assert get_tag(img, "potential_energy") == pytest.approx(float(i))

    # Interior images must not alias each other's tag dict.
    interior = images[1:-1]
    for a, b in zip(interior, interior[1:], strict=False):
        assert a.info["key_value_pairs"] is not b.info["key_value_pairs"]
    # ...nor the endpoint band image's dict.
    assert images[1].info["key_value_pairs"] is not images[0].info["key_value_pairs"]

    # The source minima are never mutated by band tag writes.
    assert "potential_energy" not in reactant.info["key_value_pairs"]
    assert "potential_energy" not in product.info["key_value_pairs"]
    assert reactant.info["key_value_pairs"]["raw_score"] == pytest.approx(1.23)
    assert product.info["key_value_pairs"]["raw_score"] == pytest.approx(4.56)


# ---------------------------------------------------------------------------
# T3: serial NEB fallback with a non-deepcopyable calculator must not raise
# ---------------------------------------------------------------------------


def test_serial_neb_shared_calculator_fallback_reaches_neb(
    cu3_triangle, cu3_linear, temp_output_dir
):
    """A calculator that cannot be deep-copied falls back to a shared instance.

    ASE ``NEB.get_forces`` raises ``ValueError`` when images share one
    calculator unless ``allow_shared_calculator=True``. The serial fallback must
    set that flag so the run reaches (and steps) NEB construction instead of
    failing with the shared-calculator error.

    ``_finalize_neb_result`` deep-copies the TS image (including its calculator);
    it is patched here so the deliberately non-deep-copyable calculator does not
    trip that unrelated code path — the assertion is about NEB construction.
    """
    from unittest.mock import patch

    class _NoDeepcopyEMT(EMT):
        def __deepcopy__(self, memo):
            raise TypeError("this calculator cannot be deep-copied")

    reactant = cu3_triangle.copy()
    product = cu3_linear.copy()
    reactant.calc = _NoDeepcopyEMT()
    product.calc = _NoDeepcopyEMT()

    with patch("scgo.ts_search.transition_state._finalize_neb_result") as finalize_mock:
        result = find_transition_state(
            reactant,
            product,
            calculator=_NoDeepcopyEMT(),
            output_dir=temp_output_dir,
            pair_id="shared_calc",
            n_images=3,
            fmax=0.5,
            neb_steps=1,
            use_torchsim=False,
            verbosity=0,
        )

    assert "status" in result
    err = str(result.get("error") or "").lower()
    # The specific shared-calculator ASE ValueError must not appear.
    assert "share the same calculator" not in err
    # NEB was constructed and stepped (finalize was reached).
    finalize_mock.assert_called_once()
    assert int(result.get("steps_taken") or 0) >= 1


def test_idpp_priority_screen_forwards_clash_distance(
    h2_reactant, h2_product, monkeypatch
):
    """The IDPP ranking must gate paths with the resolved ``neb_prescreen_clash_distance``
    (the value the real NEB prescreen uses), not fall back to the 0.7 default.
    """
    import logging

    from scgo.ts_search import transition_state_run as ts_run

    captured: dict[str, object] = {}

    def _spy_validate(images, *, n_slab, mic, max_endpoint_mismatch, clash_distance):
        captured["clash_distance"] = clash_distance
        return None

    monkeypatch.setattr(ts_run, "validate_initial_neb_path", _spy_validate)
    monkeypatch.setattr(
        ts_run, "_evaluate_bands_in_chunks", lambda *a, **k: [[0.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(
        ts_run, "validate_initial_neb_energy_profile", lambda *a, **k: None
    )

    minima = [(0.0, h2_reactant.copy()), (0.1, h2_product.copy())]
    ts_run._prioritize_adsorbate_pairs_by_idpp(
        [(0, 1)],
        minima,
        max_pairs=1,
        relaxer=object(),
        neb_n_images=3,
        neb_interpolation_method="idpp",
        neb_interpolation_mic=False,
        neb_align_endpoints=True,
        neb_perturb_sigma=0.0,
        rng=np.random.default_rng(0),
        system_type="gas_cluster",
        n_slab=0,
        n_core_mobile=None,
        n_adsorbate_mobile=None,
        adsorbate_fragment_lengths=None,
        neb_surface_cell_remap=False,
        neb_surface_lattice_rotation=False,
        neb_surface_max_lattice_shift=0,
        max_endpoint_mismatch=float("inf"),
        neb_prescreen_clash_distance=1.0,
        min_saddle_prominence=0.40,
        neb_max_spurious_barrier=8.0,
        neb_interpolation_bond_tolerance_a=None,
        parallel_neb_max_batch_atoms=None,
        parallel_neb_max_bands=None,
        logger=logging.getLogger("test"),
    )

    assert captured["clash_distance"] == 1.0


def test_find_transition_state_skips_energy_profile_when_mismatch_unset(
    temp_output_dir, h2_reactant, h2_product, monkeypatch
):
    """Serial ASE path must not call the energy-profile screen when mismatch is None."""
    import scgo.ts_search.transition_state as ts_mod

    calls: list[object] = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ts_mod, "validate_initial_neb_energy_profile", _spy)

    find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="energy_profile_skip",
        n_images=3,
        fmax=0.1,
        neb_steps=5,
        verbosity=0,
        use_torchsim=False,
        max_endpoint_mismatch=None,
    )
    assert calls == []

    find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="energy_profile_on",
        n_images=3,
        fmax=0.1,
        neb_steps=5,
        verbosity=0,
        use_torchsim=False,
        max_endpoint_mismatch=1.25,
    )
    assert len(calls) >= 1


def test_neb_run_config_carries_promoted_thresholds():
    from scgo.param_presets import TS_DEFAULTS_BY_SYSTEM_TYPE
    from tests.ts_search.test_parallel_neb import _gas_neb_cfg

    for system_type in TS_DEFAULTS_BY_SYSTEM_TYPE:
        cfg = _gas_neb_cfg(system_type=system_type)
        assert cfg.layer_cluster_threshold_ang == 0.4
        assert cfg.neb_interpolation_bond_tolerance_a == 0.5


def _bonded_dimer(positions: np.ndarray) -> Atoms:
    atoms = Atoms("CO", positions=positions.copy())
    atoms.set_constraint(FixBondLengths([(0, 1)]))
    return atoms


def test_interpolate_path_bond_check_passes_when_preserved(caplog):
    a1 = _bonded_dimer(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]))
    a2 = _bonded_dimer(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]))
    with caplog.at_level("WARNING", logger="scgo.ts_search.transition_state"):
        images = interpolate_path(
            a1,
            a2,
            n_images=3,
            method="linear",
            align_endpoints=False,
            neb_interpolation_bond_tolerance_a=0.5,
        )
    assert len(images) == 5
    assert not any("FixBondLengths" in r.message for r in caplog.records)


def test_interpolate_path_bond_check_warns_when_stretched(caplog):
    a1 = _bonded_dimer(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    a2 = _bonded_dimer(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]))
    with caplog.at_level("WARNING", logger="scgo.ts_search.transition_state"):
        interpolate_path(
            a1,
            a2,
            n_images=3,
            method="linear",
            align_endpoints=False,
            neb_interpolation_bond_tolerance_a=0.1,
        )
    assert any("FixBondLengths" in r.message for r in caplog.records)
