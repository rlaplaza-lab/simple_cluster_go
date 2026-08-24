"""Consolidated NEB endpoint / surface alignment tests (T2-15 fold).

Merged verbatim from ``test_surface_neb_alignment.py`` and ``test_neb_blockwise_alignment.py``; all tests and markers preserved.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms

from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.system_types import AdsorbateDefinition, get_system_policy
from scgo.ts_search import transition_state as ts_mod
from scgo.ts_search.transition_state import (
    _align_endpoints_blockwise,
    _align_product_for_neb,
    _align_product_kabsch_to_reactant,
    _align_product_surface_pbc,
    _lattice_translation_candidates,
    _requires_surface_pbc_alignment,
    _validate_lattice_compatible_rotation,
    interpolate_path,
    validate_initial_neb_path,
)
from scgo.ts_search.transition_state_io import adsorbate_pair_select_cap
from scgo.utils.helpers import get_cluster_formula

# ---------------------------------------------------------------------
# from test_surface_neb_alignment.py
# ---------------------------------------------------------------------

"""Surface NEB endpoint alignment: lattice remap and compatible rotation."""


def test_system_policy_surface_enables_remap_and_rotation():
    bare = get_system_policy("surface_cluster")
    assert bare.neb_surface_cell_remap is True
    assert bare.neb_surface_lattice_rotation is True
    # Free in-plane Kabsch breaks adsorbate–slab registry; remap/MIC stay on.
    ads = get_system_policy("surface_cluster_adsorbate")
    assert ads.neb_surface_cell_remap is True
    assert ads.neb_surface_lattice_rotation is False


def test_validate_lattice_compatible_rotation_rejects_out_of_plane():
    # 90° rotation about x tilts the slab normal (z) into the plane.
    rot_bad = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    with pytest.raises(SCGOValidationError, match="surface normal"):
        _validate_lattice_compatible_rotation(rot_bad, normal_axis=2)


def test_inplane_rotation_matrix_preserves_normal_axis():
    angle = np.deg2rad(40.0)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    _validate_lattice_compatible_rotation(rot, normal_axis=2)


def test_surface_alignment_cell_remap_shortens_periodic_jump():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)

    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    raw = b.get_positions().copy()
    aligned = _align_product_surface_pbc(
        a, raw, n_slab=n_slab, enable_cell_remap=True, enable_lattice_rotation=False
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 0])) < 0.5


def test_surface_alignment_rotation_reduces_mobile_rms():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8

    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    mobile_rot[:, 0] += slab.cell[0, 0]

    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=True,
        enable_lattice_rotation=True,
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms < 0.15
    slab_disp = np.linalg.norm(aligned[:n_slab] - a.get_positions()[:n_slab], axis=1)
    assert float(np.max(slab_disp)) < 1e-6


def test_interpolate_path_surface_uses_pbc_align_entrypoint(monkeypatch):
    """Slab NEB routes endpoint alignment through ``_align_product_for_neb``."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    called = {"for_neb": 0}
    orig_for_neb = ts_mod._align_product_for_neb

    def _track_for_neb(*args, **kwargs):
        called["for_neb"] += 1
        return orig_for_neb(*args, **kwargs)

    monkeypatch.setattr(ts_mod, "_align_product_for_neb", _track_for_neb)

    interpolate_path(
        a,
        b,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    assert called["for_neb"] == 1


def test_interpolate_path_surface_unifies_product_cell():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    b.cell[0, 0] += 0.01

    images = interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    assert np.allclose(images[-1].cell, images[0].cell)
    assert list(images[-1].pbc) == list(images[0].pbc)


def test_kabsch_align_rejects_slab_systems():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    a = slab.copy() + Atoms("Pt", positions=[[0.0, 0.0, 8.0]])
    with pytest.raises(SCGORuntimeError, match="Slab NEB endpoints must use"):
        _align_product_kabsch_to_reactant(a, a.get_positions(), n_slab=len(slab))


def test_align_product_for_neb_routes_mic_alias_to_surface():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    via_for_neb = _align_product_for_neb(a, b.get_positions(), n_slab=len(slab))
    via_surface = _align_product_surface_pbc(a, b.get_positions(), n_slab=len(slab))
    np.testing.assert_allclose(via_for_neb, via_surface, atol=1e-8)


def test_interpolate_path_endpoints_unchanged_by_ase_interpolate(monkeypatch):
    """ASE ``NEB.interpolate`` must not move aligned endpoint images before optimization."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    captured: dict[str, np.ndarray] = {}
    from ase.mep import NEB as AseNEB

    _orig = AseNEB.interpolate

    def _record_endpoints(self, *args, **kwargs):
        captured["reactant"] = self.images[0].get_positions().copy()
        captured["product"] = self.images[-1].get_positions().copy()
        return _orig(self, *args, **kwargs)

    monkeypatch.setattr(AseNEB, "interpolate", _record_endpoints)

    images = interpolate_path(
        a,
        b,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    np.testing.assert_allclose(captured["reactant"], images[0].get_positions())
    np.testing.assert_allclose(captured["product"], images[-1].get_positions())


def test_run_transition_state_search_forwards_alignment_kwargs(monkeypatch, tmp_path):
    """Runner should pass slab/block dims and max lattice shift into interpolation."""
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.ts_search import transition_state_run as ts_run_mod

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.5
    react = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    prod = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    captured: dict[str, object] = {}
    pair_kwargs: dict[str, object] = {}
    max_pairs = 6

    def _fake_find_transition_state(reactant, product, calculator, **kwargs):
        captured.update(kwargs)
        return {
            "status": "failed",
            "pair_id": kwargs.get("pair_id", "stub"),
            "error": "stub",
            "neb_converged": False,
        }

    def _fake_select_pairs(minima, **kwargs):
        pair_kwargs.update(kwargs)
        # Excess survivors: bare surface must still truncate to max_pairs.
        return [(0, 1)] * adsorbate_pair_select_cap(max_pairs)

    monkeypatch.setattr(
        ts_run_mod, "find_transition_state", _fake_find_transition_state
    )
    monkeypatch.setattr(ts_run_mod, "save_neb_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ts_run_mod, "save_transition_state_results", lambda *args, **kwargs: None
    )
    full_comp = full_adsorbate_slab_composition(["Pt"], cfg)
    formula = get_cluster_formula(full_comp)
    monkeypatch.setattr(
        ts_run_mod,
        "load_minima_by_composition",
        lambda *_a, **_k: {
            formula: [
                (0.0, react),
                (0.1, prod),
            ]
        },
    )
    monkeypatch.setattr(ts_run_mod, "select_structure_pairs", _fake_select_pairs)
    monkeypatch.setattr(ts_run_mod, "get_calculator_class", lambda _n: object)
    monkeypatch.setattr(ts_run_mod, "auto_niter_ts", lambda _c: 10)

    results = ts_run_mod.run_transition_state_search(
        composition=["Pt"],
        system_type="surface_cluster",
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=cfg,
        verbosity=0,
        neb_surface_max_lattice_shift=3,
        max_pairs=max_pairs,
        max_endpoint_mismatch=1.25,
        use_torchsim=False,
        use_parallel_neb=False,
    )
    assert captured["neb_cfg"].n_slab == n_slab
    assert captured["neb_cfg"].neb_surface_max_lattice_shift == 3
    assert pair_kwargs["max_pairs"] == max_pairs
    assert len(results) == max_pairs


def test_run_transition_state_search_empty_core_sets_block_dims(
    monkeypatch, tmp_path
) -> None:
    """Empty-core adsorbate still enables blockwise dims (n_core=0, n_ads>0)."""
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.ts_search import transition_state_run as ts_run_mod

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.5
    react = slab.copy() + Atoms("OH", positions=[[0.1, 0.1, z0], [0.1, 0.1, z0 + 1.0]])
    prod = slab.copy() + Atoms(
        "OH", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0], [0.2, 0.1, z0 + 1.0]]
    )
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    captured: dict[str, object] = {}
    pair_kwargs: dict[str, object] = {}
    max_pairs = 6

    def _fake_find_transition_state(reactant, product, calculator, **kwargs):
        captured.update(kwargs)
        return {
            "status": "failed",
            "pair_id": kwargs.get("pair_id", "stub"),
            "error": "stub",
            "neb_converged": False,
        }

    def _fake_select_pairs(minima, **kwargs):
        pair_kwargs.update(kwargs)
        return [(0, 1)]

    monkeypatch.setattr(
        ts_run_mod, "find_transition_state", _fake_find_transition_state
    )
    monkeypatch.setattr(ts_run_mod, "save_neb_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ts_run_mod, "save_transition_state_results", lambda *args, **kwargs: None
    )
    full_comp = full_adsorbate_slab_composition(["O", "H"], cfg)
    formula = get_cluster_formula(full_comp)
    monkeypatch.setattr(
        ts_run_mod,
        "load_minima_by_composition",
        lambda *_a, **_k: {formula: [(0.0, react), (0.1, prod)]},
    )
    monkeypatch.setattr(ts_run_mod, "select_structure_pairs", _fake_select_pairs)
    monkeypatch.setattr(ts_run_mod, "get_calculator_class", lambda _n: object)
    monkeypatch.setattr(ts_run_mod, "auto_niter_ts", lambda _c: 10)

    ts_run_mod.run_transition_state_search(
        composition=["O", "H"],
        system_type="surface_cluster_adsorbate",
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=cfg,
        adsorbate_definition=ads_def,
        verbosity=0,
        max_pairs=max_pairs,
        max_endpoint_mismatch=1.25,
        use_torchsim=False,
        use_parallel_neb=False,
    )
    assert pair_kwargs["surface_aware"] is True
    assert pair_kwargs["use_mic"] is True
    assert pair_kwargs["adsorbate_aware"] is True
    assert pair_kwargs["n_core_mobile"] == 0
    assert pair_kwargs["max_pairs"] == adsorbate_pair_select_cap(max_pairs)
    assert captured["neb_cfg"].n_slab == n_slab
    assert captured["neb_cfg"].n_core_mobile == 0
    assert captured["neb_cfg"].n_adsorbate_mobile == 2


def test_get_ts_search_params_surface_keeps_alignment_defaults():
    from scgo.param_presets import get_ts_search_params
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_align_endpoints"] is True
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_surface_cell_remap"] is True
    assert ts["neb_surface_lattice_rotation"] is True
    assert ts["neb_surface_max_lattice_shift"] == 1


def _slab_with_mobile_pt(*, size=(2, 2, 1), mobile_xy=(0.1, 0.1)):
    slab = fcc111("Pt", size=size, vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[mobile_xy[0], mobile_xy[1], z0]])
    return slab, a, n_slab, z0


def test_surface_alignment_y_axis_periodic_jump():
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms("Pt", positions=[[0.1, slab.cell[1, 1] - 0.1, z0]])
    aligned = _align_product_surface_pbc(
        a, b.get_positions(), n_slab=n_slab, enable_lattice_rotation=False
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 1])) < 0.5


def test_surface_alignment_diagonal_two_cell_wrap():
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    shift = slab.cell[0] + slab.cell[1]
    b = slab.copy() + Atoms("Pt", positions=[[0.1 + shift[0], 0.1 + shift[1], z0]])
    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        max_lattice_shift=2,
        enable_lattice_rotation=False,
    )
    disp = aligned - a.get_positions()
    assert float(np.linalg.norm(disp[-1])) < 0.5


def test_lattice_translation_candidates_span_grows_with_max_shift():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cell = np.asarray(slab.cell.array, dtype=float)
    small = _lattice_translation_candidates(cell, 0, 1, max_shift=1)
    large = _lattice_translation_candidates(cell, 0, 1, max_shift=2)
    assert len(large) > len(small)
    assert len(small) == 9
    assert len(large) == 25


def test_surface_alignment_split_periodic_images_multi_atom():
    """Mobile atoms stored in inconsistent periodic images should still align."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    mobile_split = mobile.copy()
    mobile_split[1, 0] += slab.cell[0, 0]
    mobile_split[2, 1] += slab.cell[1, 1]
    b = slab.copy() + Atoms("Pt3", positions=mobile_split)

    aligned = _align_product_surface_pbc(
        a, b.get_positions(), n_slab=n_slab, enable_lattice_rotation=False
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms < 0.2


def test_surface_alignment_remap_only_shortens_x_wrap():
    slab, a, n_slab, _z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms(
        "Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, a.get_positions()[-1, 2]]]
    )
    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=True,
        enable_lattice_rotation=False,
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 0])) < 0.5


def test_surface_alignment_remap_disabled_leaves_rotated_cluster_misaligned():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    mobile_rot[:, 0] += slab.cell[0, 0]
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=False,
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms > 0.15


def test_surface_alignment_rotation_reduces_rms_on_rotated_cluster():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    no_rot = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=False,
    )
    with_rot = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=True,
    )
    rms_no = float(
        np.sqrt(np.mean((no_rot[n_slab:] - a.get_positions()[n_slab:]) ** 2))
    )
    rms_yes = float(
        np.sqrt(np.mean((with_rot[n_slab:] - a.get_positions()[n_slab:]) ** 2))
    )
    assert rms_yes < rms_no
    assert rms_yes < 0.15


def test_interpolate_path_forwards_max_lattice_shift(monkeypatch):
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms("Pt", positions=[[0.1 + 2.0 * slab.cell[0, 0], 0.1, z0]])
    captured: dict[str, int] = {}

    def _spy_align(reactant, product_positions, **kwargs):
        captured["max_shift"] = kwargs.get("max_lattice_shift", -1)
        return _align_product_surface_pbc(reactant, product_positions, **kwargs)

    monkeypatch.setattr(ts_mod, "_align_product_surface_pbc", _spy_align)
    interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
        neb_surface_max_lattice_shift=2,
    )
    assert captured["max_shift"] == 2


def test_get_ts_search_params_surface_max_lattice_shift_default():
    from scgo.param_presets import get_ts_search_params
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_surface_max_lattice_shift"] == 1


def test_interpolate_path_fixed_slab_anchors_under_surface_align():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    fixed_idx = list(range(len(slab)))

    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    a.set_constraint(FixAtoms(indices=fixed_idx))
    b.set_constraint(FixAtoms(indices=fixed_idx))

    images = interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        system_type="surface_cluster",
    )
    disp = images[-1].get_positions() - images[0].get_positions()
    assert float(np.max(np.linalg.norm(disp[fixed_idx], axis=1))) < 1e-2


# ---------------------------------------------------------------------
# from test_neb_blockwise_alignment.py
# ---------------------------------------------------------------------

"""NEB: blockwise endpoint matching (slab + core + adsorbate)."""


def test_blockwise_reorders_adsorbate_block_to_reactant() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.0, 0.0, 0.1],
            [1.2, 0.7, 0.3],
        ],
    )
    react = Atoms(symbols=["Pt", "Pt", "O", "H"], positions=pos, pbc=False)
    prod = Atoms(
        symbols=["Pt", "Pt", "H", "O"],
        positions=np.vstack([pos[:2], pos[3:4], pos[2:3]]),
        pbc=False,
    )
    _align_endpoints_blockwise(react, prod, n_slab=1, n_core=1, n_ads=2)
    np.testing.assert_array_almost_equal(
        prod.get_positions()[2:4], react.get_positions()[2:4]
    )


def test_interpolate_path_accepts_block_dims_for_gas_adsorbate() -> None:
    sym = ["Pt", "Pt", "H"]
    pos = np.random.default_rng(0).random((3, 3))
    a1 = Atoms(symbols=sym, positions=pos, pbc=False, cell=[20, 20, 20])
    a2 = Atoms(symbols=sym, positions=pos.copy(), pbc=False, cell=[20, 20, 20])
    out = interpolate_path(
        a1,
        a2,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=2,
        n_adsorbate_mobile=1,
    )
    assert len(out) == 2 + 2
    assert len(out[0]) == 3 and len(out[-1]) == 3


def test_interpolate_path_blockwise_mic_on_periodic_surface() -> None:
    """Blockwise matching + MIC on slab/core/adsorbate under in-plane PBC."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.5

    core_pos = np.array([[0.5, 0.5, z0], [1.5, 0.6, z0]])
    ads_pos = np.array([[1.0, 1.2, z0 + 0.2], [1.1, 1.3, z0 + 0.9]])
    react = slab.copy() + Atoms(
        symbols=["Pt", "Pt", "O", "H"], positions=np.vstack([core_pos, ads_pos])
    )
    prod_ads = ads_pos[[1, 0]]
    prod_core = core_pos + np.array([slab.cell[0, 0] - 0.1, 0.0, 0.0])
    prod = slab.copy() + Atoms(
        symbols=["Pt", "Pt", "H", "O"],
        positions=np.vstack([prod_core, prod_ads]),
    )

    images = interpolate_path(
        react,
        prod,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
        n_core_mobile=2,
        n_adsorbate_mobile=2,
    )

    disp = images[-1].get_positions() - images[0].get_positions()
    assert float(np.max(np.linalg.norm(disp[:n_slab], axis=1))) < 1e-2
    mobile_disp = np.linalg.norm(disp[n_slab:], axis=1)
    assert float(np.max(mobile_disp)) < 0.25
    rms = float(np.sqrt(np.mean(mobile_disp**2)))
    assert rms < 0.15


def test_fragment_wise_matching_swaps_crossed_oh() -> None:
    """Two OH fragments crossed on product are restored by COM fragment matching."""
    core = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0]])
    oh1 = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 2.46]])
    oh2 = np.array([[2.0, 0.0, 1.5], [2.0, 0.0, 2.46]])
    react = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=np.vstack([core, oh1, oh2]),
        pbc=False,
    )
    # Product: same core, but OH fragments swapped (and H/O order swapped in frag0).
    prod = Atoms(
        symbols=["Pt", "Pt", "Pt", "H", "O", "H", "O"],
        positions=np.vstack([core, oh2[[1, 0]], oh1[[1, 0]]]),
        pbc=False,
    )
    _align_endpoints_blockwise(
        react,
        prod,
        n_slab=0,
        n_core=3,
        n_ads=4,
        adsorbate_fragment_lengths=[2, 2],
    )
    np.testing.assert_allclose(prod.get_positions()[3:5], oh1, atol=1e-8)
    np.testing.assert_allclose(prod.get_positions()[5:7], oh2, atol=1e-8)
    assert list(prod.numbers[3:7]) == list(react.numbers[3:7])


def test_requires_surface_pbc_alignment_ignores_gas_vacuum_pbc() -> None:
    """3D pbc on a gas cluster is a vacuum box, not slab PBC alignment."""
    gas = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], pbc=True)
    gas.set_cell([20.0, 20.0, 20.0])
    assert _requires_surface_pbc_alignment(gas, n_slab=0) is False
    slab_like = gas.copy()
    slab_like.pbc = [True, True, False]
    assert _requires_surface_pbc_alignment(slab_like, n_slab=0) is True
    assert _requires_surface_pbc_alignment(gas, n_slab=1) is True


def test_align_product_for_neb_gas_vacuum_pbc_uses_3d_kabsch() -> None:
    """Gas pbc=True still 3D-Kabsches; MIC-only would leave a rotated core."""
    react = Atoms(
        symbols=["Pt", "Pt", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 1.5],
            [1.0, 0.0, 2.46],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    rot = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    core_com = react.get_positions()[:2].mean(axis=0)
    prod_pos = (react.get_positions() - core_com) @ rot.T + core_com
    aligned = _align_product_for_neb(react, prod_pos, n_slab=0, n_core_mobile=2)
    core_rms = float(
        np.sqrt(np.mean(np.sum((aligned[:2] - react.positions[:2]) ** 2, axis=1)))
    )
    assert core_rms < 0.05


def test_interpolate_path_matches_oh_fragments_after_gas_rotation() -> None:
    """Fragment COM matching runs after core Kabsch so a rotation cannot swap OH."""
    core = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0]])
    oh1 = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 2.46]])
    oh2 = np.array([[2.0, 0.0, 1.5], [2.0, 0.0, 2.46]])
    react = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=np.vstack([core, oh1, oh2]),
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    rot = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    center = core.mean(axis=0)
    prod_pos = np.vstack([core, oh2, oh1])
    prod_pos = (prod_pos - center) @ rot.T + center
    prod = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=prod_pos,
        cell=react.cell,
        pbc=True,
    )
    images = interpolate_path(
        react,
        prod,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=3,
        n_adsorbate_mobile=4,
        adsorbate_fragment_lengths=[2, 2],
    )
    aligned = images[-1].get_positions()
    ref = images[0].get_positions()
    np.testing.assert_allclose(aligned, ref, atol=1e-6)


def test_core_anchored_kabsch_ignores_adsorbate_drag() -> None:
    """Kabsch fit on core should not be pulled by a large adsorbate hop."""
    react = Atoms(
        symbols=["Pt", "Pt", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.96],
        ],
        pbc=False,
    )
    # Product: core slightly rotated/translated; OH hopped far away.
    prod_pos = np.array(
        [
            [0.1, 0.05, 0.0],
            [2.05, -0.05, 0.0],
            [4.5, 3.0, 2.0],
            [4.5, 3.0, 2.96],
        ]
    )
    aligned_full = _align_product_kabsch_to_reactant(
        react, prod_pos, n_slab=0, n_core_mobile=None
    )
    aligned_core = _align_product_kabsch_to_reactant(
        react, prod_pos, n_slab=0, n_core_mobile=2
    )
    core_rms_full = float(
        np.sqrt(np.mean(np.sum((aligned_full[:2] - react.positions[:2]) ** 2, axis=1)))
    )
    core_rms_core = float(
        np.sqrt(np.mean(np.sum((aligned_core[:2] - react.positions[:2]) ** 2, axis=1)))
    )
    assert core_rms_core <= core_rms_full + 1e-9
    assert core_rms_core < 0.15


def test_validate_initial_neb_path_rejects_clash() -> None:
    # Three images so the middle one is treated as an interior clash check.
    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mid = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    b = a.copy()
    with pytest.raises(SCGOValidationError, match="clashing/discontinuous"):
        validate_initial_neb_path(
            [a, mid, b], max_endpoint_mismatch=1.25, clash_distance=0.7
        )


def test_validate_initial_neb_path_rejects_huge_residual() -> None:
    a = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    b = Atoms("Pt2", positions=[[8.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    with pytest.raises(SCGOValidationError, match="clashing/discontinuous"):
        validate_initial_neb_path([a, b], max_endpoint_mismatch=1.25)


def test_validate_initial_neb_path_noop_without_mismatch_gate() -> None:
    # Without max_endpoint_mismatch the endpoint-displacement gate is skipped,
    # but the always-on clash check must still pass (H-H at 1.0 A is non-clashing).
    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    validate_initial_neb_path([a, a.copy()], max_endpoint_mismatch=None)


def test_validate_initial_neb_energy_profile_rejects_huge_barrier() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    with pytest.raises(SCGOValidationError, match="discontinuous"):
        validate_initial_neb_energy_profile(
            [0.0, 10.0, 20.0, 0.2], max_spurious_barrier=8.0
        )


def test_validate_initial_neb_energy_profile_allows_endpoint_max() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    # Adsorbate OH hops often start endpoint-max on IDPP; climb can still succeed.
    validate_initial_neb_energy_profile([0.0, 0.2, 0.5, 1.0], max_spurious_barrier=8.0)


def test_validate_initial_neb_energy_profile_accepts_modest_barrier() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    validate_initial_neb_energy_profile([0.0, 0.5, 1.1, 0.2], max_spurious_barrier=8.0)


def test_idpp_band_optimization_priority_prefers_robust_interior() -> None:
    from scgo.ts_search.transition_state import idpp_band_optimization_priority

    robust = idpp_band_optimization_priority([0.0, 0.5, 1.2, 0.2])
    endpoint = idpp_band_optimization_priority([0.0, 0.2, 0.5, 1.0])
    soft = idpp_band_optimization_priority([0.0, 0.35, 0.45, 0.4])
    assert robust[0] == 2
    assert endpoint[0] == 1
    assert soft[0] == 0
    assert robust > endpoint > soft


def test_neb_uses_two_stage_climb_skips_soft_interior_barriers() -> None:
    from scgo.ts_search.transition_state import neb_uses_two_stage_climb

    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.2, 0.5, 1.0])
        is False
    )
    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.4, 0.9, 0.2])
        is False
    )
    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.5, 1.2, 0.2])
        is True
    )


def test_validate_initial_neb_energy_profile_rejects_endpoint_drift() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    with pytest.raises(SCGOValidationError, match="product energy drifted"):
        validate_initial_neb_energy_profile(
            [0.0, 0.5, 1.0, 0.2],
            reference_reactant_energy=0.0,
            reference_product_energy=-6.0,
            max_endpoint_energy_drift=0.5,
        )


def test_validate_initial_neb_energy_profile_rejects_one_sided_slide() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    # Interior max only 0.1 eV above the higher endpoint.
    with pytest.raises(SCGOValidationError, match="prominence"):
        validate_initial_neb_energy_profile(
            [0.0, 0.3, 0.5, 0.4],
            reference_reactant_energy=0.0,
            reference_product_energy=0.4,
            min_saddle_prominence=0.40,
        )


def test_copy_atoms_isolates_nested_info_from_metadata_writes() -> None:
    from scgo.metadata.atoms import set_tags
    from scgo.utils.helpers import copy_atoms, extract_energy_from_atoms

    src = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    src.info["key_value_pairs"] = {"raw_score": 1.0}
    clone = copy_atoms(src)
    set_tags(clone, raw_score=-9.0, potential_energy=9.0)
    assert extract_energy_from_atoms(src) == pytest.approx(-1.0)
    assert extract_energy_from_atoms(clone) == pytest.approx(9.0)
    # ASE Atoms.copy() alone would have shared the nested dicts.
    shallow = src.copy()
    set_tags(shallow, raw_score=-3.0)
    assert extract_energy_from_atoms(src) == pytest.approx(3.0)


def test_prepare_neb_endpoints_slab_search_empty_core_uses_deposit_prefix() -> None:
    """Empty-core surface_adsorbate NEB prep must not treat top-layer as adsorbate."""
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.surface.partition import prepare_slab_search_surface_config
    from scgo.ts_search.neb_endpoints import prepare_neb_endpoints
    from scgo.utils.ts_runner_kwargs import NebRunConfig

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        adsorption_height_min=1.0,
        adsorption_height_max=3.0,
        comparator_use_mic=True,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    n_fixed = int(part.n_fixed)
    n_full = len(cfg.slab)
    z_top = float(np.max(cfg.slab.positions[:, 2]))
    xy = np.mean(cfg.slab.positions[n_fixed:, :2], axis=0)

    def _combined(dx: float) -> Atoms:
        oh = Atoms(
            "OH",
            positions=[
                [xy[0] + dx, xy[1], z_top + 1.8],
                [xy[0] + dx, xy[1], z_top + 2.76],
            ],
            cell=cfg.slab.cell,
            pbc=cfg.slab.pbc,
        )
        combined = cfg.slab.copy() + oh
        tags = np.zeros(len(combined), dtype=int)
        tags[n_full:] = 1
        combined.set_tags(tags)
        return combined

    ads = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    neb_cfg = NebRunConfig(
        neb_n_images=5,
        neb_spring_constant=0.1,
        neb_fmax=0.5,
        neb_steps=10,
        neb_climb=False,
        neb_interpolation_method="linear",
        neb_align_endpoints=True,
        neb_perturb_sigma=0.0,
        neb_interpolation_mic=True,
        neb_tangent_method="improvedtangent",
        neb_surface_cell_remap=False,
        neb_surface_lattice_rotation=False,
        neb_surface_max_lattice_shift=0,
        n_slab=n_fixed,
        n_core_mobile=n_full - n_fixed,
        n_adsorbate_mobile=2,
        adsorbate_fragment_lengths=[2],
        max_endpoint_mismatch=3.0,
        neb_prescreen_clash_distance=0.5,
        min_saddle_prominence=0.0,
        neb_max_spurious_barrier=10.0,
        layer_cluster_threshold_ang=0.4,
        neb_interpolation_bond_tolerance_a=0.5,
        adsorbate_definition=ads,
        connectivity_factor=1.4,
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
        enforce_adsorbate_subgraph_integrity=True,
        system_type="surface_adsorbate",
        surface_config=cfg,
        torchsim_params=None,
    )
    react, prod = prepare_neb_endpoints(_combined(0.0), _combined(1.2), neb_cfg)
    assert len(react) == n_full + 2
    assert len(prod) == n_full + 2
