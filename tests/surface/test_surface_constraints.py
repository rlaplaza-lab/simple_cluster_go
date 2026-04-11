"""Unit tests for slab FixAtoms layering helpers and SurfaceSystemConfig rules."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.algorithms.geneticalgorithm_go_torchsim import _torchsim_prepare_relaxed_copy
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import (
    attach_slab_constraints,
    attach_slab_constraints_from_surface_config,
    surface_slab_constraint_summary,
)


def _three_layer_slab_positions() -> np.ndarray:
    """Six atoms: two per layer at z=0, 1, 2 (axis 2)."""
    pos = np.zeros((6, 3))
    pos[0:2, 2] = 0.0
    pos[2:4, 2] = 1.0
    pos[4:6, 2] = 2.0
    return pos


def _fix_indices(combined: Atoms) -> list[int]:
    assert len(combined.constraints) == 1
    c = combined.constraints[0]
    return sorted(int(i) for i in c.index)


def test_attach_slab_constraints_nothing_frozen() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)
    combined = slab + ads
    attach_slab_constraints(
        combined,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=None,
        n_relax_top_slab_layers=None,
        surface_normal_axis=2,
    )
    assert combined.constraints == []


def test_attach_slab_constraints_fix_bottom_two_layers() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)
    combined = slab + ads
    attach_slab_constraints(
        combined,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=2,
        surface_normal_axis=2,
    )
    assert _fix_indices(combined) == [0, 1, 2, 3]


def test_attach_slab_constraints_relax_top_two_matches_fix_bottom_one() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)

    c1 = slab + ads
    attach_slab_constraints(
        c1,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=1,
        surface_normal_axis=2,
    )
    idx_bottom = _fix_indices(c1)

    c2 = slab + ads
    attach_slab_constraints(
        c2,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=None,
        n_relax_top_slab_layers=2,
        surface_normal_axis=2,
    )
    idx_top_mode = _fix_indices(c2)

    assert idx_bottom == idx_top_mode == [0, 1]


def test_attach_slab_constraints_relax_top_all_layers_no_fixatoms() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)
    combined = slab + ads
    attach_slab_constraints(
        combined,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=None,
        n_relax_top_slab_layers=3,
        surface_normal_axis=2,
    )
    assert combined.constraints == []


def test_attach_slab_constraints_both_layer_modes_rejected() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    combined = slab + Atoms("Pt", positions=[[0, 0, 5]], cell=slab.cell, pbc=False)
    with pytest.raises(ValueError, match="at most one"):
        attach_slab_constraints(
            combined,
            len(slab),
            fix_all_slab_atoms=False,
            n_fix_bottom_slab_layers=1,
            n_relax_top_slab_layers=1,
            surface_normal_axis=2,
        )


def test_surface_config_relax_top_with_fix_all_rejected() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    with pytest.raises(ValueError, match="incompatible"):
        SurfaceSystemConfig(
            slab=slab,
            fix_all_slab_atoms=True,
            n_relax_top_slab_layers=1,
        )


def test_surface_config_both_layer_specs_rejected() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    with pytest.raises(ValueError, match="at most one"):
        SurfaceSystemConfig(
            slab=slab,
            fix_all_slab_atoms=False,
            n_fix_bottom_slab_layers=1,
            n_relax_top_slab_layers=1,
        )


def test_attach_slab_constraints_from_surface_config_matches_explicit() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=2,
    )
    c_cfg = slab + ads
    attach_slab_constraints_from_surface_config(c_cfg, cfg)
    c_exp = slab + ads
    attach_slab_constraints(
        c_exp,
        len(slab),
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=2,
        n_relax_top_slab_layers=None,
        surface_normal_axis=2,
    )
    assert _fix_indices(c_cfg) == _fix_indices(c_exp)


def test_surface_slab_constraint_summary_json_safe() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=2,
    )
    s = surface_slab_constraint_summary(cfg)
    assert s["n_slab_atoms"] == len(slab)
    assert s["surface_normal_axis"] == 2
    assert s["fix_all_slab_atoms"] is False
    assert s["n_fix_bottom_slab_layers"] is None
    assert s["n_relax_top_slab_layers"] == 2


def test_torchsim_prepare_relaxed_copy_attaches_fixatoms() -> None:
    pos = _three_layer_slab_positions()
    slab = Atoms("Pt6", positions=pos, cell=[10, 10, 10], pbc=False)
    ads = Atoms("Pt", positions=[[0.0, 0.0, 5.0]], cell=slab.cell, pbc=False)
    cand = slab + ads
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    out = _torchsim_prepare_relaxed_copy(cand, cfg, len(slab))
    assert out is not cand
    assert _fix_indices(out) == [0, 1, 2, 3]
