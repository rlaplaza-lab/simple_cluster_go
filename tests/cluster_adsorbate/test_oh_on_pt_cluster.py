"""Tests for OH placement and relaxation on small Pt clusters (EMT)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.cluster_adsorbate import (
    ClusterOHConfig,
    attach_oh_bond_constraint,
    combine_core_adsorbate,
    place_oh_on_cluster,
    relax_metal_cluster_with_oh,
)
from scgo.utils.ts_provenance import CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION


def _pt_linear_dimer() -> Atoms:
    return Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]],
        cell=[18.0, 18.0, 18.0],
        pbc=False,
    )


def _pt_triangle() -> Atoms:
    return Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [1.15, 2.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )


def test_place_oh_succeeds_pt3_fixed_seed() -> None:
    core = _pt_triangle()
    rng = default_rng(42)
    cfg = ClusterOHConfig(max_placement_attempts=200)
    oh = place_oh_on_cluster(core, rng, cfg)
    assert oh is not None
    assert oh.get_chemical_symbols() == ["O", "H"]
    combined = combine_core_adsorbate(core, oh)
    assert len(combined) == 5


def test_attach_oh_bond_constraint_rejects_bad_symbols() -> None:
    a = Atoms("OH", positions=[[0, 0, 0], [0.96, 0, 0]], cell=[10, 10, 10], pbc=False)
    attach_oh_bond_constraint(a, 0, 1)
    b = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]], cell=[10, 10, 10], pbc=False)
    with pytest.raises(ValueError, match="must be O"):
        attach_oh_bond_constraint(b, 0, 1)


def test_attach_oh_bond_constraint_rejects_bad_indices() -> None:
    a = Atoms("OH", positions=[[0, 0, 0], [0.96, 0, 0]], cell=[10, 10, 10], pbc=False)
    with pytest.raises(ValueError, match="Invalid"):
        attach_oh_bond_constraint(a, 0, 5)


def test_oh_relax_reports_connected_structure_emt() -> None:
    core = _pt_linear_dimer()
    d0 = 0.96
    n = np.array([0.0, 0.0, 1.0])
    # O within typical Pt–O connectivity threshold relative to the dimer
    o_pos = np.array([1.15, 0.0, 1.55])
    h_pos = o_pos + d0 * n
    pre = Atoms(
        "OH", positions=np.vstack([o_pos, h_pos]), cell=core.get_cell(), pbc=False
    )

    relaxed, info = relax_metal_cluster_with_oh(
        core,
        EMT(),
        preplaced=pre,
        fix_core=True,
        fmax=0.15,
        steps=120,
        config=ClusterOHConfig(cell_margin=8.0),
    )
    assert np.isfinite(info["final_energy"])
    assert info["structure_ok_initial"] is True
    assert info["structure_ok_final"] is True
    assert "oh_distance" in info
    prov = info["provenance"]
    assert (
        prov["cluster_adsorbate_schema_version"]
        == CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION
    )
    assert prov["calculator_class"] == "EMT"
    assert prov["n_frag"] == 2


def test_relax_metal_cluster_with_oh_with_placement_emt() -> None:
    core = _pt_triangle()
    rng = default_rng(123)
    cfg = ClusterOHConfig(max_placement_attempts=300, height_min=0.85, height_max=2.0)
    relaxed, info = relax_metal_cluster_with_oh(
        core,
        EMT(),
        rng=rng,
        config=cfg,
        fix_core=True,
        fmax=0.2,
        steps=100,
    )
    assert len(relaxed) == 5
    assert np.isfinite(info["final_energy"])
    assert info["structure_ok_initial"] is True
    assert info["structure_ok_final"] is True


def test_preplaced_wrong_length_raises() -> None:
    core = _pt_linear_dimer()
    bad = Atoms("O", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
    with pytest.raises(ValueError, match="fragment_template"):
        relax_metal_cluster_with_oh(core, EMT(), preplaced=bad)
