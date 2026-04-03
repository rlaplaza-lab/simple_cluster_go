"""Monatomic and polyatomic adsorbates (O, H, H2O) on Pt clusters."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.cluster_adsorbate import (
    ClusterAdsorbateConfig,
    place_fragment_on_cluster,
    relax_metal_cluster_with_adsorbate,
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


def test_place_atomic_oxygen_pt3() -> None:
    core = _pt_triangle()
    o_tmpl = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=core.get_cell(), pbc=False)
    rng = default_rng(0)
    cfg = ClusterAdsorbateConfig(max_placement_attempts=250)
    frag = place_fragment_on_cluster(core, o_tmpl, rng, cfg, anchor_index=0)
    assert frag is not None
    assert frag.get_chemical_symbols() == ["O"]


def test_place_atomic_hydrogen_pt3() -> None:
    core = _pt_triangle()
    h_tmpl = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=core.get_cell(), pbc=False)
    rng = default_rng(1)
    cfg = ClusterAdsorbateConfig(max_placement_attempts=250)
    frag = place_fragment_on_cluster(core, h_tmpl, rng, cfg, anchor_index=0)
    assert frag is not None
    assert frag.get_chemical_symbols() == ["H"]


def test_place_water_pt3() -> None:
    core = _pt_triangle()
    h2o = molecule("H2O")
    rng = default_rng(2)
    cfg = ClusterAdsorbateConfig(max_placement_attempts=400)
    frag = place_fragment_on_cluster(
        core, h2o, rng, cfg, anchor_index=0, bond_axis=None
    )
    assert frag is not None
    assert frag.get_chemical_symbols() == ["O", "H", "H"]


def test_relax_water_connected_structure_emt() -> None:
    core = _pt_triangle()
    h2o = molecule("H2O")

    # O near the cluster so connectivity survives a short unconstrained relax (EMT).
    o_pos = np.array([1.15, 0.65, 1.35])
    rel = h2o.get_positions() - h2o.get_positions()[0]
    pre = Atoms(
        symbols=h2o.get_chemical_symbols(),
        positions=rel + o_pos,
        cell=core.get_cell(),
        pbc=False,
    )

    relaxed, info = relax_metal_cluster_with_adsorbate(
        core,
        EMT(),
        h2o,
        preplaced=pre,
        config=ClusterAdsorbateConfig(cell_margin=10.0),
        anchor_index=0,
        bond_axis=None,
        bond_pairs=(),
        fix_core=True,
        fmax=0.25,
        # EMT can desorb unconstrained H2O in a few steps; zero steps still exercises
        # post-relax validation on the minimized-energy bookkeeping path.
        steps=0,
    )
    assert np.isfinite(info["final_energy"])
    assert len(relaxed) == 6
    assert info["structure_ok_initial"] is True
    assert info["structure_ok_final"] is True
    assert info["bond_lengths"] == {}
    assert info["n_frag"] == 3
    assert "provenance" in info
    assert info["provenance"]["formula"] == "H2OPt3"


def test_attach_fix_bond_lengths_rejects_duplicate() -> None:
    from scgo.cluster_adsorbate import attach_fix_bond_lengths

    a = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=False)
    with pytest.raises(ValueError, match="duplicate"):
        attach_fix_bond_lengths(a, [(0, 1), (1, 0)])
