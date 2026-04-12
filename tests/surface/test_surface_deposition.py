"""Tests for slab + cluster deposition and GA wiring."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import closest_distances_generator, get_all_atom_types
from numpy.random import default_rng

from scgo.algorithms.ga_common import create_ga_pairing
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.surface.deposition import create_deposited_cluster, slab_surface_extreme
from scgo.surface.objectives import adsorption_energy


@pytest.fixture
def pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def test_slab_surface_extreme(pt_slab: Atoms) -> None:
    zmax = slab_surface_extreme(pt_slab, 2, upper=True)
    assert zmax == pytest.approx(np.max(pt_slab.get_positions()[:, 2]))


def test_create_deposited_cluster_len_and_order(pt_slab: Atoms) -> None:
    composition = ["Pt", "Pt"]
    n_slab = len(pt_slab)
    dummy = np.vstack([pt_slab.get_positions(), np.zeros((2, 3))])
    tmpl = Atoms(
        symbols=list(pt_slab.get_chemical_symbols()) + composition,
        positions=dummy,
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    idx_top = range(n_slab, n_slab + len(composition))
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, idx_top),
        ratio_of_covalent_radii=0.7,
    )
    cfg = SurfaceSystemConfig(
        slab=pt_slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.5,
        max_placement_attempts=500,
    )
    rng = default_rng(12345)
    ads_sys = create_deposited_cluster(composition, pt_slab, blmin, rng, cfg)
    assert ads_sys is not None
    assert len(ads_sys) == n_slab + len(composition)
    # Adsorbate atoms should lie above the slab top along z
    z_top_slab = slab_surface_extreme(pt_slab, 2, upper=True)
    ads_pos = ads_sys.get_positions()[n_slab:]
    assert np.min(ads_pos[:, 2]) > z_top_slab - 0.1


def test_create_ga_pairing_surface_requires_matching_template(pt_slab: Atoms) -> None:
    composition = ["Pt"]
    n_slab = len(pt_slab)
    wrong = Atoms("Pt", positions=[[0, 0, 0]], cell=pt_slab.cell, pbc=pt_slab.pbc)
    with pytest.raises(ValueError, match="surface GA"):
        create_ga_pairing(wrong, len(composition), default_rng(0), slab_atoms=pt_slab)

    dummy = np.vstack([pt_slab.get_positions(), np.zeros((1, 3))])
    tmpl = Atoms(
        symbols=list(pt_slab.get_chemical_symbols()) + composition,
        positions=dummy,
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    pairing = create_ga_pairing(
        tmpl, len(composition), default_rng(0), slab_atoms=pt_slab
    )
    if hasattr(pairing, "primary"):
        assert len(pairing.primary.slab) == n_slab
    else:
        assert len(pairing.slab) == n_slab


def test_attach_slab_constraints_fix_all(pt_slab: Atoms) -> None:
    top = Atoms("Pt", positions=[[0.0, 0.0, 20.0]], cell=pt_slab.cell, pbc=pt_slab.pbc)
    combined = pt_slab + top
    attach_slab_constraints(
        combined,
        len(pt_slab),
        fix_all_slab_atoms=True,
        n_fix_bottom_slab_layers=None,
        surface_normal_axis=2,
    )
    assert len(combined.constraints) == 1


def test_adsorption_energy_sign() -> None:
    e = adsorption_energy(-10.0, -5.0, -4.0)
    assert e == pytest.approx(-1.0)
