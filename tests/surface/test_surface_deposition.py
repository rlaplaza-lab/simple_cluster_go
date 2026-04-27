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
from scgo.surface.deposition import (
    create_deposited_cluster,
    create_deposited_cluster_batch,
    slab_surface_extreme,
)
from scgo.surface.objectives import adsorption_energy


@pytest.fixture
def pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def _build_surface_template(slab: Atoms, composition: list[str]) -> Atoms:
    n_adsorbate = len(composition)
    padded_positions = np.vstack([slab.get_positions(), np.zeros((n_adsorbate, 3))])
    return Atoms(
        symbols=list(slab.get_chemical_symbols()) + composition,
        positions=padded_positions,
        cell=slab.cell,
        pbc=slab.pbc,
    )


def _build_surface_blmin(slab: Atoms, composition: list[str]) -> dict:
    n_slab = len(slab)
    template = _build_surface_template(slab, composition)
    top_indices = range(n_slab, n_slab + len(composition))
    return closest_distances_generator(
        get_all_atom_types(template, top_indices),
        ratio_of_covalent_radii=0.7,
    )


def _surface_config(slab: Atoms, *, max_placement_attempts: int) -> SurfaceSystemConfig:
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.5,
        max_placement_attempts=max_placement_attempts,
    )


def test_slab_surface_extreme(pt_slab: Atoms) -> None:
    zmax = slab_surface_extreme(pt_slab, 2, upper=True)
    assert zmax == pytest.approx(np.max(pt_slab.get_positions()[:, 2]))


def test_create_deposited_cluster_len_and_order(pt_slab: Atoms) -> None:
    composition = ["Pt", "Pt"]
    n_slab = len(pt_slab)
    blmin = _build_surface_blmin(pt_slab, composition)
    cfg = _surface_config(pt_slab, max_placement_attempts=500)
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
        create_ga_pairing(
            wrong,
            len(composition),
            default_rng(0),
            slab_atoms=pt_slab,
            system_type="surface_cluster",
        )

    tmpl = _build_surface_template(pt_slab, composition)
    pairing = create_ga_pairing(
        tmpl,
        len(composition),
        default_rng(0),
        slab_atoms=pt_slab,
        system_type="surface_cluster",
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


def test_create_deposited_cluster_batch_threaded_is_seed_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    pt_slab: Atoms,
) -> None:
    composition = ["Pt", "Pt"]
    blmin = _build_surface_blmin(pt_slab, composition)
    cfg = _surface_config(pt_slab, max_placement_attempts=50)

    def _fake_create_deposited_cluster(
        composition: list[str],
        slab: Atoms,
        blmin: dict,
        rng: np.random.Generator,
        config: SurfaceSystemConfig,
        previous_search_glob: str = "**/*.db",
        **kwargs: object,
    ) -> Atoms:
        _ = (composition, blmin, config, previous_search_glob, kwargs)
        marker = float(rng.integers(0, 1_000_000))
        out = slab.copy()
        out.info["task_marker"] = marker
        return out

    monkeypatch.setattr(
        "scgo.surface.deposition.create_deposited_cluster",
        _fake_create_deposited_cluster,
    )

    rng1 = default_rng(2026)
    rng2 = default_rng(2026)
    batch1 = create_deposited_cluster_batch(
        composition,
        pt_slab,
        blmin,
        n_structures=12,
        rng=rng1,
        config=cfg,
        n_jobs=4,
    )
    batch2 = create_deposited_cluster_batch(
        composition,
        pt_slab,
        blmin,
        n_structures=12,
        rng=rng2,
        config=cfg,
        n_jobs=4,
    )

    marks1 = [atoms.info["task_marker"] for atoms in batch1]
    marks2 = [atoms.info["task_marker"] for atoms in batch2]
    assert marks1 == marks2


def test_create_deposited_cluster_preserves_adsorbate_symbol_order_with_two_oh(
    pt_slab: Atoms,
) -> None:
    composition = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"]
    n_slab = len(pt_slab)
    dummy = np.vstack([pt_slab.get_positions(), np.zeros((len(composition), 3))])
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
    rng = default_rng(2026)

    # Use adsorbate definition to preserve order via hierarchical initialization
    adsorbate_definition = {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
    }

    # Create a simple fragment template for the adsorbate
    from ase import Atoms as AtomsClass

    adsorbate_fragment_template = AtomsClass(
        symbols=["O", "H", "O", "H"],
        positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]],
        cell=[10, 10, 10],
        pbc=False,
    )

    ads_sys = create_deposited_cluster(
        composition,
        pt_slab,
        blmin,
        rng,
        cfg,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=adsorbate_fragment_template,
    )
    assert ads_sys is not None
    assert ads_sys.get_chemical_symbols()[n_slab:] == composition


def test_create_deposited_cluster_batch_preserves_adsorbate_symbol_order_with_two_oh(
    pt_slab: Atoms,
) -> None:
    composition = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"]
    n_slab = len(pt_slab)
    dummy = np.vstack([pt_slab.get_positions(), np.zeros((len(composition), 3))])
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

    # Use adsorbate definition to preserve order via hierarchical initialization
    adsorbate_definition = {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
    }

    # Create a simple fragment template for the adsorbate
    from ase import Atoms as AtomsClass

    adsorbate_fragment_template = AtomsClass(
        symbols=["O", "H", "O", "H"],
        positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]],
        cell=[10, 10, 10],
        pbc=False,
    )

    batch = create_deposited_cluster_batch(
        composition,
        pt_slab,
        blmin,
        n_structures=5,
        rng=default_rng(2026),
        config=cfg,
        n_jobs=2,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=adsorbate_fragment_template,
    )
    assert len(batch) == 5
    for atoms in batch:
        assert atoms.get_chemical_symbols()[n_slab:] == composition
