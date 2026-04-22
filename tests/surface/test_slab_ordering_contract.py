"""Tests for slab-first atom ordering required by SurfaceSystemConfig."""

from __future__ import annotations

import json

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.database.metadata import add_metadata
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints_from_surface_config
from scgo.surface.deposition import combine_slab_adsorbate
from scgo.surface.validation import (
    validate_stored_slab_adsorbate_metadata,
    validate_surface_config_slab_prefix,
)


def test_validate_surface_config_slab_prefix_accepts_matching_combine() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab)
    ads = Atoms("Pt", positions=[[1.0, 1.0, 8.0]], cell=slab.cell, pbc=slab.pbc)
    combined = combine_slab_adsorbate(slab, ads)
    validate_surface_config_slab_prefix(combined, cfg)


def test_validate_surface_config_slab_prefix_rejects_swapped_adsorbate_first() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab)
    ads = Atoms("Au", positions=[[1.0, 1.0, 8.0]], cell=slab.cell, pbc=slab.pbc)
    bad = ads + slab.copy()
    with pytest.raises(ValueError, match="Slab-first ordering contract"):
        validate_surface_config_slab_prefix(bad, cfg)


def test_validate_surface_config_slab_prefix_rejects_too_short() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab)
    short = Atoms("Pt", positions=[[0.0, 0.0, 0.0]], cell=slab.cell, pbc=slab.pbc)
    with pytest.raises(ValueError, match="at least"):
        validate_surface_config_slab_prefix(short, cfg)


def test_attach_slab_constraints_from_surface_config_validates_first() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab)
    ads = Atoms("Au", positions=[[1.0, 1.0, 8.0]], cell=slab.cell, pbc=slab.pbc)
    bad = ads + slab.copy()
    with pytest.raises(ValueError, match="Slab-first ordering contract"):
        attach_slab_constraints_from_surface_config(bad, cfg)


def test_validate_stored_slab_adsorbate_metadata_with_json() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    ads = Atoms("Cu", positions=[[1.0, 1.0, 8.0]], cell=slab.cell, pbc=slab.pbc)
    combined = combine_slab_adsorbate(slab, ads)
    syms = slab.get_chemical_symbols()
    add_metadata(
        combined,
        n_slab_atoms=len(slab),
        system_type="surface_cluster_adsorbate",
        slab_chemical_symbols_json=json.dumps(syms),
    )
    validate_stored_slab_adsorbate_metadata(combined)

    combined2 = combined.copy()
    # scramble first slab atom symbol in a copy (invalid)
    nums = combined2.get_atomic_numbers()
    nums[0] = 79  # Au vs Pt
    combined2.set_atomic_numbers(nums)
    with pytest.raises(ValueError, match="slab_chemical_symbols_json"):
        validate_stored_slab_adsorbate_metadata(combined2)


def test_validate_stored_slab_adsorbate_metadata_skips_without_json() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    ads = Atoms("Cu", positions=[[1.0, 1.0, 8.0]], cell=slab.cell, pbc=slab.pbc)
    combined = combine_slab_adsorbate(slab, ads)
    add_metadata(
        combined,
        n_slab_atoms=len(slab),
        system_type="surface_cluster_adsorbate",
    )
    validate_stored_slab_adsorbate_metadata(combined)


def test_metadata_validation_keeps_adsorbate_order_for_pt5_two_oh() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    ads = Atoms(
        "Pt5OHOH",
        positions=[
            [1.0, 1.0, 8.0],
            [2.0, 1.0, 8.2],
            [3.0, 1.0, 8.4],
            [1.5, 2.0, 8.6],
            [2.5, 2.0, 8.8],
            [1.2, 1.2, 9.3],
            [1.6, 1.2, 9.7],
            [2.2, 1.2, 9.4],
            [2.6, 1.2, 9.8],
        ],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    combined = combine_slab_adsorbate(slab, ads)
    add_metadata(
        combined,
        n_slab_atoms=len(slab),
        system_type="surface_cluster_adsorbate",
        slab_chemical_symbols_json=json.dumps(slab.get_chemical_symbols()),
    )
    validate_stored_slab_adsorbate_metadata(combined)
    assert combined.get_chemical_symbols()[len(slab) :] == [
        "Pt",
        "Pt",
        "Pt",
        "Pt",
        "Pt",
        "O",
        "H",
        "O",
        "H",
    ]
