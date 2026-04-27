"""Tests for runtime mobile core+adsorbate symbol order vs adsorbate_definition."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.surface.deposition import combine_slab_adsorbate
from scgo.system_types import (
    validate_mobile_symbols_match_adsorbate_definition,
    validate_structure_for_system_type,
)


def _def_pt5_oh() -> dict:
    return {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
    }


def _well_spaced_pt5_oh_positions() -> np.ndarray:
    # Avoid clashes for validate_combined_cluster_structure
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [0.0, 2.8, 0.0],
            [2.6, 2.8, 0.0],
            [1.3, 1.4, 2.0],
            [1.0, 0.5, 3.2],
            [1.5, 2.0, 3.0],
        ],
        dtype=float,
    )


def test_validate_mobile_symbols_gas_accept() -> None:
    atoms = Atoms(
        "Pt5OH",
        positions=[[0, 0, 0]] * 7,
        cell=[20, 20, 20],
        pbc=False,
    )
    validate_mobile_symbols_match_adsorbate_definition(atoms, 0, _def_pt5_oh())


def test_validate_mobile_symbols_gas_reject_wrong_order() -> None:
    atoms = Atoms(
        "OHPt5",
        positions=[[0, 0, 0]] * 7,
        cell=[20, 20, 20],
        pbc=False,
    )
    with pytest.raises(ValueError, match="Mobile symbols mismatch"):
        validate_mobile_symbols_match_adsorbate_definition(atoms, 0, _def_pt5_oh())


def test_validate_structure_gas_adsorbate_accepts() -> None:
    atoms = Atoms(
        "Pt5OH",
        positions=_well_spaced_pt5_oh_positions(),
        cell=[24, 24, 24],
        pbc=False,
    )
    validate_structure_for_system_type(
        atoms,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=_def_pt5_oh(),
    )


def test_validate_mobile_symbols_after_slab_prefix_accepts() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    n_s = len(slab)
    ads = Atoms(
        "Pt5OHOH",
        positions=[[1.0, 1.0, 10.0 + 0.2 * i] for i in range(9)],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    combined = combine_slab_adsorbate(slab, ads)
    ad_def: dict = {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
    }
    validate_mobile_symbols_match_adsorbate_definition(combined, n_s, ad_def)


def test_validate_mobile_symbols_after_slab_prefix_rejects_wrong_block_order() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    n_s = len(slab)
    bad_mobile = Atoms(
        "OHOHPt5",
        positions=[[1.0, 1.0, 10.0 + 0.2 * i] for i in range(9)],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    combined = combine_slab_adsorbate(slab, bad_mobile)
    ad_def: dict = {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
    }
    with pytest.raises(ValueError, match="Mobile symbols mismatch"):
        validate_mobile_symbols_match_adsorbate_definition(combined, n_s, ad_def)
