"""Tests for supported-cluster deposit validation (surface contact + connectivity)."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.surface.deposition import combine_slab_adsorbate, slab_surface_extreme
from scgo.surface.validation import validate_supported_cluster_deposit


@pytest.fixture
def pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def test_validate_supported_cluster_deposit_accepts_typical_deposit(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    # Single Pt adsorbate directly above a surface atom (approximate atop).
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top + 2.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert ok, msg


def test_validate_supported_cluster_deposit_rejects_no_surface_contact(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    # Far above the slab: no pair within connectivity distance.
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top + 12.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "No adsorbate–slab pair" in msg


def test_validate_supported_cluster_deposit_rejects_disconnected_adsorbate(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    # Two Pt atoms far apart in the adsorbate (two components).
    ads = Atoms(
        "Pt2",
        positions=[
            [0.0, 0.0, z_top + 2.0],
            [5.0, 5.0, z_top + 2.0],
        ],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "Adsorbate validation failed" in msg


def test_validate_supported_cluster_deposit_rejects_penetration(pt_slab: Atoms) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top - 1.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "penetrates" in msg
