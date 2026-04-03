"""Tests for ASE FixAtoms -> TorchSim FixAtoms batch mapping."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.constraints import FixAtoms as ASEFixAtoms

from scgo.calculators.torchsim_helpers import (
    build_torchsim_fixatoms_from_ase_batch,
    collect_ase_fixatoms_indices,
)


def test_collect_ase_fixatoms_indices_empty() -> None:
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert collect_ase_fixatoms_indices(a) == []


def test_collect_ase_fixatoms_indices_one_constraint() -> None:
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a.set_constraint(ASEFixAtoms(indices=[0]))
    assert collect_ase_fixatoms_indices(a) == [0]


def test_build_torchsim_fixatoms_batch_global_indices() -> None:
    torch = pytest.importorskip("torch")

    s1 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s1.set_constraint(ASEFixAtoms(indices=[0]))
    s2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s2.set_constraint(ASEFixAtoms(indices=[1]))

    c = build_torchsim_fixatoms_from_ase_batch([s1, s2], device=torch.device("cpu"))
    assert c is not None
    assert c.atom_idx.tolist() == [0, 3]


def test_build_torchsim_fixatoms_batch_none_when_unconstrained() -> None:
    torch = pytest.importorskip("torch")

    a = Atoms("H", positions=[[0, 0, 0]])
    assert (
        build_torchsim_fixatoms_from_ase_batch([a], device=torch.device("cpu")) is None
    )
