"""Tests for TorchSim vs ASE NEB policy (MACE + UMA/FairChem)."""

from __future__ import annotations

import importlib.util

import pytest

from scgo.exceptions import SCGONotImplementedError, SCGOValidationError
from scgo.param_presets import get_ts_search_params
from scgo.utils.torchsim_policy import (
    calculator_name_supports_torchsim_batched_neb,
    is_uma_like_calculator,
    is_upet_like_calculator,
    resolve_ts_torchsim_flags,
)
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs


@pytest.mark.requires_uma
def test_is_uma_like_calculator():
    from scgo.calculators.uma_helpers import UMA

    assert is_uma_like_calculator(None) is False
    assert is_uma_like_calculator(object.__new__(UMA)) is True


@pytest.mark.requires_upet
def test_is_upet_like_calculator():
    from scgo.calculators.upet_helpers import UPET

    assert is_upet_like_calculator(None) is False
    assert is_upet_like_calculator(object.__new__(UPET)) is True


def test_calculator_name_supports_torchsim_only_mace():
    assert calculator_name_supports_torchsim_batched_neb("MACE") is True
    assert calculator_name_supports_torchsim_batched_neb("mace") is True
    assert calculator_name_supports_torchsim_batched_neb("UMA") is True
    assert calculator_name_supports_torchsim_batched_neb("UPET") is True
    assert calculator_name_supports_torchsim_batched_neb("EMT") is False


def test_resolve_ts_torchsim_flags_uma_requires_stacks(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name in ("torch_sim", "fairchem"):
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    us, up = resolve_ts_torchsim_flags("UMA", True, True)
    assert us is True and up is True


def test_resolve_ts_torchsim_flags_emt_rejects_torchsim_request():
    with pytest.raises(SCGOValidationError, match="does not support TorchSim"):
        resolve_ts_torchsim_flags("EMT", True, False)


@pytest.mark.parametrize(
    "mace_available",
    [True, False],
    ids=["mace_env", "no_mace_env"],
)
def test_resolve_ts_mace_depends_on_torch_sim_importability(
    mace_available: bool, monkeypatch
):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name == "torch_sim":
            return object() if mace_available else None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    if mace_available:
        us, up = resolve_ts_torchsim_flags("MACE", True, True)
        assert us is True and up is True
    else:
        with pytest.raises(SCGONotImplementedError, match="TorchSim was requested"):
            resolve_ts_torchsim_flags("MACE", True, True)


def test_get_ts_search_params_uma_has_torchsim_when_available(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name in ("torch_sim", "fairchem"):
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    ts = get_ts_search_params(
        calculator="UMA", calculator_kwargs={}, system_type="gas_cluster"
    )
    assert ts["use_torchsim"] is True
    assert ts["use_parallel_neb"] is True


def test_resolve_ts_torchsim_flags_parallel_defaults_on_when_none(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name == "torch_sim":
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)
    us, up = resolve_ts_torchsim_flags("MACE", True, None)
    assert us is True and up is True
    us, up = resolve_ts_torchsim_flags("MACE", True, False)
    assert us is True and up is False
    us, up = resolve_ts_torchsim_flags("MACE", False, None)
    assert us is False and up is False

    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name in ("torch_sim", "upet", "metatomic_torchsim"):
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    us, up = resolve_ts_torchsim_flags("UPET", True, True)
    assert us is True and up is True


def test_coerce_ts_params_upet_sets_model_fields(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name in ("torch_sim", "upet", "metatomic_torchsim"):
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    ts = get_ts_search_params(
        calculator="UPET", calculator_kwargs={}, system_type="gas_cluster"
    )
    kw = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kw["use_torchsim"] is True
    tsp = kw["torchsim_params"]
    assert tsp["model_kind"] == "upet"
    assert tsp["upet_model_name"] == "pet-mad-s"
    assert tsp["upet_version"] == "1.5.0"


def test_coerce_ts_params_uma_sets_fairchem_model_fields(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_spec(name: str):
        if name in ("torch_sim", "fairchem"):
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)

    ts = get_ts_search_params(
        calculator="UMA", calculator_kwargs={}, system_type="gas_cluster"
    )
    kw = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kw["use_torchsim"] is True
    assert kw["use_parallel_neb"] is True
    tsp = kw["torchsim_params"]
    assert tsp["model_kind"] == "fairchem"
    assert tsp["fairchem_model_name"] == "uma-s-1p2"
    assert tsp["fairchem_task_name"] == "oc25"
