"""Tests for TorchSim vs ASE NEB policy (MACE-only batched paths)."""

from __future__ import annotations

import importlib.util

import pytest

from scgo.param_presets import get_ts_run_kwargs, get_ts_search_params
from scgo.utils.torchsim_policy import (
    calculator_name_supports_torchsim_batched_neb,
    coerce_find_transition_state_torchsim,
    is_uma_like_calculator,
    mace_torchsim_stack_available,
    resolve_ts_torchsim_flags,
)


def test_is_uma_like_calculator():
    from scgo.calculators.uma_helpers import UMA

    assert is_uma_like_calculator(None) is False
    assert is_uma_like_calculator(object.__new__(UMA)) is True


def test_calculator_name_supports_torchsim_only_mace():
    assert calculator_name_supports_torchsim_batched_neb("MACE") is True
    assert calculator_name_supports_torchsim_batched_neb("mace") is True
    assert calculator_name_supports_torchsim_batched_neb("UMA") is False
    assert calculator_name_supports_torchsim_batched_neb("EMT") is False


def test_resolve_ts_torchsim_flags_uma_disables_both():
    us, up = resolve_ts_torchsim_flags("UMA", True, True, logger=None)
    assert us is False and up is False


def test_resolve_ts_torchsim_flags_emt_disables_torchsim():
    us, up = resolve_ts_torchsim_flags("EMT", True, False, logger=None)
    assert us is False and up is False


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
    assert mace_torchsim_stack_available() is mace_available

    us, up = resolve_ts_torchsim_flags("MACE", True, True, logger=None)
    assert us is mace_available
    assert up == (mace_available and True)


def test_get_ts_search_params_uma_has_no_torchsim():
    ts = get_ts_search_params(calculator="UMA", calculator_kwargs={})
    assert ts["use_torchsim"] is False
    assert ts["use_parallel_neb"] is False


def test_get_ts_run_kwargs_uma_disables_torchsim():
    ts = get_ts_search_params(calculator="UMA", calculator_kwargs={})
    kw = get_ts_run_kwargs(ts)
    assert kw["use_torchsim"] is False
    assert kw["use_parallel_neb"] is False


def test_coerce_find_transition_state_torchsim_with_uma_calculator():
    from unittest.mock import MagicMock

    from scgo.calculators.uma_helpers import UMA

    log = MagicMock()
    calc = object.__new__(UMA)

    assert (
        coerce_find_transition_state_torchsim(
            use_torchsim=True,
            calculator=calc,
            pair_id="0_1",
            logger=log,
        )
        is False
    )
    log.warning.assert_called()
