"""Tests for UMA / fairchem calculator wrapper."""

from __future__ import annotations

import importlib.util
import warnings

import pytest


def test_uma_class_is_ase_calculator():
    from ase.calculators.calculator import Calculator

    from scgo.calculators.uma_helpers import UMA

    assert issubclass(UMA, Calculator)


def test_get_calculator_class_uma():
    from scgo.utils.run_helpers import get_calculator_class

    cls = get_calculator_class("UMA")
    assert cls.__name__ == "UMA"


def test_get_default_uma_params():
    from scgo.param_presets import get_default_uma_params

    p = get_default_uma_params()
    assert p["calculator"] == "UMA"
    assert p["calculator_kwargs"]["model_name"] == "uma-s-1"


def test_get_ts_search_params_uma_disables_torchsim():
    from scgo.param_presets import get_ts_search_params_uma

    ts = get_ts_search_params_uma()
    assert ts["calculator"] == "UMA"
    assert ts["use_torchsim"] is False
    assert ts["use_parallel_neb"] is False


def test_both_mlip_stacks_warns_when_both_importable():
    if (
        importlib.util.find_spec("mace") is None
        or importlib.util.find_spec("fairchem") is None
    ):
        pytest.skip("needs both mace and fairchem importable")

    from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        ensure_mace_uma_not_both_installed()
    assert any("MACE stack" in str(w.message) for w in wrec)
