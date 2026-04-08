"""Tests for TS parameter presets and run-kwargs mapping."""

import pytest

from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_NEB_TANGENT_METHOD
from scgo.param_presets import get_ts_run_kwargs, get_ts_search_params


def test_ts_search_params_expose_dedupe_and_tolerance_defaults():
    ts = get_ts_search_params()

    # Defaults must exist and be coherent with project-wide defaults
    assert ts.get("dedupe_minima", None) is True
    assert ts.get("minima_energy_tolerance", None) == pytest.approx(
        DEFAULT_ENERGY_TOLERANCE
    )

    # Ensure get_ts_run_kwargs maps those keys through to the TS runner kwargs
    kwargs = get_ts_run_kwargs(ts)
    assert kwargs["dedupe_minima"] is True
    assert kwargs["minima_energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
    assert kwargs.get("neb_interpolation_mic") is False
    assert kwargs.get("neb_tangent_method") == DEFAULT_NEB_TANGENT_METHOD
    assert kwargs.get("similarity_pair_cor_max") == pytest.approx(0.1)


def test_ts_search_params_allow_overrides():
    ts = get_ts_search_params()
    ts["dedupe_minima"] = False
    ts["minima_energy_tolerance"] = 0.05

    kwargs = get_ts_run_kwargs(ts)
    assert kwargs["dedupe_minima"] is False
    assert kwargs["minima_energy_tolerance"] == pytest.approx(0.05)


def test_ts_search_surface_regime_mic_and_fmax():
    ts = get_ts_search_params(regime="surface")
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_n_images"] == 5
    assert ts["neb_fmax"] == pytest.approx(0.05)
    assert ts["torchsim_fmax"] == pytest.approx(0.05)
    assert ts["neb_climb"] is True
    assert ts["neb_align_endpoints"] is False
    kwargs = get_ts_run_kwargs(ts)
    assert kwargs["neb_interpolation_mic"] is True
    assert kwargs["neb_n_images"] == 5
    assert kwargs["neb_climb"] is True
    assert kwargs["neb_align_endpoints"] is False


def test_ts_search_regime_invalid():
    with pytest.raises(ValueError, match="regime"):
        get_ts_search_params(regime="invalid")  # type: ignore[arg-type]


def test_ts_search_step_defaults_can_be_auto():
    """TS presets should expose 'auto' for NEB/TorchSim max-steps and pass-through to run kwargs."""
    ts = get_ts_search_params()

    # Defaults changed to 'auto' so callers can request size-dependent steps
    assert ts.get("neb_maxsteps") == "auto"
    assert ts.get("torchsim_maxsteps") == "auto"

    kwargs = get_ts_run_kwargs(ts)
    assert kwargs["neb_steps"] == "auto"
    assert kwargs["torchsim_params"]["max_steps"] == "auto"


def test_loaders_default_to_final_unique_minima():
    """Public loaders should default to final_unique_minimum rows only."""
    from scgo.database.helpers import (
        extract_minima_from_database_file,
        extract_transition_states_from_database_file,
        load_previous_run_results,
    )
    from scgo.ts_search.transition_state_io import (
        load_minima_by_composition,
        load_transition_states_by_composition,
    )

    # require_final
    assert extract_minima_from_database_file.__defaults__[1] is True
    # prefer_final_unique
    assert load_previous_run_results.__defaults__[3] is True
    assert load_minima_by_composition.__defaults__[1] is True
    # require_final_unique_ts
    assert extract_transition_states_from_database_file.__defaults__[1] is True
    assert load_transition_states_by_composition.__defaults__[1] is True
