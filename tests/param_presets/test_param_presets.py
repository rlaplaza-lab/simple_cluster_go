"""Tests for TS parameter presets and run-kwargs mapping."""

import pytest
from ase.build import fcc111

from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_NEB_TANGENT_METHOD
from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.run_helpers import resolve_auto_params
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs


def test_ts_search_params_expose_dedupe_and_tolerance_defaults():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("dedupe_minima", None) is True
    assert ts.get("minima_energy_tolerance", None) == pytest.approx(
        DEFAULT_ENERGY_TOLERANCE
    )

    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs["dedupe_minima"] is True
    assert kwargs["minima_energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
    assert kwargs.get("neb_interpolation_mic") is False
    assert kwargs.get("neb_tangent_method") == DEFAULT_NEB_TANGENT_METHOD
    assert kwargs.get("similarity_pair_cor_max") == pytest.approx(0.1)


def test_ts_search_params_allow_overrides():
    ts = get_ts_search_params(system_type="gas_cluster")
    ts["dedupe_minima"] = False
    ts["minima_energy_tolerance"] = 0.05

    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs["dedupe_minima"] is False
    assert kwargs["minima_energy_tolerance"] == pytest.approx(0.05)


def test_ts_search_params_surface_config_forwarded_to_run_kwargs():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts.get("surface_config") is cfg
    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs.get("surface_config") is cfg


def test_coerce_ts_surface_config_defaults_to_none():
    ts = get_ts_search_params(system_type="gas_cluster")
    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs.get("surface_config") is None


def test_ts_search_surface_regime_mic_and_fmax():
    ts = get_ts_search_params(system_type="surface_cluster")
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_n_images"] == 5
    assert ts["neb_fmax"] == pytest.approx(0.1)
    assert ts["torchsim_fmax"] == pytest.approx(0.1)
    assert ts["neb_steps"] == 500
    assert ts["torchsim_max_steps"] == 500
    assert ts["neb_climb"] is False
    assert ts["neb_interpolation_method"] == "idpp"
    assert ts["neb_align_endpoints"] is False
    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs["neb_interpolation_mic"] is True
    assert kwargs["neb_n_images"] == 5
    assert kwargs["neb_climb"] is False
    assert kwargs["neb_fmax"] == pytest.approx(0.1)
    assert kwargs["neb_steps"] == 500
    assert kwargs["neb_interpolation_method"] == "idpp"
    assert kwargs["torchsim_params"]["force_tol"] == pytest.approx(0.1)
    assert kwargs["torchsim_params"]["max_steps"] == 500
    assert kwargs["neb_align_endpoints"] is False


def test_ts_search_step_defaults_can_be_auto():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("neb_steps") == "auto"
    assert ts.get("torchsim_max_steps") == "auto"

    kwargs = coerce_ts_params_to_runner_kwargs(ts)
    assert kwargs["neb_steps"] == "auto"
    assert kwargs["torchsim_params"]["max_steps"] == "auto"


def test_loaders_default_to_final_unique_minima():
    """Public loaders should default to final_unique_minimum rows only."""
    from scgo.database.helpers import (
        extract_minima_from_database_file,
        extract_transition_states_from_database_file,
        load_previous_run_results,
    )
    from scgo.ts_search.transition_state_io import load_minima_by_composition

    # require_final
    assert extract_minima_from_database_file.__defaults__[1] is True
    # prefer_final_unique
    assert load_previous_run_results.__defaults__[3] is True
    assert load_minima_by_composition.__defaults__[1] is True
    # require_final_unique_ts
    assert extract_transition_states_from_database_file.__defaults__[1] is True


def _fake_torchsim_go(seed: int) -> dict:
    from scgo.param_presets import get_default_params

    p = get_default_params()
    p["seed"] = seed
    return p


def _build_mace_go_ts_like_runner(
    seed: int,
    *,
    niter: int,
    population_size: int,
    max_pairs: int,
    system_type: str,
    surface_config: SurfaceSystemConfig | None = None,
) -> tuple[dict, dict]:
    """Same shape as ``runners/run_pt5_*.py`` (MACE + TorchSim base + TS preset)."""
    go = get_torchsim_ga_params(seed)
    go["calculator"] = "MACE"
    ga = go["optimizer_params"]["ga"]
    ga["system_type"] = system_type
    ga["niter"] = niter
    ga["population_size"] = population_size
    if surface_config is not None:
        ga["surface_config"] = surface_config
    ts = get_ts_search_params(
        system_type=system_type,
        surface_config=surface_config,
    )
    ts["max_pairs"] = max_pairs
    return go, ts


def test_production_style_mace_go_ts_gas(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    go, ts = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="gas_cluster",
    )
    ga = go["optimizer_params"]["ga"]
    assert ga["niter"] == 8
    assert ga["population_size"] == 18
    assert ts["max_pairs"] == 12
    assert ts.get("surface_config") is None
    kw = coerce_ts_params_to_runner_kwargs(ts)
    assert kw["max_pairs"] == 12


def test_production_style_mace_go_ts_surface_has_surface_config(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    go, ts = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="surface_cluster",
        surface_config=cfg,
    )
    ga = go["optimizer_params"]["ga"]
    assert ga["surface_config"] is cfg
    assert resolve_auto_params(ga, ["Pt"] * 5, "ga")["niter_local_relaxation"] >= 400
    assert ts["surface_config"] is cfg
    assert coerce_ts_params_to_runner_kwargs(ts).get("surface_config") is cfg
