"""Tests for TS parameter presets and run-kwargs mapping."""

import pytest
from ase.build import fcc111

import scgo.param_presets as param_presets_module
from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_NEB_TANGENT_METHOD
from scgo.exceptions import SCGOValidationError
from scgo.pair_selection_defaults import (
    DEFAULT_PAIR_CORE_RMS_MAX_GAS,
    DEFAULT_PAIR_CORE_RMS_MAX_SURFACE,
)
from scgo.param_presets import (
    TS_DEFAULTS_BY_SYSTEM_TYPE,
    get_default_params,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    get_torchsim_ga_params,
    get_ts_defaults,
    get_ts_search_params,
    low_effort_neb_steps,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SYSTEM_TYPE_POLICIES, get_system_policy
from scgo.utils.run_helpers import initialize_ts_params, prepare_algorithm_kwargs
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs
from tests.constants import TS_FMAX_CONVERGED


def _surface_config_for_test() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    return SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)


def _ts_search_params_for(system_type: str) -> dict:
    if get_system_policy(system_type).uses_surface:
        return get_ts_search_params(
            system_type=system_type, surface_config=_surface_config_for_test()
        )
    return get_ts_search_params(system_type=system_type)


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_ts_defaults_match_system_policy_align_and_mic(system_type):
    """`TS_DEFAULTS_BY_SYSTEM_TYPE` must agree with `SystemPolicy` flags."""
    defaults = get_ts_defaults(system_type)
    policy = SYSTEM_TYPE_POLICIES[system_type]
    assert defaults["neb_align_endpoints"] is (not policy.neb_disable_alignment)
    assert defaults["neb_interpolation_mic"] is policy.neb_force_mic
    assert defaults["neb_surface_cell_remap"] is policy.neb_surface_cell_remap
    assert (
        defaults["neb_surface_lattice_rotation"] is policy.neb_surface_lattice_rotation
    )


def test_ts_defaults_keys_match_system_type_policies():
    """`TS_DEFAULTS_BY_SYSTEM_TYPE` keys must match `SYSTEM_TYPE_POLICIES`."""
    assert set(TS_DEFAULTS_BY_SYSTEM_TYPE) == set(SYSTEM_TYPE_POLICIES)


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_ts_defaults_fmax_matches_shared_constant(system_type):
    """Force-convergence thresholds are shared, not per system type."""
    defaults = get_ts_defaults(system_type)
    shared = float(param_presets_module.TS_NEB_FMAX)
    assert shared == TS_FMAX_CONVERGED, (
        f"Production TS_NEB_FMAX={shared} drifted from pinned "
        f"TS_FMAX_CONVERGED={TS_FMAX_CONVERGED}"
    )
    assert float(defaults["neb_fmax"]) == shared
    assert float(defaults["torchsim_fmax"]) == shared
    assert float(defaults["neb_fmax"]) == float(defaults["torchsim_fmax"])


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_get_ts_search_params_seeds_from_per_system_defaults(system_type):
    """Each system type's preset reflects its `get_ts_defaults` block."""
    ts = _ts_search_params_for(system_type)
    defaults = get_ts_defaults(system_type)
    for key, expected in defaults.items():
        assert ts[key] == expected, (
            f"{system_type}: ts_params[{key!r}]={ts[key]!r} != defaults[{key!r}]={expected!r}"
        )


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_coerce_sparse_ts_params_falls_back_to_per_system_defaults(system_type):
    """A sparse `ts_params` flows through with policy-coherent NEB defaults."""
    sparse: dict = {"calculator": "MACE"}
    if get_system_policy(system_type).uses_surface:
        sparse["surface_config"] = _surface_config_for_test()
    kwargs = coerce_ts_params_to_runner_kwargs(sparse, system_type=system_type)
    defaults = get_ts_defaults(system_type)
    for key in (
        "neb_align_endpoints",
        "neb_interpolation_mic",
        "neb_n_images",
        "neb_spring_constant",
        "neb_fmax",
        "neb_steps",
        "neb_climb",
        "neb_perturb_sigma",
        "neb_interpolation_method",
        "neb_tangent_method",
        "neb_surface_cell_remap",
        "neb_surface_lattice_rotation",
        "neb_surface_max_lattice_shift",
        "max_endpoint_mismatch",
    ):
        expected = defaults[key]
        assert kwargs[key] == expected, (
            f"{system_type}: kwargs[{key!r}]={kwargs[key]!r} != defaults[{key!r}]={expected!r}"
        )
    assert kwargs["torchsim_params"]["force_tol"] == defaults["torchsim_fmax"]
    assert kwargs["torchsim_params"]["max_steps"] == defaults["torchsim_max_steps"]
    assert "torchsim_fmax" not in kwargs
    assert "torchsim_max_steps" not in kwargs


def test_initialize_ts_params_then_coerce_matches_full_preset():
    initialized = initialize_ts_params(
        {"calculator": "EMT", "use_torchsim": False},
        system_type="gas_cluster",
    )
    kwargs = coerce_ts_params_to_runner_kwargs(initialized, system_type="gas_cluster")
    assert kwargs["params"]["calculator"] == "EMT"
    assert kwargs["use_torchsim"] is False


def test_ts_search_params_accepts_seed():
    ts = get_ts_search_params(system_type="gas_cluster", seed=99)
    assert ts["seed"] == 99


def test_ts_search_params_expose_dedupe_and_tolerance_defaults():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("dedupe_minima", None) is True
    assert ts.get("dedupe_ts", None) is True
    assert ts.get("minima_energy_tolerance", None) == pytest.approx(
        DEFAULT_ENERGY_TOLERANCE
    )
    assert ts.get("ts_energy_tolerance", None) == pytest.approx(
        DEFAULT_ENERGY_TOLERANCE
    )

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["dedupe_minima"] is True
    assert kwargs["minima_energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
    assert kwargs["dedupe_ts"] is True
    assert kwargs["ts_energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
    assert kwargs.get("neb_interpolation_mic") is False
    assert kwargs.get("neb_tangent_method") == DEFAULT_NEB_TANGENT_METHOD
    assert kwargs.get("similarity_pair_cor_max") == pytest.approx(0.1)


def test_ts_search_params_expose_adsorbate_subgraph_integrity_default():
    ts = get_ts_search_params(system_type="gas_cluster_adsorbate")
    assert ts["enforce_adsorbate_subgraph_integrity"] is True
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster_adsorbate")
    assert kwargs["enforce_adsorbate_subgraph_integrity"] is True


def test_ts_defaults_expose_promoted_thresholds():
    for system_type in TS_DEFAULTS_BY_SYSTEM_TYPE:
        d = get_ts_defaults(system_type)
        assert d["binding_penetration_tolerance_a"] == 0.1
        assert d["layer_cluster_threshold_ang"] == 0.4
        assert d["neb_interpolation_bond_tolerance_a"] == 0.5
        if system_type == "surface":
            assert d["neb_max_spurious_barrier"] == 50.0
            assert d["max_endpoint_mismatch"] == pytest.approx(3.0)
            assert d["neb_prescreen_clash_distance"] == pytest.approx(0.35)
        elif system_type == "surface_cluster":
            assert d["neb_max_spurious_barrier"] == 8.0
            assert d["max_endpoint_mismatch"] == pytest.approx(2.5)
        else:
            assert d["neb_max_spurious_barrier"] == 8.0


def test_ts_search_params_allow_cluster_fragmentation_for_surface_regimes():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    assert (
        get_ts_search_params(system_type="gas_cluster")["allow_cluster_fragmentation"]
        is False
    )
    assert (
        get_ts_search_params(system_type="surface_cluster", surface_config=cfg)[
            "allow_cluster_fragmentation"
        ]
        is True
    )
    assert (
        get_ts_search_params(system_type="surface", surface_config=cfg)[
            "allow_cluster_fragmentation"
        ]
        is True
    )
    assert (
        get_ts_search_params(
            system_type="surface_cluster_adsorbate", surface_config=cfg
        )["allow_cluster_fragmentation"]
        is False
    )
    assert (
        get_ts_search_params(system_type="surface_adsorbate", surface_config=cfg)[
            "allow_cluster_fragmentation"
        ]
        is True
    )


def test_low_effort_surface_cluster_ts_defaults_match_example_path():
    """Example ``surface_cluster`` TS must keep production physics, not recover knobs.

    ``example_pt5_graphite`` only overrides ``max_pairs`` / ``connectivity_factor``.
    Fragmentation + endpoint mismatch must already be on so low-effort NEBs can
    converge without a separate loosened re-run.
    """
    cfg = _surface_config_for_test()
    production = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    low = get_low_effort_ts_search_params(
        system_type="surface_cluster", surface_config=cfg
    )
    assert low["allow_cluster_fragmentation"] is True
    assert low["max_endpoint_mismatch"] == pytest.approx(2.5)
    assert low["neb_max_spurious_barrier"] == pytest.approx(8.0)
    assert (
        low["neb_prescreen_clash_distance"]
        == production["neb_prescreen_clash_distance"]
    )
    assert low["neb_climb"] is False
    assert low["neb_fmax"] == pytest.approx(production["neb_fmax"])
    assert low["neb_n_images"] == production["neb_n_images"]
    assert low["neb_steps"] < production["neb_steps"]


def test_adsorbate_ts_presets_enable_climb_and_mismatch_gate():
    gas = get_ts_search_params(system_type="gas_cluster_adsorbate")
    assert gas["neb_climb"] is True
    assert gas["neb_spring_constant"] == pytest.approx(0.5)
    assert gas["neb_fmax"] == pytest.approx(0.20)
    assert gas["torchsim_fmax"] == pytest.approx(0.20)
    assert gas["neb_n_images"] == 7
    assert gas["neb_steps"] == 4000
    assert gas["use_parallel_neb"] is True
    assert gas["max_endpoint_mismatch"] == pytest.approx(1.25)
    assert gas["energy_gap_threshold"] == pytest.approx(0.75)
    assert gas["pair_core_rms_max"] == pytest.approx(1.5)
    assert gas["pair_score_w_core"] == pytest.approx(0.30)
    assert gas["pair_score_gap_center"] == pytest.approx(0.50)

    bare = get_ts_search_params(system_type="gas_cluster")
    assert bare["neb_climb"] is False
    assert bare["use_parallel_neb"] is True
    assert bare["parallel_neb_max_bands"] is None
    # No explicit band cap: gas bands are chunked by the atom budget instead.
    assert bare["parallel_neb_max_batch_atoms"] == 6000
    assert bare["neb_fmax"] == pytest.approx(0.20)
    assert bare["max_endpoint_mismatch"] is None
    assert bare["energy_gap_threshold"] == pytest.approx(2.0)
    assert bare["pair_core_rms_max"] is None
    assert bare["pair_score_w_core"] == pytest.approx(0.0)

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    surf = get_ts_search_params(
        system_type="surface_cluster_adsorbate", surface_config=cfg
    )
    assert surf["neb_climb"] is True
    assert surf["neb_steps"] == 4000
    assert surf["neb_fmax"] == pytest.approx(0.20)
    assert surf["torchsim_fmax"] == pytest.approx(0.20)
    assert surf["use_parallel_neb"] is True
    assert surf["parallel_neb_max_bands"] == 4
    assert surf["parallel_neb_max_batch_atoms"] == 4000
    assert "torchsim_batch_size" not in surf
    assert surf["max_endpoint_mismatch"] == pytest.approx(1.5)
    assert surf["neb_n_images"] == 7
    assert surf["energy_gap_threshold"] == pytest.approx(0.75)
    assert surf["pair_core_rms_max"] == pytest.approx(2.0)
    assert surf["pair_score_gap_center"] == pytest.approx(0.55)
    assert surf["neb_surface_cell_remap"] is True
    assert surf["neb_surface_lattice_rotation"] is False

    bare_ads = get_ts_search_params(system_type="surface_adsorbate", surface_config=cfg)
    assert bare_ads["max_endpoint_mismatch"] == pytest.approx(3.0)
    assert bare_ads["neb_climb"] is True
    assert bare_ads["neb_surface_lattice_rotation"] is False


@pytest.mark.parametrize(
    ("system_type", "expected"),
    [
        ("gas_cluster", None),
        ("surface_cluster", None),
        ("surface", None),
        ("gas_cluster_adsorbate", DEFAULT_PAIR_CORE_RMS_MAX_GAS),
        ("surface_cluster_adsorbate", DEFAULT_PAIR_CORE_RMS_MAX_SURFACE),
        ("surface_adsorbate", DEFAULT_PAIR_CORE_RMS_MAX_SURFACE),
    ],
)
def test_pair_core_rms_max_default_matches_regime(system_type, expected):
    """Core-RMS hard gate is set only for adsorbate+core regimes (gas 1.5, surface 2.0).

    ``surface_adsorbate`` has no metal core, so the gate is unused at runtime
    (``n_core_mobile == 0``), but the regime default stays the surface-adsorbate
    value from :func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`.
    """
    ts = _ts_search_params_for(system_type)
    if expected is None:
        assert ts["pair_core_rms_max"] is None
    else:
        assert ts["pair_core_rms_max"] == pytest.approx(expected)


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_ts_force_convergence_and_parallel_neb_are_shared(system_type):
    """Force tolerance and parallel NEB are defaults for every system type."""
    ts = _ts_search_params_for(system_type)
    assert ts["neb_fmax"] == pytest.approx(0.20)
    assert ts["torchsim_fmax"] == pytest.approx(0.20)
    assert ts["use_parallel_neb"] is True


def test_ts_search_params_allow_overrides():
    ts = get_ts_search_params(system_type="gas_cluster")
    ts["dedupe_minima"] = False
    ts["minima_energy_tolerance"] = 0.05
    ts["dedupe_ts"] = False
    ts["ts_energy_tolerance"] = 0.03
    ts["similarity_tolerance"] = 0.02
    ts["similarity_pair_cor_max"] = 0.05

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["dedupe_minima"] is False
    assert kwargs["minima_energy_tolerance"] == pytest.approx(0.05)
    assert kwargs["dedupe_ts"] is False
    assert kwargs["ts_energy_tolerance"] == pytest.approx(0.03)
    assert kwargs["similarity_tolerance"] == pytest.approx(0.02)
    assert kwargs["similarity_pair_cor_max"] == pytest.approx(0.05)


def test_ts_search_params_embed_surface_config_for_surface_systems():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["surface_config"] is cfg
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="surface_cluster")
    assert kwargs.get("surface_config") is cfg


def test_coerce_ts_surface_config_defaults_to_none():
    ts = get_ts_search_params(system_type="gas_cluster")
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs.get("surface_config") is None


def test_coerce_ts_requires_valid_system_type():
    ts = get_ts_search_params(system_type="gas_cluster")
    with pytest.raises(SCGOValidationError, match="Unsupported system_type"):
        coerce_ts_params_to_runner_kwargs(ts, system_type="not_a_real_type")


def test_ts_search_surface_regime_mic_and_fmax():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_n_images"] == 5
    assert ts["neb_fmax"] == pytest.approx(0.20)
    assert ts["torchsim_fmax"] == pytest.approx(0.20)
    assert ts["neb_steps"] == 2000
    assert ts["torchsim_max_steps"] == 2000
    assert ts["use_parallel_neb"] is True
    assert ts["parallel_neb_max_bands"] == 4
    assert ts["parallel_neb_max_batch_atoms"] == 4000
    assert "torchsim_batch_size" not in ts
    assert ts["neb_climb"] is False
    assert ts["neb_interpolation_method"] == "idpp"
    assert ts["neb_align_endpoints"] is True
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="surface_cluster")
    assert kwargs["neb_interpolation_mic"] is True
    assert kwargs["neb_n_images"] == 5
    assert kwargs["neb_climb"] is False
    assert kwargs["neb_fmax"] == pytest.approx(0.20)
    assert kwargs["neb_steps"] == 2000
    assert kwargs["use_parallel_neb"] is True
    assert kwargs["neb_interpolation_method"] == "idpp"
    assert kwargs["torchsim_params"]["force_tol"] == pytest.approx(0.20)
    assert kwargs["torchsim_params"]["max_steps"] == 2000
    assert kwargs["neb_align_endpoints"] is True
    assert kwargs["neb_surface_cell_remap"] is True
    assert kwargs["neb_surface_lattice_rotation"] is True
    assert kwargs["parallel_neb_max_batch_atoms"] == 4000
    # G3: the TS relaxer is sized for the largest fused NEB force batch, mirroring
    # the GO expected_max_atoms pattern so the memory-scaler cache bucket is stable.
    assert kwargs["torchsim_params"]["expected_max_atoms"] == 4000
    assert kwargs["torchsim_params"]["max_atoms_to_try"] == 4000


def test_coerce_ts_gas_relaxer_sized_for_atom_budget():
    """G3: gas presets size the relaxer from their own (larger) atom budget."""
    ts = get_ts_search_params(system_type="gas_cluster")
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["parallel_neb_max_batch_atoms"] == 6000
    assert kwargs["torchsim_params"]["expected_max_atoms"] == 6000
    assert kwargs["torchsim_params"]["max_atoms_to_try"] == 6000


def test_coerce_ts_omits_relaxer_sizing_without_atom_budget():
    """No atom budget -> no expected_max_atoms cap (torch-sim defaults apply)."""
    ts = get_ts_search_params(system_type="gas_cluster")
    ts["parallel_neb_max_batch_atoms"] = None
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["parallel_neb_max_batch_atoms"] is None
    assert "expected_max_atoms" not in kwargs["torchsim_params"]
    assert "max_atoms_to_try" not in kwargs["torchsim_params"]


def test_ts_search_step_defaults_can_be_auto():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("neb_steps") == "auto"
    assert ts.get("torchsim_max_steps") == "auto"

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["neb_steps"] == "auto"
    assert kwargs["torchsim_params"]["max_steps"] == "auto"


def test_default_go_and_default_ts_presets_share_mace_model():
    go_params = get_default_params()
    ts_params = get_ts_search_params(system_type="gas_cluster")

    assert go_params["calculator"] == "MACE"
    assert ts_params["calculator"] == "MACE"
    assert go_params["calculator_kwargs"] == {"model_name": "mace_matpes_0"}
    assert ts_params["calculator_kwargs"] == {"model_name": "mace_matpes_0"}


def test_loaders_default_to_final_unique_minima():
    """Public loaders should default to final_unique_minimum rows only."""
    import inspect

    from scgo.database.helpers import (
        extract_minima_from_database_file,
        load_previous_run_results,
    )
    from scgo.ts_search.transition_state_io import load_minima_by_composition

    assert (
        inspect.signature(extract_minima_from_database_file)
        .parameters["require_final"]
        .default
        is True
    )
    assert (
        inspect.signature(load_previous_run_results)
        .parameters["prefer_final_unique"]
        .default
        is True
    )
    assert (
        inspect.signature(load_minima_by_composition)
        .parameters["prefer_final_unique"]
        .default
        is True
    )


def _fake_torchsim_go(
    *,
    system_type: str,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str | None = None,
) -> dict:
    from scgo.param_presets import get_default_params

    p = get_default_params()
    if surface_config is not None:
        p["surface_config"] = surface_config
    if model_name is not None:
        p["calculator_kwargs"]["model_name"] = model_name
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
    go_params = param_presets_module.get_torchsim_ga_params(
        system_type=system_type, seed=seed, surface_config=surface_config
    )
    go_params["calculator"] = "MACE"
    ga = go_params["optimizer_params"]["ga"]
    ga["niter"] = niter
    ga["population_size"] = population_size
    if surface_config is not None:
        go_params["surface_config"] = surface_config
    ts_params = get_ts_search_params(
        system_type=system_type,
        surface_config=surface_config,
    )
    ts_params["max_pairs"] = max_pairs
    return go_params, ts_params


def test_production_style_mace_go_ts_gas(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    go_params, ts_params = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="gas_cluster",
    )
    ga = go_params["optimizer_params"]["ga"]
    assert ga["niter"] == 8
    assert ga["population_size"] == 18
    assert ts_params["max_pairs"] == 12
    assert "surface_config" not in ts_params
    kw = coerce_ts_params_to_runner_kwargs(ts_params, system_type="gas_cluster")
    assert kw["max_pairs"] == 12


def test_production_style_mace_go_ts_surface_has_surface_config(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    go_params, ts_params = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="surface_cluster",
        surface_config=cfg,
    )
    ga = go_params["optimizer_params"]["ga"]
    assert go_params["surface_config"] is cfg
    assert "surface_config" not in ga
    assert "system_type" not in ga
    prepared = prepare_algorithm_kwargs(
        ga,
        {"fitness_strategy": "low_energy", "surface_config": cfg},
        ["Pt"] * 5,
        "ga",
        system_type="surface_cluster",
    )
    assert prepared["niter_local_relaxation"] >= 400
    assert prepared["surface_config"] is cfg
    assert ts_params["surface_config"] is cfg
    assert (
        coerce_ts_params_to_runner_kwargs(ts_params, system_type="surface_cluster").get(
            "surface_config"
        )
        is cfg
    )


@pytest.mark.slow
def test_get_torchsim_ga_params_relaxer_uses_calculator_mace_model_name():
    """TorchSim relaxer must use the same MACE name as ``calculator_kwargs``."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    try:
        p = get_torchsim_ga_params(
            system_type="gas_cluster", seed=11, model_name="mace_mp_small"
        )
    except Exception as exc:  # pragma: no cover - environment-dependent torch/mace load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")
    assert p["calculator_kwargs"]["model_name"] == "mace_mp_small"
    relaxer = p["optimizer_params"]["ga"]["relaxer"]
    assert relaxer.mace_model_name == "mace_mp_small"


@pytest.mark.slow
def test_get_torchsim_ga_params_default_relaxer_matches_default_model():
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    try:
        p = get_torchsim_ga_params(system_type="gas_cluster", seed=3)
    except Exception as exc:  # pragma: no cover - environment-dependent torch/mace load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")
    assert p["calculator_kwargs"].get("model_name") == "mace_matpes_0"
    assert p["optimizer_params"]["ga"]["relaxer"].mace_model_name == "mace_matpes_0"


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_low_effort_ts_params_only_shrink_step_budget(system_type):
    """The low-effort TS preset must change budgets, never NEB physics."""
    production = _ts_search_params_for(system_type)
    if get_system_policy(system_type).uses_surface:
        low = get_low_effort_ts_search_params(
            system_type=system_type, surface_config=_surface_config_for_test()
        )
    else:
        low = get_low_effort_ts_search_params(system_type=system_type)

    budget_keys = {"neb_steps", "torchsim_max_steps", "write_timing_json"}
    for key, expected in production.items():
        if key in budget_keys:
            continue
        assert low[key] == expected, (
            f"{system_type}: low-effort preset changed physics key {key!r}: "
            f"{low[key]!r} != {expected!r}"
        )
    assert low["write_timing_json"] is False
    # max_pairs stays uncapped: it is the caller's cost lever.
    assert low["max_pairs"] is None


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_low_effort_neb_steps_are_floored_and_reduced(system_type):
    """NEB budgets shrink toward 25% but never below the convergence floor."""
    steps = low_effort_neb_steps(system_type)
    floor = param_presets_module._LOW_EFFORT_NEB_FLOOR
    assert steps >= floor

    production = get_ts_defaults(system_type)["neb_steps"]
    if isinstance(production, int):
        assert steps <= production
        expected = max(
            floor, round(production * param_presets_module._LOW_EFFORT_SCALE)
        )
        assert steps == expected
    else:
        # "auto" is resolved from composition at run time, so only the floor applies.
        assert production == "auto"
        assert steps == floor


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_low_effort_ts_params_preserve_n_images(system_type):
    """Adsorbate bands keep their 7 images; the budget preset must not reset them."""
    policy = get_system_policy(system_type)
    if policy.uses_surface:
        low = get_low_effort_ts_search_params(
            system_type=system_type, surface_config=_surface_config_for_test()
        )
    else:
        low = get_low_effort_ts_search_params(system_type=system_type)
    assert low["neb_n_images"] == (7 if policy.has_adsorbate else 5)


def test_low_effort_ts_params_surface_config_required():
    with pytest.raises(
        SCGOValidationError, match="requires surface_config to be provided"
    ):
        get_low_effort_ts_search_params(system_type="surface_cluster")


@pytest.mark.slow
@pytest.mark.parametrize(
    "system_type", ["gas_cluster", "gas_cluster_adsorbate", "surface_cluster"]
)
def test_low_effort_ga_params_shrink_budget_only(system_type):
    """The low-effort GA preset must only shrink the search budget."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    kwargs = {"system_type": system_type, "seed": 7}
    if get_system_policy(system_type).uses_surface:
        kwargs["surface_config"] = _surface_config_for_test()
    try:
        production = get_torchsim_ga_params(**kwargs)
        low = get_low_effort_torchsim_ga_params(**kwargs)
    except Exception as exc:  # pragma: no cover - environment-dependent model load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")

    assert low["calculator"] == production["calculator"]
    assert low["calculator_kwargs"] == production["calculator_kwargs"]
    assert low["seed"] == production["seed"]

    ga = low["optimizer_params"]["ga"]
    assert ga["niter"] == param_presets_module._LOW_EFFORT_GA_NITER
    assert ga["population_size"] == param_presets_module._LOW_EFFORT_GA_POPULATION_SIZE
    assert (
        ga["niter_local_relaxation"]
        == param_presets_module._LOW_EFFORT_GA_NITER_LOCAL_RELAXATION
    )
    assert ga["n_jobs_population_init"] == 1
    assert ga["early_stopping_niter"] == 0
    assert ga["write_timing_json"] is False
    assert ga["detailed_timing"] is False
    # Budget is genuinely below the production benchmark reference.
    base = param_presets_module._get_base_ga_benchmark_params(7)["optimizer_params"][
        "ga"
    ]
    assert ga["niter"] < base["niter"]
    assert ga["population_size"] < base["population_size"]


@pytest.mark.slow
def test_low_effort_ga_params_surface_local_relaxation_is_clamped_up():
    """Surface GO keeps production-strength local relaxation despite the low budget."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    cfg = _surface_config_for_test()
    try:
        params = get_low_effort_torchsim_ga_params(
            system_type="surface_cluster", surface_config=cfg, seed=5
        )
    except Exception as exc:  # pragma: no cover - environment-dependent model load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")

    prepared = prepare_algorithm_kwargs(
        params["optimizer_params"]["ga"],
        {
            "fitness_strategy": "low_energy",
            "surface_config": params.get("surface_config"),
        },
        ["Pt"] * 5,
        "ga",
        system_type="surface_cluster",
    )
    assert prepared["niter_local_relaxation"] >= 400
