"""Tests for parameter presets and run-helper utilities."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.exceptions import SCGOValidationError
from scgo.param_presets import (
    get_default_params,
    get_diversity_params,
    get_high_energy_params,
    get_minimal_ga_params,
    get_testing_params,
    get_uma_ga_benchmark_params,
)
from scgo.utils.run_helpers import (
    _get_calculators,
    _normalize_optimizer_class,
    _resolve_fitness_strategy,
    diff_param_overrides,
    get_calculator_class,
    initialize_params,
    initialize_ts_params,
    log_configuration,
    log_params_resolution,
    prepare_algorithm_kwargs,
    resolve_auto_params,
    resolve_diversity_params,
    validate_algorithm_params,
)


def test_get_default_params_structure():
    """get_default_params should return a dict with expected top-level keys."""
    params = get_default_params()
    for key in [
        "calculator",
        "calculator_kwargs",
        "validate_with_hessian",
        "fmax_threshold",
        "check_hessian",
        "imag_freq_threshold",
        "optimizer_params",
        "enforce_adsorbate_subgraph_integrity",
    ]:
        assert key in params
    assert set(params["optimizer_params"].keys()) == {"simple", "bh", "ga"}
    assert params["optimizer_params"]["bh"]["temperature"] == pytest.approx(1.0)
    identity_keys = {
        "system_type",
        "surface_config",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
    }
    for algo, slot in params["optimizer_params"].items():
        assert identity_keys.isdisjoint(slot), f"{algo} slot has identity keys"


def test_go_slots_share_uniqueness_defaults():
    params = get_default_params()
    for algo in ("simple", "bh", "ga"):
        slot = params["optimizer_params"][algo]
        assert slot["energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
        assert slot["comparator_tol"] == pytest.approx(DEFAULT_COMPARATOR_TOL)
        assert slot["comparator_pair_cor_max"] == pytest.approx(DEFAULT_PAIR_COR_MAX)
        assert slot["comparator_n_top"] is None


def test_torchsim_ga_params_stamps_surface_config_top_level_only():
    """Surface builders put ``surface_config`` on the top-level dict, not in slots."""
    from ase.build import fcc111

    from scgo.param_presets import get_torchsim_ga_params
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    # Avoid loading the MACE relaxer: monkeypatch via get_default_params path is
    # heavy; call the builder and strip relaxer if present, or skip on load fail.
    try:
        params = get_torchsim_ga_params(
            system_type="surface_cluster", surface_config=cfg, seed=1
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"TorchSim model load unavailable: {exc}")
    assert params["surface_config"] is cfg
    for algo in ("simple", "bh", "ga"):
        assert "surface_config" not in params["optimizer_params"][algo]
        assert "system_type" not in params["optimizer_params"][algo]


@pytest.mark.parametrize(
    "builder_name",
    [
        "get_low_effort_torchsim_ga_params",
        "get_low_effort_upet_ga_params",
        "get_low_effort_uma_ga_params",
    ],
)
def test_low_effort_builders_stamp_surface_config_top_level_only(
    monkeypatch, builder_name
):
    """Low-effort GO builders mirror TorchSim: top-level surface only, no slot identity."""
    from ase.build import fcc111

    import scgo.param_presets as presets
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)

    def _fake_torchsim(**kwargs):
        p = get_default_params()
        if kwargs.get("surface_config") is not None:
            p["surface_config"] = kwargs["surface_config"]
        return p

    def _fake_build(calculator, **kwargs):
        p = get_default_params()
        p["calculator"] = calculator
        if kwargs.get("calculator_kwargs") is not None:
            p["calculator_kwargs"] = dict(kwargs["calculator_kwargs"])
        if kwargs.get("model_name") is not None:
            p["calculator_kwargs"]["model_name"] = kwargs["model_name"]
        p["optimizer_params"]["ga"]["relaxer"] = object()
        return p

    # Bypass MLIP / TorchSim construction so this stays a cheap unit test.
    monkeypatch.setattr(presets, "get_torchsim_ga_params", _fake_torchsim)
    monkeypatch.setattr(presets, "_build_ga_calculator_params", _fake_build)

    builder = getattr(presets, builder_name)
    kwargs: dict = {"system_type": "surface_cluster", "surface_config": cfg, "seed": 1}
    if builder_name == "get_low_effort_upet_ga_params":
        kwargs["model_name"] = "pet-mad-s"
        kwargs["version"] = "1.5.0"
    elif builder_name == "get_low_effort_uma_ga_params":
        kwargs["model_name"] = "uma-s-1p2"
        kwargs["uma_task"] = "oc25"
    params = builder(**kwargs)
    assert params["surface_config"] is cfg
    for algo in ("simple", "bh", "ga"):
        assert "surface_config" not in params["optimizer_params"][algo]
        assert "system_type" not in params["optimizer_params"][algo]


def test_get_minimal_ga_params_merged_with_defaults():
    """initialize_params should deep-merge minimal GA params with defaults."""
    base = get_default_params()
    minimal = get_minimal_ga_params(seed=42, model_name="mace_mp_small")

    merged = initialize_params(minimal)

    # Top-level keys from defaults must still be present
    assert merged["validate_with_hessian"] == base["validate_with_hessian"]
    assert merged["calculator"] == "MACE"
    assert merged["seed"] == 42
    assert merged["calculator_kwargs"]["model_name"] == "mace_mp_small"

    # GA sub-dict should be a shallow override of defaults
    default_ga = base["optimizer_params"]["ga"]
    merged_ga = merged["optimizer_params"]["ga"]
    for key, default_value in default_ga.items():
        if key in minimal["optimizer_params"]["ga"]:
            assert merged_ga[key] == minimal["optimizer_params"]["ga"][key]
        else:
            assert merged_ga[key] == default_value

    # get_minimal_ga_params pins n_jobs_offspring=1 explicitly (sequential runner).
    assert merged_ga["n_jobs_offspring"] == 1


def test_initialize_params_deep_merge_user_overrides():
    """User overrides should replace only the provided nested keys."""
    user = {
        "calculator": "EMT",
        "optimizer_params": {
            "bh": {
                "niter": 5,
            },
        },
    }
    merged = initialize_params(user)

    # Calculator override is respected (and drops MACE default kwargs).
    assert merged["calculator"] == "EMT"
    assert merged["calculator_kwargs"] == {}

    # BH niter overridden, but other BH keys preserved from defaults
    bh_params = merged["optimizer_params"]["bh"]
    assert bh_params["niter"] == 5
    assert "temperature" in bh_params

    # GA params untouched except for defaults
    assert "ga" in merged["optimizer_params"]

    uma = initialize_params({"calculator": "UMA"})
    assert uma["calculator_kwargs"] == {
        "model_name": "uma-s-1p2",
        "task_name": "oc25",
    }


def test_validate_algorithm_params_raises_on_unexpected_keys():
    """validate_algorithm_params should fail on unexpected keys."""
    algo_params = {"niter": 10, "unknown_key": 123}
    with pytest.raises(SCGOValidationError, match="Unexpected BH algorithm parameters"):
        validate_algorithm_params(algo_params, chosen_go="bh")


def test_validate_algorithm_params_accepts_offspring_fraction(caplog):
    """GA-specific key `offspring_fraction` should NOT trigger an unexpected-key warning."""
    caplog.set_level("WARNING")
    algo_params = {"offspring_fraction": 0.5}

    validate_algorithm_params(algo_params, chosen_go="ga")

    warnings = [rec.message for rec in caplog.records]
    assert not any("Unexpected GA algorithm parameters" in str(msg) for msg in warnings)


def test_validate_algorithm_params_rejects_surface_config():
    """``surface_config`` is identity — forbidden in optimizer slots."""
    with pytest.raises(SCGOValidationError, match="Unexpected GA algorithm parameters"):
        validate_algorithm_params({"surface_config": None}, chosen_go="ga")


def test_get_testing_params_is_lightweight():
    """get_testing_params should favour EMT and very small iteration counts."""
    params = get_testing_params()
    defaults = get_default_params()

    assert params["calculator"] == "EMT"
    assert params["validate_with_hessian"] == defaults["validate_with_hessian"]
    assert params["tag_final_minima"] == defaults["tag_final_minima"]
    bh = params["optimizer_params"]["bh"]
    ga = params["optimizer_params"]["ga"]

    assert bh["niter"] <= 5
    assert ga["population_size"] <= 10
    assert ga["niter"] <= 5


def test_get_testing_params_merges_like_defaults():
    """Sparse overrides on get_testing_params still deep-merge with defaults."""
    merged = initialize_params({"calculator": "EMT"})
    testing = get_testing_params()
    assert set(testing.keys()) == set(get_default_params().keys())
    assert (
        merged["optimizer_params"]["ga"]["vacuum"]
        == get_default_params()["optimizer_params"]["ga"]["vacuum"]
    )


def test_initialize_ts_params_sparse_merge():
    """Sparse TS dicts deep-merge onto get_ts_search_params defaults."""
    merged = initialize_ts_params(
        {"calculator": "EMT", "neb_n_images": 7},
        system_type="gas_cluster",
    )
    base = initialize_ts_params(None, system_type="gas_cluster")
    assert merged["calculator"] == "EMT"
    assert merged["neb_n_images"] == 7
    assert merged["neb_fmax"] == base["neb_fmax"]
    assert merged["energy_gap_threshold"] == base["energy_gap_threshold"]


def test_initialize_ts_params_calculator_kwargs_deep_merge():
    user = {
        "calculator_kwargs": {"model_name": "mace_mp_small"},
    }
    merged = initialize_ts_params(user, system_type="gas_cluster")
    assert merged["calculator_kwargs"]["model_name"] == "mace_mp_small"

    # Calculator change must replace GO kwargs wholesale (no UMA task_name leak).
    go = {
        "calculator": "UMA",
        "calculator_kwargs": {"model_name": "uma-s-1p2", "task_name": "oc25"},
    }
    switched = initialize_ts_params(
        {"calculator": "MACE", "calculator_kwargs": {"model_name": "mace_mp_small"}},
        system_type="gas_cluster",
        go_params=go,
    )
    assert switched["calculator"] == "MACE"
    assert switched["calculator_kwargs"] == {"model_name": "mace_mp_small"}


def test_diff_param_overrides_nested_paths():
    base = get_default_params()
    merged = initialize_params({"optimizer_params": {"ga": {"niter": 5}}})
    overrides = diff_param_overrides(base, merged)
    assert "optimizer_params.ga.niter" in overrides
    assert overrides["optimizer_params.ga.niter"] == 5


def test_log_params_resolution_logs_overrides(caplog):
    caplog.set_level(logging.INFO)
    user = {"calculator": "EMT"}
    merged = initialize_params(user)
    log_params_resolution(
        "SCGO",
        source_label="get_default_params()",
        user_params=user,
        merged=merged,
        base=get_default_params(),
        verbosity=1,
    )
    assert any("merged user overrides" in rec.message for rec in caplog.records)
    assert any("calculator" in rec.message for rec in caplog.records)


def test_log_params_resolution_no_user_dict(caplog):
    caplog.set_level(logging.INFO)
    merged = get_default_params()
    log_params_resolution(
        "SCGO",
        source_label="get_default_params()",
        user_params=None,
        merged=merged,
        base=merged,
        verbosity=1,
    )
    assert any("no user overrides" in rec.message for rec in caplog.records)


def test_get_high_energy_params_sets_fitness_strategy():
    params = get_high_energy_params()
    assert params["fitness_strategy"] == "high_energy"
    assert params["optimizer_params"]["ga"]["population_size"] == "auto"
    assert params["optimizer_params"]["bh"]["temperature"] == pytest.approx(2.0)


def test_get_diversity_params_sets_reference_db():
    params = get_diversity_params(
        reference_db_glob="Pt*_searches/**/*.db",
        max_references=50,
        update_interval=2,
    )
    assert params["fitness_strategy"] == "diversity"
    assert params["diversity_reference_db"] == "Pt*_searches/**/*.db"
    # Slot None defaults must not shadow the top-level values.
    diversity = resolve_diversity_params(params["optimizer_params"]["ga"], params, "ga")
    assert diversity["diversity_max_references"] == 50
    assert diversity["diversity_update_interval"] == 2


@pytest.mark.requires_uma
def test_get_uma_ga_benchmark_params_structure():
    fairchem = pytest.importorskip("fairchem")
    if not hasattr(fairchem, "core"):
        pytest.skip("fairchem.core not available")
    params = _skip_if_model_unavailable(lambda: get_uma_ga_benchmark_params(seed=7))
    assert params["calculator"] == "UMA"
    assert params["seed"] == 7
    assert params["optimizer_params"]["ga"]["relaxer"] is not None


class TestResolveAutoParams:
    """Tests for resolve_auto_params function."""

    @pytest.mark.parametrize(
        "key,value,composition_len,chosen_go,expect_missing,expect_value",
        [
            pytest.param(
                "niter",
                "auto",
                5,
                "bh",
                False,
                None,
                id="niter_auto_resolves",
            ),
            pytest.param(
                "niter",
                None,
                5,
                "bh",
                False,
                None,
                id="niter_none_resolves",
            ),
            pytest.param(
                "niter",
                42,
                5,
                "bh",
                False,
                42,
                id="niter_explicit_preserved",
            ),
            pytest.param(
                "niter_local_relaxation",
                "auto",
                5,
                "bh",
                False,
                None,
                id="niter_local_relaxation_auto_resolves",
            ),
            pytest.param(
                "population_size",
                "auto",
                10,
                "ga",
                False,
                None,
                id="population_size_auto_ga_resolves",
            ),
            pytest.param(
                "population_size",
                "auto",
                10,
                "bh",
                True,
                None,
                id="population_size_ignored_non_ga",
            ),
        ],
    )
    def test_resolve_auto_params_single_key(
        self, key, value, composition_len, chosen_go, expect_missing, expect_value
    ):
        composition = ["Pt"] * composition_len
        resolved = resolve_auto_params({key: value}, composition, chosen_go)

        if expect_missing:
            assert key not in resolved
            return

        assert key in resolved
        if expect_value is not None:
            assert resolved[key] == expect_value
        else:
            assert isinstance(resolved[key], int)
            assert resolved[key] > 0

    def test_resolve_auto_params_mixed(self):
        """Test resolve_auto_params handles multiple auto parameters."""
        composition = ["Pt"] * 8
        algo_params = {
            "niter": "auto",
            "niter_local_relaxation": "auto",
            "population_size": "auto",
        }

        resolved = resolve_auto_params(algo_params, composition, "ga")

        assert isinstance(resolved["niter"], int)
        assert isinstance(resolved["niter_local_relaxation"], int)
        assert isinstance(resolved["population_size"], int)

    def test_prepare_algorithm_kwargs_surface_ga_floors_niter_local(self):
        from ase.build import fcc111

        from scgo.constants import SURFACE_GA_MIN_LOCAL_RELAX_STEPS
        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
        composition = ["Pt"] * 4
        top = {"surface_config": cfg}
        assert (
            prepare_algorithm_kwargs(
                {"niter_local_relaxation": "auto"},
                top,
                composition,
                "ga",
                system_type="surface_cluster",
            )["niter_local_relaxation"]
            >= SURFACE_GA_MIN_LOCAL_RELAX_STEPS
        )
        assert (
            prepare_algorithm_kwargs(
                {"niter_local_relaxation": 40},
                top,
                composition,
                "ga",
                system_type="surface_cluster",
            )["niter_local_relaxation"]
            == SURFACE_GA_MIN_LOCAL_RELAX_STEPS
        )


class TestNormalizeOptimizerClass:
    """Tests for _normalize_optimizer_class helper function."""

    def test_normalize_optimizer_string(self):
        """Test _normalize_optimizer_class converts optimizer string to class."""
        optimizer_class = _normalize_optimizer_class("LBFGS")

        # Optimizer should be converted to class
        assert not isinstance(optimizer_class, str)
        assert callable(optimizer_class)

    def test_normalize_optimizer_class(self):
        """Test _normalize_optimizer_class preserves optimizer class."""
        from ase.optimize import LBFGS

        optimizer_class = _normalize_optimizer_class(LBFGS)

        assert optimizer_class is LBFGS


class TestResolveFitnessStrategy:
    """Tests for _resolve_fitness_strategy helper function."""

    def test_resolve_fitness_strategy_from_top_level(self):
        """Test _resolve_fitness_strategy inherits from top-level params."""
        algo_params = {}
        params = {"fitness_strategy": "high_energy"}

        strategy = _resolve_fitness_strategy(algo_params, params)

        assert strategy == "high_energy"

    def test_resolve_fitness_strategy_none_inherits_from_top_level(self):
        """Preset None in optimizer_params should inherit from top-level."""
        algo_params = {"fitness_strategy": None}
        params = {"fitness_strategy": "high_energy"}

        strategy = _resolve_fitness_strategy(algo_params, params)

        assert strategy == "high_energy"

    def test_resolve_fitness_strategy_algorithm_override(self):
        """Test _resolve_fitness_strategy uses algorithm-specific override."""
        algo_params = {"fitness_strategy": "diversity"}
        params = {"fitness_strategy": "high_energy"}

        strategy = _resolve_fitness_strategy(algo_params, params)

        assert strategy == "diversity"


class TestResolveDiversityParams:
    """Tests for resolve_diversity_params function."""

    def test_resolve_diversity_params_from_algo_params(self):
        """Test resolve_diversity_params extracts from algo_params."""
        algo_params = {
            "diversity_reference_db": "test.db",
            "diversity_max_references": 50,
            "diversity_update_interval": 10,
        }
        params = {}

        diversity = resolve_diversity_params(algo_params, params, "ga")

        assert diversity["diversity_reference_db"] == "test.db"
        assert diversity["diversity_max_references"] == 50
        assert diversity["diversity_update_interval"] == 10

    def test_resolve_diversity_params_from_top_level(self):
        """Test resolve_diversity_params extracts from top-level params."""
        algo_params = {}
        params = {
            "diversity_reference_db": "top_level.db",
            "diversity_max_references": 75,
            "diversity_update_interval": 15,
        }

        diversity = resolve_diversity_params(algo_params, params, "ga")

        assert diversity["diversity_reference_db"] == "top_level.db"
        assert diversity["diversity_max_references"] == 75
        assert diversity["diversity_update_interval"] == 15

    def test_resolve_diversity_params_algo_overrides_top_level(self):
        """Test algo_params override top-level params."""
        algo_params = {
            "diversity_reference_db": "algo.db",
            "diversity_max_references": 30,
        }
        params = {
            "diversity_reference_db": "top.db",
            "diversity_max_references": 100,
            "diversity_update_interval": 5,
        }

        diversity = resolve_diversity_params(algo_params, params, "ga")

        assert diversity["diversity_reference_db"] == "algo.db"  # Algo overrides
        assert diversity["diversity_max_references"] == 30  # Algo overrides
        assert diversity["diversity_update_interval"] == 5  # From top-level

    def test_resolve_diversity_params_defaults(self):
        """Test resolve_diversity_params uses defaults when not provided."""
        algo_params = {"diversity_reference_db": "test.db"}
        params = {}

        diversity = resolve_diversity_params(algo_params, params, "ga")

        assert diversity["diversity_reference_db"] == "test.db"
        assert diversity["diversity_max_references"] == 100  # Default
        assert diversity["diversity_update_interval"] == 5  # Default

    def test_resolve_diversity_params_missing_reference_db_raises(self):
        """Test resolve_diversity_params raises error when reference_db missing."""
        algo_params = {}
        params = {}

        with pytest.raises(
            SCGOValidationError, match="diversity_reference_db is required"
        ):
            resolve_diversity_params(algo_params, params, "ga")

    def test_resolve_diversity_params_error_message_includes_algorithm(self):
        """Test error message includes algorithm name."""
        algo_params = {}
        params = {}

        with pytest.raises(
            SCGOValidationError, match="diversity_reference_db is required"
        ) as exc_info:
            resolve_diversity_params(algo_params, params, "bh")

        error_msg = str(exc_info.value)
        assert "bh" in error_msg


class TestLogConfiguration:
    """Tests for log_configuration function."""

    def test_log_configuration_output(self, caplog):
        """Test log_configuration logs configuration details."""
        caplog.set_level(logging.INFO)
        params = {
            "calculator": "EMT",
            "validate_with_hessian": False,
            "check_hessian": True,
            "fmax_threshold": 0.05,
            "imag_freq_threshold": 50.0,
        }
        optimizer_kwargs = {"niter": 10, "temperature": 0.01}

        log_configuration(
            params=params,
            chosen_go="bh",
            cluster_formula="Pt3",
            n_atoms=3,
            global_optimizer_kwargs=optimizer_kwargs,
            verbosity=1,
        )

        log_output = caplog.text
        assert "Pt3" in log_output
        assert "BH" in log_output or "bh" in log_output
        assert "EMT" in log_output

    def test_log_configuration_quiet_mode(self, caplog):
        """Test log_configuration doesn't log in quiet mode."""
        caplog.set_level(logging.INFO)
        params = {"calculator": "EMT"}
        optimizer_kwargs = {"niter": 10}

        log_configuration(
            params=params,
            chosen_go="bh",
            cluster_formula="Pt3",
            n_atoms=3,
            global_optimizer_kwargs=optimizer_kwargs,
            verbosity=0,
        )

        # Should not log anything in quiet mode
        assert len(caplog.records) == 0

    def test_log_configuration_redacts_relaxer_model_dump(self, caplog):
        """Test log_configuration keeps relaxer logging compact."""
        caplog.set_level(logging.INFO)
        params = {"calculator": "EMT"}

        class _VerboseRelaxer:
            def __repr__(self):
                return "VerboseRelaxer(model=VERY_LONG_MODEL_DUMP)"

        optimizer_kwargs = {"relaxer": _VerboseRelaxer()}

        log_configuration(
            params=params,
            chosen_go="ga",
            cluster_formula="Pt3",
            n_atoms=3,
            global_optimizer_kwargs=optimizer_kwargs,
            verbosity=1,
        )

        log_output = caplog.text
        assert "SCGO optimizer: relaxer=<_VerboseRelaxer>" in log_output
        assert "VERY_LONG_MODEL_DUMP" not in log_output


def test_cleanup_torch_cuda_runs_safely():
    """cleanup_torch_cuda should be callable and not raise if torch absent."""
    from scgo.utils.run_helpers import cleanup_torch_cuda

    # Should not raise in environments without torch; if torch is available
    # it should still be safe to call.
    cleanup_torch_cuda()


class TestGetCalculatorClass:
    """Tests for get_calculator_class function."""

    def test_get_calculator_class_valid(self):
        """Test get_calculator_class returns class for valid calculator name."""
        calc_cls = get_calculator_class("EMT")
        assert calc_cls is not None

    def test_get_calculator_class_unknown_raises(self):
        """Test get_calculator_class raises error for unknown calculator."""
        with pytest.raises(SCGOValidationError, match="Unknown calculator"):
            get_calculator_class("UNKNOWN_CALC")

    def test_get_calculator_class_unavailable_raises(self, monkeypatch):
        """Test get_calculator_class raises error for unavailable calculator."""
        emt = _get_calculators()["EMT"]
        monkeypatch.setattr(
            "scgo.utils.run_helpers._get_calculators",
            lambda: {"EMT": emt, "TEST": None},
        )
        with pytest.raises(SCGOValidationError, match="not available"):
            get_calculator_class("TEST")


def test_get_low_effort_upet_ga_params_structure():
    """UPET low-effort GO preset: reduced budget + attached relaxer (no CUDA)."""
    pytest.importorskip("upet")
    from scgo.param_presets import get_low_effort_upet_ga_params

    params = get_low_effort_upet_ga_params(
        system_type="gas_cluster",
        seed=42,
        model_name="pet-mad-s",
        version="1.5.0",
    )
    assert params["calculator"] == "UPET"
    assert params["n_jobs"] == 1
    assert params["calculator_kwargs"] == {
        "model_name": "pet-mad-s",
        "version": "1.5.0",
    }
    ga = params["optimizer_params"]["ga"]
    assert ga["niter"] == 3
    assert ga["population_size"] == 13
    assert ga["niter_local_relaxation"] == 70
    assert ga["n_jobs_population_init"] == 1
    assert ga["early_stopping_niter"] == 0
    assert ga["relaxer"] is not None
    assert ga["relaxer"].upet_model_name == "pet-mad-s"
    assert ga["relaxer"].upet_version == "1.5.0"
    assert ga["relaxer"].max_steps is None


def test_get_low_effort_upet_ga_params_relaxer_matches_nondefault_model(monkeypatch):
    """Non-default model_name must reach attach (same PES as ASE calc)."""
    from scgo.param_presets import get_low_effort_upet_ga_params

    captured: dict = {}

    def _fake_attach(ga, calculator_kwargs, **kwargs):
        captured["calculator_kwargs"] = dict(calculator_kwargs)
        ga["relaxer"] = SimpleNamespace(max_steps=kwargs.get("max_steps"))

    monkeypatch.setattr("scgo.param_presets._attach_torchsim_relaxer", _fake_attach)
    params = get_low_effort_upet_ga_params(
        system_type="gas_cluster",
        model_name="pet-mad-xl",
        version="2.0.0",
    )
    assert captured["calculator_kwargs"] == {
        "model_name": "pet-mad-xl",
        "version": "2.0.0",
    }
    assert params["optimizer_params"]["ga"]["relaxer"].max_steps is None


@pytest.mark.requires_uma
def test_get_low_effort_uma_ga_params_structure():
    """UMA low-effort GO preset: reduced budget + attached relaxer (no CUDA)."""
    pytest.importorskip("fairchem")
    from scgo.param_presets import get_low_effort_uma_ga_params

    params = _skip_if_model_unavailable(
        lambda: get_low_effort_uma_ga_params(
            system_type="gas_cluster",
            seed=42,
            model_name="uma-s-1p2",
            uma_task="oc25",
        )
    )
    assert params["calculator"] == "UMA"
    assert params["calculator_kwargs"] == {
        "model_name": "uma-s-1p2",
        "task_name": "oc25",
    }
    ga = params["optimizer_params"]["ga"]
    assert ga["niter"] == 3
    assert ga["population_size"] == 13
    assert ga["niter_local_relaxation"] == 70
    assert ga["n_jobs_population_init"] == 1
    assert ga["early_stopping_niter"] == 0
    assert ga["relaxer"] is not None


def test_low_effort_ts_search_params_upet_floors_neb_steps():
    """UPET TS low-effort path returns the same floored neb_steps as MACE."""
    from scgo.param_presets import (
        get_low_effort_ts_search_params,
        low_effort_neb_steps,
    )

    mace = get_low_effort_ts_search_params(
        "MACE", None, system_type="gas_cluster", seed=42
    )
    upet = get_low_effort_ts_search_params(
        "UPET",
        {"model_name": "pet-mad-s", "version": "1.5.0"},
        system_type="gas_cluster",
        seed=42,
    )
    assert upet["neb_steps"] == low_effort_neb_steps("gas_cluster")
    assert upet["neb_steps"] == mace["neb_steps"]
    assert upet["calculator"] == "UPET"


def _skip_if_model_unavailable(get_params):
    """Build params; skip the test if the relaxer model cannot load here."""
    pytest.importorskip("torch")
    try:
        return get_params()
    except Exception as exc:  # pragma: no cover - env-dependent torch model load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")


def test_get_torchsim_ga_params_relaxer_invariants():
    """T1.3: MACE TorchSim preset exposes the expected relaxer invariants."""
    pytest.importorskip("mace")
    from scgo.param_presets import get_torchsim_ga_params

    params = _skip_if_model_unavailable(
        lambda: get_torchsim_ga_params(system_type="gas_cluster", seed=11)
    )
    ga = params["optimizer_params"]["ga"]
    relaxer = ga["relaxer"]
    assert relaxer.model_kind == "mace"
    assert relaxer.autobatcher is True
    assert relaxer.expected_max_atoms == 600
    assert relaxer.max_steps is None
    assert ga["niter"] == "auto"
    assert ga["population_size"] == "auto"


def test_get_uma_ga_benchmark_params_relaxer_invariants():
    """T1.3: UMA benchmark preset exposes the expected relaxer invariants."""
    pytest.importorskip("fairchem")
    from scgo.param_presets import get_uma_ga_benchmark_params

    params = _skip_if_model_unavailable(lambda: get_uma_ga_benchmark_params(seed=7))
    ga = params["optimizer_params"]["ga"]
    relaxer = ga["relaxer"]
    assert relaxer.autobatcher is True
    assert relaxer.expected_max_atoms == 600
    assert relaxer.max_steps is None


def test_get_default_uma_params_relaxer_invariants():
    """T1.3: UMA default preset exposes the expected relaxer invariants."""
    pytest.importorskip("fairchem")
    from scgo.param_presets import get_default_uma_params

    params = _skip_if_model_unavailable(get_default_uma_params)
    ga = params["optimizer_params"]["ga"]
    relaxer = ga["relaxer"]
    assert relaxer.autobatcher is None
    assert relaxer.expected_max_atoms is None
    assert relaxer.max_steps is None


def test_get_upet_ga_benchmark_params_relaxer_invariants():
    """T1.3: UPET benchmark preset exposes the expected relaxer invariants."""
    pytest.importorskip("upet")
    from scgo.param_presets import get_upet_ga_benchmark_params

    params = _skip_if_model_unavailable(lambda: get_upet_ga_benchmark_params(seed=7))
    ga = params["optimizer_params"]["ga"]
    relaxer = ga["relaxer"]
    assert relaxer.autobatcher is True
    assert relaxer.expected_max_atoms == 600
    assert relaxer.max_steps is None


def test_get_default_upet_params_relaxer_invariants():
    """T1.3: UPET default preset exposes the expected relaxer invariants."""
    pytest.importorskip("upet")
    from scgo.param_presets import get_default_upet_params

    params = _skip_if_model_unavailable(get_default_upet_params)
    ga = params["optimizer_params"]["ga"]
    relaxer = ga["relaxer"]
    assert relaxer.autobatcher is None
    assert relaxer.expected_max_atoms is None
    assert relaxer.max_steps is None
