"""Tests for parameter presets and run-helper utilities."""

from __future__ import annotations

import logging

import pytest

from scgo.param_presets import (
    get_default_params,
    get_minimal_ga_params,
    get_testing_params,
)
from scgo.utils.run_helpers import (
    _normalize_optimizer_class,
    _resolve_fitness_strategy,
    initialize_params,
    log_configuration,
    resolve_auto_params,
    resolve_diversity_params,
    validate_algorithm_params,
    validate_calculator,
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
        "n_trials",
        "optimizer_params",
    ]:
        assert key in params
    assert set(params["optimizer_params"].keys()) == {"simple", "bh", "ga"}


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

    # Calculator override is respected
    assert merged["calculator"] == "EMT"

    # BH niter overridden, but other BH keys preserved from defaults
    bh_params = merged["optimizer_params"]["bh"]
    assert bh_params["niter"] == 5
    assert "temperature" in bh_params

    # GA params untouched except for defaults
    assert "ga" in merged["optimizer_params"]


def test_validate_algorithm_params_raises_on_unexpected_keys():
    """validate_algorithm_params should fail on unexpected keys."""
    algo_params = {"niter": 10, "unknown_key": 123}
    with pytest.raises(ValueError, match="Unexpected BH algorithm parameters"):
        validate_algorithm_params(algo_params, chosen_go="bh", verbosity=1)


def test_validate_algorithm_params_accepts_offspring_fraction(caplog):
    """GA-specific key `offspring_fraction` should NOT trigger an unexpected-key warning."""
    caplog.set_level("WARNING")
    algo_params = {"offspring_fraction": 0.5}

    validate_algorithm_params(algo_params, chosen_go="ga", verbosity=1)

    warnings = [rec.message for rec in caplog.records]
    assert not any("Unexpected GA algorithm parameters" in str(msg) for msg in warnings)


def test_validate_algorithm_params_accepts_surface_config(caplog):
    """`surface_config` is a recognized GA key for adsorbate-on-slab runs."""
    caplog.set_level("WARNING")
    algo_params = {"surface_config": None}

    validate_algorithm_params(algo_params, chosen_go="ga", verbosity=1)

    warnings = [rec.message for rec in caplog.records]
    assert not any("Unexpected GA algorithm parameters" in str(msg) for msg in warnings)


def test_get_testing_params_is_lightweight():
    """get_testing_params should favour EMT and very small iteration counts."""
    params = get_testing_params()

    assert params["calculator"] == "EMT"
    bh = params["optimizer_params"]["bh"]
    ga = params["optimizer_params"]["ga"]

    assert bh["niter"] <= 5
    assert ga["population_size"] <= 10
    assert ga["niter"] <= 5


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

        with pytest.raises(ValueError, match="diversity_reference_db is required"):
            resolve_diversity_params(algo_params, params, "ga")

    def test_resolve_diversity_params_error_message_includes_algorithm(self):
        """Test error message includes algorithm name."""
        algo_params = {}
        params = {}

        with pytest.raises(ValueError) as exc_info:
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
            n_trials=1,
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
            n_trials=1,
            cluster_formula="Pt3",
            n_atoms=3,
            global_optimizer_kwargs=optimizer_kwargs,
            verbosity=0,
        )

        # Should not log anything in quiet mode
        assert len(caplog.records) == 0

    def test_log_configuration_multiple_trials(self, caplog):
        """Test log_configuration logs trial count when > 1."""
        caplog.set_level(logging.INFO)
        params = {"calculator": "EMT"}
        optimizer_kwargs = {"niter": 10}

        log_configuration(
            params=params,
            chosen_go="bh",
            n_trials=5,
            cluster_formula="Pt3",
            n_atoms=3,
            global_optimizer_kwargs=optimizer_kwargs,
            verbosity=1,
        )

        log_output = caplog.text
        assert "5" in log_output or "Trials" in log_output

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
            n_trials=1,
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


class TestValidateCalculator:
    """Tests for validate_calculator function."""

    def test_validate_calculator_valid(self):
        """Test validate_calculator passes for valid calculator name."""
        validate_calculator("EMT")

    def test_validate_calculator_unknown_raises(self):
        """Test validate_calculator raises error for unknown calculator."""
        with pytest.raises(ValueError, match="Unknown calculator"):
            validate_calculator("UNKNOWN_CALC")

    def test_validate_calculator_unavailable_raises(self):
        """Test validate_calculator raises error for unavailable calculator."""
        # Use custom calculators dict with None value
        calculators_dict = {"TEST": None}

        with pytest.raises(ValueError, match="not available"):
            validate_calculator("TEST", calculators_dict=calculators_dict)
