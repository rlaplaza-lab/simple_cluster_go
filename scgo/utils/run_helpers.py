"""Helper functions for running SCGO campaigns.

This module provides utility functions used by the high-level API in scgo.run_minima
to eliminate code duplication and improve maintainability.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Any

import numpy as np
from ase.calculators.emt import EMT

from scgo.calculators.mace_helpers import MACE
from scgo.constants import BOLTZMANN_K_EV_PER_K
from scgo.param_presets import get_default_params
from scgo.utils.fitness_strategies import validate_fitness_strategy
from scgo.utils.helpers import (
    auto_niter,
    auto_niter_local_relaxation,
    auto_population_size,
    deep_merge_dicts,
    filter_dict_keys,
)
from scgo.utils.logging import get_logger
from scgo.utils.optimizer_utils import get_optimizer_class

_CALCULATORS = {
    "EMT": EMT,
    "MACE": MACE,
}


def initialize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Initialize and merge params with defaults.

    Handles None check and deep merge with default parameters.

    Args:
        params: User-provided parameters dict or None.

    Returns:
        Deep copy of params merged with defaults.
    """
    default_params = get_default_params()
    if params is None:
        return default_params

    return deep_merge_dicts(default_params, params)


def get_calculator_class(calculator_name: str) -> type:
    """Get calculator class by name.

    Args:
        calculator_name: Name of the calculator (e.g., "MACE", "EMT").

    Returns:
        Calculator class.

    Raises:
        ValueError: If calculator name is unknown or not available.
    """
    if calculator_name not in _CALCULATORS:
        raise ValueError(
            f"Unknown calculator: {calculator_name}. "
            f"Available calculators: {list(_CALCULATORS.keys())}",
        )

    calculator_class = _CALCULATORS[calculator_name]

    return calculator_class


def validate_calculator(
    calculator_name: str, calculators_dict: dict[str, Any] | None = None
) -> None:
    """Validate calculator availability.

    Args:
        calculator_name: Name of the calculator to validate.
        calculators_dict: Dictionary of available calculators. If None, uses
            internal _CALCULATORS.

    Raises:
        ValueError: If calculator name is unknown or not available.
    """
    calculators_dict = calculators_dict or _CALCULATORS

    if calculator_name not in calculators_dict:
        raise ValueError(
            f"Unknown calculator: {calculator_name}. "
            f"Available calculators: {list(calculators_dict.keys())}",
        )

    if calculators_dict[calculator_name] is None:
        raise ValueError(
            f"Calculator {calculator_name} is not available. "
            f"Please install the required dependencies.",
        )


def validate_algorithm_params(
    algo_params: dict[str, Any],
    chosen_go: str,
    verbosity: int,
) -> None:
    """Validate algorithm-specific parameters.

    Args:
        algo_params: Dictionary of algorithm-specific parameters.
        chosen_go: Name of chosen algorithm ('simple', 'bh', or 'ga').
        verbosity: Logging verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
    """
    valid_algo_params = {
        "simple": {"optimizer", "fmax", "niter", "niter_local_relaxation"},
        "bh": {
            "optimizer",
            "fmax",
            "n_trials",
            "niter",
            "niter_local_relaxation",
            "temperature",
            "dr",
            "move_fraction",
            "move_strategy",
            "deduplicate",
            "energy_tolerance",
            "comparator_tol",
            "comparator_pair_cor_max",
            "comparator_n_top",
            "fitness_strategy",
            "diversity_reference_db",
            "diversity_max_references",
            "diversity_update_interval",
        },
        "ga": {
            "optimizer",
            "fmax",
            "n_trials",
            "niter",
            "niter_local_relaxation",
            "population_size",
            "offspring_fraction",
            "n_jobs_population_init",
            "mutation_probability",
            "max_mutation_probability",
            "vacuum",
            "energy_tolerance",
            "use_torchsim",
            "use_adaptive_mutations",
            "stagnation_trigger",
            "stagnation_full_trigger",
            "aggressive_burst_multiplier",
            "recovery_window",
            "early_stopping_niter",
            "batch_size",
            "relaxer",
            "fitness_strategy",
            "diversity_reference_db",
            "diversity_max_references",
            "diversity_update_interval",
            "surface_config",
        },
    }

    if chosen_go in valid_algo_params:
        unexpected_algo_keys = set(algo_params.keys()) - valid_algo_params[chosen_go]
        if unexpected_algo_keys:
            raise ValueError(
                f"Unexpected {chosen_go.upper()} algorithm parameters: "
                f"{sorted(unexpected_algo_keys)}. "
                f"Allowed keys: {sorted(valid_algo_params[chosen_go])}"
            )


def resolve_auto_params(
    algo_params: dict[str, Any],
    composition: list[str],
    chosen_go: str,
) -> dict[str, Any]:
    """Resolve 'auto' parameter values.

    Args:
        algo_params: Dictionary of algorithm-specific parameters.
        composition: List of atomic symbols.
        chosen_go: Name of chosen algorithm.

    Returns:
        Dictionary with resolved values to merge into global_optimizer_kwargs.
    """
    niter_val = algo_params.get("niter")
    resolved = {
        "niter": auto_niter(composition) if niter_val in ("auto", None) else niter_val
    }

    # Resolve niter_local_relaxation for all algorithms
    niter_local_val = algo_params.get("niter_local_relaxation")
    resolved["niter_local_relaxation"] = (
        auto_niter_local_relaxation(composition)
        if niter_local_val in ("auto", None)
        else niter_local_val
    )

    if chosen_go == "ga":
        pop_size_val = algo_params.get("population_size")
        resolved["population_size"] = (
            auto_population_size(composition)
            if pop_size_val in ("auto", None)
            else pop_size_val
        )

    return resolved


def _normalize_optimizer_class(optimizer: str | type) -> type:
    """Normalize optimizer parameter to class.

    Converts optimizer string name to class if needed, otherwise returns as-is.

    Args:
        optimizer: Optimizer name (string) or class.

    Returns:
        Optimizer class.
    """
    if isinstance(optimizer, str):
        return get_optimizer_class(optimizer)
    return optimizer


def _resolve_fitness_strategy(
    algo_params: dict[str, Any], params: dict[str, Any]
) -> str:
    """Resolve fitness strategy with validation.

    Algorithm-specific fitness_strategy overrides top-level default.
    Both are validated.

    Args:
        algo_params: Algorithm-specific parameter dictionary.
        params: Top-level parameter dictionary.

    Returns:
        Resolved fitness strategy string.

    Raises:
        ValueError: If fitness strategy is invalid.
    """
    top_level_fitness_strategy = params.get("fitness_strategy", "low_energy")
    validate_fitness_strategy(top_level_fitness_strategy)

    algo_fitness_strategy = algo_params.get("fitness_strategy")
    if algo_fitness_strategy is None:
        return top_level_fitness_strategy  # Inherit from top-level

    validate_fitness_strategy(algo_fitness_strategy)
    return algo_fitness_strategy


def resolve_diversity_params(
    algo_params: dict[str, Any],
    params: dict[str, Any],
    chosen_go: str,
) -> dict[str, Any]:
    """Resolve diversity parameters for fitness strategy.

    Extracts diversity parameters from algorithm-specific params or top-level params,
    with algorithm-specific taking precedence. Raises ValueError if required
    diversity_reference_db is missing.

    Args:
        algo_params: Algorithm-specific parameter dictionary.
        params: Top-level parameter dictionary.
        chosen_go: Name of chosen algorithm (for error messages).

    Returns:
        Dictionary with resolved diversity parameters:
        - diversity_reference_db (required)
        - diversity_max_references (default: 100)
        - diversity_update_interval (default: 5)

    Raises:
        ValueError: If diversity_reference_db is not provided.
    """
    diversity_params = {}

    # Resolve reference_db (algorithm-specific overrides top-level)
    algo_reference_db = algo_params.get("diversity_reference_db")
    if algo_reference_db is None:
        algo_reference_db = params.get("diversity_reference_db")

    if algo_reference_db is None:
        raise ValueError(
            f"diversity_reference_db is required for fitness_strategy='diversity'. "
            f"Set params['diversity_reference_db'] or "
            f"params['optimizer_params']['{chosen_go}']['diversity_reference_db']"
        )

    diversity_params["diversity_reference_db"] = algo_reference_db
    diversity_params["diversity_max_references"] = algo_params.get(
        "diversity_max_references"
    ) or params.get("diversity_max_references", 100)
    diversity_params["diversity_update_interval"] = algo_params.get(
        "diversity_update_interval"
    ) or params.get("diversity_update_interval", 5)

    return diversity_params


def prepare_algorithm_kwargs(
    algo_params: dict[str, Any],
    params: dict[str, Any],
    composition: list[str],
    chosen_go: str,
) -> dict[str, Any]:
    """Unified parameter preparation for algorithm execution.

    Resolves "auto" parameter values, converts optimizer string names to
    classes, resolves fitness strategy, and filters out top-level keys that
    shouldn't be passed to algorithms.

    Args:
        algo_params: Dictionary of algorithm-specific parameters from optimizer_params.
        params: Full top-level parameter dictionary (for fitness strategy and diversity resolution).
        composition: List of atomic symbols.
        chosen_go: Name of chosen algorithm ('simple', 'bh', or 'ga').

    Returns:
        Dictionary ready for direct algorithm execution.
    """
    resolved = resolve_auto_params(algo_params, composition, chosen_go)

    base_kwargs = filter_dict_keys(
        algo_params, {"n_trials", "niter", "population_size"}
    )
    base_kwargs.update(resolved)

    if "optimizer" in base_kwargs:
        base_kwargs["optimizer"] = _normalize_optimizer_class(base_kwargs["optimizer"])

    base_kwargs["fitness_strategy"] = _resolve_fitness_strategy(algo_params, params)

    if base_kwargs["fitness_strategy"] == "diversity":
        diversity_params = resolve_diversity_params(algo_params, params, chosen_go)
        base_kwargs.update(diversity_params)

    return base_kwargs


def log_configuration(
    params: dict[str, Any],
    chosen_go: str,
    n_trials: int,
    cluster_formula: str,
    n_atoms: int,
    global_optimizer_kwargs: dict[str, Any],
    verbosity: int,
) -> None:
    """Log final configuration.

    Args:
        params: Full parameter dictionary.
        chosen_go: Name of chosen algorithm.
        n_trials: Number of trials to run.
        cluster_formula: Chemical formula string.
        n_atoms: Number of atoms.
        global_optimizer_kwargs: Resolved algorithm parameters.
        verbosity: Logging verbosity level.
    """
    logger = get_logger(__name__)

    if verbosity < 1:
        return

    logger.info(
        "SCGO config: composition=%s atoms=%d algorithm=%s calculator=%s",
        cluster_formula,
        n_atoms,
        chosen_go.upper(),
        params["calculator"],
    )
    if n_trials > 1:
        logger.info("SCGO config: trials=%d", n_trials)

    calculator_kwargs = params.get("calculator_kwargs", {})
    if calculator_kwargs:
        logger.info("SCGO config: calculator_kwargs=%s", calculator_kwargs)

    logger.info(
        "SCGO config: validate_with_hessian=%s check_hessian=%s fmax_threshold=%s imag_freq_threshold=%s",
        params.get("validate_with_hessian", False),
        params.get("check_hessian", True),
        params.get("fmax_threshold", 0.05),
        params.get("imag_freq_threshold", 50.0),
    )

    for key, value in sorted(global_optimizer_kwargs.items()):
        # Convert numpy types to native Python types for cleaner output
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)

        if key == "temperature" and isinstance(value, float):
            temp_k = value / BOLTZMANN_K_EV_PER_K
            logger.info("SCGO optimizer: %s=%0.6f eV (%0.1f K)", key, value, temp_k)
        elif isinstance(value, float) and abs(value) < 0.001:
            logger.info("SCGO optimizer: %s=%0.6f", key, value)
        elif isinstance(value, int | float):
            logger.info("SCGO optimizer: %s=%s", key, value)
        else:
            logger.info("SCGO optimizer: %s=%s", key, value)


def cleanup_torch_cuda(logger: Any | None = None) -> None:
    """Release PyTorch CUDA caches when available, then run GC."""
    import torch

    if torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.synchronize()
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()
        if logger is not None:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            logger.debug(
                "Post-cleanup CUDA memory: allocated=%s reserved=%s",
                allocated,
                reserved,
            )
            if reserved - allocated > 100_000_000:
                logger.debug(
                    "CUDA fragmentation detected: reserved-allocated=%s bytes",
                    reserved - allocated,
                )

    gc.collect()
