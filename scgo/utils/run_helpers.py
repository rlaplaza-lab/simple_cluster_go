"""Helper functions for running SCGO campaigns.

This module provides utility functions used by the high-level API in
scgo.runner_api
to eliminate code duplication and improve maintainability.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Any

import numpy as np
from ase.calculators.emt import EMT

from scgo.cluster_adsorbate.config import resolve_cluster_adsorbate_config
from scgo.constants import DEFAULT_FMAX_THRESHOLD, SURFACE_GA_MIN_LOCAL_RELAX_STEPS
from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.param_presets import (
    default_calculator_kwargs,
    get_default_params,
    get_ts_search_params,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    SystemType,
    get_system_policy,
    validate_system_type_settings,
)
from scgo.utils.fitness_strategies import resolve_fitness_strategy
from scgo.utils.helpers import (
    auto_niter,
    auto_niter_local_relaxation,
    auto_population_size,
    deep_merge_dicts,
)
from scgo.utils.logging import get_logger
from scgo.utils.optimizer_utils import get_optimizer_class
from scgo.utils.parallel_workers import inherit_n_jobs

logger = get_logger(__name__)

# Allowed per-algorithm parameter keys for :func:`validate_algorithm_params`.
_VALID_ALGO_PARAMS: dict[str, frozenset[str]] = {
    "simple": frozenset(
        {
            "optimizer",
            "fmax",
            "niter",
            "niter_local_relaxation",
            "energy_tolerance",
            "comparator_tol",
            "comparator_pair_cor_max",
            "comparator_n_top",
        }
    ),
    "bh": frozenset(
        {
            "optimizer",
            "fmax",
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
            "comparator_component_weights",
            "comparator_cross_weight",
            "fitness_strategy",
            "diversity_reference_db",
            "diversity_max_references",
            "diversity_update_interval",
            "n_slab",
            "write_timing_json",
            "detailed_timing",
            "enforce_adsorbate_subgraph_integrity",
            "freeze_adsorbate_internal_geometry",
        }
    ),
    "ga": frozenset(
        {
            "optimizer",
            "fmax",
            "niter",
            "niter_local_relaxation",
            "population_size",
            "offspring_fraction",
            "n_jobs_population_init",
            "n_jobs_offspring",
            "mutation_probability",
            "max_mutation_probability",
            "vacuum",
            "previous_search_glob",
            "energy_tolerance",
            "comparator_tol",
            "comparator_pair_cor_max",
            "comparator_n_top",
            "comparator_component_weights",
            "comparator_cross_weight",
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
            "write_timing_json",
            "detailed_timing",
            "enforce_adsorbate_subgraph_integrity",
            "freeze_adsorbate_internal_geometry",
        }
    ),
}


_CALCULATORS: dict[str, Any] | None = None


def _get_calculators() -> dict[str, Any]:
    """ASE calculator registry; MLIP entries are None if extras are not installed."""
    global _CALCULATORS
    if _CALCULATORS is None:
        calcs: dict[str, Any] = {"EMT": EMT}
        try:
            from scgo.calculators.mace_helpers import MACE as _MACE

            calcs["MACE"] = _MACE
        except ImportError:
            calcs["MACE"] = None
        try:
            from scgo.calculators.uma_helpers import UMA as _UMA

            calcs["UMA"] = _UMA
        except ImportError:
            calcs["UMA"] = None
        try:
            from scgo.calculators.upet_helpers import UPET as _UPET

            calcs["UPET"] = _UPET
        except ImportError:
            calcs["UPET"] = None
        _CALCULATORS = calcs
    return _CALCULATORS


def _normalize_calculator_name(name: str) -> str:
    return name.strip().upper()


def _calculator_kwargs_for_change(
    calculator: str,
    user_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Defaults for ``calculator``, overlaid with explicit user kwargs."""
    merged = default_calculator_kwargs(calculator)
    if user_kwargs:
        merged.update(dict(user_kwargs))
    return merged


def initialize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Initialize and merge params with defaults.

    When ``calculator`` changes vs defaults, ``calculator_kwargs`` are replaced
    wholesale (new-calculator defaults, then any user kwargs).
    """
    default_params = get_default_params()
    if params is None:
        return default_params

    # copy_base=False mutates default_params; capture calculator first.
    default_calc = _normalize_calculator_name(
        str(default_params.get("calculator", "MACE"))
    )
    merged = deep_merge_dicts(default_params, params, copy_base=False)
    if "calculator" in params:
        user_calc = _normalize_calculator_name(str(params["calculator"]))
        if user_calc != default_calc:
            merged["calculator_kwargs"] = _calculator_kwargs_for_change(
                str(params["calculator"]),
                params.get("calculator_kwargs"),
            )
    return merged


def initialize_ts_params(
    ts_params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    go_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize and merge TS params with :func:`~scgo.get_ts_search_params` defaults.

    When ``go_params`` is provided, calculator settings are aligned with the
    merged GO dict unless overridden in ``ts_params``. Changing ``calculator``
    in ``ts_params`` replaces ``calculator_kwargs`` wholesale (new-calculator
    defaults plus any explicit TS kwargs) so GO backend keys do not leak.
    """
    resolved_surface = surface_config
    if ts_params is not None:
        ts_sc = ts_params.get("surface_config")
        if ts_sc is not None:
            resolved_surface = resolved_surface or ts_sc
    if go_params is not None and resolved_surface is None:
        go_sc = go_params.get("surface_config")
        if go_sc is not None:
            resolved_surface = go_sc

    inherited_calc = "MACE"
    inherited_kwargs: dict[str, Any] | None = None
    if go_params is not None:
        inherited_calc = str(go_params.get("calculator", "MACE"))
        ck = go_params.get("calculator_kwargs")
        if ck:
            inherited_kwargs = dict(ck)
    elif ts_params is not None:
        if "calculator" in ts_params:
            inherited_calc = str(ts_params["calculator"])
        ck = ts_params.get("calculator_kwargs")
        if ck:
            inherited_kwargs = dict(ck)

    if ts_params is not None and "calculator" in ts_params:
        final_calc = str(ts_params["calculator"])
    else:
        final_calc = inherited_calc

    calc_changed = (
        go_params is not None
        and ts_params is not None
        and "calculator" in ts_params
        and _normalize_calculator_name(final_calc)
        != _normalize_calculator_name(inherited_calc)
    )

    if calc_changed:
        assert ts_params is not None
        calc_kwargs: dict[str, Any] | None = _calculator_kwargs_for_change(
            final_calc,
            ts_params.get("calculator_kwargs"),
        )
    else:
        calc_kwargs = inherited_kwargs

    base = get_ts_search_params(
        calculator=final_calc,
        calculator_kwargs=calc_kwargs,
        system_type=system_type,
        surface_config=resolved_surface,
    )
    if ts_params is None:
        return base

    if calc_changed:
        # Keep kwargs atomic: base already has defaults + explicit TS overlay.
        override = {k: v for k, v in ts_params.items() if k != "calculator_kwargs"}
        merged = deep_merge_dicts(base, override, copy_base=False)
        merged["calculator_kwargs"] = dict(base["calculator_kwargs"])
        return merged

    return deep_merge_dicts(base, ts_params, copy_base=False)


def diff_param_overrides(
    base: dict[str, Any],
    merged: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Return flat ``path -> value`` entries where ``merged`` differs from ``base``."""
    overrides: dict[str, Any] = {}
    for key in set(base) | set(merged):
        path = key if not prefix else f"{prefix}.{key}"
        base_val = base.get(key)
        merged_val = merged.get(key)
        if isinstance(base_val, dict) and isinstance(merged_val, dict):
            overrides.update(diff_param_overrides(base_val, merged_val, prefix=path))
        elif base_val != merged_val:
            overrides[path] = merged_val
    return overrides


def log_params_resolution(
    context: str,
    *,
    source_label: str,
    user_params: dict[str, Any] | None,
    merged: dict[str, Any],
    base: dict[str, Any],
    verbosity: int,
) -> None:
    """Log how user params were merged onto preset defaults."""
    if verbosity < 1:
        return
    if user_params is None:
        logger.info("%s params: using %s (no user overrides)", context, source_label)
        return
    overrides = diff_param_overrides(base, merged)
    if overrides:
        logger.info(
            "%s params: merged user overrides on top of %s: %s",
            context,
            source_label,
            overrides,
        )
    else:
        logger.info(
            "%s params: using %s (user dict matched defaults)",
            context,
            source_label,
        )


def get_calculator_class(calculator_name: str) -> type:
    """Get calculator class by name.

    Args:
        calculator_name: Name of the calculator (e.g., "MACE", "EMT").

    Returns:
        Calculator class.

    Raises:
        SCGOValidationError: If calculator name is unknown or not available.
    """
    calculators = _get_calculators()
    if calculator_name not in calculators:
        raise SCGOValidationError(
            f"Unknown calculator: {calculator_name}. "
            f"Available calculators: {list(calculators.keys())}",
        )

    calculator_class = calculators[calculator_name]
    if calculator_class is None:
        raise SCGOValidationError(
            f"Calculator {calculator_name} is not available. "
            "Install the matching optional dependencies."
        )
    return calculator_class


def validate_algorithm_params(
    algo_params: dict[str, Any],
    chosen_go: str,
) -> None:
    """Validate algorithm-specific parameters.

    Args:
        algo_params: Dictionary of algorithm-specific parameters.
        chosen_go: Name of chosen algorithm ('simple', 'bh', or 'ga').

    Raises:
        SCGOValidationError: If ``algo_params`` contains keys that are not
            allowed for ``chosen_go``.
    """
    valid_algo_params = _VALID_ALGO_PARAMS

    if chosen_go in valid_algo_params:
        unexpected_algo_keys = set(algo_params.keys()) - valid_algo_params[chosen_go]
        if unexpected_algo_keys:
            raise SCGOValidationError(
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
        SCGOValidationError: If fitness strategy is invalid.
    """
    top_level_fitness_strategy = params.get("fitness_strategy", "low_energy")
    return resolve_fitness_strategy(
        algo_params.get("fitness_strategy"),
        inherit_from=top_level_fitness_strategy,
    )


def resolve_diversity_params(
    algo_params: dict[str, Any],
    params: dict[str, Any],
    chosen_go: str,
) -> dict[str, Any]:
    """Resolve diversity parameters for fitness strategy.

    Extracts diversity parameters from algorithm-specific params or top-level params,
    with algorithm-specific taking precedence. Raises SCGOValidationError if required
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
        SCGOValidationError: If diversity_reference_db is not provided.
    """
    diversity_params = {}

    # Resolve reference_db (algorithm-specific overrides top-level)
    algo_reference_db = algo_params.get("diversity_reference_db")
    if algo_reference_db is None:
        algo_reference_db = params.get("diversity_reference_db")

    if algo_reference_db is None:
        raise SCGOValidationError(
            f"diversity_reference_db is required for fitness_strategy='diversity'. "
            f"Set params['diversity_reference_db'] or "
            f"params['optimizer_params']['{chosen_go}']['diversity_reference_db']"
        )

    diversity_params["diversity_reference_db"] = algo_reference_db
    max_refs = algo_params.get("diversity_max_references")
    diversity_params["diversity_max_references"] = (
        max_refs
        if max_refs is not None
        else params.get("diversity_max_references", 100)
    )
    update_interval = algo_params.get("diversity_update_interval")
    diversity_params["diversity_update_interval"] = (
        update_interval
        if update_interval is not None
        else params.get("diversity_update_interval", 5)
    )

    return diversity_params


def prepare_algorithm_kwargs(
    algo_params: dict[str, Any],
    params: dict[str, Any],
    composition: list[str],
    chosen_go: str,
    *,
    system_type: SystemType,
) -> dict[str, Any]:
    """Unified parameter preparation for algorithm execution.

    Resolves "auto" parameter values, converts optimizer string names to
    classes, resolves fitness strategy, and replaces the raw ``niter`` /
    ``population_size`` entries with their resolved values. For the GA it also
    applies the CPU parallelism cascade, so ``n_jobs_population_init`` /
    ``n_jobs_offspring`` inherit the top-level ``params["n_jobs"]`` when they are
    not set explicitly (see :mod:`scgo.utils.parallel_workers`).

    Args:
        algo_params: Dictionary of algorithm-specific parameters from optimizer_params.
        params: Full top-level parameter dictionary (for fitness strategy and diversity resolution).
        composition: List of atomic symbols.
        chosen_go: Name of chosen algorithm ('simple', 'bh', or 'ga').
        system_type: Resolved system type; validated against ``surface_config``
            and used to add system-specific keys.

    Returns:
        Dictionary ready for direct algorithm execution.
    """
    resolved = resolve_auto_params(algo_params, composition, chosen_go)
    surface_config = params.get("surface_config")
    if system_type == "gas_cluster" and surface_config is not None:
        raise SCGOValidationError(
            "system_type='gas_cluster' does not allow surface_config. "
            "Use surface_cluster or surface_cluster_adsorbate."
        )
    validate_system_type_settings(
        system_type=system_type,
        surface_config=surface_config,
    )
    if chosen_go == "simple":
        policy = get_system_policy(system_type)
        if policy.uses_surface or policy.has_adsorbate:
            raise SCGOValidationError(
                f"simple optimizer only supports system_type='gas_cluster', got {system_type!r}."
            )

    base_kwargs = {
        k: v for k, v in algo_params.items() if k not in {"niter", "population_size"}
    }
    base_kwargs.update(resolved)
    base_kwargs["system_type"] = system_type
    if surface_config is not None:
        base_kwargs["surface_config"] = surface_config
    if chosen_go == "ga":
        # CPU parallelism cascade: the per-stage keys inherit the single
        # top-level ``n_jobs`` unless they are set explicitly. Only the GA takes
        # these kwargs; ``simple`` / ``bh`` signatures reject them.
        top_n_jobs = params.get("n_jobs")
        base_kwargs["n_jobs_population_init"] = inherit_n_jobs(
            algo_params.get("n_jobs_population_init"), top_n_jobs
        )
        base_kwargs["n_jobs_offspring"] = inherit_n_jobs(
            algo_params.get("n_jobs_offspring"), top_n_jobs
        )
    policy = get_system_policy(system_type)
    if chosen_go == "ga" and policy.uses_surface:
        nlr = int(base_kwargs["niter_local_relaxation"])
        base_kwargs["niter_local_relaxation"] = max(
            SURFACE_GA_MIN_LOCAL_RELAX_STEPS, nlr
        )

    if "optimizer" in base_kwargs:
        base_kwargs["optimizer"] = _normalize_optimizer_class(base_kwargs["optimizer"])

    base_kwargs["fitness_strategy"] = _resolve_fitness_strategy(algo_params, params)

    if base_kwargs["fitness_strategy"] == "diversity":
        diversity_params = resolve_diversity_params(algo_params, params, chosen_go)
        base_kwargs.update(diversity_params)

    for key in (
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
        "connectivity_factor",
        "allow_cluster_fragmentation",
        "allow_adsorbate_surface_detachment",
        "enforce_adsorbate_subgraph_integrity",
        "freeze_adsorbate_internal_geometry",
    ):
        v = params.get(key)
        if v is not None:
            base_kwargs[key] = v

    if get_system_policy(system_type).has_adsorbate:
        base_kwargs["cluster_adsorbate_config"] = resolve_cluster_adsorbate_config(
            base_kwargs.get("cluster_adsorbate_config")
        )

    return base_kwargs


def log_ts_configuration(
    ts_params: dict[str, Any],
    coerced_kwargs: dict[str, Any],
    *,
    verbosity: int,
    user_params: dict[str, Any] | None = None,
    base: dict[str, Any] | None = None,
) -> None:
    """Log resolved transition-state search configuration."""
    if verbosity < 1:
        return

    if base is not None:
        log_params_resolution(
            "TS",
            source_label="get_ts_search_params()",
            user_params=user_params,
            merged=ts_params,
            base=base,
            verbosity=verbosity,
        )

    calc_params = coerced_kwargs.get("params") or {}
    logger.info(
        "TS config: calculator=%s calculator_kwargs=%s",
        calc_params.get("calculator"),
        calc_params.get("calculator_kwargs", {}),
    )
    for key in (
        "max_pairs",
        "energy_gap_threshold",
        "similarity_tolerance",
        "similarity_pair_cor_max",
        "pair_core_rms_max",
        "pair_score_gap_center",
        "pair_score_gap_width",
        "pair_score_cum_scale",
        "pair_score_mismatch_scale",
        "pair_score_core_rms_scale",
        "pair_score_w_gap",
        "pair_score_w_distinct",
        "pair_score_w_mismatch",
        "pair_score_w_core",
        "connectivity_factor",
        "dedupe_minima",
        "minima_energy_tolerance",
        "dedupe_ts",
        "ts_energy_tolerance",
        "use_torchsim",
        "use_parallel_neb",
        "parallel_neb_max_bands",
        "parallel_neb_max_batch_atoms",
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
    ):
        if key in coerced_kwargs and coerced_kwargs[key] is not None:
            logger.info("TS config: %s=%s", key, coerced_kwargs[key])


def log_configuration(
    params: dict[str, Any],
    chosen_go: str,
    cluster_formula: str,
    n_atoms: int,
    global_optimizer_kwargs: dict[str, Any],
    verbosity: int,
    *,
    user_params: dict[str, Any] | None = None,
    params_base: dict[str, Any] | None = None,
) -> None:
    """Log final configuration.

    Args:
        params: Full parameter dictionary.
        chosen_go: Name of chosen algorithm.
        cluster_formula: Chemical formula string.
        n_atoms: Number of atoms.
        global_optimizer_kwargs: Resolved algorithm parameters.
        verbosity: Logging verbosity level.
        user_params: Original user dict before merge (for provenance logging).
        params_base: Base defaults used for merge (defaults to ``get_default_params()``).
    """
    if verbosity < 1:
        return

    log_params_resolution(
        "SCGO",
        source_label="get_default_params()",
        user_params=user_params,
        merged=params,
        base=params_base if params_base is not None else get_default_params(),
        verbosity=verbosity,
    )

    logger.info(
        "SCGO config: composition=%s atoms=%d algorithm=%s calculator=%s",
        cluster_formula,
        n_atoms,
        chosen_go.upper(),
        params["calculator"],
    )

    calculator_kwargs = params.get("calculator_kwargs", {})
    if calculator_kwargs:
        logger.info("SCGO config: calculator_kwargs=%s", calculator_kwargs)

    logger.info(
        "SCGO config: validate_with_hessian=%s check_hessian=%s fmax_threshold=%s imag_freq_threshold=%s",
        params.get("validate_with_hessian", False),
        params.get("check_hessian", True),
        params.get("fmax_threshold", DEFAULT_FMAX_THRESHOLD),
        params.get("imag_freq_threshold", 50.0),
    )

    def _format_optimizer_log_value(key: str, value: Any) -> Any:
        """Format optimizer values to avoid overly verbose object dumps."""
        if key == "relaxer" and value is not None:
            return f"<{value.__class__.__name__}>"
        return value

    for key, value in sorted(global_optimizer_kwargs.items()):
        # Convert numpy types to native Python types for cleaner output
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)

        value = _format_optimizer_log_value(key, value)

        if key == "temperature" and isinstance(value, float):
            # BH Metropolis energy scale (eV), not a physical temperature.
            logger.info("SCGO optimizer: %s=%0.6f eV", key, value)
        elif isinstance(value, float) and abs(value) < 0.001:
            logger.info("SCGO optimizer: %s=%0.6f", key, value)
        elif isinstance(value, int | float):
            logger.info("SCGO optimizer: %s=%s", key, value)
        else:
            logger.info("SCGO optimizer: %s=%s", key, value)


def cleanup_torch_cuda(logger: Any | None = None) -> None:
    """Release PyTorch CUDA caches when available, then run GC."""
    try:
        import torch
    except ImportError:
        gc.collect()
        return

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
