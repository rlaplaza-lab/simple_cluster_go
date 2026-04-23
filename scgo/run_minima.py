"""Global optimization (minima search) campaign implementations.

For stable workflow entry points (single GO, campaign GO, TS, GO+TS), prefer
:mod:`scgo.runner_api`.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from ase import Atoms

from scgo.minima_search import run_trials
from scgo.optimization.algorithm_select import select_scgo_minima_algorithm
from scgo.system_types import SystemType
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import configure_logging, get_logger
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import (
    cleanup_torch_cuda,
    get_calculator_class,
    initialize_params,
    log_configuration,
    prepare_algorithm_kwargs,
    validate_algorithm_params,
)
from scgo.utils.run_tracking import ensure_run_id
from scgo.utils.timing_report import log_timing_summary, sum_neb_seconds_from_ts_results
from scgo.utils.validation import validate_composition


def _select_algorithm(n_atoms: int, system_type: SystemType, logger: Any) -> str:
    """Choose algorithm 'simple', 'bh' or 'ga' based on atom count."""
    chosen = select_scgo_minima_algorithm(n_atoms, system_type)
    if chosen == "simple":
        logger.info(
            f"Selected simple optimization for {n_atoms}-atom cluster (trivial structure)"
        )
    elif chosen == "bh":
        logger.info(
            f"Selected Basin Hopping for {n_atoms}-atom cluster (small cluster)"
        )
    else:
        logger.info(f"Selected Genetic Algorithm for {n_atoms}-atom cluster")
    return chosen


def _resolve_explicit_system_type(params: dict[str, Any]) -> SystemType:
    candidates = []
    for algo in ("simple", "bh", "ga"):
        value = params.get("optimizer_params", {}).get(algo, {}).get("system_type")
        if isinstance(value, str):
            candidates.append(value)
    if not candidates:
        any_surface_config = any(
            params.get("optimizer_params", {}).get(algo, {}).get("surface_config")
            is not None
            for algo in ("simple", "bh", "ga")
        )
        return "surface_cluster" if any_surface_config else "gas_cluster"
    if len(set(candidates)) != 1:
        raise ValueError(
            "All optimizer_params system_type values must match exactly for one run."
        )
    return candidates[0]


def run_scgo_trials(
    composition: list[str],
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Any | None = None,
) -> list[tuple[float, Atoms]]:
    """Run trials for a composition; return unique minima as (energy, Atoms) list sorted by energy."""
    configure_logging(verbosity)
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    # Capture user intent for n_trials before defaults are merged
    user_n_trials = params.get("n_trials") if params else None

    # Initialize and merge params with defaults
    params = initialize_params(params)

    # Validate calculator availability
    calculator_name = params.get("calculator", "MACE")
    _ = get_calculator_class(calculator_name)

    # Validate params structure - rng should not be in optimizer_params
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise ValueError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    n_atoms = len(composition)
    cluster_formula = get_cluster_formula(composition)
    main_output_dir = (
        str(Path(output_dir))
        if output_dir is not None
        else f"{cluster_formula}_searches"
    )

    # Algorithm selection: Use simple optimization for 1-2 atoms, BH for 3, GA for larger
    resolved_system_type = _resolve_explicit_system_type(params)
    chosen_go = _select_algorithm(n_atoms, resolved_system_type, logger)

    # Extract algorithm-specific parameters without mutation
    algo_params = params["optimizer_params"].get(chosen_go, {})

    # Validate algorithm-specific parameters
    validate_algorithm_params(algo_params, chosen_go, verbosity)

    # Determine n_trials: use user value if provided, otherwise use smart default
    # (params["n_trials"] contains the static default of 1, which we override for BH)
    if user_n_trials is not None:
        n_trials_param = user_n_trials
    else:
        n_trials_param = 10 if chosen_go == "bh" else 1
        # Update params for consistent logging
        params["n_trials"] = n_trials_param

    # Get calculator kwargs if provided
    calculator_kwargs = params.get("calculator_kwargs", {})

    # Unified parameter preparation (resolves auto params, fitness strategy, diversity, etc.)
    global_optimizer_kwargs = prepare_algorithm_kwargs(
        algo_params=algo_params,
        params=params,
        composition=composition,
        chosen_go=chosen_go,
    )

    # Validate that no unexpected top-level keys were provided
    expected_top_level_keys = {
        "validate_with_hessian",
        "calculator",
        "calculator_kwargs",
        "fmax_threshold",
        "check_hessian",
        "imag_freq_threshold",
        "n_trials",
        "optimizer_params",
        "fitness_strategy",
        "diversity_reference_db",
        "diversity_max_references",
        "diversity_update_interval",
        "tag_final_minima",
        "seed",  # seed is handled separately at API boundary, not passed to algorithms
    }
    unexpected_keys = set(params.keys()) - expected_top_level_keys
    if unexpected_keys:
        raise ValueError(
            f"Unexpected parameter keys: {sorted(unexpected_keys)}. "
            f"Expected keys: {sorted(expected_top_level_keys)}"
        )

    # Log the final configuration being used
    log_configuration(
        params=params,
        chosen_go=chosen_go,
        n_trials=n_trials_param,
        cluster_formula=cluster_formula,
        n_atoms=n_atoms,
        global_optimizer_kwargs=global_optimizer_kwargs,
        verbosity=verbosity,
    )

    final_unique_minima = run_trials(
        composition=composition,
        global_optimizer=chosen_go,
        global_optimizer_kwargs=global_optimizer_kwargs,
        n_trials=n_trials_param,  # Now configurable via params
        output_dir=main_output_dir,
        calculator_for_global_optimization=(
            calculator_for_global_optimization
            if calculator_for_global_optimization is not None
            else get_calculator_class(params["calculator"])(**calculator_kwargs)
        ),
        validate_with_hessian=params.get("validate_with_hessian", False),
        tag_final_minima=params.get("tag_final_minima", True),
        rng=rng,
    )

    cleanup_torch_cuda(logger=logger)

    return final_unique_minima


def parse_composition_arg(comp_str: str) -> list[str]:
    """Supports two formats:
    - Comma-separated symbols: "Pt,Pt,Au"
    - Compact formula: "Pt3Au" or "AuPt2"
    """
    comp_str = comp_str.strip()
    if "," in comp_str:
        parts = [p.strip() for p in comp_str.split(",") if p.strip()]
        # Normalize element symbols (e.g., 'pt' -> 'Pt')
        normalized = [p[0].upper() + p[1:].lower() if len(p) > 0 else p for p in parts]
        return normalized

    # Parse compact formula, e.g., "Pt3Au" or "pt3au" -> [("Pt", "3"), ("Au", "")]
    # Accept lower- or upper-case element symbols and optional integer counts
    token_re = re.compile(r"([A-Za-z]{1,2})(\d*)", flags=re.IGNORECASE)
    matches = token_re.findall(comp_str)
    if not matches:
        raise ValueError(f"Unable to parse composition string: {comp_str}")

    reconstructed = "".join(elem + count for elem, count in matches)
    if reconstructed.lower() != comp_str.lower():
        raise ValueError(f"Unable to parse composition string: {comp_str}")

    composition: list[str] = []
    for elem, count_str in matches:
        # Normalize capitalization: first letter uppercase, rest lowercase
        elem_norm = elem[0].upper() + elem[1:].lower() if len(elem) > 0 else elem
        count = int(count_str) if count_str else 1
        if count == 0:
            raise ValueError(
                f"Element '{elem_norm}' has zero count in composition string: '{comp_str}'"
            )
        composition.extend([elem_norm] * count)

    return composition


# CLI removed: programmatic API remains. Prefer :mod:`scgo.runner_api` for workflows;
# this module holds implementation entry points used by that layer.


def run_scgo_campaign_one_element(
    element: str,
    min_atoms: int,
    max_atoms: int,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run mono-element campaigns for sizes min_atoms..max_atoms; return mapping formula->minima."""
    configure_logging(verbosity)
    params = initialize_params(params)

    # Validate inputs
    if not element or not isinstance(element, str):
        raise ValueError("element must be a non-empty string")
    if min_atoms < 1:
        raise ValueError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be >= min_atoms")

    # Generate run_id once at campaign start if not provided
    logger = get_logger(__name__)
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    all_compositions = [
        [element] * n_atoms for n_atoms in range(min_atoms, max_atoms + 1)
    ]

    return run_scgo_campaign_arbitrary_compositions(
        all_compositions,
        params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


def run_scgo_campaign_two_elements(
    element1: str,
    element2: str,
    min_atoms: int,
    max_atoms: int,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run bimetallic campaigns for sizes min_atoms..max_atoms; return mapping formula->minima."""
    configure_logging(verbosity)
    params = initialize_params(params)

    # Validate inputs
    if not element1 or not isinstance(element1, str):
        raise ValueError("element1 must be a non-empty string")
    if not element2 or not isinstance(element2, str):
        raise ValueError("element2 must be a non-empty string")
    if min_atoms < 1:
        raise ValueError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be >= min_atoms")

    # Generate run_id once at campaign start if not provided
    logger = get_logger(__name__)
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    all_compositions = []
    for n_atoms in range(min_atoms, max_atoms + 1):
        for i in range(n_atoms + 1):
            composition = [element1] * i + [element2] * (n_atoms - i)
            all_compositions.append(composition)

    return run_scgo_campaign_arbitrary_compositions(
        all_compositions,
        params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


def run_scgo_campaign_arbitrary_compositions(
    compositions: Iterable[list[str]],
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run optimizations for an iterable of compositions; return mapping formula->minima."""
    params = initialize_params(params)
    configure_logging(verbosity)

    # Validate params structure early: 'rng' must not be present inside
    # optimizer-specific params. Raise ValueError so callers get immediate
    # feedback instead of having the error swallowed during campaign
    # iteration.
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise ValueError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )
    logger = get_logger(__name__)

    # Generate run_id once at campaign start if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    all_results = {}
    compositions_list = list(compositions)
    if not compositions_list:
        raise ValueError("compositions iterable must not be empty")
    num_compositions = len(compositions_list)
    logger.info(f"Starting campaign for {num_compositions} compositions.")

    # Create calculator once and reuse it for all compositions to avoid file handle leaks
    calculator_kwargs = params.get("calculator_kwargs", {})
    calculator_for_global_optimization = get_calculator_class(params["calculator"])(
        **calculator_kwargs,
    )

    for i, composition in enumerate(compositions_list):
        formula_str = get_cluster_formula(composition)
        if verbosity >= 1:
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Running minima search for {formula_str} ({i + 1}/{num_compositions})"
            )
            logger.info(f"{'=' * 60}")

        comp_seed = int(rng.integers(0, 2**63 - 1))
        trial_output_dir = (
            str(Path(output_dir) / f"{formula_str}_searches")
            if output_dir is not None
            else None
        )

        try:
            results = run_scgo_trials(
                composition,
                params,
                seed=comp_seed,
                verbosity=verbosity,
                run_id=run_id,
                clean=clean,
                output_dir=trial_output_dir,
                calculator_for_global_optimization=calculator_for_global_optimization,
            )
            # Always add results (possibly empty) so the API returns a key for each
            # requested composition; this makes the function predictable for
            # downstream consumers and tests.
            all_results[formula_str] = results
            if not results and verbosity >= 1:
                logger.warning(f"No minima found for {formula_str} (results empty)")
            if verbosity >= 1:
                logger.info(f"Finished processing {formula_str}.")
                logger.info(f"  Returned {len(results)} final minima for {formula_str}")
        except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as e:
            # Enhanced error logging for HPC debugging
            error_details = [
                f"Failed to process {formula_str}: {e}",
                f"Working directory: {os.getcwd()}",
            ]
            if trial_output_dir:
                error_details.append(f"Output directory: {trial_output_dir}")
                if os.path.exists(trial_output_dir):
                    try:
                        files = os.listdir(trial_output_dir)
                        error_details.append(f"Output dir contents: {files}")
                    except OSError:
                        error_details.append(
                            "Output dir exists but cannot list contents"
                        )
                else:
                    error_details.append("Output directory does not exist")

            logger.error(" | ".join(error_details), exc_info=(verbosity >= 2))
            raise

    # Best-effort: drop shared calculator reference and free CUDA memory to avoid
    # fragmentation when campaigns are run sequentially in the same process.
    del calculator_for_global_optimization
    cleanup_torch_cuda(logger=logger)

    return all_results


def run_scgo_go_ts_pipeline(
    composition: list[str],
    *,
    go_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run global optimization then transition-state search; return a compact run summary.

    ``go_params`` is the same global-optimization dict as ``run_go`` / ``run_go_ts``'s
    ``go_params=``. Minima are written under ``output_path / f"{formula}_searches"`` so
    :func:`~scgo.ts_search.transition_state_run.run_transition_state_search` can
    load them. For high-level entry points see :mod:`scgo.runner_api`.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    formula = get_cluster_formula(composition)
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(f"{formula}_campaign")
    )
    output_path.mkdir(parents=True, exist_ok=True)
    ts_base_dir = output_path / f"{formula}_searches"

    pipeline_t0 = perf_counter()
    merged_ga = initialize_params(None if go_params is None else dict(go_params))
    calculator_kwargs = merged_ga.get("calculator_kwargs", {})
    _ = get_calculator_class(merged_ga.get("calculator", "MACE"))
    calculator_for_global_optimization = get_calculator_class(merged_ga["calculator"])(
        **calculator_kwargs,
    )
    try:
        go_t0 = perf_counter()
        minima_list = run_scgo_trials(
            composition,
            params=merged_ga,
            seed=seed,
            verbosity=verbosity,
            output_dir=str(ts_base_dir),
            calculator_for_global_optimization=calculator_for_global_optimization,
        )
    finally:
        go_wall_s = perf_counter() - go_t0
        del calculator_for_global_optimization
        cleanup_torch_cuda(logger=logger)

    minima_by_formula = {formula: minima_list}

    ts_kwargs_local = dict(ts_kwargs)
    ts_kwargs_local.pop("base_dir", None)
    ts_kwargs_local.pop("seed", None)
    ts_kwargs_local.pop("verbosity", None)
    write_ts_json = bool(ts_kwargs_local.pop("write_timing_json", False))

    from scgo.ts_search import run_transition_state_search

    ts_results = run_transition_state_search(
        composition,
        output_dir=ts_base_dir,
        seed=seed,
        verbosity=verbosity,
        write_timing_json=write_ts_json,
        **ts_kwargs_local,
    )
    ts_success = sum(1 for result in ts_results if result.get("status") == "success")

    ts_neb = sum_neb_seconds_from_ts_results(ts_results)
    elapsed_s = perf_counter() - pipeline_t0
    go_ts_timings: dict[str, float] = {
        "total_wall_s": elapsed_s,
        "go_phase_s": go_wall_s,
        "ts_neb_sum_s": ts_neb,
        "cpu_non_relax_s": max(0.0, elapsed_s - go_wall_s - ts_neb),
    }
    log_timing_summary(logger, "go_ts", go_ts_timings, verbosity=verbosity)
    logger.info(
        "Completed GO->TS pipeline for %s: successful NEBs=%d/%d, wall_time=%.2f s",
        formula,
        ts_success,
        len(ts_results),
        elapsed_s,
    )
    return {
        "formula": formula,
        "output_dir": output_path,
        "ts_base_dir": ts_base_dir,
        "minima_by_formula": minima_by_formula,
        "ts_results": ts_results,
        "ts_success_count": ts_success,
        "ts_total_count": len(ts_results),
        "wall_time_s": elapsed_s,
        "timings_s": go_ts_timings,
    }


def run_scgo_one_element_go_ts_pipeline(
    element: str,
    n_atoms: int,
    *,
    go_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one-element GO then TS and return a compact run summary."""
    if not element or not isinstance(element, str):
        raise ValueError("element must be a non-empty string")
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    composition = [element] * n_atoms
    return run_scgo_go_ts_pipeline(
        composition,
        go_params=go_params,
        ts_kwargs=ts_kwargs,
        seed=seed,
        verbosity=verbosity,
        output_dir=output_dir,
    )


__all__ = [
    "run_scgo_trials",
    "run_scgo_campaign_one_element",
    "run_scgo_campaign_two_elements",
    "run_scgo_campaign_arbitrary_compositions",
    "run_scgo_go_ts_pipeline",
    "run_scgo_one_element_go_ts_pipeline",
    "parse_composition_arg",
]
