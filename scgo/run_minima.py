"""User-facing functions to run global optimization (minima search) campaigns."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ase import Atoms

from scgo.minima_search import run_trials
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
    validate_calculator,
)
from scgo.utils.run_tracking import ensure_run_id
from scgo.utils.validation import validate_composition


def _select_algorithm(n_atoms: int, logger: Any) -> str:
    """Choose algorithm 'simple', 'bh' or 'ga' based on atom count."""
    if n_atoms <= 2:
        chosen_go = "simple"
        logger.info(
            f"Selected simple optimization for {n_atoms}-atom cluster (trivial structure)"
        )
    elif n_atoms == 3:
        chosen_go = "bh"
        logger.info(
            f"Selected Basin Hopping for {n_atoms}-atom cluster (small cluster)"
        )
    else:
        chosen_go = "ga"
        logger.info(f"Selected Genetic Algorithm for {n_atoms}-atom cluster")

    return chosen_go


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
    validate_calculator(calculator_name)

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
    chosen_go = _select_algorithm(n_atoms, logger)

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


# CLI removed: programmatic API remains. Use `run_scgo_trials(...)` and TS APIs directly.


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


__all__ = [
    "run_scgo_trials",
    "run_scgo_campaign_one_element",
    "run_scgo_campaign_two_elements",
    "run_scgo_campaign_arbitrary_compositions",
    "parse_composition_arg",
]
