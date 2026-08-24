"""Global-optimization (GO) trial and campaign runners.

Implements algorithm selection and the low-level GO execution used by the
public ``run_go`` / ``run_go_campaign`` API in :mod:`scgo.runner_api`.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo.constants import DEFAULT_FMAX_THRESHOLD
from scgo.exceptions import (
    SCGODatabaseError,
    SCGOFileError,
    SCGOValidationError,
)
from scgo.metadata.run_dir import ensure_run_id
from scgo.minima_search import run_trials
from scgo.param_presets import default_params_top_level_keys
from scgo.runner_params import (
    _merge_adsorbate_context_into_params,
    _reject_slot_identity_keys,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    SystemType,
    as_adsorbate_definition,
    get_system_policy,
    resolve_search_mobile_composition,
    select_scgo_minima_algorithm,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import (
    configure_logging,
    get_logger,
    log_info_v,
    log_warning_v,
)
from scgo.utils.output_paths import (
    resolve_go_campaign_searches_dir,
    resolve_go_searches_dir,
)
from scgo.utils.parallel_workers import inherit_n_jobs
from scgo.utils.path_keys import resolve_run_path_key
from scgo.utils.phase_logging import log_phase_header
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import (
    cleanup_torch_cuda,
    get_calculator_class,
    initialize_params,
    log_configuration,
    prepare_algorithm_kwargs,
    validate_algorithm_params,
)
from scgo.utils.validation import validate_composition

logger = get_logger(__name__)

# Top-level param keys accepted by the GO runner that are intentionally absent
# from :func:`scgo.param_presets.get_default_params` (surface/adsorbate config and
# the validation-only ``validation_n_jobs`` knob). Kept as a single source of
# truth so the allowlist in :func:`_run_go_trials` stays in sync with defaults.
_EXTRA_ACCEPTED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "surface_config",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
        "validation_n_jobs",
    }
)

# Static allowlist: default template keys ∪ extras (no deepcopy per GO call).
_EXPECTED_TOP_LEVEL_KEYS: frozenset[str] = (
    default_params_top_level_keys() | _EXTRA_ACCEPTED_TOP_LEVEL_KEYS
)

ScgoMinimaAlgorithm = Literal["simple", "bh", "ga"]

# ``select_scgo_minima_algorithm`` lives in :mod:`scgo.system_types` (re-exported
# here) so that :mod:`scgo.runner_params` can use it without importing this
# module, avoiding a circular import between the two runner modules.


def _validate_optimizer_rng(params: dict[str, Any]) -> None:
    for algo in ("bh", "ga"):
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise SCGOValidationError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )


def _prepare_go_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Backfill preset defaults under user overrides, then validate optimizer RNG.

    Partial override dicts (e.g. ``{"calculator": "EMT"}``) must not leave
    required keys missing downstream, otherwise callers see a bare ``KeyError``
    instead of a usable run or a :class:`SCGOValidationError`.
    """
    params = initialize_params(params)
    _reject_slot_identity_keys(params)
    _validate_optimizer_rng(params)
    return params


def _resolve_go_seed(seed: int | None, params: dict[str, Any]) -> int | None:
    """Prefer explicit ``seed`` arg; fall back to ``params['seed']``; coerce to int.

    Kept local to avoid a circular import with :mod:`scgo.runner_params`
    (which imports :func:`select_scgo_minima_algorithm` from this module).
    Public GO/TS entrypoints use :func:`scgo.runner_params.resolve_workflow_seed`.
    """
    raw = seed if seed is not None else params.get("seed")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise SCGOValidationError(f"seed must be int-like, got {raw!r}") from e


def _run_go_trials(
    composition: list[str],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Calculator | None = None,
) -> list[tuple[float, Atoms]]:
    """Run global optimization for a composition; return unique minima sorted by energy."""
    configure_logging(verbosity)

    policy = get_system_policy(system_type)
    allow_empty_comp = policy.slab_is_search_target and not policy.has_adsorbate
    validate_composition(composition, allow_empty=allow_empty_comp, allow_tuple=False)

    params = _prepare_go_params(params)

    # Validate calculator availability
    calculator_name = params.get("calculator", "MACE")
    _ = get_calculator_class(calculator_name)

    seed = _resolve_go_seed(seed, params)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    surface_cfg = params.get("surface_config")
    ads_def = as_adsorbate_definition(params.get("adsorbate_definition"))
    search_comp = resolve_search_mobile_composition(
        system_type=system_type,
        composition=list(composition),
        surface_config=surface_cfg
        if isinstance(surface_cfg, SurfaceSystemConfig)
        else None,
        adsorbate_definition=ads_def,
    )
    n_atoms = len(search_comp)
    cluster_formula = (
        get_cluster_formula(composition)
        if composition
        else (getattr(surface_cfg, "name", None) or "surface")
    )
    path_key = resolve_run_path_key(composition, system_type=system_type, params=params)
    main_output_dir = str(resolve_go_searches_dir(output_dir, path_key))

    # Algorithm selection: simple optimization for 1-2 search-mobile atoms (plain
    # gas clusters only), Basin Hopping for 3 (Genetic Algorithm when adsorbates
    # are present), Genetic Algorithm otherwise
    chosen_go = select_scgo_minima_algorithm(n_atoms, system_type)
    if chosen_go == "simple":
        logger.info(
            "Selected simple optimization for %d-atom cluster (trivial structure)",
            n_atoms,
        )
    elif chosen_go == "bh":
        logger.info(
            "Selected Basin Hopping for %d search-mobile atoms (small system)", n_atoms
        )
    else:
        logger.info("Selected Genetic Algorithm for %d search-mobile atoms", n_atoms)

    # Extract algorithm-specific parameters without mutation
    algo_params = params["optimizer_params"].get(chosen_go, {})

    # Validate algorithm-specific parameters
    validate_algorithm_params(algo_params, chosen_go)

    # Get calculator kwargs if provided
    calculator_kwargs = params.get("calculator_kwargs", {})

    # Unified parameter preparation (resolves auto params, fitness strategy, diversity, etc.)
    global_optimizer_kwargs = prepare_algorithm_kwargs(
        algo_params=algo_params,
        params=params,
        composition=composition,
        chosen_go=chosen_go,
        system_type=system_type,
    )

    # Validate that no unexpected top-level keys were provided. Keep in sync with
    # defaults; a few accepted inputs (surface/adsorbate config, validation n_jobs)
    # are valid top-level keys that are intentionally absent from defaults.
    expected_top_level_keys = _EXPECTED_TOP_LEVEL_KEYS
    unexpected_keys = set(params.keys()) - expected_top_level_keys
    if unexpected_keys:
        raise SCGOValidationError(
            f"Unexpected parameter keys: {sorted(unexpected_keys)}. "
            f"Expected keys: {sorted(expected_top_level_keys)}"
        )

    # Log the final configuration being used
    log_configuration(
        params=params,
        chosen_go=chosen_go,
        cluster_formula=cluster_formula,
        n_atoms=n_atoms,
        global_optimizer_kwargs=global_optimizer_kwargs,
        verbosity=verbosity,
        user_params=None,
        params_base=None,
    )

    final_unique_minima = run_trials(
        composition=composition,
        global_optimizer=chosen_go,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=main_output_dir,
        calculator_for_global_optimization=(
            calculator_for_global_optimization
            if calculator_for_global_optimization is not None
            else get_calculator_class(params["calculator"])(**calculator_kwargs)
        ),
        validate_with_hessian=params.get("validate_with_hessian", False),
        fmax_threshold=params.get("fmax_threshold", DEFAULT_FMAX_THRESHOLD),
        check_hessian=params.get("check_hessian", True),
        imag_freq_threshold=params.get("imag_freq_threshold", 50.0),
        validation_n_jobs=inherit_n_jobs(
            params.get("validation_n_jobs"), params.get("n_jobs")
        ),
        tag_final_minima=params.get("tag_final_minima", True),
        rng=rng,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        search_mobile_count=len(search_comp),
    )

    cleanup_torch_cuda(logger=logger)

    return final_unique_minima


def _run_go_campaign_compositions(
    compositions: Iterable[list[str]],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Calculator | None = None,
    composition_adsorbate: list[tuple[AdsorbateDefinition | None, object]]
    | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run global optimization for an iterable of compositions; map path key->minima."""
    compositions_list = list(compositions)
    if not compositions_list:
        raise SCGOValidationError("compositions iterable must not be empty")
    if composition_adsorbate is not None and len(composition_adsorbate) != len(
        compositions_list
    ):
        raise SCGOValidationError(
            "composition_adsorbate must be parallel to compositions "
            f"(got {len(composition_adsorbate)} vs {len(compositions_list)})."
        )

    params = _prepare_go_params(params)
    configure_logging(verbosity)

    # Generate run_id once at campaign start if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    seed = _resolve_go_seed(seed, params)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    all_results = {}
    num_compositions = len(compositions_list)
    logger.info(
        "Starting global optimization campaign for %d compositions", num_compositions
    )

    # Create the calculator once and reuse it for all compositions to avoid file
    # handle leaks. When the caller supplies a pre-warmed calculator (e.g. a loaded
    # MLIP) it is reused as-is for every composition; the campaign only builds its
    # own when none is supplied. Note ``params["calculator"]`` is still validated
    # per composition inside ``_run_go_trials`` (``get_calculator_class(calculator_name)``),
    # so skipping the factory lookup here loses no validation. A caller-owned
    # calculator is mutated in place (its ``directory`` is set per run by
    # ``minima_search``), and ends pointing at the last composition's run dir.
    owns_calculator = calculator_for_global_optimization is None
    if owns_calculator:
        calculator_kwargs = params.get("calculator_kwargs", {})
        calculator_for_global_optimization = get_calculator_class(params["calculator"])(
            **calculator_kwargs,
        )

    for i, composition in enumerate(compositions_list):
        formula_str = get_cluster_formula(composition)
        trial_params = params
        if composition_adsorbate is not None:
            ads_def, ads_temp = composition_adsorbate[i]
            trial_params = _merge_adsorbate_context_into_params(
                params,
                adsorbate_definition=ads_def,
                adsorbate_fragment_template=ads_temp,
            )
        path_key = resolve_run_path_key(
            composition, system_type=system_type, params=trial_params
        )
        log_phase_header(
            logger,
            f"Running minima search for {path_key} ({i + 1}/{num_compositions})",
            verbosity=verbosity,
        )

        comp_seed = int(rng.integers(0, 2**63 - 1))
        trial_output_dir = resolve_go_campaign_searches_dir(output_dir, path_key)
        trial_output_dir_str = (
            str(trial_output_dir) if trial_output_dir is not None else None
        )

        try:
            results = _run_go_trials(
                composition,
                system_type,
                trial_params,
                seed=comp_seed,
                verbosity=verbosity,
                run_id=run_id,
                clean=clean,
                output_dir=trial_output_dir_str,
                calculator_for_global_optimization=calculator_for_global_optimization,
            )
            # Always add results (possibly empty) so the API returns a key for each
            # requested composition; this makes the function predictable for
            # downstream consumers and tests.
            all_results[path_key] = results
            if not results and verbosity >= 1:
                log_warning_v(
                    logger,
                    "No minima found for %s (results empty)",
                    path_key,
                    verbosity=verbosity,
                )
            log_info_v(logger, "Finished processing %s", path_key, verbosity=verbosity)
            log_info_v(
                logger,
                "  Returned %d final minima for %s",
                len(results),
                path_key,
                verbosity=verbosity,
            )
        except (
            RuntimeError,
            ValueError,
            OSError,
            sqlite3.DatabaseError,
            SCGODatabaseError,
            SCGOFileError,
            SCGOValidationError,
        ) as e:
            # Enhanced error logging for HPC debugging
            error_details = [
                f"Failed to process {path_key} ({formula_str}): {e}",
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
            all_results[path_key] = []
            log_warning_v(
                logger,
                "Skipping %s (%s) and continuing campaign (%d/%d)",
                path_key,
                formula_str,
                i + 1,
                num_compositions,
                verbosity=verbosity,
            )
            continue

    # Best-effort: free CUDA memory to avoid fragmentation when campaigns are run
    # sequentially in the same process. Only drop our own calculator reference when
    # the campaign built it; a caller-supplied calculator is left intact for reuse.
    if owns_calculator:
        del calculator_for_global_optimization
    cleanup_torch_cuda(logger=logger)

    return all_results
