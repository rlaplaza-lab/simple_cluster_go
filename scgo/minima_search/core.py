"""Core workflow functions for global optimization.

Coordinates trials, manages output, filters results, validates minima.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import Counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.io import write

from scgo.algorithms import bh_go, ga_go, simple_go
from scgo.database import SCGODatabaseManager
from scgo.database.metadata import (
    add_metadata,
    get_metadata,
    mark_final_minima_in_db,
)
from scgo.initialization import create_initial_cluster
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.helpers import (
    canonicalize_storage_frame,
    compute_final_id,
    ensure_directory_exists,
    filter_dict_keys,
    filter_unique_minima,
    get_cluster_formula,
    get_provenance,
    is_true_minimum,
)
from scgo.utils.logging import get_logger
from scgo.utils.optimizer_utils import get_optimizer_class
from scgo.utils.rng_helpers import create_child_rng
from scgo.utils.run_tracking import (
    RunMetadataJSONEncoder,
    ensure_run_id,
    save_run_metadata,
)
from scgo.utils.ts_provenance import ts_output_provenance
from scgo.utils.validation import validate_composition

# Default required calculator methods
_DEFAULT_REQUIRED_METHODS = ["get_potential_energy", "get_forces"]


def _sanitize_global_optimizer_kwargs_for_metadata(
    global_optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Copy kwargs for JSON metadata: drop non-serializable objects (relaxer, slab)."""
    gok = global_optimizer_kwargs.copy()
    gok.pop("relaxer", None)
    surface_config = gok.pop("surface_config", None)
    if surface_config is not None:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise TypeError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        slab = surface_config.slab
        n_slab = len(slab)
        gok["surface_config"] = {
            "present": True,
            "n_slab_atoms": n_slab,
            "slab_chemical_symbols": list(slab.get_chemical_symbols()),
            "surface_normal_axis": surface_config.surface_normal_axis,
            "fix_all_slab_atoms": surface_config.fix_all_slab_atoms,
            "n_fix_bottom_slab_layers": surface_config.n_fix_bottom_slab_layers,
            "n_relax_top_slab_layers": surface_config.n_relax_top_slab_layers,
            "adsorption_height_min": surface_config.adsorption_height_min,
            "adsorption_height_max": surface_config.adsorption_height_max,
            "comparator_use_mic": surface_config.comparator_use_mic,
            "cluster_init_vacuum": surface_config.cluster_init_vacuum,
            "init_mode": surface_config.init_mode,
            "max_placement_attempts": surface_config.max_placement_attempts,
        }
    return gok


# Algorithm registry with default parameters
# Note: GA uses special handling via _select_and_run_ga, so function is None
_ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "simple": {
        "function": simple_go,
        "default_niter": 1,
        "requires_calculator": False,
    },
    "bh": {
        "function": bh_go,
        "default_niter": 100,
        "requires_calculator": False,
    },
    "ga": {
        "function": None,  # Special handling via _select_and_run_ga
        "default_niter": 10,
        "requires_calculator": True,
    },
}


def _get_default_calculator() -> Calculator:
    """Get default calculator instance."""
    return EMT()


def _ensure_calculator(calculator: Calculator | None) -> Calculator:
    """Return *calculator* or a default EMT instance when None."""
    return calculator or _get_default_calculator()


def _validate_calculator_compatibility(
    calculator: Calculator,
    required_methods: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate calculator has required methods and returns expected types.

    Args:
        calculator: ASE calculator instance
        required_methods: List of method names to check (default: ["get_potential_energy", "get_forces"])

    Returns:
        tuple: (is_valid, error_message)
    """
    required_methods = required_methods or _DEFAULT_REQUIRED_METHODS

    missing_methods = [
        method_name
        for method_name in required_methods
        if not hasattr(calculator, method_name)
        or not callable(getattr(calculator, method_name))
    ]

    if missing_methods:
        return False, f"Calculator missing required methods: {missing_methods}"

    return True, "Calculator is compatible"


def _is_ml_calculator_for_torchsim(calculator: Calculator) -> bool:
    """True if the calculator looks like an MLIP served by TorchSim+MACE."""
    calculator_class_name = calculator.__class__.__name__
    # UMA/FAIRChem can be driven by TorchSim's FairChem model wrapper; do not
    # exclude it from TorchSim GA selection.
    model = getattr(calculator, "model", None)
    return hasattr(model, "forward") or calculator_class_name in (
        "MACECalculator",
        "MACE",
        "UMA",
        "FAIRChemCalculator",
    )


def _get_ga_go_torchsim():
    try:
        from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim as _impl
    except ImportError as e:
        raise ImportError(
            "TorchSim GA requires TorchSim. Install with: pip install 'scgo[mace]' "
            "(MACE) or 'scgo[uma]' (UMA) depending on the model family."
        ) from e
    return _impl


def ga_go_torchsim(*args, **kwargs):
    """TorchSim GA entry point; lazy-imports MACE deps (allows tests to monkeypatch)."""
    return _get_ga_go_torchsim()(*args, **kwargs)


def _filter_ga_kwargs_for_torchsim(
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Filter optimizer kwargs for TorchSim GA (drops ASE-only keys, including stale ``use_torchsim``)."""
    return filter_dict_keys(optimizer_kwargs, {"optimizer", "use_torchsim"})


def _filter_ga_kwargs_for_ase(
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Filter and normalize optimizer kwargs for ASE GA.

    Removes TorchSim-specific keys and converts optimizer string to class.

    Args:
        optimizer_kwargs: Dictionary of optimizer parameters.

    Returns:
        Filtered and normalized dictionary suitable for ASE GA.
    """
    ase_ga_kwargs = filter_dict_keys(
        optimizer_kwargs, {"relaxer", "batch_size", "use_torchsim"}
    )

    optimizer_name = ase_ga_kwargs.get("optimizer")
    if isinstance(optimizer_name, str):
        ase_ga_kwargs["optimizer"] = get_optimizer_class(optimizer_name)

    return ase_ga_kwargs


def _resolve_ga_backend(
    optimizer_kwargs: dict[str, Any],
    calculator: Calculator,
    logger: Any,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """TorchSim GA for MLIPs; ASE GA otherwise. Ignores any ``use_torchsim`` key (removed from public API)."""
    requested_torchsim = optimizer_kwargs.get("relaxer") is not None or bool(
        optimizer_kwargs.get("use_torchsim")
    )
    ml = _is_ml_calculator_for_torchsim(calculator)
    torchsim_kwargs = _filter_ga_kwargs_for_torchsim(optimizer_kwargs)
    ase_ga_kwargs = _filter_ga_kwargs_for_ase(optimizer_kwargs)

    if requested_torchsim:
        # Fail fast if TorchSim was explicitly requested but isn't importable.
        # This avoids silent backends switches that can hide misconfiguration.
        _ = _get_ga_go_torchsim()
        logger.debug("Using TorchSim GA (explicit request)")
        return True, torchsim_kwargs, {}

    if ml:
        opt = optimizer_kwargs.get("optimizer")
        if isinstance(opt, str) and opt.upper() != "FIRE":
            logger.debug('TorchSim GA ignores optimizer "%s"; uses FIRE.', opt)
        logger.debug("Using TorchSim GA (ML calculator)")
        return True, torchsim_kwargs, {}

    logger.debug("Using ASE GA (non-ML calculator)")
    return False, {}, ase_ga_kwargs


def _select_and_run_ga(
    composition: list[str],
    output_dir: str,
    optimizer_kwargs: dict[str, Any],
    calculator: Calculator,
    rng: np.random.Generator,
    verbosity: int,
    run_id: str | None = None,
    clean: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run TorchSim GA for MLIPs and ASE GA for classical calculators.

    Args:
        composition: List of element symbols defining the cluster composition.
        output_dir: Directory for output files.
        optimizer_kwargs: Dictionary of optimizer parameters.
        calculator: The ASE calculator to use.
        rng: Random number generator.
        verbosity: Verbosity level.

    Returns:
        List of ``(energy, Atoms)`` tuples from the GA run.
    """
    logger = get_logger(__name__)

    use_torchsim, torchsim_kwargs, ase_ga_kwargs = _resolve_ga_backend(
        optimizer_kwargs, calculator, logger
    )

    if use_torchsim:
        return ga_go_torchsim(
            composition=composition,
            output_dir=output_dir,
            calculator=calculator,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **torchsim_kwargs,
        )

    return ga_go(
        composition=composition,
        output_dir=output_dir,
        calculator=calculator,
        rng=rng,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        **ase_ga_kwargs,
    )


def scgo(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    trial_id: int = 1,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run a single global optimization trial for a fixed composition.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name ("simple", "bh", or "ga").
        global_optimizer_kwargs: Optimizer parameters.
        output_dir: Trial output directory.
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        trial_id: Trial identifier.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        run_id: Optional run ID.
        clean: Start fresh if True.

    Returns:
        List of (energy, Atoms) for minima.

    Raises:
        ValueError: For invalid parameters.
    """
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    if not isinstance(global_optimizer, str):
        raise ValueError("global_optimizer must be a string")

    if not isinstance(global_optimizer_kwargs, dict):
        raise ValueError("global_optimizer_kwargs must be a dictionary")

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")

    # RNG must be a numpy Generator (required for deterministic behavior)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")

    if not isinstance(trial_id, int) or trial_id < 1:
        raise ValueError("trial_id must be a positive integer")

    if not isinstance(verbosity, int) or verbosity not in (0, 1, 2, 3):
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")

    calculator_for_global_optimization = _ensure_calculator(
        calculator_for_global_optimization
    )

    # Ensure file-based calculators run in the trial directory to avoid collisions
    if hasattr(calculator_for_global_optimization, "directory"):
        calculator_for_global_optimization.directory = output_dir

    is_valid, error_msg = _validate_calculator_compatibility(
        calculator_for_global_optimization
    )
    if not is_valid:
        calc_type = type(calculator_for_global_optimization).__name__
        calc_module = type(calculator_for_global_optimization).__module__
        raise ValueError(
            f"Calculator validation failed: {error_msg}. "
            f"Calculator type: {calc_type} (from {calc_module}). "
            f"Ensure the calculator implements get_potential_energy() and get_forces() methods."
        )

    # Filter keys handled at scgo/run_trials level so **optimizer_kwargs cannot
    # override explicit run_id/clean.
    optimizer_kwargs = filter_dict_keys(
        global_optimizer_kwargs, {"n_trials", "run_id", "clean"}
    )

    ensure_directory_exists(output_dir)

    optimizer_name_lower = global_optimizer.lower()
    if optimizer_name_lower not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown global_optimizer: {global_optimizer}. "
            f"Must be one of {list(_ALGORITHM_REGISTRY.keys())}"
        )

    algo_config = _ALGORITHM_REGISTRY[optimizer_name_lower]
    algo_function = algo_config["function"]

    if optimizer_name_lower == "ga":
        # GA requires special handling with calculator
        all_minima = _select_and_run_ga(
            composition=composition,
            output_dir=output_dir,
            optimizer_kwargs=optimizer_kwargs,
            calculator=calculator_for_global_optimization,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
        )
    else:
        # Other algorithms use standard function signature and need atoms
        atoms = create_initial_cluster(composition, rng=rng)
        atoms.calc = calculator_for_global_optimization
        all_minima = algo_function(
            atoms=atoms,
            output_dir=output_dir,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **optimizer_kwargs,
        )

    if not all_minima:
        logger.info("Global optimization finished but found no valid minima.")
        return []

    for _, atoms_obj in all_minima:
        add_metadata(atoms_obj, run_id=run_id, trial_id=trial_id)

    return all_minima


def run_trials(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    n_trials: int,
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    validate_with_hessian: bool = True,
    tag_final_minima: bool = True,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run multiple global optimization trials, filter and validate results.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name (e.g., "bh", "ga").
        global_optimizer_kwargs: Optimizer parameters.
        n_trials: Number of trials.
        output_dir: Output directory.
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        validate_with_hessian: Whether to validate with Hessian.
        verbosity: Verbosity level.
        run_id: Optional run ID.
        clean: Start fresh if True.

    Returns:
        List of (energy, Atoms) for unique minima.
    """
    logger = get_logger(__name__)

    # Validate inputs early
    validate_composition(composition, allow_empty=False, allow_tuple=False)

    if not isinstance(global_optimizer, str):
        raise ValueError("global_optimizer must be a string")

    if not isinstance(global_optimizer_kwargs, dict):
        raise ValueError("global_optimizer_kwargs must be a dictionary")

    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")

    if not isinstance(validate_with_hessian, bool):
        raise ValueError("validate_with_hessian must be a boolean")

    if not isinstance(verbosity, int) or verbosity not in (0, 1, 2, 3):
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")

    calculator_for_global_optimization = _ensure_calculator(
        calculator_for_global_optimization
    )

    # Generate run_id if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    # Create run-specific output directory
    run_output_dir = os.path.join(output_dir, run_id)
    ensure_directory_exists(run_output_dir)

    # Ensure final unique minima directory exists even if no minima are found
    final_xyz_dir = os.path.join(output_dir, "final_unique_minima")
    ensure_directory_exists(final_xyz_dir)

    # Cache cluster formula (used multiple times)
    composition_str = get_cluster_formula(composition)

    # Save run metadata (include formula and run parameters for traceability)
    gok_for_metadata = _sanitize_global_optimizer_kwargs_for_metadata(
        global_optimizer_kwargs
    )
    params = {
        "global_optimizer": global_optimizer,
        "global_optimizer_kwargs": gok_for_metadata,
        "n_trials": n_trials,
        "validate_with_hessian": validate_with_hessian,
        "verbosity": verbosity,
        "clean": clean,
        "calculator": calculator_for_global_optimization.__class__.__name__
        if calculator_for_global_optimization
        else None,
    }
    save_run_metadata(
        run_output_dir,
        run_id,
        metadata={
            "composition": composition,
            "formula": composition_str,
            "params": params,
        },
    )

    # Load previous run results BEFORE running trials (better UX)
    previous_minima = []
    if not clean:
        # Use database manager for efficient loading with caching
        with SCGODatabaseManager(
            base_dir=output_dir, enable_caching=True
        ) as db_manager:
            previous_minima = db_manager.load_previous_results(
                composition=composition,
                current_run_id=run_id,
                prefer_final_unique=True,
            )
            if previous_minima:
                logger.info(
                    f"Loaded {len(previous_minima)} minima from previous runs "
                    f"(excluding current run {run_id})"
                )

    all_raw_minima = []

    for trial_idx in range(n_trials):
        trial_rng = create_child_rng(rng)

        trial_dir = os.path.join(run_output_dir, f"trial_{trial_idx + 1}")
        logger.info(f"Running trial {trial_idx + 1}/{n_trials}")

        trial_results = scgo(
            composition=composition,
            global_optimizer=global_optimizer,
            global_optimizer_kwargs=global_optimizer_kwargs,
            output_dir=trial_dir,
            rng=trial_rng,
            calculator_for_global_optimization=calculator_for_global_optimization,
            trial_id=trial_idx + 1,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
        )

        if trial_results:
            all_raw_minima.extend(trial_results)

    # Combine all results (previous + current) before deduplication
    if previous_minima:
        all_minima_for_filtering = previous_minima + all_raw_minima
        logger.info(
            f"Combined {len(previous_minima)} previous + {len(all_raw_minima)} current minima"
        )
    else:
        all_minima_for_filtering = all_raw_minima

    if not all_minima_for_filtering:
        logger.info("No minima found.")
        return []

    logger.info(
        f"All trials complete. Found {len(all_raw_minima)} raw minima from current run."
    )
    logger.info("Filtering for unique structures across all runs...")
    unique_candidates = filter_unique_minima(all_minima_for_filtering)
    logger.info(f"Found {len(unique_candidates)} unique candidates.")

    if not unique_candidates:
        return []

    if validate_with_hessian:
        logger.info(
            f"Validating {len(unique_candidates)} unique candidates to confirm they are true minima...",
        )

        # Ensure validation runs in a separate directory to avoid overwriting trial files
        if hasattr(calculator_for_global_optimization, "directory"):
            val_dir = os.path.join(output_dir, "validation")
            ensure_directory_exists(val_dir)
            calculator_for_global_optimization.directory = val_dir

        validated_minima = []
        for i, (energy, atoms) in enumerate(unique_candidates):
            logger.info(
                f"Validating candidate {i + 1}/{len(unique_candidates)} (E={energy:.4f} eV)...",
            )
            try:
                is_valid = is_true_minimum(
                    atoms=atoms,
                    calculator=calculator_for_global_optimization,
                    check_hessian=True,
                )
                if is_valid:
                    validated_minima.append((energy, atoms))
                else:
                    logger.info(f"Candidate {i + 1} rejected")
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    f"Validation failed for candidate {i + 1} (E={energy:.4f} eV): {e}"
                )

        if not validated_minima:
            logger.info(
                "Validation finished. No candidates were confirmed as true minima."
            )
            return []

        final_minima = validated_minima
    else:
        final_minima = unique_candidates

    best_energy, _ = final_minima[0]
    logger.info(f"Process complete. Found {len(final_minima)} final unique minima.")
    logger.info(f"Best potential energy: {best_energy:.4f} eV")

    final_xyz_dir = os.path.join(output_dir, "final_unique_minima")
    ensure_directory_exists(final_xyz_dir)
    logger.info(
        f'Writing {len(final_minima)} final structures to "{os.path.basename(final_xyz_dir)}"'
    )

    # Write results summary file (composition_str already cached above)
    _write_results_summary(
        output_dir=output_dir,
        final_minima=final_minima,
        composition_str=composition_str,
        run_id=run_id,
        params=params,
    )

    final_minima_info: list[dict] = []
    for i, (_energy, atoms) in enumerate(final_minima):
        provenance = get_provenance(atoms)
        trial_id = provenance.get("trial", "N/A")
        atoms_run_id = provenance.get("run_id", run_id)

        # Format: Pt2_minimum_01_run_20260120_003007_trial_1.xyz
        # (run_id already contains "run_" prefix, so don't add it again)
        filename = (
            f"{composition_str}_minimum_{i + 1:02d}_{atoms_run_id}_trial_{trial_id}.xyz"
        )
        filepath = os.path.join(final_xyz_dir, filename)

        atoms_clean = atoms.copy()
        atoms_clean.calc = None
        n_slab_meta = get_metadata(atoms_clean, "n_slab_atoms", 0) or 0
        if get_metadata(atoms_clean, "system_kind") == "slab_adsorbate" and n_slab_meta:
            canonicalize_storage_frame(
                atoms_clean,
                pbc_aware=True,
                center=False,
                n_slab=int(n_slab_meta),
            )
        else:
            canonicalize_storage_frame(atoms_clean)
        if "tags" in atoms_clean.arrays:
            del atoms_clean.arrays["tags"]

        write(filepath, atoms_clean)

        try:
            final_id = compute_final_id(atoms_clean, _energy)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"compute_final_id failed for {filepath}: {e}")
            final_id = None

        final_minima_info.append(
            {
                "atoms": atoms,
                "energy": _energy,
                "rank": i + 1,
                "final_written": filepath,
                "final_id": final_id,
            }
        )

    # Mark final minima in DB (if enabled) to avoid re-scanning later
    if tag_final_minima:
        try:
            mark_final_minima_in_db(final_minima_info, base_dir=output_dir)
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
            # Consider DB tagging a systemic failure -- surface it after logging
            logger.warning(f"Failed to tag final minima in DB: {e}")
            raise

    return final_minima


def _write_results_summary(
    output_dir: str,
    final_minima: list[tuple[float, Atoms]],
    composition_str: str,
    run_id: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Write a summary file of results by run.

    Args:
        output_dir: Base output directory.
        final_minima: List of final unique minima.
        composition_str: Chemical formula string.
        run_id: Current run ID.
        params: Same snapshot as ``run_*/metadata.json`` (optimizer, trials, etc.).
    """
    logger = get_logger(__name__)

    # Count structures by run_id
    run_counts = Counter()
    for _, atoms in final_minima:
        provenance = get_provenance(atoms)
        run_id_from_atoms = provenance.get("run_id", run_id)
        run_counts[run_id_from_atoms] += 1

    summary = ts_output_provenance()
    summary.update(
        {
            "composition": composition_str,
            "total_unique_minima": len(final_minima),
            "minima_by_run": dict(run_counts),
            "current_run_id": run_id,
            "params": params,
            "run_metadata_relpath": (f"{run_id}/metadata.json" if run_id else None),
        }
    )

    summary_file = os.path.join(output_dir, "results_summary.json")
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, cls=RunMetadataJSONEncoder)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Wrote results summary to {summary_file}")
    except (OSError, TypeError) as e:
        logger.warning(f"Failed to write results summary: {e}")
        raise
