"""Core workflow functions for global optimization.

Coordinates datetime-tagged runs, manages output, filters results, validates minima.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase_ga.utilities import get_all_atom_types

from scgo.algorithms import bh_go, ga_go, simple_go
from scgo.cluster_adsorbate.hierarchical import (
    build_hierarchical_core_fragment_cluster,
)
from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_FMAX_THRESHOLD
from scgo.database import SCGODatabaseManager
from scgo.exceptions import (
    SCGODatabaseError,
    SCGOFileError,
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.initialization import create_initial_cluster
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.metadata.atoms import ensure_final_id, get_tag, get_tags, set_tags
from scgo.metadata.persist import mark_final_minima_in_db
from scgo.metadata.provenance import output_json_provenance
from scgo.metadata.run_dir import (
    RunDirJSONEncoder,
    ensure_run_id,
    save_run_dir_record,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.surface.validation import (
    validate_stored_mobile_partition_metadata,
    validate_stored_slab_adsorbate_metadata,
)
from scgo.system_types import (
    as_adsorbate_definition,
    get_system_policy,
    resolve_mobile_composition,
    resolve_structure_mic,
    validate_adsorbate_definition,
    validate_minimum_structure,
    validate_system_type_settings,
)
from scgo.system_types.dedup_geometry import resolve_uniqueness_geometry
from scgo.utils.comparators import ComparatorBlocks, uniqueness_settings_from_mapping
from scgo.utils.fitness_strategies import resolve_fitness_strategy
from scgo.utils.helpers import (
    adsorbate_primary_cell_shift,
    apply_primary_cell_shift,
    canonicalize_storage_frame,
    ensure_directory_exists,
    filter_unique_minima,
    get_cluster_formula,
    is_true_minimum,
)
from scgo.utils.logging import get_logger, log_debug_v, log_info_v
from scgo.utils.parallel_workers import resolve_n_jobs_for_tasks
from scgo.utils.path_keys import resolve_run_path_key
from scgo.utils.phase_logging import format_count_summary
from scgo.utils.rng_helpers import create_child_rng
from scgo.utils.validation import validate_composition

logger = get_logger(__name__)

_SURFACE_SYSTEM_TYPES = frozenset(
    {
        "surface_cluster",
        "surface_cluster_adsorbate",
        "surface",
        "surface_adsorbate",
    }
)

_VALIDATION_CALCULATOR: Calculator | None = None
_MIN_PARALLEL_VALIDATION_CANDIDATES = 4


def _init_validation_worker(calculator: Calculator) -> None:
    global _VALIDATION_CALCULATOR
    _VALIDATION_CALCULATOR = calculator


def _validate_minimum_worker(
    payload: tuple[float, Atoms, float, bool, float],
) -> tuple[float, Atoms] | None:
    energy, atoms, fmax_threshold, check_hessian, imag_freq_threshold = payload
    if _VALIDATION_CALCULATOR is None:
        raise SCGORuntimeError("Validation worker calculator not initialized")
    if is_true_minimum(
        atoms=atoms,
        calculator=_VALIDATION_CALCULATOR,
        fmax_threshold=fmax_threshold,
        check_hessian=check_hessian,
        imag_freq_threshold=imag_freq_threshold,
    ):
        return (energy, atoms)
    return None


def _validate_candidates_parallel(
    calculator: Calculator,
    payloads: list[tuple[float, Atoms, float, bool, float]],
    n_workers: int,
) -> tuple[bool, list[tuple[float, Atoms]]]:
    """Validate candidates in a process pool; return ``(ok, minima)``.

    Returns ``(False, [])`` when parallel startup is skipped or fails so the
    caller can fall back to sequential validation. Successful runs return
    ``(True, validated_minima)``.
    """
    if n_workers <= 1 or not payloads:
        return False, []

    try:
        deepcopy(calculator)
    except Exception as e:
        logger.warning(
            "Calculator is not deep-copyable (%s); "
            "falling back to sequential validation",
            e,
        )
        return False, []

    logger.info(
        "Validating %d unique candidates with up to %d parallel workers...",
        len(payloads),
        n_workers,
    )
    validated_minima: list[tuple[float, Atoms]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_validation_worker,
            initargs=(calculator,),
        ) as executor:
            futures = [
                executor.submit(_validate_minimum_worker, payload)
                for payload in payloads
            ]
            for i, future in enumerate(futures, 1):
                try:
                    validated = future.result()
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    SCGOValidationError,
                ) as e:
                    logger.warning("Parallel validation task %d failed: %s", i, e)
                    continue
                if validated is not None:
                    validated_minima.append(validated)
    except (
        TypeError,
        AttributeError,
        OSError,
        RuntimeError,
        pickle.PicklingError,
        BrokenProcessPool,
    ) as e:
        logger.warning(
            "Parallel validation failed to start (%s); falling back to sequential",
            e,
        )
        return False, []

    return True, validated_minima


def _create_surface_initialized_atoms(
    *,
    composition: list[str],
    surface_config: SurfaceSystemConfig,
    rng: np.random.Generator,
    adsorbate_definition: Any = None,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: Any = None,
    system_type: str | None = None,
) -> Atoms:
    working_config = surface_config
    policy = get_system_policy(system_type) if system_type is not None else None
    n_fixed = 0
    if policy is not None and policy.slab_is_search_target:
        working_config, partition = prepare_slab_search_surface_config(surface_config)
        n_fixed = partition.n_fixed
        if not policy.has_adsorbate:
            atoms = working_config.slab.copy()
            pos = atoms.get_positions()
            pos[n_fixed:] += rng.normal(0.0, 0.35, size=pos[n_fixed:].shape)
            atoms.set_positions(pos)
            return atoms
        # Adsorbate-only deposit onto the reordered slab.
        ads_def = as_adsorbate_definition(adsorbate_definition)
        deposit_composition = (
            list(ads_def.adsorbate_symbols) if ads_def is not None else []
        )
    else:
        deposit_composition = list(composition)

    slab = working_config.slab
    n_slab = len(slab)
    n_top = len(deposit_composition)
    if n_top == 0 and policy is not None and policy.slab_is_search_target:
        raise SCGORuntimeError(
            "surface_adsorbate initialization requires adsorbate symbols."
        )
    template = Atoms(
        symbols=list(slab.get_chemical_symbols()) + deposit_composition,
        positions=np.vstack([slab.get_positions(), np.zeros((n_top, 3))]),
        cell=slab.cell,
        pbc=slab.pbc,
    )
    idx_top = range(n_slab, n_slab + n_top)
    blmin = build_blmin_from_zs(
        get_all_atom_types(template, list(idx_top)),
        ratio=BLMIN_RATIO_DEFAULT,
    )
    deposited = create_deposited_cluster(
        composition=deposit_composition,
        slab=slab,
        blmin=blmin,
        rng=rng,
        config=working_config,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=adsorbate_fragment_template,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    if deposited is None:
        raise SCGORuntimeError("Failed to create initial surface-supported structure.")
    if policy is not None and policy.slab_is_search_target and n_fixed > 0:
        pos = deposited.get_positions()
        pos[n_fixed:n_slab] += rng.normal(0.0, 0.25, size=pos[n_fixed:n_slab].shape)
        deposited.set_positions(pos)
    return deposited


def _create_gas_cluster_adsorbate_initial_atoms(
    *,
    composition: list[str],
    rng: np.random.Generator,
    adsorbate_definition: Any,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: Any = None,
    vacuum: float = 10.0,
    init_mode: str = "smart",
    max_hierarchical_attempts: int = 200,
    previous_search_glob: str = "**/*.db",
    verbosity: int = 1,
) -> Atoms:
    """Build hierarchical gas-phase core+fragment seed for adsorbate runs."""
    atoms = build_hierarchical_core_fragment_cluster(
        adsorbate_definition,
        rng,
        previous_search_glob,
        adsorbate_fragment_template,
        cluster_adsorbate_config,
        cluster_init_vacuum=vacuum,
        init_mode=init_mode,
        max_placement_attempts=max_hierarchical_attempts,
        verbosity=verbosity,
    )
    if atoms is None:
        raise SCGORuntimeError(
            "Failed to build hierarchical gas-phase core+fragment seed; "
            "increase max_hierarchical_attempts or relax fragment placement."
        )
    return atoms


def _is_slab_surface_minimum(atoms: Atoms) -> tuple[bool, int]:
    """Return whether ``atoms`` is a slab surface minimum and its ``n_slab_atoms``."""
    system_type = get_tag(atoms, "system_type")
    n_slab = int(get_tag(atoms, "n_slab_atoms", 0) or 0)
    if system_type in _SURFACE_SYSTEM_TYPES and n_slab > 0:
        return True, n_slab
    return False, n_slab


def _resolve_surface_alignment_kwargs(
    global_optimizer_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve slab final-write alignment knobs from GO kwargs and system policy."""
    system_type = global_optimizer_kwargs.get("system_type")
    if not isinstance(system_type, str):
        raise SCGOValidationError(
            "system_type must be set in global_optimizer_kwargs for surface alignment."
        )
    policy = get_system_policy(system_type)  # type: ignore[arg-type]
    if not policy.uses_surface:
        return None

    cell_remap = bool(
        global_optimizer_kwargs.get(
            "neb_surface_cell_remap", policy.neb_surface_cell_remap
        )
    )
    lattice_rotation = bool(
        global_optimizer_kwargs.get(
            "neb_surface_lattice_rotation", policy.neb_surface_lattice_rotation
        )
    )
    max_shift = int(global_optimizer_kwargs.get("neb_surface_max_lattice_shift", 1))
    cell_remap = policy.neb_surface_cell_remap and cell_remap
    lattice_rotation = policy.neb_surface_lattice_rotation and lattice_rotation
    return {
        "enable_cell_remap": cell_remap,
        "enable_lattice_rotation": lattice_rotation,
        "max_lattice_shift": max_shift,
    }


def _resolve_n_core_mobile_for_alignment(
    atoms: Atoms,
    global_optimizer_kwargs: dict[str, Any],
) -> int | None:
    """Resolve core mobile count for surface PBC final-write alignment."""
    n_core_meta = get_tag(atoms, "n_core_atoms", None)
    if n_core_meta is not None:
        try:
            n_core = int(n_core_meta)
        except (TypeError, ValueError):
            n_core = None
        else:
            if n_core >= 0:
                return n_core

    ads_def = as_adsorbate_definition(
        global_optimizer_kwargs.get("adsorbate_definition")
    )
    if ads_def is not None:
        return int(ads_def.n_core)
    return None


def _align_slab_minimum_to_reference(
    reference: Atoms,
    candidate: Atoms,
    *,
    n_slab: int,
    enable_cell_remap: bool,
    enable_lattice_rotation: bool,
    max_lattice_shift: int,
    n_core_mobile: int | None = None,
) -> None:
    """Align ``candidate`` to ``reference`` using the TS slab PBC protocol (in-place).

    Writes the already-computed aligned coordinates without running constraint
    projectors (same contract as NEB ``interpolate(..., apply_constraint=False)``).
    """
    from scgo.ts_search.transition_state import _align_product_surface_pbc

    aligned = _align_product_surface_pbc(
        reference,
        candidate.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=enable_cell_remap,
        enable_lattice_rotation=enable_lattice_rotation,
        max_lattice_shift=max_lattice_shift,
        n_core_mobile=n_core_mobile,
    )
    candidate.set_positions(aligned, apply_constraint=False)
    candidate.set_cell(reference.cell)
    candidate.pbc = reference.pbc


# Consumed by ``scgo`` for hierarchical/surface init only; not passed to simple/BH.
_INIT_ONLY_OPTIMIZER_KWARGS = frozenset(
    {
        "adsorbate_fragment_template",
        "vacuum",
        "init_mode",
        "max_hierarchical_attempts",
        "previous_search_glob",
    }
)


def _write_timing_json_enabled(global_optimizer_kwargs: dict[str, Any]) -> bool:
    return bool(global_optimizer_kwargs.get("write_timing_json", False))


def _optimizer_kwargs_for_algorithm_call(
    optimizer_kwargs: dict[str, Any],
    *,
    global_optimizer: str,
) -> dict[str, Any]:
    """Return kwargs safe to pass to simple/BH after initial structure construction."""
    if global_optimizer == "ga":
        return optimizer_kwargs
    return {
        key: value
        for key, value in optimizer_kwargs.items()
        if key not in _INIT_ONLY_OPTIMIZER_KWARGS
    }


def _sanitize_global_optimizer_kwargs_for_metadata(
    global_optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Copy GO kwargs for JSON metadata, dropping non-serializable objects.

    ``relaxer``, ``adsorbate_fragment_template`` and ``cluster_adsorbate_config``
    are dropped; ``surface_config`` is replaced by a plain-dict slab summary.

    Raises:
        SCGOValidationError: If ``surface_config`` is set but is not a
            :class:`~scgo.surface.config.SurfaceSystemConfig`.
    """
    gok = global_optimizer_kwargs.copy()
    gok.pop("relaxer", None)
    gok.pop("adsorbate_fragment_template", None)
    gok.pop("cluster_adsorbate_config", None)
    surface_config = gok.pop("surface_config", None)
    if surface_config is not None:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise SCGOValidationError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        slab = surface_config.slab
        n_slab = len(slab)
        gok["surface_config"] = {
            "present": True,
            "n_slab_atoms": n_slab,
            "slab_chemical_symbols": slab.get_chemical_symbols(),
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


# Algorithm registry
_ALGORITHM_REGISTRY: dict[str, Callable[..., list[tuple[float, Atoms]]]] = {
    "simple": simple_go,
    "bh": bh_go,
    "ga": ga_go,
}


def _require_calculator(calculator: Calculator | None) -> Calculator:
    """Require an explicit ASE calculator for global optimization."""
    if calculator is None:
        raise SCGOValidationError(
            "calculator_for_global_optimization is required. "
            "Pass an ASE calculator (e.g. EMT() in tests)."
        )
    return calculator


def _validate_common_run_inputs(
    *,
    composition: list[str],
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    verbosity: int,
    require_system_type: bool = False,
) -> None:
    """Validate arguments shared by :func:`scgo` and :func:`run_trials`."""
    system_type = global_optimizer_kwargs.get("system_type")
    allow_empty = False
    if isinstance(system_type, str):
        try:
            policy = get_system_policy(system_type)
            allow_empty = policy.slab_is_search_target and not policy.has_adsorbate
        except KeyError:
            allow_empty = False
    validate_composition(composition, allow_empty=allow_empty, allow_tuple=False)

    if require_system_type and not isinstance(
        global_optimizer_kwargs.get("system_type"), str
    ):
        raise SCGOValidationError(
            "system_type must be set in global_optimizer_kwargs "
            "(e.g. 'gas_cluster', 'surface_cluster')."
        )

    if not isinstance(output_dir, str) or not output_dir:
        raise SCGOValidationError("output_dir must be a non-empty string")

    if not isinstance(rng, np.random.Generator):
        raise SCGOValidationError("rng must be a numpy.random.Generator")

    if not isinstance(verbosity, int) or verbosity not in (0, 1, 2, 3):
        raise SCGOValidationError("verbosity must be one of 0, 1, 2, or 3")


def _compact_structural_gate_reason(message: str) -> str:
    """Short label for grouping final-gate drops (strip wrapper and details)."""
    text = message.removeprefix("Mobile-region validation failed: ").strip()
    return text.split(" (", 1)[0].strip().rstrip(".") or "structural gate"


def _gate_structurally_valid_candidates(
    candidates: list[tuple[float, Atoms]],
    system_type: str,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int | None,
    global_optimizer_kwargs: dict[str, Any],
    cluster_adsorbate_config: object,
    *,
    n_slab_deposit: int | None = None,
    verbosity: int = 1,
) -> list[tuple[float, Atoms]]:
    """Run the final structural gate over dedup'd candidates.

    ``n_slab`` is applied uniformly when given; for non-surface systems it is
    ``None`` and resolved per-candidate from the ``n_slab_atoms`` tag.
    ``n_slab_deposit`` (slab-search types) marks the frozen bottom-layer prefix
    so the layer-stacking cutoff treats mobile top layers as search core —
    matching the GA storage gate. Default verbosity logs one drop summary;
    per-candidate detail is DEBUG.
    """
    valid: list[tuple[float, Atoms]] = []
    dropped: list[tuple[float, str]] = []
    for energy, atoms in candidates:
        resolved_n_slab = (
            n_slab if n_slab is not None else (get_tag(atoms, "n_slab_atoms") or 0)
        )
        try:
            validate_minimum_structure(
                atoms,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=resolved_n_slab,
                adsorbate_definition=global_optimizer_kwargs.get(
                    "adsorbate_definition"
                ),
                connectivity_factor=global_optimizer_kwargs.get("connectivity_factor"),
                cluster_adsorbate_config=cluster_adsorbate_config,
                allow_cluster_fragmentation=global_optimizer_kwargs.get(
                    "allow_cluster_fragmentation", False
                ),
                allow_adsorbate_surface_detachment=global_optimizer_kwargs.get(
                    "allow_adsorbate_surface_detachment", False
                ),
                enforce_adsorbate_subgraph_integrity=global_optimizer_kwargs.get(
                    "enforce_adsorbate_subgraph_integrity", True
                ),
                n_slab_deposit=n_slab_deposit,
            )
        except SCGOValidationError as exc:
            dropped.append((energy, str(exc)))
            continue
        valid.append((energy, atoms))
    if dropped:
        log_info_v(
            logger,
            "Final structural gate: kept %d/%d candidates (%s)",
            len(valid),
            len(candidates),
            format_count_summary(
                Counter(_compact_structural_gate_reason(msg) for _, msg in dropped)
            ),
        )
        for energy, msg in dropped:
            log_debug_v(
                logger,
                "Dropping dedup'd candidate (E=%.4f eV) failing final structural gate: %s",
                energy,
                msg,
            )
    return valid


def scgo(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    timing_output_dir: str | None = None,
    timing_collector: list[dict[str, Any]] | None = None,
) -> list[tuple[float, Atoms]]:
    """Run global optimization for a fixed composition into one run directory.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name ("simple", "bh", or "ga").
        global_optimizer_kwargs: Optimizer parameters.
        output_dir: Run output directory (typically ``run_*/``).
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        run_id: Optional run ID.
        clean: Start fresh if True.

    Returns:
        List of (energy, Atoms) for minima.

    Raises:
        SCGOValidationError: For invalid parameters (unknown ``global_optimizer``,
            missing ``system_type``, missing calculator, or unusable
            ``surface_config`` / ``adsorbate_definition``).
    """
    _validate_common_run_inputs(
        composition=composition,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=output_dir,
        rng=rng,
        verbosity=verbosity,
    )

    calculator_for_global_optimization = _require_calculator(
        calculator_for_global_optimization
    )

    # Ensure file-based calculators run in the trial directory to avoid collisions
    if hasattr(calculator_for_global_optimization, "directory"):
        calculator_for_global_optimization.directory = output_dir

    # Filter keys handled at scgo/run_trials level so **optimizer_kwargs cannot
    # override explicit run_id/clean.
    optimizer_name_lower = global_optimizer.lower()
    if optimizer_name_lower not in _ALGORITHM_REGISTRY:
        raise SCGOValidationError(
            f"Unknown global_optimizer: {global_optimizer}. "
            f"Must be one of {list(_ALGORITHM_REGISTRY.keys())}"
        )
    optimizer_kwargs = {
        k: v
        for k, v in global_optimizer_kwargs.items()
        if k not in {"run_id", "clean", "timing_output_dir", "timing_collector"}
    }
    timing_kwargs: dict[str, Any] = {}
    if timing_output_dir is not None:
        timing_kwargs["timing_output_dir"] = timing_output_dir
    if timing_collector is not None:
        timing_kwargs["timing_collector"] = timing_collector
    if "fitness_strategy" in optimizer_kwargs:
        optimizer_kwargs["fitness_strategy"] = resolve_fitness_strategy(
            optimizer_kwargs["fitness_strategy"], allow_none=False
        )
    system_type = optimizer_kwargs.get("system_type")
    if not isinstance(system_type, str):
        raise SCGOValidationError(
            "system_type must be set in global_optimizer_kwargs "
            "(e.g. 'gas_cluster', 'surface_cluster')."
        )
    policy = get_system_policy(system_type)
    surface_cfg = optimizer_kwargs.get("surface_config")
    validate_system_type_settings(
        system_type=system_type,
        surface_config=surface_cfg
        if isinstance(surface_cfg, SurfaceSystemConfig)
        else None,
    )
    ads_def = as_adsorbate_definition(optimizer_kwargs.get("adsorbate_definition"))
    if ads_def is not None:
        optimizer_kwargs["adsorbate_definition"] = ads_def
    if ads_def is not None and policy.has_adsorbate:
        composition, ads_def = resolve_mobile_composition(
            composition, ads_def, context="scgo"
        )
        optimizer_kwargs["adsorbate_definition"] = ads_def
    validate_adsorbate_definition(
        system_type=system_type,
        composition=composition,
        adsorbate_definition=ads_def,
        context="scgo",
    )
    if (
        policy.has_adsorbate
        and not policy.uses_surface
        and ads_def is not None
        and ads_def.n_core == 0
    ):
        logger.info(
            "Gas adsorbate run with empty core_symbols: skipping global optimization"
        )
        return []

    ensure_directory_exists(output_dir)

    algo_function = _ALGORITHM_REGISTRY[optimizer_name_lower]

    if optimizer_name_lower == "ga":
        all_minima = ga_go(
            composition=composition,
            output_dir=output_dir,
            calculator=calculator_for_global_optimization,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **{**optimizer_kwargs, **timing_kwargs},
        )
    else:
        # Non-GA algorithms need explicit starting atoms.
        if policy.uses_surface:
            surface_config = optimizer_kwargs.get("surface_config")
            if not isinstance(surface_config, SurfaceSystemConfig):
                raise SCGOValidationError(
                    f"system_type={system_type!r} requires surface_config for "
                    f"{optimizer_name_lower.upper()} initialization."
                )
            atoms = _create_surface_initialized_atoms(
                composition=composition,
                surface_config=surface_config,
                rng=rng,
                adsorbate_definition=optimizer_kwargs.get("adsorbate_definition"),
                adsorbate_fragment_template=optimizer_kwargs.get(
                    "adsorbate_fragment_template"
                ),
                cluster_adsorbate_config=optimizer_kwargs.get(
                    "cluster_adsorbate_config"
                ),
                system_type=system_type,
            )
            if policy.slab_is_search_target:
                prepared, _part = prepare_slab_search_surface_config(surface_config)
                optimizer_kwargs["surface_config"] = prepared
                optimizer_kwargs.setdefault("n_slab", len(prepared.slab))
            else:
                optimizer_kwargs.setdefault("n_slab", len(surface_config.slab))
        elif policy.has_adsorbate:
            ads_def = as_adsorbate_definition(
                optimizer_kwargs.get("adsorbate_definition")
            )
            if ads_def is None:
                raise SCGOValidationError(
                    f"system_type={system_type!r} requires adsorbate_definition in "
                    f"global_optimizer_kwargs for {optimizer_name_lower.upper()}."
                )
            optimizer_kwargs["adsorbate_definition"] = ads_def
            if optimizer_kwargs.get("adsorbate_fragment_template") is None:
                raise SCGOValidationError(
                    f"system_type={system_type!r} requires adsorbate_fragment_template "
                    "for hierarchical adsorbate initialization."
                )
            vac = float(optimizer_kwargs.get("vacuum", 10.0))
            mode = str(optimizer_kwargs.get("init_mode", "smart"))
            max_h = int(optimizer_kwargs.get("max_hierarchical_attempts", 200))
            glb = str(optimizer_kwargs.get("previous_search_glob", "**/*.db"))
            atoms = _create_gas_cluster_adsorbate_initial_atoms(
                composition=composition,
                rng=rng,
                adsorbate_definition=ads_def,
                adsorbate_fragment_template=optimizer_kwargs.get(
                    "adsorbate_fragment_template"
                ),
                cluster_adsorbate_config=optimizer_kwargs.get(
                    "cluster_adsorbate_config"
                ),
                vacuum=vac,
                init_mode=mode,
                max_hierarchical_attempts=max_h,
                previous_search_glob=glb,
                verbosity=verbosity,
            )
        else:
            atoms = create_initial_cluster(composition, rng=rng, verbosity=verbosity)
        atoms.calc = calculator_for_global_optimization
        algo_kwargs = _optimizer_kwargs_for_algorithm_call(
            optimizer_kwargs,
            global_optimizer=optimizer_name_lower,
        )
        all_minima = algo_function(
            atoms=atoms,
            output_dir=output_dir,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **algo_kwargs,
            **timing_kwargs,
        )

    if not all_minima:
        logger.info("Global optimization finished but found no valid minima")
        return []

    for _, atoms_obj in all_minima:
        set_tags(atoms_obj, run_id=run_id)

    return all_minima


def run_trials(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    validate_with_hessian: bool = True,
    fmax_threshold: float = DEFAULT_FMAX_THRESHOLD,
    check_hessian: bool = True,
    imag_freq_threshold: float = 50.0,
    validation_n_jobs: int | None = None,
    tag_final_minima: bool = True,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    allow_metadata_mismatch: bool = False,
    search_mobile_count: int | None = None,
) -> list[tuple[float, Atoms]]:
    """Run global optimization once, filter and validate results across runs.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name (e.g., "bh", "ga").
        global_optimizer_kwargs: Optimizer parameters.
        output_dir: Searches directory (parent of ``run_*/`` dirs).
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        validate_with_hessian: Whether to validate with Hessian.
        check_hessian: Whether to compute the Hessian during validation.
        imag_freq_threshold: Imaginary-frequency cutoff for validation (cm^-1).
        validation_n_jobs: Parallel workers for Hessian/force validation; ``None``
            inherits the top-level ``params["n_jobs"]`` (and defaults to
            ``DEFAULT_N_JOBS`` when that is also unset).
        verbosity: Verbosity level.
        run_id: Optional run ID.
        clean: Start fresh if True.
        search_mobile_count: Optional trailing-mobile atom count used as
            ``n_top`` for uniqueness filtering. When omitted, defaults to
            ``len(composition)``. Slab-target runs pass the true mobile count
            here so dedupe does not collapse distinct top-layer geometries.

    Returns:
        List of (energy, Atoms) for unique minima.
    """
    _validate_common_run_inputs(
        composition=composition,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=output_dir,
        rng=rng,
        verbosity=verbosity,
        require_system_type=True,
    )

    if not isinstance(validate_with_hessian, bool):
        raise SCGOValidationError("validate_with_hessian must be a boolean")

    calculator_for_global_optimization = _require_calculator(
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

    # Cache cluster formula (used multiple times) and path key for filenames
    composition_str = get_cluster_formula(composition)
    system_type_for_path = global_optimizer_kwargs.get("system_type")
    path_key = resolve_run_path_key(
        composition,
        system_type=system_type_for_path,
        params=global_optimizer_kwargs,
    )
    # Slab-target runs have an empty composition (and thus empty chemical
    # formula); fall back to the directory identity so ``formula`` is never empty.
    metadata_formula = composition_str or path_key

    # Save run metadata (include formula and run parameters for traceability)
    gok_for_metadata = _sanitize_global_optimizer_kwargs_for_metadata(
        global_optimizer_kwargs
    )
    params = {
        "global_optimizer": global_optimizer,
        "global_optimizer_kwargs": gok_for_metadata,
        "validate_with_hessian": validate_with_hessian,
        "verbosity": verbosity,
        "clean": clean,
        "calculator": calculator_for_global_optimization.__class__.__name__
        if calculator_for_global_optimization
        else None,
    }
    save_run_dir_record(
        run_output_dir,
        run_id,
        record={
            "path_key": path_key,
            "composition": composition,
            "formula": metadata_formula,
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
                    "Loaded %s minima from previous runs (excluding current run %s)",
                    len(previous_minima),
                    run_id,
                )

    all_raw_minima = []
    write_timing = _write_timing_json_enabled(global_optimizer_kwargs)

    run_rng = create_child_rng(rng)
    logger.info("Running global optimization for run %s", run_id)

    all_raw_minima = scgo(
        composition=composition,
        global_optimizer=global_optimizer,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=run_output_dir,
        rng=run_rng,
        calculator_for_global_optimization=calculator_for_global_optimization,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        timing_output_dir=run_output_dir if write_timing else None,
    )

    # Combine all results (previous + current) before deduplication
    if previous_minima:
        all_minima_for_filtering = previous_minima + all_raw_minima
        logger.info(
            "Combined %s previous + %s current minima",
            len(previous_minima),
            len(all_raw_minima),
        )
    else:
        all_minima_for_filtering = all_raw_minima

    if not all_minima_for_filtering:
        logger.info("No minima found")
        _write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str=composition_str,
            run_id=run_id,
            params=params,
        )
        return []

    logger.info(
        "Run complete. Found %s raw minima from current run",
        len(all_raw_minima),
    )
    logger.info("Filtering for unique structures across all runs")
    surface_cfg = global_optimizer_kwargs.get("surface_config")
    system_type_for_mic = global_optimizer_kwargs.get("system_type")
    if not isinstance(system_type_for_mic, str):
        raise SCGOValidationError(
            "system_type must be set in global_optimizer_kwargs for minima dedupe."
        )
    dedupe_mic = resolve_structure_mic(system_type_for_mic, surface_cfg)
    uniqueness = uniqueness_settings_from_mapping(global_optimizer_kwargs)
    energy_tol = global_optimizer_kwargs.get("energy_tolerance")
    if energy_tol is None:
        energy_tol = DEFAULT_ENERGY_TOLERANCE
    comparator_n_top = global_optimizer_kwargs.get("comparator_n_top")
    if comparator_n_top is not None:
        if search_mobile_count is not None and int(comparator_n_top) != int(
            search_mobile_count
        ):
            logger.info(
                "comparator_n_top=%d overrides search_mobile_count=%d as the "
                "dedupe n_top window",
                int(comparator_n_top),
                int(search_mobile_count),
            )
        dedupe_n_top = int(comparator_n_top)
    elif search_mobile_count is not None:
        dedupe_n_top = int(search_mobile_count)
    else:
        dedupe_n_top = len(composition)

    # Block-aware geometry mirrors the GA/BH in-search comparators; an explicit
    # comparator_n_top forces the legacy trailing-window comparison instead.
    # Adsorbate types without any split info stay on the legacy rule too.
    dedupe_blocks: ComparatorBlocks | None = None
    dedupe_weights: dict[str, float] | None = None
    dedupe_cross_weight = 1.0
    dedupe_settings = uniqueness
    ads_def_for_dedupe = global_optimizer_kwargs.get("adsorbate_definition")
    ads_info_available = (
        not get_system_policy(system_type_for_mic).has_adsorbate
        or ads_def_for_dedupe is not None
    )
    if comparator_n_top is None and ads_info_available:
        resolved_geo = resolve_uniqueness_geometry(
            system_type=system_type_for_mic,
            n_atoms=len(all_minima_for_filtering[0][1]),
            surface_config=surface_cfg,
            adsorbate_definition=ads_def_for_dedupe,
            settings=uniqueness,
        )
        dedupe_blocks = resolved_geo.blocks
        dedupe_settings = resolved_geo.settings
        if dedupe_blocks is not None:
            dedupe_weights = resolved_geo.component_weights
            dedupe_cross_weight = resolved_geo.cross_weight

    unique_candidates = filter_unique_minima(
        all_minima_for_filtering,
        float(energy_tol),
        n_top=dedupe_n_top,
        mic=dedupe_mic,
        comparator_tol=dedupe_settings.comparator_tol,
        comparator_pair_cor_max=dedupe_settings.comparator_pair_cor_max,
        blocks=dedupe_blocks,
        component_weights=dedupe_weights,
        cross_weight=dedupe_cross_weight,
    )
    logger.info("Found %s unique candidates", len(unique_candidates))

    # Final structural gate on the dedup'd candidates so *every* final minimum
    # passes connectivity / connected-components checks regardless of which
    # algorithm produced it (simple/BH/GA already gate internally; this is a
    # defense-in-depth backstop before the physical hessian/vibration gate).
    gate_system_type = str(system_type_for_mic)
    gate_policy = get_system_policy(gate_system_type)
    gate_surface_config_raw = global_optimizer_kwargs.get("surface_config")
    gate_cluster_adsorbate_config_raw = global_optimizer_kwargs.get(
        "cluster_adsorbate_config"
    )
    if gate_policy.uses_surface:
        sc = gate_surface_config_raw
        if sc is None:
            logger.warning(
                "Surface system %s missing surface_config at final gate; "
                "skipping structural validation",
                gate_system_type,
            )
            structurally_valid = list(unique_candidates)
        else:
            gate_n_slab_deposit: int | None = None
            if gate_policy.slab_is_search_target:
                prepared_sc, gate_part = prepare_slab_search_surface_config(sc)
                gate_surface_config = prepared_sc
                # Match GA storage: the frozen bottom prefix defines the deposit
                # boundary so mobile top layers count as search core.
                gate_n_slab_deposit = int(gate_part.n_fixed)
            else:
                gate_surface_config = sc
            gate_n_slab = len(gate_surface_config.slab)
            structurally_valid = _gate_structurally_valid_candidates(
                unique_candidates,
                gate_system_type,
                gate_surface_config,
                gate_n_slab,
                global_optimizer_kwargs,
                gate_cluster_adsorbate_config_raw,
                n_slab_deposit=gate_n_slab_deposit,
                verbosity=verbosity,
            )
    else:
        structurally_valid = _gate_structurally_valid_candidates(
            unique_candidates,
            gate_system_type,
            gate_surface_config_raw,
            None,
            global_optimizer_kwargs,
            gate_cluster_adsorbate_config_raw,
            verbosity=verbosity,
        )
    unique_candidates = structurally_valid
    if not unique_candidates:
        logger.info(
            "All dedup'd candidates rejected by the final structural gate; "
            "no minima to validate."
        )
        _write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str=composition_str,
            run_id=run_id,
            params=params,
        )
        return []

    if validate_with_hessian:
        logger.info(
            "Validating %s unique candidates to confirm they are true minima...",
            len(unique_candidates),
        )

        # Ensure validation runs in a separate directory to avoid overwriting run files
        if hasattr(calculator_for_global_optimization, "directory"):
            val_dir = os.path.join(output_dir, "validation")
            ensure_directory_exists(val_dir)
            calculator_for_global_optimization.directory = val_dir

        validated_minima = []
        payloads = [
            (energy, atoms, fmax_threshold, check_hessian, imag_freq_threshold)
            for energy, atoms in unique_candidates
        ]
        n_validate_workers = (
            resolve_n_jobs_for_tasks(validation_n_jobs, len(payloads))
            if check_hessian
            else 1
        )

        parallel_ok = False
        if (
            n_validate_workers > 1
            and len(payloads) >= _MIN_PARALLEL_VALIDATION_CANDIDATES
        ):
            parallel_ok, validated_minima = _validate_candidates_parallel(
                calculator_for_global_optimization,
                payloads,
                n_validate_workers,
            )

        if not parallel_ok:
            for i, (energy, atoms) in enumerate(unique_candidates):
                logger.info(
                    "Validating candidate %s/%s (E=%.4f eV)...",
                    i + 1,
                    len(unique_candidates),
                    energy,
                )
                try:
                    is_valid = is_true_minimum(
                        atoms=atoms,
                        calculator=calculator_for_global_optimization,
                        fmax_threshold=fmax_threshold,
                        check_hessian=check_hessian,
                        imag_freq_threshold=imag_freq_threshold,
                    )
                    if is_valid:
                        validated_minima.append((energy, atoms))
                    else:
                        logger.info("Candidate %s rejected", i + 1)
                except (SCGOValidationError, OSError, RuntimeError, ValueError) as e:
                    logger.warning(
                        "Validation failed for candidate %s (E=%.4f eV): %s",
                        i + 1,
                        energy,
                        e,
                        exc_info=(verbosity >= 2),
                    )

        if not validated_minima:
            logger.info(
                "Validation finished. No candidates were confirmed as true minima"
            )
            _write_results_summary(
                output_dir=output_dir,
                final_minima=[],
                composition_str=composition_str,
                run_id=run_id,
                params=params,
            )
            return []

        final_minima = validated_minima
    else:
        final_minima = unique_candidates

    best_energy, _ = final_minima[0]
    logger.info("Process complete. Found %s final unique minima", len(final_minima))
    logger.info("Best potential energy: %.4f eV", best_energy)

    final_xyz_dir = os.path.join(output_dir, "final_unique_minima")
    logger.info(
        'Writing %s final structures to "%s"',
        len(final_minima),
        os.path.basename(final_xyz_dir),
    )

    # Write results summary file (composition_str already cached above)
    _write_results_summary(
        output_dir=output_dir,
        final_minima=final_minima,
        composition_str=composition_str,
        run_id=run_id,
        params=params,
    )

    align_kwargs_source = dict(global_optimizer_kwargs)
    if not isinstance(align_kwargs_source.get("system_type"), str):
        raise SCGOValidationError(
            "system_type must be set in global_optimizer_kwargs for result alignment."
        )
    surface_align_kwargs = _resolve_surface_alignment_kwargs(align_kwargs_source)
    reference_atoms: Atoms | None = None
    reference_n_slab = 0
    reference_primary_cell_shift: np.ndarray | None = None
    if surface_align_kwargs and final_minima:
        _best_energy, best_atoms = final_minima[0]
        is_slab_ref, reference_n_slab = _is_slab_surface_minimum(best_atoms)
        if is_slab_ref:
            reference_atoms = best_atoms.copy()
            reference_atoms.calc = None
            reference_primary_cell_shift = adsorbate_primary_cell_shift(
                reference_atoms, n_slab=reference_n_slab
            )

    final_minima_info: list[dict] = []
    written_xyz: set[Path] = set()
    for i, (_energy, atoms) in enumerate(final_minima):
        provenance = get_tags(atoms)
        atoms_run_id = provenance.get("run_id", run_id)

        filename = f"{path_key}_minimum_{i + 1:02d}_{atoms_run_id}.xyz"
        filepath = os.path.join(final_xyz_dir, filename)

        # Match DB rows by pre-alignment geometry (same frame as relaxed candidates).
        final_id = ensure_final_id(atoms, _energy)

        atoms_clean = atoms.copy()
        atoms_clean.calc = None
        n_slab_meta = get_tag(atoms_clean, "n_slab_atoms", 0) or 0
        system_type = get_tag(atoms_clean, "system_type")
        try:
            validate_stored_slab_adsorbate_metadata(atoms_clean)
            validate_stored_mobile_partition_metadata(atoms_clean)
        except ValueError as e:
            if allow_metadata_mismatch:
                logger.warning("Structure metadata check before write: %s", e)
            else:
                raise SCGOValidationError(
                    f"Refusing to write minimum with invalid metadata: {e}"
                ) from e
        aligned_to_surface_reference = False
        if reference_atoms is not None and surface_align_kwargs is not None:
            is_slab_candidate, _ = _is_slab_surface_minimum(atoms_clean)
            if is_slab_candidate:
                n_core_mobile = _resolve_n_core_mobile_for_alignment(
                    atoms_clean, align_kwargs_source
                )
                _align_slab_minimum_to_reference(
                    reference_atoms,
                    atoms_clean,
                    n_slab=reference_n_slab,
                    enable_cell_remap=surface_align_kwargs["enable_cell_remap"],
                    enable_lattice_rotation=surface_align_kwargs[
                        "enable_lattice_rotation"
                    ],
                    max_lattice_shift=surface_align_kwargs["max_lattice_shift"],
                    n_core_mobile=n_core_mobile,
                )
                aligned_to_surface_reference = True
                if reference_primary_cell_shift is not None and np.any(
                    reference_primary_cell_shift != 0
                ):
                    apply_primary_cell_shift(atoms_clean, reference_primary_cell_shift)
        if not aligned_to_surface_reference:
            if (
                system_type in {"surface_cluster", "surface_cluster_adsorbate"}
                and n_slab_meta
            ):
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
        written_xyz.add(Path(filepath))

        final_minima_info.append(
            {
                "atoms": atoms,
                "energy": _energy,
                "rank": i + 1,
                "final_written": filepath,
                "final_id": final_id,
            }
        )

    # Drop superseded XYZ files so the folder mirrors the deduplicated final set.
    for stale in Path(final_xyz_dir).glob(f"{path_key}_minimum_*.xyz"):
        if stale not in written_xyz:
            stale.unlink(missing_ok=True)

    # Mark final minima in DB (if enabled) to avoid re-scanning later
    if tag_final_minima:
        try:
            tag_summary = mark_final_minima_in_db(
                final_minima_info, base_dir=output_dir
            )
            rows_updated = int(tag_summary.get("rows_updated", 0))
            n_final = len(final_minima_info)
            log_fn = (
                logger.warning if rows_updated == 0 and n_final > 0 else logger.info
            )
            log_fn(
                "Tagged %d/%d final minima in DB under %s (dbs_touched=%d)",
                rows_updated,
                n_final,
                output_dir,
                int(tag_summary.get("dbs_touched", 0)),
            )
        except (
            sqlite3.DatabaseError,
            OSError,
            SCGODatabaseError,
            SCGOFileError,
        ) as e:
            # Consider DB tagging a systemic failure -- surface it after logging
            logger.warning("Failed to tag final minima in DB: %s", e)
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
    # Count structures by run_id
    run_counts = Counter()
    for _, atoms in final_minima:
        provenance = get_tags(atoms)
        run_id_from_atoms = provenance.get("run_id", run_id)
        run_counts[run_id_from_atoms] += 1

    summary = output_json_provenance()
    timing_relpath = f"{run_id}/timing.json" if run_id else None
    if run_id and not os.path.isfile(os.path.join(output_dir, run_id, "timing.json")):
        timing_relpath = None
    summary.update(
        {
            "composition": composition_str,
            "total_unique_minima": len(final_minima),
            "minima_by_run": dict(run_counts),
            "current_run_id": run_id,
            "params": params,
            "run_metadata_relpath": (f"{run_id}/metadata.json" if run_id else None),
            "run_timing_relpath": timing_relpath,
        }
    )

    summary_file = os.path.join(output_dir, "results_summary.json")
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, cls=RunDirJSONEncoder)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Wrote results summary to %s", summary_file)
    except (OSError, TypeError) as e:
        logger.warning("Failed to write results summary: %s", e)
        raise
