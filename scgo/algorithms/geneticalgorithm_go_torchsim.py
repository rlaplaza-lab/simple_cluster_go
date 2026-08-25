"""TorchSim-enhanced Genetic Algorithm global optimization for clusters.

Genetic Algorithm global optimization with batched relaxations (TorchSim for MLIPs,
ASE sequential batch relaxer for classical calculators). Database interaction
remains single-threaded to protect against SQLite locking issues.
"""

from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase_ga.data import DataConnection
from ase_ga.utilities import get_all_atom_types
from scipy.spatial.distance import cdist
from tqdm import tqdm

from scgo.algorithms.ga_common import (
    ClusterStartGenerator,
    EnergyAndStructureComparator,
    SurfaceClusterStartGenerator,
    SurfaceSlabStartGenerator,
    create_ga_pairing,
    create_mutation_operators,
    extract_constraint_index_lists,
    ga_run_metadata_extras,
    log_early_stopping_info,
    maybe_apply_mobile_core_ads_tags,
    reseed_mutation_operator_rngs,
    select_population_class,
    setup_diversity_scorer,
    sort_minima_by_fitness,
    update_early_stopping_state_unified,
    update_mutation_weights,
    validate_ga_common_params,
    validate_structure_for_ga_storage,
)
from scgo.algorithms.run_context import validate_and_resolve_run_context
from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
    _assert_offspring_integrity,
)
from scgo.ase_ga_patches.population import Population
from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer
from scgo.calculators.torchsim_helpers import (
    TorchSimBatchRelaxer,
    build_torchsim_relaxer,
)
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import prepare_atoms_for_local_relax
from scgo.cluster_adsorbate.rigid import enforce_frozen_adsorbate_geometry
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_FMAX_THRESHOLD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database import (
    RetryConfig,
    close_data_connection,
    database_retry,
    setup_database,
)
from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.initialization import compute_cell_side
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.metadata.atoms import filter_by_tags, get_tag, get_tags, set_tags
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    SystemType,
    get_system_policy,
    resolve_search_mobile_composition,
    resolve_structure_mic,
    uses_surface,
    validate_minimum_structure,
)
from scgo.system_types.dedup_geometry import resolve_uniqueness_geometry
from scgo.utils.comparators import (
    ComparatorBlocks,
    UniquenessSettings,
    create_geometry_comparator,
)
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
)
from scgo.utils.helpers import (
    extract_minima_from_database,
)
from scgo.utils.logging import (
    drain_inductor_filelock_summary,
    get_logger,
    log_debug_v,
    log_info_v,
    should_show_progress,
)
from scgo.utils.mutation_weights import get_adaptive_mutation_config
from scgo.utils.parallel_workers import (
    resolve_n_jobs,
    resolve_n_jobs_for_tasks,
)
from scgo.utils.phase_logging import (
    compact_ga_ineligible_reason,
    format_count_summary,
    log_generation_offspring_summaries,
    log_phase_subheader,
)
from scgo.utils.rng_helpers import (
    create_child_rng,
    ensure_rng_or_create,
    offspring_rng_triple,
)
from scgo.utils.timing_report import (
    build_timing_payload,
    cpu_non_relax_seconds_from_timings,
    emit_timing_data,
    ga_relax_seconds_from_timings,
    log_timing_summary,
)
from scgo.utils.torchsim_policy import (
    is_ml_calculator,
)
from scgo.utils.validation import validate_composition

logger = get_logger(__name__)


_PREFILTER_BLMIN_FACTOR = 0.55

# Cache by (unique Z, id(blmin)). Cleared each GA generation so recycled ids and
# per-generation empty ``{}`` (prefilter off) cannot accumulate stale entries.
_BLMIN_THRESH_CACHE: dict[
    tuple[tuple[int, ...], int], tuple[np.ndarray, dict[int, int]]
] = {}


def _blmin_threshold_matrix(
    atomic_numbers: np.ndarray, blmin: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Map atomic numbers to a dense Z-pair clash-threshold matrix."""
    unique_z = tuple(sorted(int(z) for z in np.unique(atomic_numbers)))
    cache_key = (unique_z, id(blmin))
    cached = _BLMIN_THRESH_CACHE.get(cache_key)
    if cached is None:
        z_to_i = {z: i for i, z in enumerate(unique_z)}
        n_u = len(unique_z)
        min_allowed = np.zeros((n_u, n_u), dtype=float)
        for i, zi in enumerate(unique_z):
            for j, zj in enumerate(unique_z):
                min_allowed[i, j] = float(blmin.get((zi, zj), blmin.get((zj, zi), 0.0)))
        mask = min_allowed > 0.0
        thresh = np.zeros((n_u, n_u), dtype=float)
        thresh[mask] = _PREFILTER_BLMIN_FACTOR * min_allowed[mask]
        _BLMIN_THRESH_CACHE[cache_key] = (thresh, z_to_i)
    else:
        thresh, z_to_i = cached

    index = np.array([z_to_i[int(z)] for z in atomic_numbers], dtype=int)
    return thresh, index


# Cache of upper-triangle index pairs for the mobile–mobile clash prefilter,
# avoiding a fresh O(n²) boolean mask allocation on every offspring.
_TRIU_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _triu_cache(n: int) -> tuple[np.ndarray, np.ndarray]:
    return _TRIU_CACHE.setdefault(n, np.triu_indices(n, k=1))


def _fails_fast_geometric_prefilter(
    atoms: Atoms, blmin: dict, *, n_slab: int = 0
) -> bool:
    """Return True when a severe clash is detected quickly.

    Only mobile atoms (indices ``n_slab:``) participate: mobile–mobile and
    mobile–slab pairs are checked; slab–slab pairs are skipped.
    """
    n_atoms = len(atoms)
    if n_atoms < 2:
        return False
    n_slab_i = max(0, min(int(n_slab), n_atoms))
    n_mobile = n_atoms - n_slab_i
    if n_mobile < 1:
        return False

    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    thresh, z_index = _blmin_threshold_matrix(numbers, blmin)
    mobile_pos = positions[n_slab_i:]
    mobile_idx = z_index[n_slab_i:]

    # Mobile–mobile pairs (upper triangle, cached index pass).
    if n_mobile >= 2:
        mm = cdist(mobile_pos, mobile_pos)
        pair_thresh = thresh[np.ix_(mobile_idx, mobile_idx)]
        iu, ju = _triu_cache(n_mobile)
        mm_u = mm[iu, ju]
        pt_u = pair_thresh[iu, ju]
        if np.any((pt_u > 0.0) & (mm_u < pt_u)):
            return True

    # Mobile–slab pairs.
    if n_slab_i > 0:
        slab_pos = positions[:n_slab_i]
        slab_idx = z_index[:n_slab_i]
        ms = cdist(mobile_pos, slab_pos)
        pair_thresh = thresh[np.ix_(mobile_idx, slab_idx)]
        if np.any((pair_thresh > 0.0) & (ms < pair_thresh)):
            return True

    return False


def _picklable_atoms_copy(atoms: Atoms | None) -> Atoms | None:
    """Return an Atoms copy safe for process-pool pickling (no calculator)."""
    if atoms is None:
        return None
    copy = atoms.copy()
    copy.calc = None
    return copy


def _mobile_only_copy(atoms: Atoms, n_frozen_prefix: int) -> Atoms:
    """Return only the trailing mobile atoms, dropping the frozen slab prefix.

    The slab is available in the worker via the cached pairing operator so it
    does not need to be transmitted per job.  ``info`` is shallow-copied so
    that ``confid`` (needed by ``get_new_individual``) travels with the frame.
    """
    mobile = atoms[n_frozen_prefix:].copy()
    mobile.calc = None
    mobile.info = dict(atoms.info)
    return mobile


def _reconstruct_full_frame(mobile: Atoms, slab: Atoms) -> Atoms:
    """Reconstruct the full slab+mobile frame from a mobile-only job payload.

    The slab is taken from the cached pairing operator; mobile carries ``info``
    (including ``confid``) copied at job-build time.
    """
    full = slab + mobile
    full.info = dict(mobile.info)
    return full


def _picklable_fragment_templates(
    templates: AdsorbateFragmentInput | None,
) -> list[Atoms] | None:
    if templates is None:
        return None
    if isinstance(templates, Atoms):
        copied = _picklable_atoms_copy(templates)
        return [copied] if copied is not None else None
    out: list[Atoms] = []
    for frag in templates:
        copied = _picklable_atoms_copy(frag)
        if copied is not None:
            out.append(copied)
    return out or None


@dataclass(frozen=True)
class OffspringBuildContext:
    """Picklable snapshot of per-generation offspring build inputs."""

    atoms_template: Atoms
    n_to_optimize: int
    composition: list[str]
    blmin: dict
    system_type: SystemType
    n_slab: int
    n_frozen_prefix: int
    slab_for_pairing: Atoms | None
    surface_normal_axis: int
    adsorbate_definition: AdsorbateDefinition | None
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor | None
    allow_cluster_fragmentation: bool
    allow_adsorbate_surface_detachment: bool
    enforce_adsorbate_subgraph_integrity: bool
    freeze_adsorbate_internal_geometry: bool
    adsorbate_fragment_templates: list[Atoms] | None
    surface_config: SurfaceSystemConfig | None
    adaptive_config: dict[str, Any]
    current_mutation_probability: float
    operators_list: list
    name_map: dict[str, int]
    operators_epoch: int
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None


_OFFSPRING_WORKER_STATE: dict[str, Any] = {}


def _reseed_pairing_rng(pairing: Any, rng: np.random.Generator) -> None:
    if isinstance(pairing, DualCutAndSplicePairing):
        pairing.rng = create_child_rng(rng)
        pairing.primary.rng = create_child_rng(rng)
        pairing.exploratory.rng = create_child_rng(rng)
        return
    if isinstance(pairing, CutAndSplicePairing):
        pairing.rng = create_child_rng(rng)


def _load_offspring_worker_state(ctx: OffspringBuildContext) -> None:
    """Build pairing and operators once per worker process / generation."""
    placeholder_rng = np.random.default_rng(0)
    pairing = create_ga_pairing(
        ctx.atoms_template,
        ctx.n_to_optimize,
        placeholder_rng,
        slab_atoms=ctx.slab_for_pairing,
        system_type=ctx.system_type,
        composition=ctx.composition,
        adsorbate_definition=ctx.adsorbate_definition,
    )
    _OFFSPRING_WORKER_STATE["operators_epoch"] = ctx.operators_epoch
    _OFFSPRING_WORKER_STATE["pairing"] = pairing
    _OFFSPRING_WORKER_STATE["operators"] = copy.deepcopy(ctx.operators_list)
    _OFFSPRING_WORKER_STATE["name_map"] = dict(ctx.name_map)
    # The full static context is retained so per-job payloads (which only carry
    # the dynamic adaptive_config / mutation probability / epoch) can resolve
    # the static pairing/operator inputs without being re-pickled each job.
    _OFFSPRING_WORKER_STATE["static_ctx"] = ctx


def _offspring_worker_bootstrap_init(ctx: OffspringBuildContext) -> None:
    _offspring_worker_init()
    _load_offspring_worker_state(ctx)


def _ensure_offspring_worker_state(ctx: OffspringBuildContext) -> None:
    if _OFFSPRING_WORKER_STATE.get("operators_epoch") != ctx.operators_epoch:
        _load_offspring_worker_state(ctx)


def _offspring_worker_init() -> None:
    """Limit BLAS threading in process-pool offspring workers."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _pairing_slab(pairing: Any) -> Atoms | None:
    """Return the slab held by the (possibly dual) pairing operator."""
    if isinstance(pairing, DualCutAndSplicePairing):
        return pairing.primary.slab if len(pairing.primary.slab) > 0 else None
    if isinstance(pairing, CutAndSplicePairing):
        return pairing.slab if len(pairing.slab) > 0 else None
    return None


def _pairing_last_attempt_count(pairing: Any) -> int:
    """Return the inner cut-config attempt count from the last cross() call.

    ``CutAndSplicePairing.last_attempt_count`` tracks how many (outer × cut)
    combinations were tested.  For ``DualCutAndSplicePairing`` the active
    variant's count is returned (max of both, since only one ran).
    """
    if isinstance(pairing, DualCutAndSplicePairing):
        return max(
            getattr(pairing.primary, "last_attempt_count", 0),
            getattr(pairing.exploratory, "last_attempt_count", 0),
        )
    return int(getattr(pairing, "last_attempt_count", 0))


def _build_offspring_worker(
    job: dict[str, Any],
) -> dict[str, Any]:
    """Build one GA offspring (crossover + optional mutation) in an isolated worker.

    The static build context (pairing, operator list, name map) is held in the
    worker process via ``_OFFSPRING_WORKER_STATE`` (loaded once at bootstrap or
    per generation for the in-process path). Only the lightweight per-job payload
    (parent frames, task seed, and the dynamic adaptive config / mutation
    probability / epoch) is pickled per ``submit``, avoiding the per-job pickle
    of the slab ``Atoms`` and the operator list.

    When ``job["mobile_only"]`` is True the parent frames contain only the
    trailing mobile atoms; the slab prefix is reattached here from the cached
    pairing operator before crossover and integrity checks.
    """
    ctx: OffspringBuildContext = _OFFSPRING_WORKER_STATE["static_ctx"]
    pairing_rng, operator_rng, decision_rng = offspring_rng_triple(job["task_seed"])
    setup_t0 = perf_counter()
    # Refresh the cached pairing/operators only when the generation (epoch) advances.
    if _OFFSPRING_WORKER_STATE.get("operators_epoch") != job["operators_epoch"]:
        _load_offspring_worker_state(ctx)
    local_pairing = _OFFSPRING_WORKER_STATE["pairing"]
    _reseed_pairing_rng(local_pairing, pairing_rng)
    local_ops = _OFFSPRING_WORKER_STATE["operators"]
    reseed_mutation_operator_rngs(local_ops, operator_rng)
    local_mutations = update_mutation_weights(
        operators_list=local_ops,
        name_map=_OFFSPRING_WORKER_STATE["name_map"],
        adaptive_config=job["adaptive_config"],
        rng=decision_rng,
    )
    operator_setup_s = perf_counter() - setup_t0

    # Reconstruct full frames for surface runs where only mobile atoms were sent.
    if job.get("mobile_only"):
        _slab = _pairing_slab(local_pairing)
        if _slab is None or len(_slab) == 0:
            raise SCGORuntimeError(
                "mobile_only job received but pairing has no slab - "
                "reconstruction impossible"
            )
        a1_full = _reconstruct_full_frame(job["a1"], _slab)
        a2_full = _reconstruct_full_frame(job["a2"], _slab)
    else:
        a1_full = job["a1"]
        a2_full = job["a2"]

    crossover_t0 = perf_counter()
    child, desc = local_pairing.get_new_individual([a1_full, a2_full])
    crossover_s = perf_counter() - crossover_t0
    pairing_attempt_count = _pairing_last_attempt_count(local_pairing)
    mutation_s = 0.0
    mutation_applied = False
    if child is None:
        return {
            "index": job["index"],
            "child": None,
            "desc": None,
            "failure_reason": "pairing_failed",
            "mutation_applied": False,
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
            "pairing_attempt_count": pairing_attempt_count,
        }
    if _fails_fast_geometric_prefilter(child, ctx.blmin, n_slab=ctx.n_frozen_prefix):
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "too_close_prefilter",
            "mutation_applied": False,
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
            "pairing_attempt_count": pairing_attempt_count,
        }
    if decision_rng.random() < job["current_mutation_probability"]:
        mutation_t0 = perf_counter()
        mutated = local_mutations.get_operator().mutate(child)
        mutation_s = perf_counter() - mutation_t0
        if mutated is not None:
            child = mutated
            mutation_applied = True
    if ctx.freeze_adsorbate_internal_geometry:
        enforce_frozen_adsorbate_geometry(
            child,
            n_slab=ctx.n_slab,
            adsorbate_definition=ctx.adsorbate_definition,
            fragment_templates=ctx.adsorbate_fragment_templates,
        )
    maybe_apply_mobile_core_ads_tags(
        child,
        ctx.n_slab,
        ctx.composition,
        ctx.adsorbate_definition,
        ctx.system_type,
    )
    # Post-operator atom-count + stoichiometry guard (covers crossover + mutation).
    # Use the reconstructed full frame so the length check is against the full
    # slab+mobile count, not the mobile-only payload length.
    _assert_offspring_integrity(child, a1_full)
    try:
        # Pre-relax geometric screen (raw frame); eligibility is decided post-relax
        # via validate_structure_for_ga_storage after canonicalization.
        validate_minimum_structure(
            child,
            system_type=ctx.system_type,
            surface_config=ctx.surface_config,
            n_slab=ctx.n_slab,
            adsorbate_definition=ctx.adsorbate_definition,
            connectivity_factor=ctx.connectivity_factor,
            cluster_adsorbate_config=ctx.cluster_adsorbate_config,
            allow_cluster_fragmentation=ctx.allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=ctx.allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=ctx.enforce_adsorbate_subgraph_integrity,
        )
    except (ValueError, SCGOValidationError) as exc:
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "validation_failed",
            "validation_error": str(exc),
            "mutation_applied": mutation_applied,
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
            "pairing_attempt_count": pairing_attempt_count,
        }
    return {
        "index": job["index"],
        "child": child,
        "desc": desc,
        "failure_reason": None,
        "mutation_applied": mutation_applied,
        "operator_setup_s": operator_setup_s,
        "crossover_s": crossover_s,
        "mutation_s": mutation_s,
        "pairing_attempt_count": pairing_attempt_count,
    }


def _torchsim_prepare_relaxed_copy(
    cand: Atoms,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
    *,
    surface_mode: bool,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None,
) -> Atoms:
    """Copy a candidate and attach slab / adsorbate constraints before relaxation."""
    return prepare_atoms_for_local_relax(
        cand,
        surface_mode=surface_mode,
        surface_config=surface_config,
        n_slab=n_slab,
        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_templates=adsorbate_fragment_templates,
    )


def _record_relax_batch_steps(
    relaxer: TorchSimBatchRelaxer,
    profiling: dict[str, float] | None,
    counters: dict[str, int] | None,
    n_structures: int,
) -> None:
    steps_list = getattr(relaxer, "last_batch_relax_steps", None) or []
    if not steps_list or profiling is None:
        return
    step_val = steps_list[0]
    profiling["relax_steps_sum"] = profiling.get("relax_steps_sum", 0.0) + float(
        step_val * n_structures
    )
    profiling["relax_steps_max"] = max(
        float(profiling.get("relax_steps_max", 0.0)), float(step_val)
    )
    if counters is not None:
        counters["relax_batches"] = counters.get("relax_batches", 0) + 1
        counters["relax_structures"] = counters.get("relax_structures", 0) + int(
            n_structures
        )


@dataclass(frozen=True)
class GAWriteContext:
    """Tunneled context for :func:`_write_relaxed_candidate`."""

    n_slab: int
    n_frozen_prefix: int
    composition: list[str] | None
    adsorbate_definition: AdsorbateDefinition | None
    system_type: SystemType
    surface_mode: bool
    surface_config: SurfaceSystemConfig | None
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor | None
    allow_cluster_fragmentation: bool
    allow_adsorbate_surface_detachment: bool
    enforce_adsorbate_subgraph_integrity: bool
    freeze_adsorbate_internal_geometry: bool = False
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None


def _write_relaxed_candidate(
    da: DataConnection,
    original: Atoms,
    relaxed: Atoms,
    energy: float,
    ctx: GAWriteContext,
    *,
    generation: int | None = None,
    run_id: str | None = None,
) -> str | None:
    """Write a single relaxed candidate to the database.

    Returns the validation error string when the structure fails GA storage
    validation, or ``None`` when it is eligible for GA evolution.
    """
    original.set_cell(relaxed.get_cell(), scale_atoms=False)
    original.set_pbc(relaxed.get_pbc())
    original.set_positions(relaxed.get_positions())

    if ctx.composition is not None:
        maybe_apply_mobile_core_ads_tags(
            original,
            ctx.n_slab,
            ctx.composition,
            ctx.adsorbate_definition,
            ctx.system_type,
        )
    validation_error = validate_structure_for_ga_storage(
        original,
        surface_mode=ctx.surface_mode,
        n_slab=ctx.n_slab,
        n_slab_deposit=(
            ctx.n_frozen_prefix
            if get_system_policy(ctx.system_type).slab_is_search_target
            else None
        ),
        system_type=ctx.system_type,
        surface_config=ctx.surface_config,
        adsorbate_definition=ctx.adsorbate_definition,
        connectivity_factor=ctx.connectivity_factor,
        cluster_adsorbate_config=ctx.cluster_adsorbate_config,
        allow_cluster_fragmentation=ctx.allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=ctx.allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=ctx.enforce_adsorbate_subgraph_integrity,
    )

    if "forces" in relaxed.arrays:
        original.arrays["forces"] = relaxed.arrays["forces"].copy()

    set_tags(
        original,
        **(get_tags(relaxed) or {"potential_energy": energy, "raw_score": -energy}),
    )
    set_tags(
        original,
        ga_eligible=(validation_error is None),
    )
    if validation_error is not None:
        set_tags(
            original,
            ga_ineligible_reason=validation_error,
        )

    comp_meta = list(ctx.composition) if ctx.composition is not None else []
    constraint_lists = extract_constraint_index_lists(relaxed)
    fix_atoms_indices = constraint_lists["fix_atoms_indices"]
    fix_bond_lengths_pairs = constraint_lists["fix_bond_lengths_pairs"]
    if not fix_atoms_indices and not fix_bond_lengths_pairs and ctx.surface_mode:
        try:
            derived = prepare_atoms_for_local_relax(
                original,
                surface_mode=ctx.surface_mode,
                surface_config=ctx.surface_config,
                n_slab=ctx.n_slab,
                freeze_adsorbate_internal_geometry=ctx.freeze_adsorbate_internal_geometry,
                adsorbate_definition=ctx.adsorbate_definition,
                adsorbate_fragment_templates=ctx.adsorbate_fragment_templates,
            )
            derived_lists = extract_constraint_index_lists(derived)
            fix_atoms_indices = derived_lists["fix_atoms_indices"]
            fix_bond_lengths_pairs = derived_lists["fix_bond_lengths_pairs"]
        except (
            SCGOValidationError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            RuntimeError,
        ):
            logger.debug(
                "Could not derive constraint index lists from context; "
                "storing only the constraints found on the relaxed structure"
            )
    extra = ga_run_metadata_extras(
        ctx.surface_config,
        ctx.n_slab,
        ctx.system_type,
        comp_meta,
        adsorbate_definition=ctx.adsorbate_definition,
        fix_atoms_indices=fix_atoms_indices,
        fix_bond_lengths_pairs=fix_bond_lengths_pairs,
    )
    if generation is not None:
        set_tags(
            original,
            generation=generation,
            run_id=run_id,
            **extra,
        )
    elif run_id is not None:
        set_tags(original, run_id=run_id, **extra)

    # Root-cause fix: the unrelaxed ``original`` carried no constraints, so the
    # native DB round-trip otherwise discards the slab FixAtoms / adsorbate
    # FixBondLengths that relaxation enforced. Persisting them here aligns the
    # TorchSim GA path with the basinhopping path (which writes the constrained
    # ``a_trial`` directly). The metadata tags above are a reorder-safe backstop
    # consumed on load.
    if relaxed.constraints:
        original.set_constraint(copy.deepcopy(relaxed.constraints))

    original.calc = SinglePointCalculator(original, energy=energy)
    da.add_relaxed_step(original)
    return validation_error


_META_COLS = {"include_data": False, "columns": ["id", "mtime", "key_value_pairs"]}


def _read_candidate_batch(
    da: DataConnection,
    to_take: int | None,
) -> list[Atoms]:
    """Read a batch of unrelaxed candidates under a single DB connection.

    Uses metadata-only (no blob) indexed ``relaxed`` / ``queued`` filters to
    build the exclusion set, then ``get_atoms`` only for the taken gaids.  This
    avoids deserializing position / force blobs for every row on every call.

    Rows without a ``gaid`` (e.g. the stoichiometry template row written at
    database setup) are skipped. A gaid is excluded when *any* of its rows is
    relaxed or queued, because ``add_relaxed_step`` appends a new ``relaxed=1``
    row and leaves the original ``relaxed=0`` row in place. Up to ``to_take``
    gaids, sorted by gaid, are returned as ``Atoms`` with ``confid`` set.

    Parameters
    ----------
    to_take:
        Maximum number of gaids to return.  Pass ``None`` to return all
        pending gaids (no cap).
    """
    with da.c:
        excluded_gaids: set[int] = set()
        for r in da.c.select(relaxed=1, **_META_COLS):
            gaid = getattr(r, "gaid", None)
            if gaid is not None:
                excluded_gaids.add(int(gaid))
        for r in da.c.select(queued=1, **_META_COLS):
            gaid = getattr(r, "gaid", None)
            if gaid is not None:
                excluded_gaids.add(int(gaid))

        rows_by_gaid: dict[int, list] = {}
        for r in da.c.select(relaxed=0, **_META_COLS):
            gaid = getattr(r, "gaid", None)
            if gaid is None:
                continue
            gaid_i = int(gaid)
            if gaid_i in excluded_gaids:
                continue
            rows_by_gaid.setdefault(gaid_i, []).append(r)

        all_gaids = sorted(rows_by_gaid.keys())
        gaids = all_gaids if to_take is None else all_gaids[:to_take]
        out: list[Atoms] = []
        for gaid in gaids:
            latest = max(rows_by_gaid[gaid], key=lambda row: (row.mtime, row.id))
            atoms = da.get_atoms(latest.id)
            atoms.info["confid"] = gaid
            atoms.info.setdefault("data", {})
            out.append(atoms)
        return out


def _relax_unrelaxed_candidates(
    da: DataConnection,
    relaxer: TorchSimBatchRelaxer,
    *,
    population: Population | None = None,
    max_batch: int | None = None,
    force: bool = False,
    generation: int | None = None,
    run_id: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int = 0,
    n_frozen_prefix: int = 0,
    system_type: SystemType = "gas_cluster",
    profiling: dict[str, float] | None = None,
    counters: dict[str, int] | None = None,
    composition: list[str] | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None,
) -> tuple[int, int, dict[str, int]]:
    """Relax unrelaxed candidates in batches and commit them to the database.

    Returns:
        Tuple of (GA-eligible count, ineligible count, compact ineligible
        reason counts) for this relax call.
    """
    # Avoid a separate get_number_of_unrelaxed_candidates call (three blob selects);
    # _read_candidate_batch does the same three selects metadata-only.
    batch_cap: int | None = None if (force or max_batch is None) else max_batch

    # Batch read candidates under a single database connection
    t0 = perf_counter()
    batch = database_retry(
        lambda: _read_candidate_batch(da, batch_cap),
        config=RetryConfig(max_retries=5),
        operation_name="read_candidate_batch",
    )
    if profiling is not None:
        profiling["db_read_s"] = profiling.get("db_read_s", 0.0) + (perf_counter() - t0)

    if not batch:
        return (0, 0, {})

    t0 = perf_counter()
    surface_mode = uses_surface(system_type)
    relaxed_results = relaxer.relax_batch(
        [
            _torchsim_prepare_relaxed_copy(
                cand,
                surface_config,
                n_slab,
                surface_mode=surface_mode,
                freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_templates=adsorbate_fragment_templates,
            )
            for cand in batch
        ]
    )
    if profiling is not None:
        profiling["relax_batch_s"] = profiling.get("relax_batch_s", 0.0) + (
            perf_counter() - t0
        )
    _record_relax_batch_steps(relaxer, profiling, counters, len(batch))
    if len(relaxed_results) != len(batch):
        raise SCGORuntimeError("TorchSim relaxer returned mismatched batch size")

    # Batch write results under a single database connection.
    # Structures failing validation are persisted but marked ineligible for GA
    # evolution. Reset counters each retry so a mid-batch SQLite rollback does
    # not double-count.
    successful_count = 0
    ineligible_count = 0
    ineligible_reasons: dict[str, int] = {}
    eligible_relaxed: list[Atoms] = []

    def _write_batch_under_connection():
        """Write relaxed results under a single connection."""
        nonlocal \
            ineligible_count, \
            successful_count, \
            ineligible_reasons, \
            eligible_relaxed
        successful_count = 0
        ineligible_count = 0
        ineligible_reasons = {}
        eligible_relaxed = []
        with da.c:
            for idx, (original, (energy, relaxed)) in enumerate(
                zip(batch, relaxed_results, strict=True)
            ):
                validation_error = _write_relaxed_candidate(
                    da,
                    original,
                    relaxed,
                    energy,
                    GAWriteContext(
                        n_slab=n_slab,
                        n_frozen_prefix=n_frozen_prefix,
                        composition=composition,
                        adsorbate_definition=adsorbate_definition,
                        system_type=system_type,
                        surface_mode=surface_mode,
                        surface_config=surface_config,
                        connectivity_factor=connectivity_factor,
                        cluster_adsorbate_config=cluster_adsorbate_config,
                        allow_cluster_fragmentation=allow_cluster_fragmentation,
                        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                        adsorbate_fragment_templates=adsorbate_fragment_templates,
                    ),
                    generation=generation,
                    run_id=run_id,
                )
                if validation_error is not None:
                    ineligible_count += 1
                    reason = compact_ga_ineligible_reason(validation_error)
                    ineligible_reasons[reason] = ineligible_reasons.get(reason, 0) + 1
                    label = (
                        "Offspring" if generation is not None else "Initial candidate"
                    )
                    logger.debug(
                        "%s %d/%d failed validation after relaxation; storing "
                        "but excluding from GA population: %s",
                        label,
                        idx + 1,
                        len(batch),
                        validation_error,
                    )
                else:
                    successful_count += 1
                    eligible_relaxed.append(original)

    t0 = perf_counter()
    database_retry(
        _write_batch_under_connection,
        config=RetryConfig(max_retries=5),
        operation_name="write_relaxed_batch",
    )
    if profiling is not None:
        profiling["db_write_s"] = profiling.get("db_write_s", 0.0) + (
            perf_counter() - t0
        )

    if population is not None:
        t0 = perf_counter()
        population.update(new_cand=eligible_relaxed)
        if profiling is not None:
            profiling["population_update_s"] = profiling.get(
                "population_update_s", 0.0
            ) + (perf_counter() - t0)

    return (successful_count, ineligible_count, ineligible_reasons)


def ga_go(
    composition: list[str],
    output_dir: str,
    rng: np.random.Generator | None,
    calculator: Any,
    *,
    niter: int = 10,
    fmax: float = DEFAULT_FMAX_THRESHOLD,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = FIRE,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    comparator_tol: float = DEFAULT_COMPARATOR_TOL,
    comparator_pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
    comparator_n_top: int | None = None,
    comparator_component_weights: dict[str, float] | None = None,
    comparator_cross_weight: float = 1.0,
    mutation_probability: float = 0.4,
    population_size: int = 10,
    offspring_fraction: float = 0.5,
    n_jobs_population_init: int | None = None,
    n_jobs_offspring: int | None = None,
    vacuum: float = 10.0,
    previous_search_glob: str = "**/*.db",
    use_adaptive_mutations: bool = True,
    stagnation_trigger: int = 4,
    stagnation_full_trigger: int = 8,
    recovery_window: int = 2,
    aggressive_burst_multiplier: float = 1.8,
    max_mutation_probability: float = 0.65,
    early_stopping_niter: int = 10,
    relaxer: TorchSimBatchRelaxer | None = None,
    batch_size: int | None = None,
    torchsim_dtype: str | None = None,
    verbosity: int = 1,
    elite_fraction: float = 0.1,
    run_id: str | None = None,
    clean: bool = False,
    fitness_strategy: str = "low_energy",
    diversity_reference_db: str | None = None,
    diversity_max_references: int = 100,
    diversity_update_interval: int = 5,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType = "gas_cluster",
    write_timing_json: bool = False,
    detailed_timing: bool = False,
    timing_output_dir: str | None = None,
    timing_collector: list[dict[str, Any]] | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    freeze_adsorbate_internal_geometry: bool = False,
    ga_adaptive_retry_enabled: bool = True,
    ga_retry_floor_multiplier: int = 4,
    ga_retry_ceiling_multiplier: int = 15,
    ga_fast_prefilter_enabled: bool = True,
    db_enable_expression_indexes: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run the GA using TorchSim for batched relaxations.

    Genetic algorithm with batched relaxations (TorchSim for MLIPs, ASE batch otherwise).
    The ``relaxer`` argument controls batching; when omitted the function builds a
    :class:`TorchSimBatchRelaxer` for MLIP calculators and an
    :class:`~scgo.calculators.ase_batch_relaxer.AseBatchRelaxer` otherwise, using
    ``fmax`` as the force tolerance and ``niter_local_relaxation`` as the step cap.

    Args:
        composition: List of element symbols defining the cluster composition.
        calculator: ASE calculator for energy/force evaluations.
        previous_search_glob: Glob pattern used to discover previous database
            files for seed-based initialization.
        n_jobs_population_init: Parallel workers for population initialization.
            ``None`` uses the project default (single worker; opt in with ``-1``
            for all CPUs or ``-2`` for all but one). ``>= 1`` sets an explicit
            worker count.
        n_jobs_offspring: Parallel workers for offspring construction, with the
            same semantics as ``n_jobs_population_init``.
        early_stopping_niter: Number of consecutive generations with no improvement
                              before stopping early. Uses fitness for non-low_energy
                              strategies, energy for low_energy. If 0, no early stopping
                              is applied. Default 10.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Defaults to 1.
        elite_fraction: Fraction of population to preserve as elite candidates
                         (top performers by fitness). Default 0.1 (top 10%).
        run_id: Optional run ID for tracking.
        clean: If True, remove an existing GA database and auxiliary files in the
            output directory.
        fitness_strategy: Fitness strategy to use. One of: "low_energy", "high_energy", "diversity".
            Defaults to "low_energy" (minimize energy).
        diversity_reference_db: Glob pattern for reference structure databases (for diversity strategy).
            Required when fitness_strategy="diversity", ignored otherwise.
        diversity_max_references: Maximum number of reference structures to load (for performance).
        diversity_update_interval: Number of generations between reference updates (for diversity strategy).
        surface_config: Optional slab + adsorbate configuration for surface GA runs.
        write_timing_json: If True, write ``timing.json`` (see ``timing_output_dir``).
            Set in ``optimizer_params['ga']`` inside ``go_params``/``params``.
        detailed_timing: If True, include ``per_generation`` rows in ``timing.json``.
            Requires ``write_timing_json=True``.
        timing_output_dir: Directory for ``timing.json`` (defaults to ``output_dir``).
            ``run_trials`` sets this to the run directory alongside ``metadata.json``.
        timing_collector: Optional list appended with the timing payload after the run.
        torchsim_dtype: Optional TorchSim compute dtype, ``"float32"`` or ``"float64"``.
            Defaults to ``None``, which keeps the :class:`TorchSimBatchRelaxer`
            default of ``float64``. Set ``"float32"`` for much faster FP32/TF32 GPU
            kernels at the cost of some numerical accuracy. Only applies when this
            function builds the relaxer; ignored when ``relaxer`` is supplied.
    """
    profile_t0 = perf_counter()
    profile_timings: dict[str, float] = {}
    profile_counters: dict[str, int] = {
        "offspring_created": 0,
        "offspring_relaxed": 0,
        "offspring_worker_failures": 0,
        "offspring_attempts_total": 0,
    }
    profile_retry_failures: dict[str, int] = {}
    per_generation: list[dict[str, Any]] | None = [] if detailed_timing else None

    run_ctx = validate_and_resolve_run_context(
        system_type=system_type,
        surface_config=surface_config,
        connectivity_factor=connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        fitness_strategy=fitness_strategy,
    )
    connectivity_factor = run_ctx.connectivity_factor
    policy = run_ctx.policy
    fitness_strategy = run_ctx.fitness_strategy
    # Bare ``surface`` uses an empty cluster composition; search-mobile symbols
    # come from the top slab layers via ``resolve_search_mobile_composition``.
    validate_composition(
        composition,
        allow_empty=policy.slab_is_search_target and not policy.has_adsorbate,
        allow_tuple=False,
    )
    # Weave the project-wide parallelism default in once, here at the top-level
    # knob, so every downstream helper receives a concrete worker setting.
    n_jobs_population_init = resolve_n_jobs(
        n_jobs_population_init, "n_jobs_population_init"
    )
    n_jobs_offspring = resolve_n_jobs(n_jobs_offspring, "n_jobs_offspring")
    validate_ga_common_params(
        niter=niter,
        population_size=population_size,
        n_jobs_population_init=n_jobs_population_init,
        calculator=calculator,
        mutation_probability=mutation_probability,
        offspring_fraction=offspring_fraction,
        vacuum=vacuum,
        fmax=fmax,
    )

    if batch_size is not None and batch_size <= 0:
        batch_size = None

    # Resolve the optional TorchSim dtype knob for the auto-built relaxer.
    # ``None`` keeps the TorchSimBatchRelaxer default (float64). Callers that
    # pass their own ``relaxer`` set its dtype directly and ignore this.
    if torchsim_dtype is not None and torchsim_dtype not in ("float32", "float64"):
        raise SCGOValidationError(
            f"torchsim_dtype must be 'float32' or 'float64', got {torchsim_dtype!r}"
        )
    torchsim_dtype_resolved = (
        getattr(torch, torchsim_dtype) if torchsim_dtype is not None else None
    )

    # Normalize RNG early and enforce Generator-only policy
    rng = ensure_rng_or_create(rng)
    surface_mode = uses_surface(system_type)
    n_fixed = 0
    search_composition = list(composition)
    deposit_composition = list(composition)
    n_mobile_slab = 0

    if surface_mode:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise SCGOValidationError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        if policy.slab_is_search_target:
            from scgo.surface.partition import prepare_slab_search_surface_config

            surface_config, partition = prepare_slab_search_surface_config(
                surface_config
            )
            n_fixed = partition.n_fixed
            n_mobile_slab = partition.n_mobile_slab
            search_composition = resolve_search_mobile_composition(
                system_type=system_type,
                composition=list(composition),
                surface_config=surface_config,
                adsorbate_definition=adsorbate_definition,
            )
            if policy.has_adsorbate:
                ads = (
                    adsorbate_definition.adsorbate_symbols
                    if adsorbate_definition
                    else []
                )
                deposit_composition = (
                    [str(s) for s in ads] if isinstance(ads, list) else []
                )
            else:
                deposit_composition = []
        slab_ref = surface_config.slab.copy()
        n_slab = len(slab_ref)
        if not policy.slab_is_search_target:
            n_fixed = n_slab
            search_composition = list(composition)
            deposit_composition = list(composition)
        n_to_optimize = len(search_composition)
        if n_to_optimize < 1:
            raise SCGOValidationError(
                f"system_type={system_type!r} has no search-mobile atoms."
            )
        if policy.slab_is_search_target and not policy.has_adsorbate:
            atoms_template = slab_ref.copy()
        elif policy.slab_is_search_target:
            ads_syms = list(search_composition[n_mobile_slab:])
            if ads_syms:
                dummy_top = [[0.0, 0.0, 0.0] for _ in range(len(ads_syms))]
                atoms_template = Atoms(
                    symbols=list(slab_ref.get_chemical_symbols()) + ads_syms,
                    positions=np.vstack(
                        [slab_ref.get_positions(), np.asarray(dummy_top)]
                    ),
                    cell=slab_ref.get_cell(),
                    pbc=slab_ref.get_pbc(),
                )
            else:
                atoms_template = slab_ref.copy()
        else:
            dummy_top = [[0.0, 0.0, 0.0] for _ in range(n_to_optimize)]
            atoms_template = Atoms(
                symbols=list(slab_ref.get_chemical_symbols()) + list(composition),
                positions=np.vstack([slab_ref.get_positions(), np.asarray(dummy_top)]),
                cell=slab_ref.get_cell(),
                pbc=slab_ref.get_pbc(),
            )
    else:
        n_slab = 0
        slab_ref = None
        n_to_optimize = len(composition)
        search_composition = list(composition)
        cell_side = compute_cell_side(composition, vacuum=vacuum)
        atoms_template = Atoms(
            symbols=composition,
            positions=[[0, 0, 0] for _ in range(n_to_optimize)],  # Dummy positions
            cell=[cell_side] * 3,
            pbc=False,
        )

    pop_for_probe = population_size if population_size is not None else 32
    expected_max_atoms = (n_to_optimize + n_fixed) * pop_for_probe

    if relaxer is None:
        if is_ml_calculator(calculator):
            relaxer = build_torchsim_relaxer(
                calculator,
                fmax=fmax,
                max_steps=niter_local_relaxation,
                expected_max_atoms=expected_max_atoms,
                dtype=torchsim_dtype_resolved,
            )
        else:
            relaxer = AseBatchRelaxer(
                calculator,
                optimizer=optimizer,
                force_tol=fmax,
                max_steps=niter_local_relaxation,
                surface_mode=surface_mode,
                n_slab=n_fixed,
            )
    elif (
        isinstance(niter_local_relaxation, int) and niter_local_relaxation > 0
    ) or relaxer.max_steps is None:
        relaxer.max_steps = niter_local_relaxation

    if isinstance(relaxer, TorchSimBatchRelaxer):
        atoms_template.calc = None
    else:
        atoms_template.calc = calculator

    # Diversity scorer needs surface-aware mic; set up after operators / comp_mic.
    if surface_mode and slab_ref is not None:
        slab_for_pairing = slab_ref[:n_fixed].copy() if n_fixed > 0 else slab_ref.copy()
    else:
        slab_for_pairing = None

    adaptive_config = get_adaptive_mutation_config(
        composition=search_composition,
        current_generation=0,
        total_generations=niter,
        use_adaptive=use_adaptive_mutations,
        generations_without_improvement=0,
        stagnation_trigger=stagnation_trigger,
        stagnation_full_trigger=stagnation_full_trigger,
        recovery_window=recovery_window,
        aggressive_burst_multiplier=aggressive_burst_multiplier,
        max_mutation_probability=max_mutation_probability,
        system_type=system_type,
        adsorbate_definition=adsorbate_definition,
    )

    idx_top = (
        range(n_fixed, n_fixed + n_to_optimize)
        if surface_mode
        else range(n_to_optimize)
    )
    top_z = list({int(atoms_template[i].number) for i in idx_top})
    all_atom_types = get_all_atom_types(atoms_template, top_z)
    blmin = build_blmin_from_zs(all_atom_types, ratio=BLMIN_RATIO_DEFAULT)

    operators_list, name_map = create_mutation_operators(
        composition=search_composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=rng,
        use_adaptive=use_adaptive_mutations,
        system_type=system_type,
        n_slab=n_fixed if policy.slab_is_search_target else n_slab,
        surface_normal_axis=(surface_config.surface_normal_axis if surface_mode else 2),
        adsorbate_definition=adsorbate_definition,
        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
        adsorbate_fragment_template=adsorbate_fragment_template,
        cluster_adsorbate_config=cluster_adsorbate_config,
        connectivity_factor=connectivity_factor,
    )

    _ = update_mutation_weights(
        operators_list=operators_list,
        name_map=name_map,
        adaptive_config=adaptive_config,
        rng=rng,
    )
    # Use user-provided mutation_probability when adaptive mutations are disabled
    current_mutation_probability = (
        mutation_probability
        if not use_adaptive_mutations
        else adaptive_config["mutation_probability"]
    )

    comp_mic = resolve_structure_mic(system_type, surface_config)
    uniqueness_n_top = (
        int(comparator_n_top) if comparator_n_top is not None else n_to_optimize
    )
    user_geometry = UniquenessSettings(
        comparator_tol=comparator_tol,
        comparator_pair_cor_max=comparator_pair_cor_max,
        component_weights=comparator_component_weights,
        cross_weight=comparator_cross_weight,
    )
    # comparator_n_top forces the legacy trailing-window comparison (documented
    # escape hatch); otherwise dedupe uses type-aware role blocks.
    resolved_geo = None
    uniqueness_blocks: ComparatorBlocks | None = None
    geometry: UniquenessSettings = user_geometry
    if comparator_n_top is None:
        if policy.slab_is_search_target:
            # search_composition already contains the mobile top-layer atoms.
            n_total = int(n_fixed) + len(search_composition)
        elif surface_mode:
            n_total = len(surface_config.slab) + len(search_composition)
        else:
            n_total = len(composition)
        resolved_geo = resolve_uniqueness_geometry(
            system_type=system_type,
            n_atoms=n_total,
            surface_config=surface_config,
            adsorbate_definition=adsorbate_definition,
            settings=user_geometry,
        )
        uniqueness_blocks = resolved_geo.blocks
        geometry = resolved_geo.settings
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=search_composition,
        n_to_optimize=uniqueness_n_top,
        diversity_max_references=diversity_max_references,
        logger=logger,
        base_dir=output_dir,
        mic=comp_mic,
        uniqueness=geometry,
        blocks=uniqueness_blocks,
    )
    if resolved_geo is not None:
        structure_comparator = resolved_geo.build_comparator(
            n_top=uniqueness_n_top, mic=comp_mic
        )
    else:
        structure_comparator = create_geometry_comparator(
            n_top=uniqueness_n_top,
            mic=comp_mic,
            settings=geometry,
        )
    comp = EnergyAndStructureComparator(energy_tolerance, structure_comparator)

    t0_batch_build = perf_counter()
    if surface_mode:
        if slab_ref is None:
            raise TypeError("slab_ref is required in surface_mode")
        if policy.slab_is_search_target and not policy.has_adsorbate:
            start_generator = SurfaceSlabStartGenerator(
                slab_ref,
                n_fixed=n_fixed,
                rng=rng,
                calculator=None,
                population_size=population_size,
                verbosity=verbosity,
            )
        else:
            start_generator = SurfaceClusterStartGenerator(
                deposit_composition,
                slab_ref,
                surface_config,
                blmin,
                rng=rng,
                calculator=None,
                population_size=population_size,
                previous_search_glob=previous_search_glob,
                n_jobs=n_jobs_population_init,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
                verbosity=verbosity,
            )
    else:
        start_generator = ClusterStartGenerator(
            composition,
            vacuum,
            rng=rng,
            calculator=None,  # Do not attach calculator to initial population to avoid pickling issues
            population_size=population_size,
            mode="smart",
            previous_search_glob=previous_search_glob,
            n_jobs=n_jobs_population_init,
            system_type=system_type,
            adsorbate_definition=adsorbate_definition,
            adsorbate_fragment_template=adsorbate_fragment_template,
            cluster_adsorbate_config=cluster_adsorbate_config,
            verbosity=verbosity,
        )
    profile_timings["initial_population_batch_build_s"] = (
        perf_counter() - t0_batch_build
    )
    t0 = perf_counter()
    initial_population = [
        start_generator.get_new_candidate() for _ in range(population_size)
    ]
    profile_timings["initial_population_generation_s"] = perf_counter() - t0

    log_info_v(
        logger,
        "Generated initial population of %d candidates (batched, parallel: %s)",
        population_size,
        f"{resolve_n_jobs_for_tasks(n_jobs_population_init, population_size)} workers",
        verbosity=verbosity,
    )

    # Do not pass initial_population to setup_database (avoids formula keys in
    # key_value_pairs). Insert unrelaxed starters via the low-level API, then
    # batch-relax them and tag generation=0.
    da = setup_database(
        output_dir=output_dir,
        db_filename="ga_go.db",
        atoms_template=atoms_template,
        remove_existing=clean,
        remove_aux_files=clean,
        enable_expression_indexes=db_enable_expression_indexes,
        run_id=run_id,
    )

    # Declared before the `try` so the `finally` cleanup below can never raise
    # UnboundLocalError (which would mask an earlier failure, e.g. one raised
    # during the initial population relaxation). Created lazily in the loop.
    offspring_executor: ProcessPoolExecutor | None = None

    try:
        log_info_v(
            logger,
            "Relaxing initial population of up to %d candidates",
            population_size,
            verbosity=verbosity,
        )

        logger.debug(
            "Using GA database at %s",
            os.path.join(output_dir, "ga_go.db"),
        )

        initial_pop_count = 0
        initial_discarded_count = 0
        initial_ineligible_relaxed_count = 0
        initial_ineligible_reasons: dict[str, int] = {}
        inserted_initial_population: list[Atoms] = []

        def _insert_unrelaxed(cand):
            cand.info.setdefault("data", {})
            gaid = da.c.write(
                cand,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
                description="initial",
            )
            da.c.update(gaid, gaid=gaid)
            cand.info["confid"] = gaid

        t0 = perf_counter()
        with da.c:
            for cand in initial_population:
                if adsorbate_definition is None and not surface_mode:
                    cand = reorder_cluster_to_composition(cand, list(composition))
                maybe_apply_mobile_core_ads_tags(
                    cand,
                    n_slab,
                    composition,
                    adsorbate_definition,
                    system_type,
                )
                if freeze_adsorbate_internal_geometry:
                    enforce_frozen_adsorbate_geometry(
                        cand,
                        n_slab=n_slab,
                        adsorbate_definition=adsorbate_definition,
                        fragment_templates=adsorbate_fragment_template,
                    )
                validation_error = validate_structure_for_ga_storage(
                    cand,
                    surface_mode=surface_mode,
                    n_slab=n_slab,
                    n_slab_deposit=(n_fixed if policy.slab_is_search_target else None),
                    system_type=system_type,
                    surface_config=surface_config,
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=connectivity_factor,
                    cluster_adsorbate_config=cluster_adsorbate_config,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                )
                if validation_error is not None:
                    initial_discarded_count += 1
                    logger.debug(
                        "Discarding invalid initial candidate before DB insert: %s",
                        validation_error,
                    )
                    continue
                database_retry(
                    lambda _cand=cand: _insert_unrelaxed(_cand),
                    config=RetryConfig(max_retries=5),
                    operation_name="insert_unrelaxed_candidate",
                )
                inserted_initial_population.append(cand)
        profile_timings["initial_unrelaxed_insert_s"] = perf_counter() - t0

        if not inserted_initial_population:
            logger.error(
                "No valid initial GA population after validation (%d discarded)",
                initial_discarded_count,
            )
            return []

        # Write a relaxed batch under one connection. Per-attempt counters reset
        # inside the writer so SQLite retries do not double-count.
        def _write_relaxed_batch(batch, relaxed_results):
            nonlocal batch_ineligible_count, batch_ineligible_reasons
            batch_ineligible_count = 0
            batch_ineligible_reasons = {}
            with da.c:
                for original, (energy, relaxed) in zip(
                    batch, relaxed_results, strict=True
                ):
                    validation_error = _write_relaxed_candidate(
                        da,
                        original,
                        relaxed,
                        energy,
                        GAWriteContext(
                            n_slab=n_slab,
                            n_frozen_prefix=n_fixed,
                            composition=composition,
                            adsorbate_definition=adsorbate_definition,
                            system_type=system_type,
                            surface_mode=surface_mode,
                            surface_config=surface_config,
                            connectivity_factor=connectivity_factor,
                            cluster_adsorbate_config=cluster_adsorbate_config,
                            allow_cluster_fragmentation=allow_cluster_fragmentation,
                            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                            freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                            adsorbate_fragment_templates=adsorbate_fragment_template,
                        ),
                        generation=0,
                        run_id=run_id,
                    )
                    if validation_error is not None:
                        batch_ineligible_count += 1
                        reason = compact_ga_ineligible_reason(validation_error)
                        batch_ineligible_reasons[reason] = (
                            batch_ineligible_reasons.get(reason, 0) + 1
                        )
                        logger.debug(
                            "Initial candidate failed validation after "
                            "relaxation; storing but excluding from GA "
                            "population: %s",
                            validation_error,
                        )

        # Process starting population in batches (only candidates inserted above).
        batch_size_internal = batch_size or len(inserted_initial_population)
        t0_relax = 0.0
        t0_write = 0.0
        batch_ineligible_count = 0
        batch_ineligible_reasons: dict[str, int] = {}
        for i in range(0, len(inserted_initial_population), batch_size_internal):
            batch = inserted_initial_population[i : i + batch_size_internal]
            t_start = perf_counter()
            relaxed_results = relaxer.relax_batch(
                [
                    _torchsim_prepare_relaxed_copy(
                        c,
                        surface_config,
                        n_slab,
                        surface_mode=surface_mode,
                        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                        adsorbate_definition=adsorbate_definition,
                        adsorbate_fragment_templates=adsorbate_fragment_template,
                    )
                    for c in batch
                ]
            )
            t0_relax += perf_counter() - t_start
            _record_relax_batch_steps(
                relaxer, profile_timings, profile_counters, len(batch)
            )
            if len(relaxed_results) != len(batch):
                raise SCGORuntimeError(
                    "TorchSim relaxer returned mismatched batch size"
                )

            t_start = perf_counter()
            database_retry(
                lambda _batch=batch, _results=relaxed_results: _write_relaxed_batch(
                    _batch, _results
                ),
                config=RetryConfig(max_retries=5),
                operation_name="write_initial_relaxed_batch",
            )
            initial_ineligible_relaxed_count += batch_ineligible_count
            for reason, count in batch_ineligible_reasons.items():
                initial_ineligible_reasons[reason] = (
                    initial_ineligible_reasons.get(reason, 0) + count
                )
            t0_write += perf_counter() - t_start

            initial_pop_count += len(batch)
        profile_timings["initial_relax_batch_s"] = t0_relax
        profile_timings["initial_relaxed_write_s"] = t0_write

        if initial_pop_count > 0:
            logger.debug(
                "Tagged %s GA population members with generation=0",
                initial_pop_count,
            )

        log_file = os.path.join(output_dir, "population.log")

        with suppress(FileNotFoundError):
            os.remove(log_file)

        # Select appropriate Population class based on fitness strategy
        PopulationClass, population_kwargs = select_population_class(
            fitness_strategy=fitness_strategy,
            diversity_scorer=diversity_scorer,
            diversity_update_interval=diversity_update_interval,
            logger=logger,
        )

        population = PopulationClass(
            data_connection=da,
            population_size=population_size,
            comparator=comp,
            logfile=log_file,
            rng=rng,  # type: ignore[arg-type]
            elite_fraction=elite_fraction,
            run_id=run_id,
            **population_kwargs,
        )
        population._write_log()
        eligible_initial = initial_pop_count - initial_ineligible_relaxed_count
        ineligible_detail = format_count_summary(initial_ineligible_reasons)
        ineligible_suffix = f" ({ineligible_detail})" if ineligible_detail else ""
        log_info_v(
            logger,
            "Initial population: size=%d, %d GA-eligible, %d discarded "
            "pre-relax, %d ineligible post-relax%s",
            len(population.pop),
            eligible_initial,
            initial_discarded_count,
            initial_ineligible_relaxed_count,
            ineligible_suffix,
            verbosity=verbosity,
        )
        log_debug_v(
            logger,
            "Initial population confids=%s",
            [a.info.get("confid") for a in population.pop],
            verbosity=verbosity,
        )

        log_early_stopping_info(
            verbosity=verbosity,
            fitness_strategy=fitness_strategy,
            early_stopping_niter=early_stopping_niter,
            niter=niter,
            logger=logger,
        )

        # Track best value for early stopping (energy or fitness)
        best_value = None  # Energy for low_energy, fitness for others
        generations_without_improvement = 0
        recent_acceptance_ratios: list[float] = []

        # The offspring ProcessPoolExecutor is hoisted above the `try` and created
        # lazily below, so it is forked + pickled once instead of every generation.
        # Workers reload their pairing/operator state per generation via
        # `_build_offspring_worker` (keyed on `operators_epoch`), so reuse is
        # correctness-preserving.

        for generation in tqdm(
            range(niter),
            desc=f"  GA generations for {n_to_optimize} mobile atoms",
            disable=not should_show_progress(verbosity),
        ):
            if use_adaptive_mutations:
                adaptive_config = get_adaptive_mutation_config(
                    composition=search_composition,
                    current_generation=generation,
                    total_generations=niter,
                    use_adaptive=True,
                    generations_without_improvement=generations_without_improvement,
                    stagnation_trigger=stagnation_trigger,
                    stagnation_full_trigger=stagnation_full_trigger,
                    recovery_window=recovery_window,
                    aggressive_burst_multiplier=aggressive_burst_multiplier,
                    max_mutation_probability=max_mutation_probability,
                    system_type=system_type,
                    adsorbate_definition=adsorbate_definition,
                )
                _ = update_mutation_weights(
                    operators_list=operators_list,
                    name_map=name_map,
                    adaptive_config=adaptive_config,
                    rng=rng,
                )
                current_mutation_probability = adaptive_config["mutation_probability"]

            # Create up to `n_offspring` unrelaxed candidates for this generation;
            # TorchSim will handle batching/relaxation later.
            n_offspring = max(1, math.ceil(population_size * offspring_fraction))
            created = 0
            attempts = 0
            max_attempts = max(10, n_offspring * 10)
            if ga_adaptive_retry_enabled:
                recent_ratio = (
                    float(np.mean(recent_acceptance_ratios[-5:]))
                    if recent_acceptance_ratios
                    else 0.35
                )
                target_ratio = max(0.05, min(0.95, recent_ratio))
                estimated_needed = int(math.ceil(n_offspring / target_ratio))
                floor_attempts = max(10, n_offspring * int(ga_retry_floor_multiplier))
                ceil_attempts = max(
                    floor_attempts, n_offspring * int(ga_retry_ceiling_multiplier)
                )
                max_attempts = max(floor_attempts, min(estimated_needed, ceil_attempts))

            t_loop = perf_counter()
            t_parent_select_gen = 0.0
            t_operator_setup_gen = 0.0
            t_crossover_gen = 0.0
            t_mutation_gen = 0.0
            t_db_unrelaxed_gen = 0.0
            t_offspring_parallel_wall_gen = 0.0
            worker_failures_gen = 0
            worker_failure_types_gen: dict[str, int] = {}
            retry_failure_reasons_gen: dict[str, int] = {}
            pairing_cuts_gen = 0
            generation_all_job_results: list[dict[str, Any]] = []
            total_crossover_jobs_gen = 0

            log_phase_subheader(
                logger,
                f"Generation {generation}",
                verbosity=verbosity,
            )

            _BLMIN_THRESH_CACHE.clear()
            offspring_ctx = OffspringBuildContext(
                atoms_template=_picklable_atoms_copy(atoms_template),
                n_to_optimize=n_to_optimize,
                composition=composition,
                blmin=blmin if ga_fast_prefilter_enabled else {},
                system_type=system_type,
                n_slab=n_slab,
                n_frozen_prefix=n_fixed,
                slab_for_pairing=_picklable_atoms_copy(slab_for_pairing),
                surface_normal_axis=(
                    surface_config.surface_normal_axis if surface_mode else 2
                ),
                adsorbate_definition=adsorbate_definition,
                connectivity_factor=connectivity_factor,
                cluster_adsorbate_config=cluster_adsorbate_config,
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                adsorbate_fragment_templates=_picklable_fragment_templates(
                    adsorbate_fragment_template
                ),
                surface_config=surface_config,
                adaptive_config=adaptive_config,
                current_mutation_probability=current_mutation_probability,
                operators_list=operators_list,
                name_map=name_map,
                operators_epoch=generation,
            )
            n_workers_offspring = resolve_n_jobs_for_tasks(
                n_jobs_offspring, max(1, n_offspring)
            )
            # Create the (hoisted) pool once on first need; reuse across generations.
            if n_workers_offspring > 1 and offspring_executor is None:
                offspring_executor = ProcessPoolExecutor(
                    max_workers=n_workers_offspring,
                    initializer=_offspring_worker_bootstrap_init,
                    initargs=(offspring_ctx,),
                )
            # The parent process also needs its own state: a batch that yields a
            # single job falls back to the in-process path below, which calls
            # ``_build_offspring_worker`` here rather than in a pool worker.
            _ensure_offspring_worker_state(offspring_ctx)

            try:
                while created < n_offspring and attempts < max_attempts:
                    attempts_remaining = max_attempts - attempts
                    if attempts_remaining <= 0:
                        break
                    jobs_target = min(n_offspring - created, attempts_remaining)
                    jobs: list[dict[str, Any]] = []
                    for _ in range(jobs_target):
                        attempts += 1
                        t0 = perf_counter()
                        candidates = population.get_two_candidates()
                        t_parent_select_gen += perf_counter() - t0
                        if candidates is None:
                            continue
                        a1, a2 = candidates
                        task_seed = int(rng.integers(0, 2**31 - 1))
                        _nfp = offspring_ctx.n_frozen_prefix
                        if _nfp > 0:
                            _a1 = _mobile_only_copy(a1, _nfp)
                            _a2 = _mobile_only_copy(a2, _nfp)
                        else:
                            _a1 = a1.copy()
                            _a1.calc = None
                            _a2 = a2.copy()
                            _a2.calc = None
                        jobs.append(
                            {
                                "index": len(jobs),
                                "a1": _a1,
                                "a2": _a2,
                                "mobile_only": _nfp > 0,
                                "task_seed": task_seed,
                                "operators_epoch": offspring_ctx.operators_epoch,
                                "adaptive_config": offspring_ctx.adaptive_config,
                                "current_mutation_probability": (
                                    offspring_ctx.current_mutation_probability
                                ),
                            }
                        )
                    if not jobs:
                        continue

                    n_workers = resolve_n_jobs_for_tasks(n_jobs_offspring, len(jobs))

                    t_parallel = perf_counter()
                    job_results: dict[int, dict[str, Any]] = {}
                    worker_exceptions: list[BaseException] = []
                    if n_workers == 1:
                        for job in jobs:
                            try:
                                result = _build_offspring_worker(job)
                            except (RuntimeError, ValueError, TypeError) as exc:
                                worker_failures_gen += 1
                                err_name = type(exc).__name__
                                worker_failure_types_gen[err_name] = (
                                    worker_failure_types_gen.get(err_name, 0) + 1
                                )
                                reason = f"worker_exception_{err_name}"
                                retry_failure_reasons_gen[reason] = (
                                    retry_failure_reasons_gen.get(reason, 0) + 1
                                )
                                worker_exceptions.append(exc)
                                logger.exception(
                                    "Offspring crossover/mutation worker failed (%s)",
                                    err_name,
                                )
                                continue
                            job_results[result["index"]] = result
                    else:
                        if offspring_executor is None:
                            raise SCGORuntimeError(
                                "offspring_executor is None but n_jobs_offspring > 1"
                            )
                        futures = [
                            offspring_executor.submit(_build_offspring_worker, job)
                            for job in jobs
                        ]
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                            except (RuntimeError, ValueError, TypeError) as exc:
                                worker_failures_gen += 1
                                err_name = type(exc).__name__
                                worker_failure_types_gen[err_name] = (
                                    worker_failure_types_gen.get(err_name, 0) + 1
                                )
                                reason = f"worker_exception_{err_name}"
                                retry_failure_reasons_gen[reason] = (
                                    retry_failure_reasons_gen.get(reason, 0) + 1
                                )
                                worker_exceptions.append(exc)
                                logger.exception(
                                    "Offspring crossover/mutation worker failed (%s)",
                                    err_name,
                                )
                                continue
                            job_results[result["index"]] = result
                    total_crossover_jobs_gen += len(job_results)
                    generation_all_job_results.extend(job_results.values())
                    if len(jobs) > 0 and len(job_results) == 0 and worker_exceptions:
                        first = worker_exceptions[0]
                        if not all(
                            isinstance(e, ValueError) for e in worker_exceptions
                        ):
                            raise SCGORuntimeError(
                                f"All {len(jobs)} parallel offspring workers failed"
                            ) from first
                    t_offspring_parallel_wall_gen += perf_counter() - t_parallel
                    if worker_failures_gen:
                        profile_counters["offspring_worker_failures"] += (
                            worker_failures_gen
                        )
                        failure_limit = max(3, len(jobs) // 2)
                        if worker_failures_gen >= failure_limit:
                            logger.warning(
                                "Generation %s offspring worker failures: %d/%d (%s)",
                                generation,
                                worker_failures_gen,
                                len(jobs),
                                worker_failure_types_gen,
                            )

                    pending_inserts: list[tuple[Atoms, str]] = []
                    for idx in range(len(jobs)):
                        if created >= n_offspring:
                            break
                        result = job_results.get(idx)
                        if result is None:
                            continue
                        t_operator_setup_gen += float(result["operator_setup_s"])
                        t_crossover_gen += float(result["crossover_s"])
                        t_mutation_gen += float(result["mutation_s"])
                        pairing_cuts_gen += int(result.get("pairing_attempt_count", 0))
                        child = result["child"]
                        if child is None:
                            reason = result.get("failure_reason") or "unknown"
                            retry_failure_reasons_gen[reason] = (
                                retry_failure_reasons_gen.get(reason, 0) + 1
                            )
                            continue
                        pending_inserts.append((child, result["desc"]))
                    if pending_inserts:
                        t0 = perf_counter()
                        with da.c:
                            for child, desc in pending_inserts:
                                database_retry(
                                    lambda _a3=child, _desc=desc: (
                                        da.add_unrelaxed_candidate(
                                            _a3, description=_desc
                                        )
                                    ),
                                    config=RetryConfig(max_retries=5),
                                    operation_name="add_unrelaxed_offspring",
                                )
                                created += 1
                        t_db_unrelaxed_gen += perf_counter() - t0
            finally:
                # The pool is reused across generations; only clear the cached
                # worker state. Shutdown happens once after the generational loop.
                _OFFSPRING_WORKER_STATE.clear()

            generation_acceptance = created / max(attempts, 1)
            recent_acceptance_ratios.append(generation_acceptance)
            profile_counters["offspring_attempts_total"] += attempts
            profile_counters["pairing_cut_attempts_total"] = (
                profile_counters.get("pairing_cut_attempts_total", 0) + pairing_cuts_gen
            )
            for reason, count in retry_failure_reasons_gen.items():
                profile_retry_failures[reason] = (
                    profile_retry_failures.get(reason, 0) + count
                )
            profile_timings["offspring_mutation_queue_s"] = profile_timings.get(
                "offspring_mutation_queue_s", 0.0
            ) + (perf_counter() - t_loop)
            profile_timings["offspring_parent_select_s"] = (
                profile_timings.get("offspring_parent_select_s", 0.0)
                + t_parent_select_gen
            )
            profile_timings["offspring_operator_setup_s"] = (
                profile_timings.get("offspring_operator_setup_s", 0.0)
                + t_operator_setup_gen
            )
            profile_timings["offspring_crossover_s"] = (
                profile_timings.get("offspring_crossover_s", 0.0) + t_crossover_gen
            )
            profile_timings["offspring_mutation_s"] = (
                profile_timings.get("offspring_mutation_s", 0.0) + t_mutation_gen
            )
            profile_timings["offspring_unrelaxed_insert_s"] = (
                profile_timings.get("offspring_unrelaxed_insert_s", 0.0)
                + t_db_unrelaxed_gen
            )
            profile_timings["offspring_parallel_wall_s"] = (
                profile_timings.get("offspring_parallel_wall_s", 0.0)
                + t_offspring_parallel_wall_gen
            )
            profile_counters["offspring_created"] += created

            log_generation_offspring_summaries(
                logger,
                verbosity=verbosity,
                job_results=generation_all_job_results,
                total_jobs=total_crossover_jobs_gen,
                created=created,
                n_offspring=n_offspring,
                attempts=attempts,
            )

            # Ask TorchSim relaxer to process available unrelaxed candidates now.
            # Enforce a per-generation limit: when `batch_size` is None, target the
            # full population so a single relax_batch submission keeps the autobatcher's
            # in-flight swap reservoir full (its budget tracks the systems handed to
            # one call). This maximizes GPU utilization; a user-set `batch_size`
            # (non-None) still caps the call as before.
            per_gen_max = (
                batch_size
                if batch_size is not None
                else max(n_offspring, population_size)
            )
            pre_db_read = float(profile_timings.get("db_read_s", 0.0))
            pre_relax = float(profile_timings.get("relax_batch_s", 0.0))
            pre_db_write = float(profile_timings.get("db_write_s", 0.0))
            pre_pop_update = float(profile_timings.get("population_update_s", 0.0))
            t0_relax_call = perf_counter()
            eligible_count, ineligible_count, ineligible_reasons = (
                _relax_unrelaxed_candidates(
                    da,
                    relaxer,
                    population=population,
                    max_batch=per_gen_max,
                    generation=generation,
                    run_id=run_id,
                    surface_config=surface_config,
                    n_slab=n_slab,
                    n_frozen_prefix=n_fixed,
                    system_type=system_type,
                    profiling=profile_timings,
                    counters=profile_counters,
                    composition=composition,
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=connectivity_factor,
                    cluster_adsorbate_config=cluster_adsorbate_config,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                    adsorbate_fragment_templates=adsorbate_fragment_template,
                )
            )
            offspring_count = eligible_count
            relax_call_wall_s = perf_counter() - t0_relax_call
            post_db_read = float(profile_timings.get("db_read_s", 0.0))
            post_relax = float(profile_timings.get("relax_batch_s", 0.0))
            post_db_write = float(profile_timings.get("db_write_s", 0.0))
            post_pop_update = float(profile_timings.get("population_update_s", 0.0))
            gen_db_read_s = max(0.0, post_db_read - pre_db_read)
            gen_relax_s = max(0.0, post_relax - pre_relax)
            gen_db_write_s = max(0.0, post_db_write - pre_db_write)
            gen_pop_update_s_from_relax = max(0.0, post_pop_update - pre_pop_update)
            pop_update_s = gen_pop_update_s_from_relax
            if verbosity >= 1 and (eligible_count + ineligible_count) > 0:
                reason_detail = format_count_summary(ineligible_reasons)
                reason_suffix = f" ({reason_detail})" if reason_detail else ""
                log_info_v(
                    logger,
                    "Relaxation: %d/%d GA-eligible, %d ineligible%s",
                    eligible_count,
                    eligible_count + ineligible_count,
                    ineligible_count,
                    reason_suffix,
                    verbosity=verbosity,
                )
            if offspring_count > 0:
                profile_counters["offspring_relaxed"] += int(offspring_count)

            if per_generation is not None:
                per_generation.append(
                    {
                        "generation": int(generation),
                        "n_offspring_target": int(n_offspring),
                        "offspring_created": int(created),
                        "attempts": int(attempts),
                        "acceptance_ratio": float(generation_acceptance),
                        "offspring_relaxed_this_call": int(offspring_count),
                        "retry_failures": dict(retry_failure_reasons_gen),
                        "timings_s": {
                            "parent_select_s": t_parent_select_gen,
                            "operator_setup_s": t_operator_setup_gen,
                            "crossover_s": t_crossover_gen,
                            "mutation_s": t_mutation_gen,
                            "db_unrelaxed_insert_s": t_db_unrelaxed_gen,
                            "offspring_parallel_wall_s": t_offspring_parallel_wall_gen,
                            "torchsim_db_read_s": gen_db_read_s,
                            "torchsim_relax_s": gen_relax_s,
                            "torchsim_db_write_s": gen_db_write_s,
                            "torchsim_relax_call_wall_s": relax_call_wall_s,
                            "population_update_s": pop_update_s,
                            "population_update_s_from_relax": gen_pop_update_s_from_relax,
                            "offspring_loop_wall_s": perf_counter() - t_loop,
                        },
                    }
                )

            if early_stopping_niter > 0:
                best_value, generations_without_improvement, should_stop = (
                    update_early_stopping_state_unified(
                        population=population,
                        fitness_strategy=fitness_strategy,
                        best_value=best_value,
                        generations_without_improvement=generations_without_improvement,
                        early_stopping_niter=early_stopping_niter,
                    )
                )
                if should_stop:
                    stopping_metric = (
                        "fitness"
                        if fitness_strategy != FitnessStrategy.LOW_ENERGY
                        else "energy"
                    )
                    log_info_v(
                        logger,
                        "Early stopping triggered: no %s improvement for %d generations (best %s: %.6f)",
                        stopping_metric,
                        generations_without_improvement,
                        stopping_metric,
                        best_value,
                        verbosity=verbosity,
                    )
                    break

        _relax_unrelaxed_candidates(
            da,
            relaxer,
            population=population,
            max_batch=batch_size,
            force=True,
            run_id=run_id,
            surface_config=surface_config,
            n_slab=n_slab,
            n_frozen_prefix=n_fixed,
            system_type=system_type,
            profiling=profile_timings,
            counters=profile_counters,
            composition=composition,
            adsorbate_definition=adsorbate_definition,
            connectivity_factor=connectivity_factor,
            cluster_adsorbate_config=cluster_adsorbate_config,
            allow_cluster_fragmentation=allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
            adsorbate_fragment_templates=adsorbate_fragment_template,
        )

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            config=RetryConfig(max_retries=5),
            operation_name="get_final_all_relaxed_candidates",
        )
        if run_id is not None:
            all_candidates = filter_by_tags(all_candidates, run_id=run_id)
        all_candidates = [
            cand
            for cand in all_candidates
            if bool(get_tag(cand, "ga_eligible", default=True))
        ]
        all_minima = extract_minima_from_database(all_candidates)

        log_info_v(
            logger,
            "GA evolution complete: found %d minima",
            len(all_minima),
            verbosity=verbosity,
        )
        drain_inductor_filelock_summary(logger)

        # Sort by fitness (highest first) for non-default strategies
        sort_minima_by_fitness(
            all_minima=all_minima,
            fitness_strategy=fitness_strategy,
            logger=logger,
        )
        profile_timings["total_wall_s"] = perf_counter() - profile_t0
        profile_timings["kind"] = "ga"
        relax_total = ga_relax_seconds_from_timings(profile_timings)
        profile_timings["relax_total_s"] = relax_total
        profile_timings["cpu_non_relax_s"] = cpu_non_relax_seconds_from_timings(
            profile_timings
        )
        log_timing_summary(logger, "torchsim_ga", profile_timings, verbosity=verbosity)
        extra_payload: dict[str, Any] = {
            "counters": profile_counters,
            "retry_failures": profile_retry_failures,
        }
        if per_generation is not None:
            extra_payload["per_generation"] = per_generation
        timing_dir = timing_output_dir if timing_output_dir is not None else output_dir
        run_id_for_timing = os.path.basename(str(timing_dir).rstrip(os.sep))
        out_payload = build_timing_payload(
            backend="torchsim_ga",
            timings_s=profile_timings,
            run_id=run_id_for_timing,
            extra=extra_payload,
        )
        emit_timing_data(
            out_payload,
            write_timing_json=write_timing_json,
            output_dir=output_dir,
            timing_output_dir=timing_output_dir,
            timing_collector=timing_collector,
        )

        return all_minima

    finally:
        # Shut down the hoisted offspring pool once (it was created lazily and
        # reused across generations) before closing the data connection.
        if offspring_executor is not None:
            offspring_executor.shutdown(wait=True)
        close_data_connection(da, log_errors=False)
