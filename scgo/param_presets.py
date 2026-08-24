"""Parameter presets for SCGO campaigns."""

from __future__ import annotations

import copy
from functools import cache
from typing import Any

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_FMAX_THRESHOLD,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
    DEFAULT_TS_PAIR_COR_MAX,
)
from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.pair_selection_defaults import pair_selection_param_defaults
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    GLOptimizerParams,
    SystemType,
    get_system_policy,
)
from scgo.utils.parallel_workers import DEFAULT_N_JOBS

# Available MACE model names for use in calculator_kwargs["model_name"]
AVAILABLE_MACE_MODELS = [
    "mace_matpes_0",  # r2scan variant (default in MACE class)
    "mace_mp_small",  # Small MACE-MP
    "mace_mpa_medium",  # Medium MACE-MPA
    "mace_off_small",  # Small MACE-OFF
]

# Common fairchem pretrained names (see fairchem.core.calculate.pretrained_mlip)
AVAILABLE_UMA_MODELS = [
    "uma-s-1p2",
    "uma-s-1p1",
    "uma-m-1p1",
]

# Common UPET model identifiers (see upet.list_upet)
AVAILABLE_UPET_MODELS = [
    "pet-mad-s",
    "pet-mad-xs",
    "pet-oam-xl",
    "pet-omat-s",
    "pet-spice-s",
]

__all__ = [
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    "AVAILABLE_UPET_MODELS",
    "TS_DEFAULTS_BY_SYSTEM_TYPE",
    "TS_NEB_FMAX",
    "TS_POSTPROCESS_DEFAULTS",
    "default_calculator_kwargs",
    "default_energy_gap_threshold",
    "get_default_params",
    "default_params_top_level_keys",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_torchsim_ga_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_low_effort_torchsim_ga_params",
    "get_low_effort_upet_ga_params",
    "get_low_effort_uma_ga_params",
    "get_low_effort_ts_search_params",
    "get_ts_defaults",
    "get_ts_search_params",
    "low_effort_neb_steps",
    "get_default_uma_params",
    "get_default_upet_params",
    "get_uma_ga_benchmark_params",
    "get_upet_ga_benchmark_params",
]


# Shared NEB force tolerance for every system type. Pairing / climb / springs /
# step budgets may differ by type; force convergence must not.
# 0.20 eV/Å is the attainable MACE CI-NEB floor for soft adsorbate MEPs; tighter
# values often collapse interior saddles to endpoints before forces reach 0.05.
TS_NEB_FMAX: float = 0.20

# Backwards-compatible alias (pre-0.9.0 private name).
_TS_NEB_FMAX = TS_NEB_FMAX

# --- Low-effort ("~25% of production") preset knobs -------------------------
# Consumed by `get_low_effort_torchsim_ga_params` / `get_low_effort_ts_search_params`.
# These are the single tuning surface for the examples and the Kaggle GPU CI
# matrix, which both build their params from those two functions.

# Fraction of the production budget the low-effort presets aim for.
_LOW_EFFORT_SCALE: float = 0.25

# GA generations / population, scaled from the production benchmark reference
# in `_get_base_ga_benchmark_params` (niter=10, population_size=50).
_LOW_EFFORT_GA_NITER: int = 3
_LOW_EFFORT_GA_POPULATION_SIZE: int = 13
# Local relaxation steps per candidate. Surface system types clamp this up to
# `SURFACE_GA_MIN_LOCAL_RELAX_STEPS` (400) in
# `scgo.utils.run_helpers.prepare_algorithm_kwargs`, so surface GO stays
# production-strength; only gas types actually run at this value.
_LOW_EFFORT_GA_NITER_LOCAL_RELAXATION: int = 70

# NEB step floor for low-effort presets. Bare gas uses ``neb_steps="auto"``
# (~372 for Pt5); 25% of that does not converge. Floor keeps bands reaching
# ``neb_fmax`` and an interior saddle peak for CI assertions.
_LOW_EFFORT_NEB_FLOOR: int = 1000

# Per-system-type NEB defaults consumed by `get_ts_search_params` and
# `coerce_ts_params_to_runner_kwargs`. Keep `neb_interpolation_mic` coherent with
# `SystemPolicy.neb_force_mic` (enforced in tests via
# ``assert defaults["neb_interpolation_mic"] is policy.neb_force_mic``).
# Other knobs (n_images, steps, climb, alignment, pairing gates, ...) are
# independent per type.
# ``neb_fmax`` / ``torchsim_fmax`` are always ``TS_NEB_FMAX`` (enforced in tests).
_GAS_TS_NEB_DEFAULTS: dict[str, Any] = {
    "neb_align_endpoints": True,
    "neb_interpolation_mic": False,
    "neb_surface_cell_remap": False,
    "neb_surface_lattice_rotation": False,
    "neb_surface_max_lattice_shift": 1,
    "neb_n_images": 5,
    "neb_spring_constant": 0.1,
    "neb_fmax": TS_NEB_FMAX,
    "neb_steps": "auto",
    "neb_climb": False,
    "neb_perturb_sigma": 0.0,
    "neb_interpolation_method": "idpp",
    "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
    "torchsim_fmax": TS_NEB_FMAX,
    "torchsim_max_steps": "auto",
    "max_endpoint_mismatch": None,
    # Bare gas clusters get a looser clash gate and a tighter saddle-prominence
    # floor (flat-PES rearrangements are common for small gas clusters).
    "neb_prescreen_clash_distance": 1.0,
    "min_saddle_prominence": 0.10,
    "neb_max_spurious_barrier": 8.0,
    "binding_penetration_tolerance_a": 0.1,
    "layer_cluster_threshold_ang": 0.4,
    "neb_interpolation_bond_tolerance_a": 0.5,
    # None = all selected bands in one ParallelNEBBatch.
    "parallel_neb_max_bands": None,
    # Atom budget per fused force batch (sum of n_images * n_atoms), used when
    # parallel_neb_max_bands is None. Gas cells are small, so a larger budget
    # keeps the GPU saturated; parallel_neb_max_bands still overrides it.
    "parallel_neb_max_batch_atoms": 6000,
}

_SURFACE_TS_NEB_DEFAULTS: dict[str, Any] = {
    "neb_align_endpoints": True,
    "neb_interpolation_mic": True,
    "neb_surface_cell_remap": True,
    "neb_surface_lattice_rotation": True,
    "neb_surface_max_lattice_shift": 1,
    "neb_n_images": 5,
    "neb_spring_constant": 0.1,
    "neb_fmax": TS_NEB_FMAX,
    # Shared fmax with MIC/remap paths: keep a larger step budget than gas auto.
    "neb_steps": 2000,
    "neb_climb": False,
    "neb_perturb_sigma": 0.0,
    "neb_interpolation_method": "idpp",
    "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
    "torchsim_fmax": TS_NEB_FMAX,
    "torchsim_max_steps": 2000,
    # Surface clusters newly gain the endpoint-displacement gate (was unset).
    "neb_prescreen_clash_distance": 0.7,
    "min_saddle_prominence": 0.40,
    "neb_max_spurious_barrier": 8.0,
    "binding_penetration_tolerance_a": 0.1,
    "layer_cluster_threshold_ang": 0.4,
    "neb_interpolation_bond_tolerance_a": 0.5,
    "max_endpoint_mismatch": 1.25,
    # Surface OOM safety: chunk parallel NEB + CUDA cleanup between chunks.
    # 4 bands/force-batch trades some headroom for throughput. A chunk that
    # still OOMs is retried once at half the atom budget before its bands fail,
    # so lower this (down to 1) only for very large slab cells.
    "parallel_neb_max_bands": 4,
    # Atom budget per fused force batch (sum of n_images * n_atoms), applied
    # when parallel_neb_max_bands is cleared to None. Kept at/below the previous
    # 4-band x ~130-atom x 7-image path (~3.6k) so the on-disk memory-scaler
    # cache bucket is reused instead of re-probed.
    "parallel_neb_max_batch_atoms": 4000,
}

# Supported clusters: Pt/metal rearrangements on graphite routinely exceed the
# shared 1.25 Å * 3 cartesian gate (~6 Å). Widen the hard gate so near-minima
# pairs are not discarded before NEB.
_SURFACE_CLUSTER_TS_NEB_DEFAULTS: dict[str, Any] = {
    **_SURFACE_TS_NEB_DEFAULTS,
    "max_endpoint_mismatch": 2.5,
}

# Bare-slab vacancy/rearrangement NEBs: top-layer atoms can approach closely on
# linear/IDPP paths; keep a softer clash floor and a wider displacement gate.
_SURFACE_BARE_TS_NEB_DEFAULTS: dict[str, Any] = {
    **_SURFACE_TS_NEB_DEFAULTS,
    "max_endpoint_mismatch": 3.0,
    "neb_prescreen_clash_distance": 0.35,
    "neb_max_spurious_barrier": 50.0,
}

# Adsorbate paths need climb, stiffer springs, a hard geometric pair gate (Å),
# and a larger step budget. Keep neb_fmax and torchsim_fmax equal to the shared
# tolerance so ASE and TorchSim stay synced. Parallel multi-band NEB is on for
# every system type via get_ts_search_params.
_GAS_ADSORBATE_TS_NEB_DEFAULTS: dict[str, Any] = {
    **_GAS_TS_NEB_DEFAULTS,
    "neb_n_images": 7,
    "neb_spring_constant": 0.5,
    # Two-stage climb needs a larger budget than bare-cluster NEBs.
    "neb_steps": 4000,
    "neb_climb": True,
    "torchsim_max_steps": 4000,
    "max_endpoint_mismatch": 1.25,  # Å; also enables pre-NEB path gates
    # Explicitly preserve the adsorbate clash/prominence/barrier behavior (the
    # bare-gas base dict loosens these; do not inherit the loose values).
    "neb_prescreen_clash_distance": 0.7,
    "min_saddle_prominence": 0.40,
    "neb_max_spurious_barrier": 8.0,
    "binding_penetration_tolerance_a": 0.1,
    "layer_cluster_threshold_ang": 0.4,
    "neb_interpolation_bond_tolerance_a": 0.5,
}

_SURFACE_ADSORBATE_TS_NEB_DEFAULTS: dict[str, Any] = {
    **_SURFACE_TS_NEB_DEFAULTS,
    "neb_n_images": 7,
    "neb_spring_constant": 0.5,
    "neb_steps": 4000,
    "neb_climb": True,
    "torchsim_max_steps": 4000,
    "max_endpoint_mismatch": 1.5,  # Å; also enables pre-NEB path gates
    # Inherited surface defaults enable free in-plane Kabsch; that shifts
    # adsorbates off registry. Remap/MIC stay on via the surface base dict.
    "neb_surface_lattice_rotation": False,
    # Explicitly preserve the adsorbate clash/prominence/barrier behavior.
    "neb_prescreen_clash_distance": 0.7,
    "min_saddle_prominence": 0.40,
    "neb_max_spurious_barrier": 8.0,
    "binding_penetration_tolerance_a": 0.1,
    "layer_cluster_threshold_ang": 0.4,
    "neb_interpolation_bond_tolerance_a": 0.5,
}

# Bare-slab + adsorbate (no metal core): pair selection gates on adsorbate
# Cartesian hop, not core fingerprint. Graphite hollow/bridge site hops are
# ~2.5 Å, so keep a wider hard gate than the cluster+adsorbate 1.5 Å core gate.
_SURFACE_ONLY_ADSORBATE_TS_NEB_DEFAULTS: dict[str, Any] = {
    **_SURFACE_ADSORBATE_TS_NEB_DEFAULTS,
    "max_endpoint_mismatch": 3.0,
}

TS_DEFAULTS_BY_SYSTEM_TYPE: dict[SystemType, dict[str, Any]] = {
    "gas_cluster": dict(_GAS_TS_NEB_DEFAULTS),
    "gas_cluster_adsorbate": dict(_GAS_ADSORBATE_TS_NEB_DEFAULTS),
    "surface_cluster": dict(_SURFACE_CLUSTER_TS_NEB_DEFAULTS),
    "surface_cluster_adsorbate": dict(_SURFACE_ADSORBATE_TS_NEB_DEFAULTS),
    "surface": dict(_SURFACE_BARE_TS_NEB_DEFAULTS),
    "surface_adsorbate": dict(_SURFACE_ONLY_ADSORBATE_TS_NEB_DEFAULTS),
}


@cache
def _get_default_params_template() -> GLOptimizerParams:
    """Return the default SCGO parameter dictionary template.

    This is a cached function to avoid recreating the large dict on every call.
    Thread-safe and immutable pattern.
    """
    return {
        "validate_with_hessian": False,
        "calculator": "MACE",
        "seed": None,  # Will be overridden by function parameter
        "calculator_kwargs": {"model_name": "mace_matpes_0"},
        "fmax_threshold": DEFAULT_FMAX_THRESHOLD,
        "check_hessian": True,
        "imag_freq_threshold": 50.0,
        "n_jobs": DEFAULT_N_JOBS,  # Single CPU knob (see DEFAULT_N_JOBS); opt in with -1/-2
        "tag_final_minima": True,
        "connectivity_factor": CONNECTIVITY_FACTOR,  # Default for cluster validation
        "allow_cluster_fragmentation": False,
        "allow_adsorbate_surface_detachment": False,
        "enforce_adsorbate_subgraph_integrity": True,
        "freeze_adsorbate_internal_geometry": False,
        "fitness_strategy": "low_energy",  # Default: minimize energy
        "diversity_reference_db": None,  # For diversity strategy
        "diversity_max_references": 100,  # Performance limit
        "diversity_update_interval": 5,  # Update references every N iterations/generations
        "optimizer_params": {
            "simple": {
                "optimizer": "FIRE",
                "fmax": DEFAULT_FMAX_THRESHOLD,
                "niter": 1,
                "niter_local_relaxation": "auto",
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "comparator_tol": DEFAULT_COMPARATOR_TOL,
                "comparator_pair_cor_max": DEFAULT_PAIR_COR_MAX,
                "comparator_n_top": None,
            },
            "bh": {
                "optimizer": "FIRE",
                "temperature": 1.0,  # Metropolis energy scale (eV), ASE-style
                "fmax": DEFAULT_FMAX_THRESHOLD,
                "niter": "auto",
                "dr": 0.2,
                "move_fraction": 0.3,
                "niter_local_relaxation": "auto",
                "move_strategy": "random",
                "deduplicate": True,
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "comparator_tol": DEFAULT_COMPARATOR_TOL,
                "comparator_pair_cor_max": DEFAULT_PAIR_COR_MAX,
                "comparator_n_top": None,
                "fitness_strategy": None,  # None = inherit from top-level
                "diversity_reference_db": None,  # For diversity strategy
                "diversity_max_references": None,  # None = inherit from top-level
                "diversity_update_interval": None,  # None = inherit from top-level
            },
            "ga": {
                "optimizer": "FIRE",
                "population_size": "auto",
                "niter": "auto",
                "niter_local_relaxation": "auto",
                "mutation_probability": 0.4,
                "offspring_fraction": 0.5,
                "fmax": DEFAULT_FMAX_THRESHOLD,
                "vacuum": 10.0,
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "comparator_tol": DEFAULT_COMPARATOR_TOL,
                "comparator_pair_cor_max": DEFAULT_PAIR_COR_MAX,
                "comparator_n_top": None,
                "use_adaptive_mutations": True,
                "stagnation_trigger": 4,
                "stagnation_full_trigger": 8,
                "recovery_window": 2,
                "aggressive_burst_multiplier": 1.8,
                "max_mutation_probability": 0.65,
                "early_stopping_niter": 10,  # Stop if no improvement after N generations
                "n_jobs_population_init": None,  # None = inherit top-level "n_jobs"
                "n_jobs_offspring": None,  # None = inherit top-level "n_jobs"
                "batch_size": None,
                "relaxer": None,
                "fitness_strategy": None,  # None = inherit from top-level
                "diversity_reference_db": None,  # For diversity strategy
                "diversity_max_references": None,  # None = inherit from top-level
                "diversity_update_interval": None,  # None = inherit from top-level
            },
        },
    }


def default_calculator_kwargs(calculator: str) -> dict[str, Any]:
    """Return a fresh dict of default ``calculator_kwargs`` for ``calculator``.

    Unknown / non-ML calculators (e.g. EMT) get an empty dict.
    """
    calc_u = str(calculator).strip().upper()
    if calc_u == "MACE":
        return {"model_name": "mace_matpes_0"}
    if calc_u == "UMA":
        return {"model_name": "uma-s-1p2", "task_name": "oc25"}
    if calc_u == "UPET":
        return {"model_name": "pet-mad-s", "version": "1.5.0"}
    return {}


def get_ts_defaults(system_type: SystemType) -> dict[str, Any]:
    """Return a fresh copy of NEB knob defaults for one system type.

    Single source of truth read by :func:`~scgo.get_ts_search_params` and
    :func:`~scgo.utils.ts_runner_kwargs.coerce_ts_params_to_runner_kwargs`.
    """
    if system_type not in TS_DEFAULTS_BY_SYSTEM_TYPE:
        raise SCGOValidationError(
            f"Unsupported system_type={system_type!r}; expected one of "
            f"{sorted(TS_DEFAULTS_BY_SYSTEM_TYPE)!r}."
        )
    return dict(TS_DEFAULTS_BY_SYSTEM_TYPE[system_type])


def default_energy_gap_threshold(has_adsorbate: bool) -> float:
    """Endpoint energy-gap cap (eV): adsorbate NEBs pair across a tighter window.

    Single source for :func:`get_ts_search_params`,
    ``coerce_ts_params_to_runner_kwargs`` callers, and the runner-side fallback.
    """
    return 0.75 if has_adsorbate else 2.0


TS_POSTPROCESS_DEFAULTS: dict[str, Any] = {
    "dedupe_minima": True,
    "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
    "dedupe_ts": True,
    "ts_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
}
"""System-type-agnostic TS post-processing defaults (single specification).

Consumed by ``scgo.utils.ts_runner_kwargs.coerce_ts_params_to_runner_kwargs``;
``run_transition_state_search`` keeps importing ``DEFAULT_ENERGY_TOLERANCE``
directly for its signature.
"""


def get_default_params() -> GLOptimizerParams:
    """Return the default SCGO parameter dictionary for global optimization.

    Suitable for ``run_go`` / ``run_go_ts`` as ``params`` / ``go_params``; pass
    as-is or override keys (omitted keys are filled via
    :func:`scgo.utils.run_helpers.initialize_params`).

    CPU parallelism is a single top-level knob: ``params["n_jobs"]`` defaults to
    ``DEFAULT_N_JOBS`` (sequential) and is inherited by GA population init, GA
    offspring construction, and post-GO validation. Set it to ``-2`` (all but one
    CPU) or ``-1`` (all CPUs) to parallelize every CPU stage at once.
    """
    return copy.deepcopy(_get_default_params_template())


@cache
def default_params_top_level_keys() -> frozenset[str]:
    """Top-level keys of :func:`get_default_params` without a full deepcopy."""
    return frozenset(_get_default_params_template().keys())


def get_minimal_ga_params(
    seed: int | None = None,
    model_name: str | None = None,
) -> GLOptimizerParams:
    """Return compact GA-focused parameters derived from defaults.

    Sequential population init and offspring (explicit ``n_jobs_* = 1``) so
    runners stay easy to reason about; the top-level ``n_jobs`` default is
    also ``1``, so nothing parallelizes unless the caller opts in. Pass as-is to
    ``run_*`` or override keys; omitted keys are filled via
    :func:`scgo.utils.run_helpers.initialize_params`.
    """
    params = get_default_params()

    # Override GA-specific settings for faster/leaner runs
    params["optimizer_params"]["ga"].update(
        {
            "niter": "auto",
            "population_size": "auto",
            "mutation_probability": 0.4,
            "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
            "niter_local_relaxation": "auto",
            "n_jobs_population_init": 1,  # Sequential for runners (explicit control)
            "n_jobs_offspring": 1,  # Match init: avoid parallel offspring when init is serial
        }
    )

    # Set model name if provided
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name

    # Set seed if provided
    if seed is not None:
        params["seed"] = seed

    return params


def get_testing_params() -> GLOptimizerParams:
    """Return fast, low-cost parameters for tests (EMT, fewer iterations).

    Complete preset based on :func:`get_default_params`; pass as-is to ``run_*``
    or override keys (omitted keys are filled via
    :func:`scgo.utils.run_helpers.initialize_params`).
    """
    params = get_default_params()
    params["calculator"] = "EMT"
    params["calculator_kwargs"] = {}
    params["optimizer_params"]["simple"].update(
        {
            "niter": 1,
            "niter_local_relaxation": 2,
        }
    )
    params["optimizer_params"]["bh"].update(
        {
            "niter": 5,
            "niter_local_relaxation": 2,
        }
    )
    params["optimizer_params"]["ga"].update(
        {
            "population_size": 5,
            "offspring_fraction": 0.5,
            "niter": 2,
            "niter_local_relaxation": 2,
        }
    )
    return params


def _get_base_ga_benchmark_params(seed: int) -> GLOptimizerParams:
    """Return GA benchmark parameters derived from defaults."""
    params = get_default_params()
    params["seed"] = seed
    params["calculator_kwargs"]["default_dtype"] = "float32"
    # HPC: one knob for every CPU stage (population init, offspring, validation).
    params["n_jobs"] = -2

    # Customize GA parameters for benchmarking
    params["optimizer_params"]["ga"].update(
        {
            "fmax": 0.05,
            "niter_local_relaxation": 200,
            "niter": 10,
            "population_size": 50,
            "n_jobs_population_init": -2,  # HPC: all but one CPU (same as top-level)
            "n_jobs_offspring": -2,  # HPC: parallel offspring, aligned with init
        },
    )

    return params


def _attach_torchsim_relaxer(
    ga: dict[str, Any],
    calculator_kwargs: dict[str, Any],
    *,
    relaxer_kind: str,
    seed: int | None = None,
    max_steps: int | None,
    autobatcher: bool | None = None,
    expected_max_atoms: int | None = None,
    dtype: Any | None = None,
) -> None:
    """Set ``ga["relaxer"]`` to a calculator-backed TorchSimBatchRelaxer.

    One parameterized builder for the ``fairchem`` / ``upet`` / ``mace``
    relaxer kinds. ``dtype=None`` keeps the model default (callers pass
    ``torch.float32`` for speed); the UPET branch syncs its device from CUDA
    availability and mirrors ``expected_max_atoms`` into
    ``max_atoms_to_try``.
    """
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    fmax_val = float(ga.get("fmax", DEFAULT_FMAX_THRESHOLD))
    common_kwargs: dict[str, Any] = {
        "force_tol": fmax_val,
        "optimizer_name": "fire",
        "max_steps": max_steps,
        "dtype": dtype,
        "autobatcher": autobatcher,
        "expected_max_atoms": expected_max_atoms,
    }
    if relaxer_kind == "fairchem":
        ga["relaxer"] = TorchSimBatchRelaxer(
            model_kind="fairchem",
            fairchem_model_name=calculator_kwargs["model_name"],
            fairchem_task_name=calculator_kwargs.get("task_name"),
            **common_kwargs,
        )
    elif relaxer_kind == "upet":
        import torch

        on_cuda = torch.cuda.is_available()
        ga["relaxer"] = TorchSimBatchRelaxer(
            model_kind="upet",
            upet_model_name=calculator_kwargs.get("model_name"),
            upet_version=calculator_kwargs.get("version"),
            upet_checkpoint_path=calculator_kwargs.get("checkpoint_path"),
            upet_non_conservative=bool(
                calculator_kwargs.get("non_conservative", False)
            ),
            device=torch.device("cuda") if on_cuda else torch.device("cpu"),
            max_atoms_to_try=expected_max_atoms,
            **common_kwargs,
        )
    elif relaxer_kind == "mace":
        mace_model = calculator_kwargs.get("model_name", "mace_matpes_0")
        ga["relaxer"] = TorchSimBatchRelaxer(
            mace_model_name=mace_model,
            seed=seed,
            **common_kwargs,
        )
    else:
        raise SCGOValidationError(f"Unknown relaxer_kind={relaxer_kind!r}")


def _build_ga_calculator_params(
    calculator: str,
    *,
    effort: str,
    seed: int | None = None,
    model_name: str | None = None,
    calculator_kwargs: dict[str, Any] | None = None,
    relaxer_kind: str,
) -> GLOptimizerParams:
    """Build GA GO params for a calculator variant with a TorchSim relaxer.

    Shared by the UMA/UPET benchmark + default presets and the MACE TorchSim
    preset. ``effort="benchmark"`` starts from ``_get_base_ga_benchmark_params``
    (fixed 200-step local relaxation, ``n_jobs=-2``, float32 default dtype);
    ``effort="default"`` starts from :func:`get_default_params` (``"auto"`` local
    relaxation). The relaxer is attached via the matching ``_attach_*_torchsim_relaxer``
    helper with ``max_steps=None``; the GA assigns ``relaxer.max_steps`` from
    ``niter_local_relaxation`` at run time.
    """
    import torch

    if effort == "benchmark":
        base_seed = 0 if seed is None else int(seed)
        params = _get_base_ga_benchmark_params(base_seed)
    elif effort == "default":
        params = get_default_params()
    else:
        raise SCGOValidationError(f"Unknown effort={effort!r}")

    params["calculator"] = calculator
    if calculator_kwargs is not None:
        params["calculator_kwargs"] = dict(calculator_kwargs)
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name

    ga = params["optimizer_params"]["ga"]
    if effort == "benchmark":
        autobatcher: bool | None = True
        expected_max_atoms: int | None = 600
    else:
        autobatcher = None
        expected_max_atoms = None

    _attach_torchsim_relaxer(
        ga,
        params["calculator_kwargs"],
        relaxer_kind=relaxer_kind,
        seed=seed,
        max_steps=None,
        autobatcher=autobatcher,
        expected_max_atoms=expected_max_atoms,
        dtype=torch.float32,
    )

    return params


def get_uma_ga_benchmark_params(
    seed: int,
    *,
    model_name: str = "uma-s-1p2",
    uma_task: str = "oc25",
) -> GLOptimizerParams:
    """GA benchmark parameters matching ``_get_base_ga_benchmark_params`` but with UMA.

    Tuned for regression and profiling alongside the MACE TorchSim benchmark preset
    (:func:`get_torchsim_ga_params`): fixed local relaxation budget from the base
    preset (200 steps, not ``"auto"``), with autobatching and ``expected_max_atoms=600``
    for stable GPU memory behavior. Pass as-is to ``run_*`` or override keys.
    For general UMA runs with default GA ``"auto"`` local steps, use
    :func:`get_default_uma_params` instead.
    """
    return _build_ga_calculator_params(
        "UMA",
        effort="benchmark",
        seed=seed,
        model_name=model_name,
        calculator_kwargs={"model_name": model_name, "task_name": uma_task},
        relaxer_kind="fairchem",
    )


def get_default_uma_params() -> GLOptimizerParams:
    """Default SCGO parameters using the UMA calculator (fairchem-core).

    Pass as-is to ``run_*`` or override keys. For typical campaigns with default
    GA settings: ``niter_local_relaxation`` is ``"auto"`` and the TorchSim relaxer
    uses 250 max steps in that case. Autobatcher and memory-probe defaults follow
    :class:`~scgo.calculators.TorchSimBatchRelaxer` (``autobatcher`` None: CUDA on, CPU off). Use
    :func:`get_uma_ga_benchmark_params` when you need the same structure as the MACE
    benchmark preset (fixed local steps, explicit autobatcher/expected_max_atoms).
    """
    return _build_ga_calculator_params(
        "UMA",
        effort="default",
        model_name="uma-s-1p2",
        calculator_kwargs={"model_name": "uma-s-1p2", "task_name": "oc25"},
        relaxer_kind="fairchem",
    )


def get_upet_ga_benchmark_params(
    seed: int,
    *,
    model_name: str = "pet-mad-s",
    version: str = "1.5.0",
) -> GLOptimizerParams:
    """GA benchmark parameters using UPET with TorchSim (mirrors UMA benchmark preset).

    Tuned for regression alongside MACE/UMA benchmark presets: fixed local relaxation
    budget (200 steps), autobatching, and ``expected_max_atoms=600``. Pass as-is to
    ``run_*`` or override keys. For general UPET runs with default GA ``"auto"``
    local steps, use :func:`get_default_upet_params` instead.
    """
    return _build_ga_calculator_params(
        "UPET",
        effort="benchmark",
        seed=seed,
        model_name=model_name,
        calculator_kwargs={"model_name": model_name, "version": version},
        relaxer_kind="upet",
    )


def get_default_upet_params() -> GLOptimizerParams:
    """Default SCGO parameters using the UPET calculator (metatomic-torchsim).

    Pass as-is to ``run_*`` or override keys. Default model is ``pet-mad-s`` v1.5.0.
    For benchmark-style fixed local steps, use :func:`get_upet_ga_benchmark_params`.
    """
    return _build_ga_calculator_params(
        "UPET",
        effort="default",
        model_name="pet-mad-s",
        calculator_kwargs={"model_name": "pet-mad-s", "version": "1.5.0"},
        relaxer_kind="upet",
    )


def get_torchsim_ga_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str | None = None,
) -> GLOptimizerParams:
    """Return GO params using TorchSim relaxer (requires ``scgo[mace]``).

    Mirrors :func:`get_ts_search_params` call style by requiring ``system_type``
    and accepting ``surface_config`` / ``seed`` explicitly. Pass as-is to ``run_*``
    or override keys.
    When ``model_name`` is set, it is written to ``calculator_kwargs`` and the
    :class:`~scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer` uses the
    same MACE model name as the ASE calculator.
    """
    policy = get_system_policy(system_type)
    if policy.uses_surface and not isinstance(surface_config, SurfaceSystemConfig):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to be provided "
            "as a SurfaceSystemConfig when building go_params."
        )

    params = _build_ga_calculator_params(
        "MACE",
        effort="benchmark",
        seed=seed,
        model_name=model_name,
        relaxer_kind="mace",
    )
    if seed is None:
        params["seed"] = None

    # Keep TorchSim as the explicit relaxer backend, but let campaign scale
    # iteration budget and population size with composition complexity.
    params["optimizer_params"]["ga"]["niter"] = "auto"
    params["optimizer_params"]["ga"]["population_size"] = "auto"

    if policy.uses_surface:
        params["surface_config"] = surface_config

    return params


def get_diversity_params(
    reference_db_glob: str = "**/*.db",
    max_references: int = 100,
    update_interval: int = 5,
) -> GLOptimizerParams:
    """Return params for diversity-based optimization (reference DB, intervals).

    Pass as-is to ``run_*`` or override keys. ``reference_db_glob`` must match at
    least one database with reference structures when you run; there is no runtime
    check that the glob is non-empty. Values are written at top-level and into
    the BH/GA optimizer slots so slot ``None`` defaults cannot shadow them.
    """
    params = get_default_params()
    params["fitness_strategy"] = "diversity"
    params["diversity_reference_db"] = reference_db_glob
    params["diversity_max_references"] = max_references
    params["diversity_update_interval"] = update_interval
    # Also stamp algo slots so slot defaults cannot shadow top-level values.
    for algo in ("bh", "ga"):
        slot = params["optimizer_params"][algo]
        slot["diversity_reference_db"] = reference_db_glob
        slot["diversity_max_references"] = max_references
        slot["diversity_update_interval"] = update_interval

    # Diversity strategy works better with larger populations
    # Keep auto settings but note they will scale appropriately

    return params


def get_high_energy_params() -> GLOptimizerParams:
    """Return params that bias exploration toward high-energy structures.

    Pass as-is to ``run_*`` or override keys. Sets top-level ``fitness_strategy``
    to ``high_energy`` (used by BH and GA). Basin hopping additionally uses a
    higher temperature. GA hyperparameters are otherwise unchanged—override
    ``optimizer_params["ga"]`` if you need stronger exploration there.
    """
    params = get_default_params()
    params["fitness_strategy"] = "high_energy"

    # Higher Metropolis scale than the default 1.0 eV for more uphill acceptance.
    params["optimizer_params"]["bh"]["temperature"] = 2.0

    return params


def get_ts_search_params(
    calculator: str = "MACE",
    calculator_kwargs: dict[str, Any] | None = None,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """TS-only settings (NEB, calculator, pairing). Not merged with GO defaults.

    Suitable for ``run_ts_search`` / :func:`~scgo.run_go_ts` as ``ts_params``; pass as-is or
    override keys (omitted keys are filled via
    :func:`scgo.utils.run_helpers.initialize_ts_params`).

    For EMT or other non-TorchSim calculators, set ``use_torchsim=False`` on the
    returned dict before running.
    `system_type` is used to shape technical defaults.
    For surface system types, `surface_config` is required and stored in the
    returned dictionary so TS loading/validation always receives explicit slab
    context (no guessing).
    If ``seed`` is set, it is stored in the returned dict; :func:`~scgo.run_go_ts` / ``run_ts_*``
    require it to be consistent with ``go_params["seed"]`` and the ``seed=`` run argument.
    The ``connectivity_factor`` key sets the global connectivity threshold for cluster
    validation (default 1.4). It accepts a float or a dict of per-element and/or
    per-pair multipliers (see :mod:`scgo.system_types.connectivity_factor`); the
    same spec is used by GO algorithm gates, the ``run_trials`` final gate, and TS.

    NEB endpoint alignment is on by default (``neb_align_endpoints=True``). Surface
    system types also enable ``neb_interpolation_mic``, ``neb_surface_cell_remap``,
    and ``neb_surface_max_lattice_shift`` (default ``1``). Free in-plane
    ``neb_surface_lattice_rotation`` is on for the bare surface types
    (``surface_cluster``, ``surface``) and off for the adsorbate surface types
    (``surface_cluster_adsorbate``, ``surface_adsorbate``) to stay registry-safe.
    """
    policy = get_system_policy(system_type)
    if policy.uses_surface and not isinstance(surface_config, SurfaceSystemConfig):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to be provided "
            "as a SurfaceSystemConfig when building ts_params."
        )

    if not calculator_kwargs:
        calculator_kwargs = default_calculator_kwargs(calculator)

    params: dict[str, Any] = {
        "calculator": calculator,
        "calculator_kwargs": dict(calculator_kwargs),
        "connectivity_factor": CONNECTIVITY_FACTOR,
        # Bare-slab search and supported clusters often pass through temporarily
        # fragmented mobile geometries on NEB paths; keep adsorbate metal-cores
        # connected by default.
        "allow_cluster_fragmentation": bool(
            policy.slab_is_search_target or system_type == "surface_cluster"
        ),
        "allow_adsorbate_surface_detachment": False,
        "enforce_adsorbate_subgraph_integrity": True,
        "max_pairs": None,
        # Adsorbate NEBs need closer pairs; bare clusters keep the wider window.
        "energy_gap_threshold": default_energy_gap_threshold(policy.has_adsorbate),
        "similarity_tolerance": DEFAULT_COMPARATOR_TOL,
        "similarity_pair_cor_max": DEFAULT_TS_PAIR_COR_MAX,
        "use_torchsim": True,
        # Surface OOM safety: chunk parallel NEB + CUDA cleanup between chunks.
        # parallel_neb_max_bands defaults to 4 bands/force-batch (set in
        # _SURFACE_TS_NEB_DEFAULTS); lower it for very large slab cells.
        "use_parallel_neb": True,
        "dedupe_minima": True,
        "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
        "dedupe_ts": True,
        "ts_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
    }
    params.update(
        pair_selection_param_defaults(
            surface_aware=policy.uses_surface,
            adsorbate_aware=policy.has_adsorbate,
        )
    )
    params.update(get_ts_defaults(system_type))

    if policy.uses_surface:
        params["surface_config"] = surface_config

    if seed is not None:
        params["seed"] = int(seed)

    return params


def _apply_low_effort_ga_budget(params: GLOptimizerParams) -> GLOptimizerParams:
    """Shrink GA search budget for demos/CI; leave calculator/relaxer untouched."""
    params["n_jobs"] = 1
    params["optimizer_params"]["ga"].update(
        {
            "niter": _LOW_EFFORT_GA_NITER,
            "population_size": _LOW_EFFORT_GA_POPULATION_SIZE,
            "niter_local_relaxation": _LOW_EFFORT_GA_NITER_LOCAL_RELAXATION,
            "offspring_fraction": 0.5,
            "n_jobs_population_init": 1,
            "n_jobs_offspring": 1,
            "early_stopping_niter": 0,
            "write_timing_json": False,
            "detailed_timing": False,
        }
    )
    return params


def _stamp_surface_config_on_params(
    params: GLOptimizerParams,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> None:
    """Require and stamp top-level ``surface_config`` for surface system types."""
    policy = get_system_policy(system_type)
    if policy.uses_surface and not isinstance(surface_config, SurfaceSystemConfig):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to be provided "
            "as a SurfaceSystemConfig when building go_params."
        )
    if policy.uses_surface:
        params["surface_config"] = surface_config


def get_low_effort_torchsim_ga_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str | None = None,
) -> GLOptimizerParams:
    """Return reduced-budget GO params (~25% of production) for demos and CI.

    Thin wrapper over :func:`get_torchsim_ga_params`: the calculator, TorchSim
    relaxer, autobatcher, ``expected_max_atoms`` and float32 dtype are all
    inherited unchanged, so the *physics* matches a production run. Only the
    search budget shrinks:

    - ``niter`` / ``population_size`` are scaled to ~25% of the production
      benchmark reference in ``_get_base_ga_benchmark_params``
      (``niter=10`` / ``population_size=50``).
    - ``niter_local_relaxation`` drops to
      ``_LOW_EFFORT_GA_NITER_LOCAL_RELAXATION``. Surface system types clamp it
      back up to ``SURFACE_GA_MIN_LOCAL_RELAX_STEPS`` at run time, so supported
      and slab searches keep production-strength local relaxation.
    - Population init runs sequentially, early stopping is disabled (so the run
      length is deterministic), and timing JSON export is off.

    Used by ``examples/example_*.py`` and mirrored by the Kaggle GPU example
    matrix in ``tests/integration/test_gpu_examples_integration.py`` so the two
    cannot drift. Pass as-is to ``run_*`` or override individual keys.
    """
    params = get_torchsim_ga_params(
        system_type=system_type,
        surface_config=surface_config,
        seed=seed,
        model_name=model_name,
    )
    return _apply_low_effort_ga_budget(params)


def get_low_effort_upet_ga_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str = "pet-mad-s",
    version: str = "1.5.0",
) -> GLOptimizerParams:
    """Return reduced-budget GO params (~25%) for UPET demos and CI.

    Mirrors :func:`get_low_effort_torchsim_ga_params` but on the UPET calculator.
    The TorchSim relaxer is attached after ``model_name`` / ``version`` are set
    so it matches the ASE calculator PES. Only the search budget shrinks:

    - ``niter`` / ``population_size`` are scaled to ~25% of the production
      benchmark reference in ``_get_base_ga_benchmark_params``
      (``niter=10`` / ``population_size=50``).
    - ``niter_local_relaxation`` drops to
      ``_LOW_EFFORT_GA_NITER_LOCAL_RELAXATION``. Surface system types clamp it
      back up to ``SURFACE_GA_MIN_LOCAL_RELAX_STEPS`` at run time, so supported
      and slab searches keep production-strength local relaxation — the same
      clamp :func:`get_low_effort_torchsim_ga_params` relies on.
    - Population init runs sequentially, early stopping is disabled (so the run
      length is deterministic), and timing JSON export is off.

    Used by the Kaggle GPU example matrix
    (``tests/integration/test_gpu_examples_integration.py``) so UPET stays in
    lockstep with the MACE low-effort path. Pass as-is to ``run_*`` or override
    individual keys.
    """
    params = _build_ga_calculator_params(
        "UPET",
        effort="default",
        seed=seed,
        model_name=model_name,
        calculator_kwargs={"model_name": model_name, "version": version},
        relaxer_kind="upet",
    )
    if seed is not None:
        params["seed"] = int(seed)
    _stamp_surface_config_on_params(
        params, system_type=system_type, surface_config=surface_config
    )
    return _apply_low_effort_ga_budget(params)


def get_low_effort_uma_ga_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str = "uma-s-1p2",
    uma_task: str = "oc25",
) -> GLOptimizerParams:
    """Return reduced-budget GO params (~25%) for UMA demos and CI.

    Mirrors :func:`get_low_effort_upet_ga_params` but on the UMA calculator
    (fairchem). The FairChem-backed TorchSim relaxer is attached after
    ``model_name`` / ``uma_task`` are set so it matches the ASE calculator PES.
    Only the search budget shrinks to the same ~25% GA reduction as the
    other low-effort wrappers, with the surface local-relaxation clamp preserved
    at run time and timing JSON export off.

    Delivered for API / docs completeness, GitHub-Actions UMA smoke, and local
    runs — UMA is intentionally omitted from the Kaggle GPU matrix (HuggingFace
    auth for fairchem weights is unavailable there). Pass as-is to ``run_*`` or
    override individual keys.
    """
    params = _build_ga_calculator_params(
        "UMA",
        effort="default",
        seed=seed,
        model_name=model_name,
        calculator_kwargs={"model_name": model_name, "task_name": uma_task},
        relaxer_kind="fairchem",
    )
    if seed is not None:
        params["seed"] = int(seed)
    _stamp_surface_config_on_params(
        params, system_type=system_type, surface_config=surface_config
    )
    return _apply_low_effort_ga_budget(params)


def low_effort_neb_steps(system_type: SystemType) -> int:
    """Return the low-effort ``neb_steps`` budget for one system type.

    ~25% of the production budget, floored so every band can still converge to
    ``neb_fmax``. Gas system types use ``neb_steps="auto"`` in production (a
    composition-dependent value resolved at run time), which cannot be scaled
    here, so they fall back to the floor directly.
    """
    floor = _LOW_EFFORT_NEB_FLOOR
    base = get_ts_defaults(system_type)["neb_steps"]
    if not isinstance(base, int):
        # "auto": resolved from composition at run time, so only the floor applies.
        return floor
    return max(floor, round(base * _LOW_EFFORT_SCALE))


def get_low_effort_ts_search_params(
    calculator: str = "MACE",
    calculator_kwargs: dict[str, Any] | None = None,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return reduced-budget TS params (~25% of production) for demos and CI.

    Thin wrapper over :func:`get_ts_search_params`. Every physics knob is
    inherited unchanged — ``neb_n_images`` (7 for adsorbate types, 5 otherwise),
    ``neb_climb``, ``neb_fmax``, spring constant, MIC / cell remap / lattice
    rotation, ``max_endpoint_mismatch``, ``energy_gap_threshold``,
    ``pair_core_rms_max`` / ``pair_score_*``, and ``parallel_neb_max_bands`` — so
    a saddle found here is as valid as one from a production run. Only
    ``neb_steps`` / ``torchsim_max_steps`` shrink, to
    :func:`low_effort_neb_steps`, and timing JSON export is off.

    ``max_pairs`` is deliberately left at the preset default (``None`` = no cap):
    it is the main cost lever for the TS stage, so callers set it explicitly for
    their budget.
    """
    params = get_ts_search_params(
        calculator,
        calculator_kwargs,
        system_type=system_type,
        surface_config=surface_config,
        seed=seed,
    )
    neb_steps = low_effort_neb_steps(system_type)
    params["neb_steps"] = neb_steps
    params["torchsim_max_steps"] = neb_steps
    params["write_timing_json"] = False
    return params
