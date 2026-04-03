"""Parameter presets for SCGO campaigns."""

from __future__ import annotations

from typing import Any

import torch

from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.constants import (
    BOLTZMANN_K_EV_PER_K,
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)

# Available MACE model names for use in calculator_kwargs["model_name"]
AVAILABLE_MACE_MODELS = [
    "mace_matpes_0",  # r2scan variant (default in MACE class)
    "mace_mp_small",  # Small MACE-MP
    "mace_mpa_medium",  # Medium MACE-MPA
    "mace_off_small",  # Small MACE-OFF
]

__all__ = [
    "AVAILABLE_MACE_MODELS",
    "get_default_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_torchsim_ga_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_ts_search_params",
    "get_ts_run_kwargs",
]


def get_default_params() -> dict[str, Any]:
    """Return the default SCGO parameter dictionary."""
    return {
        "validate_with_hessian": False,
        "calculator": "MACE",
        "seed": None,  # Will be overridden by function parameter
        "calculator_kwargs": {"model_name": "mace_matpes_0"},
        "fmax_threshold": 0.05,
        "check_hessian": True,
        "imag_freq_threshold": 50.0,
        "n_trials": 1,
        "tag_final_minima": True,
        "fitness_strategy": "low_energy",  # Default: minimize energy
        "diversity_reference_db": None,  # For diversity strategy
        "diversity_max_references": 100,  # Performance limit
        "diversity_update_interval": 5,  # Update references every N iterations/generations
        "optimizer_params": {
            "simple": {
                "optimizer": "FIRE",
                "fmax": 0.05,
                "niter": 1,
                "niter_local_relaxation": "auto",
            },
            "bh": {
                "optimizer": "FIRE",
                "temperature": 500 * 8.617e-5,  # 500K in eV
                "fmax": 0.05,
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
                "diversity_max_references": 100,  # Performance limit
                "diversity_update_interval": 5,  # Update references every N iterations
            },
            "ga": {
                "optimizer": "FIRE",
                "use_torchsim": True,  # True=TorchSim GA for ML (CPU/GPU); False=force ASE GA
                "population_size": "auto",
                "niter": "auto",
                "niter_local_relaxation": "auto",
                "mutation_probability": 0.4,
                "offspring_fraction": 0.5,
                "fmax": 0.05,
                "vacuum": 10.0,
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "use_adaptive_mutations": True,
                "stagnation_trigger": 4,
                "stagnation_full_trigger": 8,
                "recovery_window": 2,
                "aggressive_burst_multiplier": 1.8,
                "max_mutation_probability": 0.65,
                "early_stopping_niter": 10,  # Stop if no improvement after N generations
                "n_jobs_population_init": -2,  # Parallel batch init: -2 = all CPUs except one
                "batch_size": None,
                "relaxer": None,
                "fitness_strategy": None,  # None = inherit from top-level
                "diversity_reference_db": None,  # For diversity strategy
                "diversity_max_references": 100,  # Performance limit
                "diversity_update_interval": 5,  # Update references every N generations
            },
        },
    }


def get_minimal_ga_params(
    seed: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return compact GA-focused parameters (merged with defaults)."""
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
        }
    )

    # Set model name if provided
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name

    # Set seed if provided
    if seed is not None:
        params["seed"] = seed

    return params


def get_testing_params() -> dict[str, Any]:
    """Return fast, low-cost parameters for tests (EMT, fewer iterations)."""
    return {
        "validate_with_hessian": False,
        "calculator": "EMT",
        "seed": None,  # Will be overridden by function parameter
        "optimizer_params": {
            "simple": {
                "optimizer": "FIRE",
                "fmax": 0.05,
                "niter": 1,
                "niter_local_relaxation": 2,
            },
            "bh": {
                "optimizer": "FIRE",
                "niter": 5,
                "dr": 0.2,
                "niter_local_relaxation": 2,
            },
            "ga": {
                "optimizer": "FIRE",
                "use_torchsim": False,  # False=force ASE GA for ML (testing); True=TorchSim GA for ML
                "population_size": 5,
                "offspring_fraction": 0.5,
                "niter": 2,
                "niter_local_relaxation": 2,
                "n_jobs_population_init": -2,  # Parallel for tests/benchmarks
            },
        },
    }


def _get_base_ga_benchmark_params(seed: int) -> dict[str, Any]:
    """Return GA benchmark parameters derived from defaults."""
    params = get_default_params()
    params["seed"] = seed
    params["calculator_kwargs"]["default_dtype"] = "float32"

    # Customize GA parameters for benchmarking
    params["optimizer_params"]["ga"].update(
        {
            "fmax": 0.05,
            "niter_local_relaxation": 200,
            "niter": 10,
            "population_size": 50,
            "n_jobs_population_init": -2,  # Parallel for benchmarks
        },
    )

    return params


def get_torchsim_ga_params(seed: int) -> dict[str, Any]:
    """Return GA params using TorchSim relaxer."""
    params = _get_base_ga_benchmark_params(seed)

    # Keep GA relaxation settings consistent across both modes
    fmax_val = params["optimizer_params"]["ga"]["fmax"]
    niter_local = params["optimizer_params"]["ga"]["niter_local_relaxation"]

    # Extract the same settings for TorchSim relaxer
    params["optimizer_params"]["ga"].update(
        {
            # TorchSim optimized for GPU: float32, no autobatching overhead for tiny batches
            "relaxer": TorchSimBatchRelaxer(
                force_tol=fmax_val,
                optimizer_name="fire",
                mace_model_name="mace_matpes_0",
                seed=seed,
                # Match baseline GA's max steps
                max_steps=niter_local,
                # Use float32 for ~10-30x faster GPU inference
                dtype=torch.float32,
                # Use binning strategy for optimal batch sizes
                autobatch_strategy="binning",
                # Optional: enable torch.compile for faster MACE (adds startup cost)
                compile_model=False,
            ),
            # Leave batch_size unset so all unrelaxed candidates are batched together
            # use_torchsim defaults True: ML calculators use TorchSim GA (CPU or GPU)
        },
    )

    return params


def get_diversity_params(
    reference_db_glob: str = "**/*.db",
    max_references: int = 100,
    update_interval: int = 5,
) -> dict[str, Any]:
    """Return params for diversity-based optimization (reference DB, intervals)."""
    params = get_default_params()
    params["fitness_strategy"] = "diversity"
    params["diversity_reference_db"] = reference_db_glob
    params["diversity_max_references"] = max_references
    params["diversity_update_interval"] = update_interval

    # Diversity strategy works better with larger populations
    # Keep auto settings but note they will scale appropriately

    return params


def get_high_energy_params() -> dict[str, Any]:
    """Return params that bias exploration toward high-energy structures."""
    params = get_default_params()
    params["fitness_strategy"] = "high_energy"

    # Increase temperature for BH to accept high-energy moves
    # Default is 500K, increase to 1000K for better high-energy exploration
    params["optimizer_params"]["bh"]["temperature"] = (
        1000 * BOLTZMANN_K_EV_PER_K
    )  # 1000K

    return params


def get_ts_search_params(
    calculator: str = "MACE",
    calculator_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return TS search parameters (NEB settings and thresholds)."""
    if calculator_kwargs is None:
        calculator_kwargs = {}

    params = get_default_params()
    params["calculator"] = calculator
    params["calculator_kwargs"] = calculator_kwargs

    # Add TS-specific parameters
    params.update(
        {
            "max_pairs": None,
            "energy_gap_threshold": 1.0,
            "similarity_tolerance": DEFAULT_COMPARATOR_TOL,
            "similarity_pair_cor_max": DEFAULT_PAIR_COR_MAX,
            # Sweep results: prefer 3 images, small spring, no interior perturbation
            "neb_n_images": 3,
            # Use a small spring constant to improve success rate while keeping
            # the band flexible (empirically best for Pt4 sweep).
            "neb_k": 0.05,
            "neb_fmax": 0.05,
            # Allow automatic scaling for NEB iterations (uses auto_niter_ts via runner)
            "neb_maxsteps": "auto",
            "neb_align_endpoints": True,
            # Disable default perturbation (perturb_sigma) — sweep showed 0.0 performed
            # better than 0.03 on Pt4.
            "neb_perturb_sigma": 0.0,
            "neb_interpolation_mic": False,
            # Enable retry/fallback when band slides to an endpoint (recommended).
            "neb_retry_on_endpoint": True,
            "use_torchsim": True,
            "torchsim_batch_size": 5,
            "torchsim_fmax": 0.05,
            # TorchSim max-steps may also be auto-resolved to match cluster size
            "torchsim_maxsteps": "auto",
            # Whether to run multiple NEBs together using `ParallelNEBBatch`.
            # Default: False (opt-in).
            "use_parallel_neb": False,
            "neb_climb": True,
            "neb_interpolation_method": "idpp",
            "validate_ts_by_frequency": False,
            "imag_freq_threshold": 50.0,
            # TS minima deduplication defaults (keeps behavior consistent with
            # global-optimization `run_minima.py` and helpers.filter_unique_minima)
            "dedupe_minima": True,
            "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
        }
    )

    return params


def get_ts_run_kwargs(ts_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a kwargs dict ready to pass to `run_transition_state_search`.

    This prevents callers from having to unpack many individual TS/NEB keys.

    Example:
        ts_params = get_ts_search_params()
        results = run_transition_state_search(composition, **get_ts_run_kwargs(ts_params))
    """
    if ts_params is None:
        ts_params = get_ts_search_params()

    return {
        "params": {
            "calculator": ts_params["calculator"],
            "calculator_kwargs": ts_params["calculator_kwargs"],
        },
        "max_pairs": ts_params.get("max_pairs"),
        "energy_gap_threshold": ts_params.get("energy_gap_threshold"),
        "similarity_tolerance": ts_params.get("similarity_tolerance"),
        "similarity_pair_cor_max": ts_params.get("similarity_pair_cor_max"),
        "neb_n_images": ts_params.get("neb_n_images"),
        "neb_spring_constant": ts_params.get("neb_k"),
        "neb_fmax": ts_params.get("neb_fmax"),
        "neb_steps": ts_params.get("neb_maxsteps"),
        "neb_align_endpoints": ts_params.get("neb_align_endpoints"),
        "neb_perturb_sigma": ts_params.get("neb_perturb_sigma"),
        "neb_interpolation_mic": ts_params.get("neb_interpolation_mic", False),
        "neb_retry_on_endpoint": ts_params.get("neb_retry_on_endpoint", True),
        "use_torchsim": ts_params.get("use_torchsim"),
        # TS-specific post-processing knobs
        "dedupe_minima": ts_params.get("dedupe_minima", True),
        "minima_energy_tolerance": ts_params.get(
            "minima_energy_tolerance", DEFAULT_ENERGY_TOLERANCE
        ),
        "torchsim_params": {
            # Map user-facing ts params to TorchSimBatchRelaxer-compatible keys
            "force_tol": ts_params.get("torchsim_fmax"),
            "max_steps": ts_params.get("torchsim_maxsteps"),
        },
        "use_parallel_neb": ts_params.get("use_parallel_neb", False),
        "neb_climb": ts_params.get("neb_climb", True),
        "neb_interpolation_method": ts_params.get("neb_interpolation_method", "idpp"),
        "validate_ts_by_frequency": ts_params.get("validate_ts_by_frequency", False),
        "imag_freq_threshold": ts_params.get("imag_freq_threshold", 50.0),
    }
