"""Parameter presets for SCGO campaigns."""

from __future__ import annotations

from typing import Any, Literal

from scgo.constants import (
    BOLTZMANN_K_EV_PER_K,
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags

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

TsSearchRegime = Literal["gas", "surface"]

__all__ = [
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    "TsSearchRegime",
    "get_default_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_torchsim_ga_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_ts_search_params",
    "get_ts_run_kwargs",
    "get_default_uma_params",
    "get_ts_search_params_uma",
    "get_uma_ga_benchmark_params",
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


def get_uma_ga_benchmark_params(
    seed: int,
    *,
    model_name: str = "uma-s-1p1",
    task_name: str = "omat",
) -> dict[str, Any]:
    """GA benchmark parameters matching :func:`_get_base_ga_benchmark_params` but with UMA (ASE GA)."""
    params = _get_base_ga_benchmark_params(seed)
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {"model_name": model_name, "task_name": task_name}
    return params


def get_default_uma_params() -> dict[str, Any]:
    """Default SCGO parameters using the UMA calculator (fairchem-core)."""
    params = get_default_params()
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {
        "model_name": "uma-s-1p1",
        "task_name": "omat",
    }
    return params


def get_ts_search_params_uma(
    *,
    regime: TsSearchRegime = "gas",
    surface_config: SurfaceSystemConfig | None = None,
    model_name: str = "uma-s-1p1",
    task_name: str | None = "omat",
) -> dict[str, Any]:
    """TS preset for UMA: ASE NEB (no TorchSim); use with ``scgo[uma]`` only."""
    return get_ts_search_params(
        calculator="UMA",
        calculator_kwargs={"model_name": model_name, "task_name": task_name},
        regime=regime,
        surface_config=surface_config,
    )


def get_torchsim_ga_params(seed: int) -> dict[str, Any]:
    """Return GA params using TorchSim relaxer (requires ``scgo[mace]``)."""
    from scgo.param_presets_torchsim import get_torchsim_ga_params_impl

    return get_torchsim_ga_params_impl(seed)


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
    *,
    regime: TsSearchRegime = "gas",
    surface_config: SurfaceSystemConfig | None = None,
) -> dict[str, Any]:
    """Return TS search parameters (NEB settings and thresholds).

    Parameters
    ----------
    calculator:
        ``"MACE"`` enables TorchSim batched NEB when ``scgo[mace]`` is installed.
        ``"UMA"`` (and other non-MACE calculators) automatically use ASE NEB
        (``use_torchsim=False``) regardless of defaults below.
    regime:
        ``"gas"`` (nanoparticle / non-periodic): ``neb_n_images=5`` for a thicker
        initial band (fewer false endpoint-TS bands than 3 images on metals),
        ``energy_gap_threshold=2.0`` eV so pairs are not dropped when relaxed
        minima span more than 1 eV, and ``neb_climb=False`` (see preset comments).
        ``"surface"`` (slab + adsorbate, periodic in-plane): MIC interpolation,
        ``neb_n_images=5``, ``neb_climb=False``, and ``fmax=0.1 eV/Å``.
        Endpoint alignment stays off by default for fixed-slab/PBC (tuned for
        supported Pt5-on-NiO MACE NEB sweeps). Override any key via the returned
        dict before calling :func:`get_ts_run_kwargs`.
    surface_config:
        The same :class:`scgo.surface.config.SurfaceSystemConfig` instance wired into
        ``optimizer_params["ga"]["surface_config"]``. Stored on the returned dict
        and forwarded by :func:`get_ts_run_kwargs` so
        :func:`run_transition_state_search` reapplies
        ``attach_slab_constraints`` on NEB endpoints (frozen vs partially relaxed
        slab stays consistent with global optimization). Omit for gas-phase or when
        minima already carry the intended constraints.
    """
    if regime not in ("gas", "surface"):
        msg = f"regime must be 'gas' or 'surface', got {regime!r}"
        raise ValueError(msg)

    if calculator_kwargs is None:
        calculator_kwargs = {}

    params = get_default_params()
    params["calculator"] = calculator
    params["calculator_kwargs"] = calculator_kwargs

    # Add TS-specific parameters
    params.update(
        {
            "max_pairs": None,
            # Wider than 1.0 eV so distinct minima (e.g. relaxed Cu clusters) are not
            # all mutually filtered when energies span >1 eV.
            "energy_gap_threshold": 2.0,
            "similarity_tolerance": DEFAULT_COMPARATOR_TOL,
            # Match ``run_transition_state_search`` default (0.1 Å), not DEFAULT_PAIR_COR_MAX.
            "similarity_pair_cor_max": 0.1,
            "pair_priority_mode": "physics",
            # Pt5 gas sweep: `neb_climb=False` beat `neb_climb=True` on convergence.
            # Five images give a smoother initial path than three for 3D metal clusters
            # (fewer bands that collapse to an endpoint before retry).
            "neb_n_images": 5,
            "neb_spring_constant": 0.1,
            "neb_fmax": 0.05,
            # Allow automatic scaling for NEB iterations (uses auto_niter_ts via runner)
            "neb_steps": "auto",
            "neb_align_endpoints": True,
            # Disable default perturbation (perturb_sigma) — sweep showed 0.0 performed
            # better than 0.03 on Pt4.
            "neb_perturb_sigma": 0.0,
            "neb_interpolation_mic": False,
            # Enable retry/fallback when band slides to an endpoint (recommended).
            "use_torchsim": True,
            "torchsim_batch_size": 5,
            "torchsim_fmax": 0.05,
            # TorchSim max-steps may also be auto-resolved to match cluster size
            "torchsim_max_steps": "auto",
            # Whether to run multiple NEBs together using `ParallelNEBBatch`.
            # Default: False (opt-in).
            "use_parallel_neb": False,
            "neb_climb": False,
            "neb_interpolation_method": "idpp",
            "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
            "validate_ts_by_frequency": False,
            "imag_freq_threshold": 50.0,
            # TS minima deduplication defaults (keeps behavior consistent with
            # global-optimization `run_minima.py` and helpers.filter_unique_minima)
            "dedupe_minima": True,
            "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
        }
    )

    if regime == "surface":
        # Supported nanoparticle-on-slab (e.g. Pt5 on NiO): recent ASE/MACE benchmark
        # sweeps favor linear interpolation with MIC and non-climbing NEB at
        # ``n_images=5`` and ``fmax=0.1`` for better success/step tradeoff.
        params.update(
            {
                "neb_interpolation_mic": True,
                "neb_n_images": 5,
                "neb_spring_constant": 0.1,
                "neb_fmax": 0.1,
                "torchsim_fmax": 0.1,
                "neb_steps": 500,
                "torchsim_max_steps": 500,
                "neb_climb": False,
                "neb_interpolation_method": "linear",
                "neb_align_endpoints": False,
            }
        )

    if surface_config is not None:
        params["surface_config"] = surface_config

    us, up = resolve_ts_torchsim_flags(
        str(params["calculator"]),
        params.get("use_torchsim"),
        params.get("use_parallel_neb"),
        logger=None,
    )
    params["use_torchsim"] = us
    params["use_parallel_neb"] = up

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

    calc_name = str(ts_params["calculator"])
    use_ts, use_pn = resolve_ts_torchsim_flags(
        calc_name,
        ts_params.get("use_torchsim"),
        ts_params.get("use_parallel_neb"),
        logger=None,
    )

    kwargs = {
        "params": {
            "calculator": ts_params["calculator"],
            "calculator_kwargs": ts_params["calculator_kwargs"],
        },
        "use_torchsim": use_ts,
        "use_parallel_neb": use_pn,
        "torchsim_params": {
            "force_tol": ts_params.get("torchsim_fmax"),
            "max_steps": ts_params.get("torchsim_max_steps"),
        },
    }

    direct_keys = [
        "max_pairs",
        "energy_gap_threshold",
        "similarity_tolerance",
        "similarity_pair_cor_max",
        "pair_priority_mode",
        "neb_n_images",
        "neb_spring_constant",
        "neb_fmax",
        "neb_steps",
        "neb_align_endpoints",
        "neb_perturb_sigma",
        "surface_config",
    ]
    for key in direct_keys:
        kwargs[key] = ts_params.get(key)

    default_keys = {
        "neb_interpolation_mic": False,
        "dedupe_minima": True,
        "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
        "neb_climb": False,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "validate_ts_by_frequency": False,
        "imag_freq_threshold": 50.0,
    }
    for key, def_val in default_keys.items():
        kwargs[key] = ts_params.get(key, def_val)

    return kwargs
