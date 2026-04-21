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
    "build_one_element_go_ts_bundle",
    "pt5_gas_go_ts_defaults",
    "pt5_graphite_go_ts_defaults",
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
                "n_jobs_offspring": -2,  # Parallel default aligned with n_jobs_population_init
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
    model_name: str = "uma-s-1p2",
    uma_task: str = "oc25",
) -> dict[str, Any]:
    """GA benchmark parameters matching :func:`_get_base_ga_benchmark_params` but with UMA.

    Installs a fairchem-backed :class:`TorchSimBatchRelaxer` so the benchmark
    runs on the TorchSim GA path (same code as the MACE preset), enabling
    apples-to-apples profiling between MACE and UMA backends.
    """
    params = _get_base_ga_benchmark_params(seed)
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {"model_name": model_name, "task_name": uma_task}

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    ga = params["optimizer_params"]["ga"]
    fmax_val = float(ga.get("fmax", 0.05))
    niter_local = ga.get("niter_local_relaxation", 200)
    max_steps = 200 if niter_local == "auto" else int(niter_local)
    ga["relaxer"] = TorchSimBatchRelaxer(
        model_kind="fairchem",
        fairchem_model_name=model_name,
        fairchem_task_name=uma_task,
        force_tol=fmax_val,
        optimizer_name="fire",
        max_steps=max_steps,
        dtype=None,
        autobatcher=True,
        expected_max_atoms=600,
    )
    return params


def get_default_uma_params() -> dict[str, Any]:
    """Default SCGO parameters using the UMA calculator (fairchem-core)."""
    params = get_default_params()
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {
        "model_name": "uma-s-1p2",
        "task_name": "oc25",
    }
    # Default UMA GA to TorchSim-backed relaxations (requires TorchSim + FairChem support).
    # Lazy import: do not require TorchSim unless this preset is used.
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    ga = params.get("optimizer_params", {}).get("ga", {})
    fmax_val = float(ga.get("fmax", 0.05))
    niter_local = ga.get("niter_local_relaxation", "auto")
    max_steps = 250 if niter_local == "auto" else int(niter_local)
    ga["relaxer"] = TorchSimBatchRelaxer(
        model_kind="fairchem",
        fairchem_model_name=params["calculator_kwargs"]["model_name"],
        fairchem_task_name=params["calculator_kwargs"].get("task_name"),
        force_tol=fmax_val,
        optimizer_name="fire",
        max_steps=max_steps,
        dtype=None,  # TorchSim default per model; keep lazy/portable
    )
    return params


def get_ts_search_params_uma(
    *,
    regime: TsSearchRegime = "gas",
    surface_config: SurfaceSystemConfig | None = None,
    model_name: str = "uma-s-1p2",
    uma_task: str | None = "oc25",
) -> dict[str, Any]:
    """TS preset for UMA (FairChem); same NEB defaults as MACE, for ``scgo[uma]``."""
    return get_ts_search_params(
        calculator="UMA",
        calculator_kwargs={"model_name": model_name, "task_name": uma_task},
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
        ``"MACE"`` / ``"UMA"``: with ``use_torchsim=True`` (the default below),
        NEB uses TorchSim when the matching extra is installed
        (``scgo[mace]`` or ``scgo[uma]``); :func:`get_ts_run_kwargs` resolves
        flags and raises if TorchSim was requested but the stack is missing.
        Calculators that do not support TorchSim NEB (e.g. ``"EMT"``) must set
        ``use_torchsim=False`` before calling :func:`get_ts_run_kwargs`.
    regime:
        ``"gas"`` (nanoparticle / non-periodic): ``neb_n_images=5`` for a thicker
        initial band (fewer false endpoint-TS bands than 3 images on metals),
        ``energy_gap_threshold=2.0`` eV so pairs are not dropped when relaxed
        minima span more than 1 eV, and ``neb_climb=False`` (see preset comments).
        ``"surface"`` (slab + adsorbate, periodic in-plane): MIC interpolation,
        ``neb_interpolation_method=idpp``, ``neb_n_images=5``, ``neb_climb=False``,
        ``fmax=0.1 eV/Å``, and ``neb_align_endpoints=False`` (Pt5-on-graphite NEB sweeps;
        see ``benchmark/benchmark_neb_knobs.py``).
        Override any key via the returned dict before calling
        :func:`get_ts_run_kwargs`, which applies
        :func:`~scgo.utils.torchsim_policy.resolve_ts_torchsim_flags` to set
        effective ``use_torchsim`` / ``use_parallel_neb`` on the runner kwargs.
    surface_config:
        The same :class:`scgo.surface.config.SurfaceSystemConfig` instance wired into
        ``optimizer_params["ga"]["surface_config"]``. Stored on the returned dict
        and forwarded by :func:`get_ts_run_kwargs` so
        :func:`~scgo.runner_api.run_transition_state_search` (``ts_kwargs=``) reapplies
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
            # TS minima deduplication defaults (keeps behavior consistent with
            # global optimization in :mod:`scgo.run_minima` / :mod:`scgo.runner_api`
            # and helpers.filter_unique_minima)
            "dedupe_minima": True,
            "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
        }
    )

    if regime == "surface":
        # Pt5-on-graphite (MACE + TorchSim NEB when ``scgo[mace]`` is installed),
        # ``benchmark_neb_knobs.py``:
        # (1) ``neb_n_images`` in {3, 5, 7} × ``neb_climb`` with linear + MIC: ``n=5``,
        #     ``climb=False`` best on composite; climbing hurt.
        # (2) linear vs idpp × ``neb_align_endpoints`` (fixed ``n=5``, ``climb=False``):
        #     idpp + align False beat linear + align False on success and composite;
        #     align True hurt both interpolators.
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
                "neb_interpolation_method": "idpp",
                "neb_align_endpoints": False,
            }
        )

    if surface_config is not None:
        params["surface_config"] = surface_config

    return params


def get_ts_run_kwargs(ts_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a dict suitable for :func:`~scgo.runner_api.run_transition_state_search` ``ts_kwargs``.

    Example:

        from scgo.runner_api import run_transition_state_search

        ts_params = get_ts_search_params()
        results = run_transition_state_search(
            composition, params=go_params, ts_kwargs=get_ts_run_kwargs(ts_params)
        )
    """
    if ts_params is None:
        ts_params = get_ts_search_params()

    calc_name = str(ts_params["calculator"])
    use_ts, use_pn = resolve_ts_torchsim_flags(
        calc_name,
        ts_params.get("use_torchsim"),
        ts_params.get("use_parallel_neb"),
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
    # TorchSim model selection: choose the correct model family lazily at runtime.
    if str(ts_params.get("calculator", "")).strip().upper() == "UMA":
        ck = ts_params.get("calculator_kwargs", {}) or {}
        kwargs["torchsim_params"].update(
            {
                "model_kind": "fairchem",
                "fairchem_model_name": ck.get("model_name", "uma-s-1p2"),
                "fairchem_task_name": ck.get("task_name", "oc25"),
            }
        )

    direct_keys = [
        "max_pairs",
        "energy_gap_threshold",
        "similarity_tolerance",
        "similarity_pair_cor_max",
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
    }
    for key, def_val in default_keys.items():
        kwargs[key] = ts_params.get(key, def_val)

    return kwargs


def build_one_element_go_ts_bundle(
    *,
    backend: str,
    seed: int,
    niter: int,
    population_size: int,
    max_pairs: int,
    regime: TsSearchRegime = "gas",
    model_name: str | None = None,
    uma_task: str = "oc25",
    surface_config: SurfaceSystemConfig | None = None,
    ga_n_jobs_population_init: int | None = None,
    ga_batch_size: int | None = None,
) -> dict[str, Any]:
    """Build GA params + TS kwargs for one-element GO->TS example workflows."""
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"mace", "uma"}:
        raise ValueError("backend must be 'mace' or 'uma'")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if niter < 1:
        raise ValueError("niter must be >= 1")
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")

    if backend_norm == "uma":
        selected_model_name = model_name or "uma-s-1p2"
        ga_params = get_uma_ga_benchmark_params(
            seed,
            model_name=selected_model_name,
            uma_task=uma_task,
        )
        ts_params = get_ts_search_params_uma(
            regime=regime,
            surface_config=surface_config,
            model_name=selected_model_name,
            uma_task=uma_task,
        )
    else:
        ga_params = get_torchsim_ga_params(seed=seed)
        ga_params["calculator"] = "MACE"
        if model_name is not None:
            ga_params["calculator_kwargs"]["model_name"] = model_name

        ts_params = get_ts_search_params(
            regime=regime,
            surface_config=surface_config,
        )
        if model_name is not None:
            ts_params["calculator_kwargs"]["model_name"] = model_name

    ga = ga_params["optimizer_params"]["ga"]
    ga["niter"] = niter
    ga["population_size"] = population_size
    if surface_config is not None:
        ga["surface_config"] = surface_config
    if ga_n_jobs_population_init is not None:
        ga["n_jobs_population_init"] = ga_n_jobs_population_init
    if ga_batch_size is not None:
        ga["batch_size"] = ga_batch_size

    ts_kwargs = get_ts_run_kwargs(ts_params)
    ts_kwargs["max_pairs"] = max_pairs

    return {
        "ga_params": ga_params,
        "ts_kwargs": ts_kwargs,
        "backend": backend_norm,
    }


def pt5_gas_go_ts_defaults() -> dict[str, int]:
    """Numeric GO+TS defaults for the Pt5 gas-phase production runner example."""
    return {"niter": 10, "population_size": 50, "max_pairs": 15}


def pt5_graphite_go_ts_defaults() -> dict[str, Any]:
    """GO+TS defaults for the Pt5 graphite-surface production runner example."""
    return {
        "niter": 6,
        "population_size": 24,
        "max_pairs": 10,
        "ga_n_jobs_population_init": -2,
        "ga_batch_size": 4,
    }
