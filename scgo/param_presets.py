"""Parameter presets for SCGO campaigns."""

from __future__ import annotations

from typing import Any

from scgo.constants import (
    BOLTZMANN_K_EV_PER_K,
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SystemType, get_system_policy

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

__all__ = [
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    "get_default_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_torchsim_ga_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_ts_search_params",
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
                "system_type": "gas_cluster",
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
                "system_type": "gas_cluster",
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
                "system_type": "gas_cluster",
            },
        },
    }


def get_minimal_ga_params(
    seed: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return compact GA-focused parameters (merged with defaults).

    Uses sequential population init and offspring work (``n_jobs_*`` set to 1) so
    runners stay easy to reason about; merge with :func:`initialize_params` to
    fill other keys from :func:`get_default_params`.
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
                "system_type": "gas_cluster",
            },
            "bh": {
                "optimizer": "FIRE",
                "niter": 5,
                "dr": 0.2,
                "niter_local_relaxation": 2,
                "system_type": "gas_cluster",
            },
            "ga": {
                "optimizer": "FIRE",
                "population_size": 5,
                "offspring_fraction": 0.5,
                "niter": 2,
                "niter_local_relaxation": 2,
                "n_jobs_population_init": -2,  # Parallel for tests/benchmarks
                "system_type": "gas_cluster",
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


def _attach_fairchem_torchsim_relaxer(
    ga: dict[str, Any],
    calculator_kwargs: dict[str, Any],
    *,
    max_steps: int,
    autobatcher: bool | None = None,
    expected_max_atoms: int | None = None,
) -> None:
    """Set ``ga['relaxer']`` to a FairChem-backed :class:`TorchSimBatchRelaxer`."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    fmax_val = float(ga.get("fmax", 0.05))
    ga["relaxer"] = TorchSimBatchRelaxer(
        model_kind="fairchem",
        fairchem_model_name=calculator_kwargs["model_name"],
        fairchem_task_name=calculator_kwargs.get("task_name"),
        force_tol=fmax_val,
        optimizer_name="fire",
        max_steps=max_steps,
        dtype=None,  # TorchSim default per model; keep lazy/portable
        autobatcher=autobatcher,
        expected_max_atoms=expected_max_atoms,
    )


def get_uma_ga_benchmark_params(
    seed: int,
    *,
    model_name: str = "uma-s-1p2",
    uma_task: str = "oc25",
) -> dict[str, Any]:
    """GA benchmark parameters matching :func:`_get_base_ga_benchmark_params` but with UMA.

    Tuned for regression and profiling alongside the MACE TorchSim benchmark preset
    (:func:`get_torchsim_ga_params`): fixed local relaxation budget from the base
    preset (200 steps, not ``"auto"``), with autobatching and ``expected_max_atoms=600``
    for stable GPU memory behaviour. For general UMA runs with default GA
    ``"auto"`` local steps, use :func:`get_default_uma_params` instead.
    """
    params = _get_base_ga_benchmark_params(seed)
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {"model_name": model_name, "task_name": uma_task}

    ga = params["optimizer_params"]["ga"]
    niter_local = ga.get("niter_local_relaxation", 200)
    max_steps = 200 if niter_local == "auto" else int(niter_local)
    _attach_fairchem_torchsim_relaxer(
        ga,
        params["calculator_kwargs"],
        max_steps=max_steps,
        autobatcher=True,
        expected_max_atoms=600,
    )
    return params


def get_default_uma_params() -> dict[str, Any]:
    """Default SCGO parameters using the UMA calculator (fairchem-core).

    For typical campaigns with default GA settings: ``niter_local_relaxation`` is
    ``"auto"`` and the TorchSim relaxer uses 250 max steps in that case. Autobatcher
    and memory-probe defaults follow :class:`TorchSimBatchRelaxer` (``autobatcher``
    None: CUDA on, CPU off). Use :func:`get_uma_ga_benchmark_params` when you need
    the same structure as the MACE benchmark preset (fixed local steps, explicit
    autobatcher/expected_max_atoms).
    """
    params = get_default_params()
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {
        "model_name": "uma-s-1p2",
        "task_name": "oc25",
    }
    ga = params.get("optimizer_params", {}).get("ga", {})
    niter_local = ga.get("niter_local_relaxation", "auto")
    max_steps = 250 if niter_local == "auto" else int(niter_local)
    _attach_fairchem_torchsim_relaxer(
        ga,
        params["calculator_kwargs"],
        max_steps=max_steps,
        autobatcher=None,
        expected_max_atoms=None,
    )
    return params


def get_ts_search_params_uma(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    model_name: str = "uma-s-1p2",
    uma_task: str | None = "oc25",
    seed: int | None = None,
) -> dict[str, Any]:
    """TS preset for UMA (FairChem); same NEB defaults as MACE, for ``scgo[uma]``."""
    return get_ts_search_params(
        calculator="UMA",
        calculator_kwargs={"model_name": model_name, "task_name": uma_task},
        system_type=system_type,
        surface_config=surface_config,
        seed=seed,
    )


def get_torchsim_ga_params(
    seed: int, *, model_name: str | None = None
) -> dict[str, Any]:
    """Return GA params using TorchSim relaxer (requires ``scgo[mace]``).

    When ``model_name`` is set, it is written to ``calculator_kwargs`` and the
    :class:`~scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer` uses the
    same MACE model name as the ASE calculator.
    """
    from scgo.param_presets_torchsim import get_torchsim_ga_params_impl

    return get_torchsim_ga_params_impl(seed, model_name=model_name)


def get_diversity_params(
    reference_db_glob: str = "**/*.db",
    max_references: int = 100,
    update_interval: int = 5,
) -> dict[str, Any]:
    """Return params for diversity-based optimization (reference DB, intervals).

    ``reference_db_glob`` must match at least one database with reference
    structures when you run; there is no runtime check that the glob is non-empty.
    """
    params = get_default_params()
    params["fitness_strategy"] = "diversity"
    params["diversity_reference_db"] = reference_db_glob
    params["diversity_max_references"] = max_references
    params["diversity_update_interval"] = update_interval

    # Diversity strategy works better with larger populations
    # Keep auto settings but note they will scale appropriately

    return params


def get_high_energy_params() -> dict[str, Any]:
    """Return params that bias exploration toward high-energy structures.

    Sets top-level ``fitness_strategy`` to ``high_energy`` (used by BH and GA).
    Basin hopping additionally uses a higher temperature. GA hyperparameters are
    otherwise unchanged—override ``optimizer_params['ga']`` if you need stronger
    exploration there.
    """
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
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """TS-only settings (NEB, calculator, pairing). Not merged with GO defaults.

    For EMT or other non-TorchSim calculators, set ``use_torchsim=False`` on the
    returned dict before running.
    System-definition inputs (`system_type`, `surface_config`) are used only to
    shape technical defaults and are intentionally not stored in the output
    dictionary; high-level run APIs own those arguments.
    If ``seed`` is set, it is stored in the returned dict; :func:`run_go_ts` / ``run_ts_*``
    require it to be consistent with ``go_params['seed']`` and the ``seed=`` run argument.
    """
    policy = get_system_policy(system_type)

    if calculator_kwargs is None:
        calc_u = str(calculator).strip().upper()
        calculator_kwargs = {"model_name": "mace_matpes_0"} if calc_u == "MACE" else {}

    params: dict[str, Any] = {
        "calculator": calculator,
        "calculator_kwargs": dict(calculator_kwargs),
        "max_pairs": None,
        "energy_gap_threshold": 2.0,
        "similarity_tolerance": DEFAULT_COMPARATOR_TOL,
        "similarity_pair_cor_max": 0.1,
        "neb_n_images": 5,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": "auto",
        "neb_align_endpoints": True,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_mic": False,
        "use_torchsim": True,
        "torchsim_batch_size": 5,
        "torchsim_fmax": 0.05,
        "torchsim_max_steps": "auto",
        "use_parallel_neb": False,
        "neb_climb": False,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "dedupe_minima": True,
        "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
    }

    if policy.uses_surface:
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

    if seed is not None:
        params["seed"] = int(seed)

    return params
