"""Flat TS dict → kwargs for :func:`~scgo.ts_search.run_transition_state_search`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.param_presets import TS_POSTPROCESS_DEFAULTS, get_ts_defaults
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import ConnectivityFactorInput, SystemType, get_system_policy
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags


@dataclass(frozen=True)
class NebRunConfig:
    """Shared NEB geometry / validation knobs for serial and parallel runners."""

    neb_n_images: int
    neb_spring_constant: float
    neb_fmax: float
    neb_steps: int | str
    neb_climb: bool
    neb_interpolation_method: str
    neb_align_endpoints: bool
    neb_perturb_sigma: float
    neb_interpolation_mic: bool
    neb_tangent_method: str
    neb_surface_cell_remap: bool
    neb_surface_lattice_rotation: bool
    neb_surface_max_lattice_shift: int
    n_slab: int
    n_core_mobile: int | None
    n_adsorbate_mobile: int | None
    adsorbate_fragment_lengths: list[int] | None
    max_endpoint_mismatch: float | None
    neb_prescreen_clash_distance: float
    min_saddle_prominence: float
    neb_max_spurious_barrier: float
    layer_cluster_threshold_ang: float
    neb_interpolation_bond_tolerance_a: float
    adsorbate_definition: Any | None
    connectivity_factor: ConnectivityFactorInput | None
    allow_cluster_fragmentation: bool
    allow_adsorbate_surface_detachment: bool
    enforce_adsorbate_subgraph_integrity: bool
    system_type: SystemType
    surface_config: SurfaceSystemConfig | None
    torchsim_params: dict[str, Any] | None
    # Per-config connectivity factor source (ClusterAdsorbateConfig). When a
    # NebRunConfig is built with connectivity_factor=None, this is honored by the
    # validation gateway (precedence: explicit value → cluster_adsorbate_config →
    # surface_config → module default). Optional for parity with the GO gate.
    cluster_adsorbate_config: Any | None = None
    # Atom budget (sum of n_images * n_atoms) for one fused parallel-NEB force
    # batch. Applied together with ``parallel_neb_max_bands`` (both bounds hold);
    # ``None`` means "no atom budget" (all bands in one batch).
    parallel_neb_max_batch_atoms: int | None = None


# Keys without per-system defaults: pass through as-is (None when missing is
# fine for the runner).
_TS_PARAM_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "write_timing_json",
    "max_pairs",
    "energy_gap_threshold",
    "similarity_tolerance",
    "similarity_pair_cor_max",
    "pair_core_rms_max",
    "pair_score_gap_center",
    "pair_score_gap_width",
    "pair_score_cum_scale",
    "pair_score_mismatch_scale",
    "pair_score_core_rms_scale",
    "pair_score_w_gap",
    "pair_score_w_distinct",
    "pair_score_w_mismatch",
    "pair_score_w_core",
    "connectivity_factor",
    "cluster_adsorbate_config",
    "allow_cluster_fragmentation",
    "allow_adsorbate_surface_detachment",
    "enforce_adsorbate_subgraph_integrity",
)

# NEB knobs that vary per system_type: fall back to the defaults table.
# torchsim_* defaults are consumed only in torchsim_params above; they are
# not valid top-level kwargs for run_transition_state_search.
_TS_PARAM_NEB_KEYS: tuple[str, ...] = (
    "neb_align_endpoints",
    "neb_interpolation_mic",
    "neb_n_images",
    "neb_spring_constant",
    "neb_fmax",
    "neb_steps",
    "neb_climb",
    "neb_perturb_sigma",
    "neb_interpolation_method",
    "neb_tangent_method",
    "neb_surface_cell_remap",
    "neb_surface_lattice_rotation",
    "neb_surface_max_lattice_shift",
    "max_endpoint_mismatch",
    "neb_prescreen_clash_distance",
    "min_saddle_prominence",
    "neb_max_spurious_barrier",
    # Consumed by the post-NEB surface geometry gate, not the NebRunConfig.
    "binding_penetration_tolerance_a",
    "layer_cluster_threshold_ang",
    "neb_interpolation_bond_tolerance_a",
    "parallel_neb_max_bands",
    "parallel_neb_max_batch_atoms",
)

# Generic (system-type-agnostic) defaults: single source in
# ``scgo.param_presets.TS_POSTPROCESS_DEFAULTS``.
_TS_PARAM_GENERIC_DEFAULTS: dict[str, Any] = dict(TS_POSTPROCESS_DEFAULTS)

# Every key ``coerce_ts_params_to_runner_kwargs`` understands: the canonical
# :func:`~scgo.param_presets.get_ts_search_params` output plus the explicit
# runner-only keys. Anything else is a typo and rejected up front.
_TS_PARAM_ALLOWLIST: frozenset[str] = (
    frozenset(_TS_PARAM_PASSTHROUGH_KEYS)
    | frozenset(_TS_PARAM_NEB_KEYS)
    | frozenset(_TS_PARAM_GENERIC_DEFAULTS)
    | {
        "calculator",
        "calculator_kwargs",
        "use_torchsim",
        "use_parallel_neb",
        "torchsim_params",
        "torchsim_fmax",
        "torchsim_max_steps",
        "surface_config",
        "seed",
        "tag_ts_in_db",
    }
)


def coerce_ts_params_to_runner_kwargs(
    ts_params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    surface_config: Any | None = None,
) -> dict[str, Any]:
    """Map initialized :func:`~scgo.get_ts_search_params` output to runner kwargs.

    Expects a fully initialized flat TS dict (see
    :func:`scgo.utils.run_helpers.initialize_ts_params`). Missing NEB knobs still
    fall back to per-system defaults in ``TS_DEFAULTS_BY_SYSTEM_TYPE`` as a safety net.
    """
    if ts_params is None:
        raise SCGOValidationError(
            "ts_params is required. Build with get_ts_search_params(system_type=...)."
        )

    calc_name = str(ts_params["calculator"])
    if system_type not in SystemType.__args__:
        raise SCGOValidationError(
            f"Unsupported system_type={system_type!r}; "
            f"expected one of {SystemType.__args__!r}."
        )
    unknown_keys = sorted(set(ts_params) - _TS_PARAM_ALLOWLIST)
    if unknown_keys:
        raise SCGOValidationError(
            f"Unexpected ts_params keys: {unknown_keys}. Expected a subset "
            f"of: {sorted(_TS_PARAM_ALLOWLIST)}."
        )
    ts_defaults = get_ts_defaults(system_type)
    use_ts, use_pn = resolve_ts_torchsim_flags(
        calc_name,
        ts_params.get("use_torchsim"),
        ts_params.get("use_parallel_neb"),
    )
    ts_surface_config = ts_params.get("surface_config")
    if (
        surface_config is not None
        and ts_surface_config is not None
        and surface_config != ts_surface_config
    ):
        raise SCGOValidationError(
            "run surface_config and ts_params['surface_config'] disagree."
        )
    resolved_surface_config = (
        surface_config if surface_config is not None else ts_surface_config
    )
    if get_system_policy(system_type).uses_surface and not isinstance(
        resolved_surface_config, SurfaceSystemConfig
    ):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config in ts_params "
            "or as the run surface_config argument."
        )

    ts_batch_atoms = ts_params.get(
        "parallel_neb_max_batch_atoms",
        ts_defaults.get("parallel_neb_max_batch_atoms"),
    )
    kwargs: dict[str, Any] = {
        "params": {
            "calculator": ts_params["calculator"],
            "calculator_kwargs": ts_params.get("calculator_kwargs") or {},
        },
        "system_type": system_type,
        "use_torchsim": use_ts,
        "use_parallel_neb": use_pn,
        "torchsim_params": {
            "force_tol": ts_params.get("torchsim_fmax", ts_defaults["torchsim_fmax"]),
            "max_steps": ts_params.get(
                "torchsim_max_steps", ts_defaults["torchsim_max_steps"]
            ),
        },
    }
    if ts_batch_atoms is not None and int(ts_batch_atoms) > 0:
        # Mirror the GO pattern (geneticalgorithm_go_torchsim sets
        # expected_max_atoms = mobile+fixed x pop_size): size the relaxer for the
        # largest fused NEB force batch so the autobatcher probe stays capped to the
        # real workload (native torch-sim estimation needs no synthetic probe).
        kwargs["torchsim_params"]["expected_max_atoms"] = int(ts_batch_atoms)
        kwargs["torchsim_params"]["max_atoms_to_try"] = int(ts_batch_atoms)
    if str(ts_params.get("calculator", "")).strip().upper() == "UMA":
        ck = ts_params.get("calculator_kwargs", {}) or {}
        model_name = ck.get("model_name")
        task_name = ck.get("task_name")
        if not model_name or not task_name:
            raise SCGOValidationError(
                "UMA transition-state search requires calculator_kwargs with "
                "'model_name' and 'task_name' (set via get_ts_search_params())."
            )
        kwargs["torchsim_params"].update(
            {
                "model_kind": "fairchem",
                "fairchem_model_name": str(model_name),
                "fairchem_task_name": str(task_name),
            }
        )
    elif str(ts_params.get("calculator", "")).strip().upper() == "UPET":
        ck = ts_params.get("calculator_kwargs", {}) or {}
        model_name = ck.get("model_name")
        if not model_name and not ck.get("checkpoint_path"):
            raise SCGOValidationError(
                "UPET transition-state search requires calculator_kwargs with "
                "'model_name' or 'checkpoint_path' (set via get_ts_search_params())."
            )
        kwargs["torchsim_params"].update(
            {
                "model_kind": "upet",
                "upet_model_name": str(model_name) if model_name else None,
                "upet_version": ck.get("version"),
                "upet_checkpoint_path": ck.get("checkpoint_path"),
                "upet_non_conservative": bool(ck.get("non_conservative", False)),
            }
        )
    user_torchsim = ts_params.get("torchsim_params")
    if isinstance(user_torchsim, dict):
        kwargs["torchsim_params"].update(user_torchsim)

    # Keys without per-system defaults: pass through as-is (None when missing
    # is fine for the runner).
    for key in _TS_PARAM_PASSTHROUGH_KEYS:
        kwargs[key] = ts_params.get(key)
    kwargs["surface_config"] = resolved_surface_config

    # NEB knobs that vary per system_type: fall back to the defaults table.
    for key in _TS_PARAM_NEB_KEYS:
        kwargs[key] = ts_params.get(key, ts_defaults[key])

    # Generic (system-type-agnostic) defaults.
    for key, def_val in _TS_PARAM_GENERIC_DEFAULTS.items():
        kwargs[key] = ts_params.get(key, def_val)

    # Boolean with a runner-side ``True`` default: route it explicitly instead
    # of the None-passthrough loop (a None there would override the default).
    kwargs["tag_ts_in_db"] = bool(ts_params.get("tag_ts_in_db", True))

    return kwargs
