"""Flat TS dict → kwargs for :func:`run_transition_state_search`."""

from __future__ import annotations

from typing import Any

from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_NEB_TANGENT_METHOD
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags


def coerce_ts_params_to_runner_kwargs(
    ts_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map ``get_ts_search_params`` output to ``run_transition_state_search`` kwargs."""
    if ts_params is None:
        raise ValueError(
            "ts_params is required. Build with get_ts_search_params(system_type=...)."
        )

    calc_name = str(ts_params["calculator"])
    use_ts, use_pn = resolve_ts_torchsim_flags(
        calc_name,
        ts_params.get("use_torchsim"),
        ts_params.get("use_parallel_neb"),
    )

    kwargs: dict[str, Any] = {
        "params": {
            "calculator": ts_params["calculator"],
            "calculator_kwargs": ts_params.get("calculator_kwargs") or {},
        },
        "system_type": ts_params["system_type"],
        "use_torchsim": use_ts,
        "use_parallel_neb": use_pn,
        "torchsim_params": {
            "force_tol": ts_params.get("torchsim_fmax"),
            "max_steps": ts_params.get("torchsim_max_steps"),
        },
    }
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
        "write_timing_json",
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
