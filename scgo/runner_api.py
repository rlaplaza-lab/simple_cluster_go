"""High-level SCGO workflows: GO, TS, GO+TS, and campaigns.

``go`` = global-optimization params; ``ts`` = flat TS preset
(:func:`scgo.param_presets.get_ts_search_params`), coerced inside this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from ase import Atoms

from scgo.run_minima import (
    parse_composition_arg,
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_go_ts_pipeline,
    run_scgo_trials,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SystemType
from scgo.ts_search.transition_state_run import (
    run_transition_state_campaign as _ts_campaign,
    run_transition_state_search as _ts_search,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs
from scgo.utils.validation import validate_composition

type CompositionInput = str | list[str] | Atoms
_ALGO_KEYS = ("simple", "bh", "ga")
_LOGGER = get_logger(__name__)


def _as_composition(composition: CompositionInput) -> list[str]:
    match composition:
        case Atoms():
            return list(composition.get_chemical_symbols())
        case str():
            return parse_composition_arg(composition)
        case list():
            validate_composition(composition, allow_empty=False, allow_tuple=False)
            return list(composition)
        case _:
            raise TypeError(
                f"composition must be str, list[str], or Atoms, got {type(composition).__name__}"
            )


def _as_composition_list(items: Iterable[CompositionInput]) -> list[list[str]]:
    out = [_as_composition(x) for x in items]
    if not out:
        raise ValueError("compositions iterable must not be empty")
    return out


def _go_campaign(
    compositions: list[list[str]],
    *,
    params: dict | None,
    seed: int | None,
    verbosity: int,
    run_id: str | None,
    clean: bool,
    output_dir: str | Path | None,
) -> dict[str, list[tuple[float, Atoms]]]:
    return run_scgo_campaign_arbitrary_compositions(
        compositions,
        params=params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


def _resolved_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _require_system_type(system_type: SystemType | None, fn_name: str) -> SystemType:
    if not isinstance(system_type, str):
        raise ValueError(f"system_type is required for {fn_name}.")
    return system_type


def _effective_write_timing_json(
    write_timing_json: bool, profile_ga: bool | None
) -> bool:
    if profile_ga is not None:
        return bool(profile_ga)
    return write_timing_json


def _with_system_type_in_optimizer_params(
    params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
) -> dict[str, Any]:
    effective = _effective_write_timing_json(write_timing_json, profile_ga)
    effective_params = dict(params or {})
    optimizer_params = dict(effective_params.get("optimizer_params", {}))
    for algo in _ALGO_KEYS:
        algo_cfg = dict(optimizer_params.get(algo, {}))
        algo_cfg["system_type"] = system_type
        if algo in ("ga", "bh"):
            algo_cfg.setdefault("write_timing_json", effective)
        optimizer_params[algo] = algo_cfg
    effective_params["optimizer_params"] = optimizer_params
    return effective_params


def _coerce_ts_for_runner(
    ts: dict[str, Any] | None,
    *,
    fn_name: str,
    system_type: SystemType,
) -> dict[str, Any]:
    if not ts:
        raise ValueError(
            f"ts is required for {fn_name}. Build with get_ts_search_params(...)."
        )
    flat = {**ts, "system_type": system_type}
    return coerce_ts_params_to_runner_kwargs(flat)


def _calculator_slug_from_go(go: dict[str, Any] | None) -> str:
    c = str((go or {}).get("calculator", "MACE")).strip().upper()
    if c in ("MACE", "UMA"):
        return c.lower()
    return c.lower() or "calc"


def _default_go_ts_output_path(
    composition: list[str],
    *,
    go: dict[str, Any],
    output_stem: str | None,
    output_root: str | Path | None,
) -> Path:
    root = output_root if output_root is not None else Path.cwd() / "scgo_runs"
    p = Path(root).expanduser().resolve()
    stem = output_stem or get_cluster_formula(composition)
    return (p / f"{stem}_{_calculator_slug_from_go(go)}").resolve()


def _log_completion(kind: str, *, elapsed_s: float, details: str) -> None:
    _LOGGER.info("%s completed in %.2f s (%s)", kind, elapsed_s, details)


def run_go(
    composition: CompositionInput,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Any | None = None,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> list[tuple[float, Atoms]]:
    system_type_local = _require_system_type(system_type, "run_go")
    effective_params = _with_system_type_in_optimizer_params(
        params,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    minima = run_scgo_trials(
        _as_composition(composition),
        params=effective_params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=out_path,
        calculator_for_global_optimization=calculator_for_global_optimization,
    )
    if log_summary:
        _log_completion(
            "run_go",
            elapsed_s=perf_counter() - t0,
            details=f"minima={len(minima)} output_dir={out_path}",
        )
    return minima


def run_go_campaign(
    compositions: Iterable[CompositionInput],
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    system_type_local = _require_system_type(system_type, "run_go_campaign")
    effective_params = _with_system_type_in_optimizer_params(
        params,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    campaign = _go_campaign(
        _as_composition_list(compositions),
        params=effective_params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=out_path,
    )
    if log_summary:
        _log_completion(
            "run_go_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(campaign)} output_dir={out_path}",
        )
    return campaign


def run_go_ts(
    composition: CompositionInput,
    *,
    go: dict[str, Any],
    ts: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    ts_composition: list[str] | None = None,
    infer_ts_composition_from_minima: bool = False,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, Any]:
    system_type_local = _require_system_type(system_type, "run_go_ts")
    go_local = _with_system_type_in_optimizer_params(
        go,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    ts_kwargs_local = _coerce_ts_for_runner(
        ts, fn_name="run_go_ts", system_type=system_type_local
    )
    comp = _as_composition(composition)
    if output_dir is None:
        out_path = _default_go_ts_output_path(
            comp, go=go, output_stem=output_stem, output_root=output_root
        )
    else:
        out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    summary = run_scgo_go_ts_pipeline(
        comp,
        go=go_local,
        ts_kwargs=ts_kwargs_local,
        seed=seed,
        verbosity=verbosity,
        output_dir=out_path,
        ts_composition=ts_composition,
        infer_ts_composition_from_minima=infer_ts_composition_from_minima,
    )
    if log_summary:
        log_go_ts_summary(_LOGGER, summary, wall_time_s=perf_counter() - t0)
    return summary


def run_go_ts_campaign(
    compositions: Iterable[CompositionInput],
    *,
    go: dict[str, Any],
    ts: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    ts_composition: list[str] | None = None,
    infer_ts_composition_from_minima: bool = False,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_go_ts_campaign")
    go_local = _with_system_type_in_optimizer_params(
        go,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    ts_kwargs_local = _coerce_ts_for_runner(
        ts, fn_name="run_go_ts_campaign", system_type=system_type_local
    )
    compositions_list = _as_composition_list(compositions)
    if output_dir is None:
        parent = _default_go_ts_output_path(
            compositions_list[0],
            go=go,
            output_stem=output_stem or "go_ts_campaign",
            output_root=output_root,
        )
    else:
        parent = _resolved_path(output_dir)
    out: dict[str, dict[str, Any]] = {}
    t0 = perf_counter()
    for composition in compositions_list:
        formula = get_cluster_formula(composition)
        root = parent / f"{formula}_campaign"
        out[formula] = run_scgo_go_ts_pipeline(
            composition,
            go=go_local,
            ts_kwargs=ts_kwargs_local,
            seed=seed,
            verbosity=verbosity,
            output_dir=root,
            ts_composition=ts_composition,
            infer_ts_composition_from_minima=infer_ts_composition_from_minima,
        )
    if log_summary:
        total = sum(int(s.get("ts_total_count") or 0) for s in out.values())
        ok = sum(int(s.get("ts_success_count") or 0) for s in out.values())
        _log_completion(
            "run_go_ts_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(out)} successful_nebs={ok}/{total}",
        )
    return out


def run_ts_search(
    composition: CompositionInput,
    *,
    ts: dict[str, Any],
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    log_summary: bool = True,
) -> list[dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_ts_search")
    merged_kwargs = _coerce_ts_for_runner(
        ts, fn_name="run_ts_search", system_type=system_type_local
    )
    if params is not None:
        merged_kwargs["params"] = params
    if surface_config is not None:
        merged_kwargs["surface_config"] = surface_config
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    results = _ts_search(
        _as_composition(composition),
        output_dir=out_path,
        seed=seed,
        verbosity=verbosity,
        **merged_kwargs,
    )
    if log_summary:
        ok = sum(1 for r in results if r.get("status") == "success")
        _log_completion(
            "run_ts_search",
            elapsed_s=perf_counter() - t0,
            details=f"successful_nebs={ok}/{len(results)} output_dir={out_path}",
        )
    return results


def run_ts_campaign(
    compositions: Iterable[CompositionInput],
    *,
    ts: dict[str, Any],
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    system_type: SystemType | None = None,
    log_summary: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    system_type_local = _require_system_type(system_type, "run_ts_campaign")
    ts_kwargs_local = _coerce_ts_for_runner(
        ts, fn_name="run_ts_campaign", system_type=system_type_local
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    campaign = _ts_campaign(
        _as_composition_list(compositions),
        output_dir=out_path,
        params=params,
        seed=seed,
        verbosity=verbosity,
        ts_kwargs=ts_kwargs_local,
    )
    if log_summary:
        total = sum(len(v) for v in campaign.values())
        ok = sum(
            1
            for result_list in campaign.values()
            for r in result_list
            if r.get("status") == "success"
        )
        _log_completion(
            "run_ts_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(campaign)} successful_nebs={ok}/{total}",
        )
    return campaign


def log_go_ts_summary(
    logger: Any,
    summary: dict[str, Any],
    *,
    wall_time_s: float | None = None,
) -> None:
    """Log NEB success counts from a ``run_go_ts*`` summary dict."""
    ts_results = summary.get("ts_results") or []
    ok = sum(1 for r in ts_results if r.get("status") == "success")
    logger.info("Successful NEBs: %d/%d", ok, len(ts_results))
    if wall_time_s is not None:
        logger.info("Total wall time: %.2f s", wall_time_s)


__all__ = [
    "CompositionInput",
    "log_go_ts_summary",
    "parse_composition_arg",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_ts_campaign",
    "run_ts_search",
]
