"""High-level SCGO workflows (GO, TS, GO+TS, campaigns).

``CompositionInput`` is a formula string, ``list[str]`` of symbols, or ``Atoms``
(only symbols are used for GO). TS options beyond ``params`` / ``output_dir`` go
in ``ts_kwargs`` (same keys as ``scgo.ts_search.transition_state_run``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

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


def _with_system_type_in_optimizer_params(
    params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    profile_ga: bool = True,
) -> dict[str, Any]:
    effective_params = dict(params or {})
    optimizer_params = dict(effective_params.get("optimizer_params", {}))
    for algo in _ALGO_KEYS:
        algo_cfg = dict(optimizer_params.get(algo, {}))
        algo_cfg["system_type"] = system_type
        if algo == "ga":
            algo_cfg.setdefault("write_profile", profile_ga)
        optimizer_params[algo] = algo_cfg
    effective_params["optimizer_params"] = optimizer_params
    return effective_params


def _require_ts_kwargs(
    ts_kwargs: dict[str, Any] | None,
    *,
    fn_name: str,
    system_type: SystemType,
) -> dict[str, Any]:
    if not ts_kwargs:
        raise ValueError(
            f"ts_kwargs is required for {fn_name}. Build it with "
            "get_ts_run_kwargs(get_ts_search_params(...))."
        )
    ts_kwargs_local = dict(ts_kwargs)
    ts_system_type = ts_kwargs_local.get("system_type")
    if not isinstance(ts_system_type, str):
        raise ValueError(f"ts_kwargs['system_type'] is required for {fn_name}.")
    if ts_system_type != system_type:
        raise ValueError(
            f"{fn_name} system_type mismatch: argument={system_type!r}, "
            f"ts_kwargs['system_type']={ts_system_type!r}."
        )
    return ts_kwargs_local


def _log_completion(kind: str, *, elapsed_s: float, details: str) -> None:
    _LOGGER.info("%s completed in %.2f s (%s)", kind, elapsed_s, details)


def normalize_backend(backend: str) -> Literal["mace", "uma"]:
    b = str(backend).strip().lower()
    if b not in ("mace", "uma"):
        raise ValueError(f"backend must be 'mace' or 'uma', got {backend!r}")
    return cast(Literal["mace", "uma"], b)


def resolve_runner_output_dir(
    *,
    default_output_parent: Path,
    default_output_subdir: str,
    backend: str,
    output_dir: str | Path | None,
) -> Path:
    backend_norm = normalize_backend(backend)
    if output_dir is None:
        return (
            default_output_parent / f"{default_output_subdir}_{backend_norm}"
        ).resolve()
    return cast(Path, _resolved_path(output_dir))


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
    profile_ga: bool = True,
    log_summary: bool = True,
) -> list[tuple[float, Atoms]]:
    system_type_local = _require_system_type(system_type, "run_go")
    effective_params = _with_system_type_in_optimizer_params(
        params, system_type=system_type_local, profile_ga=profile_ga
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
    profile_ga: bool = True,
    log_summary: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    system_type_local = _require_system_type(system_type, "run_go_campaign")
    effective_params = _with_system_type_in_optimizer_params(
        params, system_type=system_type_local, profile_ga=profile_ga
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
    ga_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    ts_composition: list[str] | None = None,
    infer_ts_composition_from_minima: bool = False,
    system_type: SystemType | None = None,
    profile_ga: bool = True,
    log_summary: bool = True,
) -> dict[str, Any]:
    system_type_local = _require_system_type(system_type, "run_go_ts")
    ga_params_local = _with_system_type_in_optimizer_params(
        ga_params, system_type=system_type_local, profile_ga=profile_ga
    )
    ts_kwargs_local = _require_ts_kwargs(
        ts_kwargs, fn_name="run_go_ts", system_type=system_type_local
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    summary = run_scgo_go_ts_pipeline(
        _as_composition(composition),
        ga_params=ga_params_local,
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
    ga_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    ts_composition: list[str] | None = None,
    infer_ts_composition_from_minima: bool = False,
    system_type: SystemType | None = None,
    profile_ga: bool = True,
    log_summary: bool = True,
) -> dict[str, dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_go_ts_campaign")
    ga_params_local = _with_system_type_in_optimizer_params(
        ga_params, system_type=system_type_local, profile_ga=profile_ga
    )
    ts_kwargs_local = _require_ts_kwargs(
        ts_kwargs, fn_name="run_go_ts_campaign", system_type=system_type_local
    )
    parent = _resolved_path(output_dir)
    out: dict[str, dict[str, Any]] = {}
    t0 = perf_counter()
    for composition in _as_composition_list(compositions):
        formula = get_cluster_formula(composition)
        root = parent / f"{formula}_campaign" if parent else None
        out[formula] = run_scgo_go_ts_pipeline(
            composition,
            ga_params=ga_params_local,
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
    ts_kwargs: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    log_summary: bool = True,
) -> list[dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_ts_search")
    merged_kwargs = _require_ts_kwargs(
        ts_kwargs, fn_name="run_ts_search", system_type=system_type_local
    )
    if params is not None and "params" not in merged_kwargs:
        merged_kwargs["params"] = params
    if surface_config is not None and "surface_config" not in merged_kwargs:
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
    ts_kwargs: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    system_type: SystemType | None = None,
    log_summary: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    system_type_local = _require_system_type(system_type, "run_ts_campaign")
    ts_kwargs_local = _require_ts_kwargs(
        ts_kwargs, fn_name="run_ts_campaign", system_type=system_type_local
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
    "normalize_backend",
    "parse_composition_arg",
    "resolve_runner_output_dir",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_ts_campaign",
    "run_ts_search",
]
