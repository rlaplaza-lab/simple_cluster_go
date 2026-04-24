"""High-level SCGO workflows: GO, TS, GO+TS, and campaigns.

``go_params`` = global-optimization params; ``ts_params`` = flat TS preset
(:func:`scgo.param_presets.get_ts_search_params`). The run ``seed`` and
``go_params['seed']`` / ``ts_params['seed']`` must agree when more than one is set
(:func:`resolve_workflow_seed`). System mode is set only by the run function
``system_type=...`` argument together with explicit ``surface_config=...`` and,
for ``*_adsorbate`` modes, core-only ``composition`` plus ``adsorbates=...``
(single or multiple ASE ``Atoms`` fragments).
System-definition keys in ``go_params`` / ``ts_params`` are rejected.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from ase import Atoms

from scgo.optimization.algorithm_select import select_scgo_minima_algorithm
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.param_presets import get_default_params, get_ts_search_params
from scgo.run_minima import (
    parse_composition_arg,
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_go_ts_pipeline,
    run_scgo_trials,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbatesInput,
    AdsorbateDefinition,
    SystemType,
    build_adsorbate_definition_from_inputs,
    get_system_policy,
    validate_adsorbate_definition,
    validate_system_type_settings,
)
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


def _merge_adsorbate_context_into_params(
    base: dict[str, Any] | None,
    *,
    adsorbate_definition: AdsorbateDefinition | None,
    adsorbate_fragment_template: Atoms | None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
) -> dict[str, Any]:
    """Attach adsorbate/surface init context for :func:`run_scgo_trials` / GA."""
    out = copy.deepcopy(base) if base is not None else {}
    if adsorbate_definition is not None:
        out["adsorbate_definition"] = adsorbate_definition
    if adsorbate_fragment_template is not None:
        out["adsorbate_fragment_template"] = adsorbate_fragment_template
    if cluster_adsorbate_config is not None:
        out["cluster_adsorbate_config"] = cluster_adsorbate_config
    return out


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
    ts_params: dict[str, Any] | None,
    *,
    fn_name: str,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> dict[str, Any]:
    if not ts_params:
        raise ValueError(
            f"ts_params is required for {fn_name}. Build with get_ts_search_params(...)."
        )
    _reject_system_definition_in_ts_params(ts_params, context=fn_name)
    if ts_params.get("system_type") is not None:
        raise ValueError(
            f"{fn_name} does not allow ts_params['system_type']. "
            "Use the run function system_type=... argument only."
        )
    if ts_params.get("surface_config") is not None:
        raise ValueError(
            f"{fn_name} does not allow ts_params['surface_config']. "
            "Use the run function surface_config=... argument only."
        )
    return coerce_ts_params_to_runner_kwargs(
        ts_params,
        system_type=system_type,
        surface_config=surface_config,
    )


def _default_ts_params_from_go(
    *,
    system_type: SystemType,
    go_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build TS defaults from preset helpers, aligned with GO calculator settings."""
    calculator = str((go_params or {}).get("calculator", "MACE"))
    calculator_kwargs = dict((go_params or {}).get("calculator_kwargs") or {})
    return get_ts_search_params(
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        system_type=system_type,
    )


def _materialize_go_ts_params(
    *,
    system_type: SystemType,
    go_params: dict[str, Any] | None,
    ts_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return concrete GO/TS params dicts using canonical preset defaults."""
    effective_go = get_default_params() if go_params is None else go_params
    effective_ts = (
        _default_ts_params_from_go(system_type=system_type, go_params=effective_go)
        if ts_params is None
        else ts_params
    )
    return effective_go, effective_ts


def _materialize_ts_params(
    *,
    system_type: SystemType,
    ts_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return concrete TS params dict using canonical TS preset defaults."""
    return (
        get_ts_search_params(system_type=system_type)
        if ts_params is None
        else ts_params
    )


def _calculator_slug_from_go_params(go_params: dict[str, Any] | None) -> str:
    c = str((go_params or {}).get("calculator", "MACE")).strip().upper()
    if c in ("MACE", "UMA"):
        return c.lower()
    return c.lower() or "calc"


def _default_go_ts_output_path(
    composition: list[str],
    *,
    go_params: dict[str, Any],
    output_stem: str | None,
    output_root: str | Path | None,
) -> Path:
    root = output_root if output_root is not None else Path.cwd() / "scgo_runs"
    p = Path(root).expanduser().resolve()
    stem = output_stem or get_cluster_formula(composition)
    return (p / f"{stem}_{_calculator_slug_from_go_params(go_params)}").resolve()


def _log_completion(kind: str, *, elapsed_s: float, details: str) -> None:
    _LOGGER.info("%s completed in %.2f s (%s)", kind, elapsed_s, details)


def _as_int_seed(label: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{label} must be int-like, got {value!r}") from e


def resolve_workflow_seed(
    *,
    seed_kw: int | None = None,
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
) -> int | None:
    """Unify run ``seed=...``, ``go_params['seed']``, and ``ts_params['seed']``; all non-null must agree."""
    parts: list[tuple[str, int]] = []
    if seed_kw is not None:
        parts.append(("run_kwd(seed=...)", _as_int_seed("run seed", seed_kw)))
    if go_params is not None and go_params.get("seed") is not None:
        parts.append(
            (
                "go_params['seed']",
                _as_int_seed("go_params['seed']", go_params.get("seed")),
            )
        )
    if ts_params is not None and ts_params.get("seed") is not None:
        parts.append(
            (
                "ts_params['seed']",
                _as_int_seed("ts_params['seed']", ts_params.get("seed")),
            )
        )
    if not parts:
        return None
    values = {v for _, v in parts}
    if len(values) > 1:
        desc = ", ".join(f"{name}={v}" for name, v in parts)
        raise ValueError(f"Inconsistent random seeds: {desc}")
    return next(iter(values))


def _with_surface_in_optimizers(
    go_params: dict[str, Any], *, surface_config: SurfaceSystemConfig | None
) -> dict[str, Any]:
    """Copy ``go_params``; fan out explicit run ``surface_config`` to optimizer slots."""
    out = copy.deepcopy(go_params)
    if surface_config is not None:
        op = out.setdefault("optimizer_params", {})
        for key in _ALGO_KEYS:
            if key not in op:
                continue
            slot = op[key]
            if not isinstance(slot, dict):
                raise ValueError(
                    f"optimizer_params['{key}'] must be a dict when using go_params['surface_config']"
                )
            ex = slot.get("surface_config")
            if ex is None:
                slot["surface_config"] = surface_config
            elif ex != surface_config:
                raise ValueError(
                    f"run argument surface_config must match "
                    f"go_params['optimizer_params']['{key}']['surface_config'] when both are set."
                )
    return out


def _reject_system_definition_in_go_params(
    go_params: dict[str, Any], *, context: str
) -> None:
    st = go_params.get("system_type")
    if st is not None:
        raise ValueError(
            f"{context} does not allow top-level go_params['system_type']={st!r}. "
            "Use the run function system_type=... argument only."
        )
    sc = go_params.get("surface_config")
    if sc is not None:
        raise ValueError(
            f"{context} does not allow top-level go_params['surface_config']. "
            "Use the run function surface_config=... argument only."
        )


def _reject_system_definition_in_ts_params(
    ts_params: dict[str, Any], *, context: str
) -> None:
    if ts_params.get("system_type") is not None:
        raise ValueError(
            f"{context} does not allow ts_params['system_type']. "
            "Use the run function system_type=... argument only."
        )
    if ts_params.get("surface_config") is not None:
        raise ValueError(
            f"{context} does not allow ts_params['surface_config']. "
            "Use the run function surface_config=... argument only."
        )


def _validate_go_ts_surface_config(
    go_prepared: dict[str, Any],
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_composition: list[str],
) -> None:
    """For surface system types, require explicit config and active GO consistency."""
    if not get_system_policy(system_type).uses_surface:
        return
    if not isinstance(surface_config, SurfaceSystemConfig):
        raise ValueError(
            f"system_type={system_type!r} requires the run surface_config argument "
            "to be a SurfaceSystemConfig."
        )
    chosen = select_scgo_minima_algorithm(len(adsorbate_composition), system_type)
    op = go_prepared.get("optimizer_params") or {}
    go_slot = op.get(chosen)
    if not isinstance(go_slot, dict):
        go_slot = {}
    go_sc = go_slot.get("surface_config")
    if not isinstance(go_sc, SurfaceSystemConfig):
        raise ValueError(
            f"system_type={system_type!r} requires go_params['optimizer_params']['{chosen}']"
            "['surface_config'] (active minima algorithm is "
            f"{chosen!r} for this adsorbate). Set top-level go_params['surface_config'] to fan out."
        )
    if go_sc != surface_config:
        raise ValueError(
            "run surface_config and go_params['optimizer_params']["
            f"'{chosen}']['surface_config'] disagree."
        )


def run_go(
    composition: CompositionInput,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Any | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> list[tuple[float, Atoms]]:
    system_type_local = _require_system_type(system_type, "run_go")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    if params is not None:
        _reject_system_definition_in_go_params(params, context="run_go")
    composition_local = _as_composition(composition)
    adsorbate_definition_local, adsorbate_fragment_template_local, full_composition = (
        build_adsorbate_definition_from_inputs(
            system_type=system_type_local,
            composition=composition_local,
            adsorbates=adsorbates,
            context="run_go",
        )
    )
    validate_adsorbate_definition(
        system_type=system_type_local,
        composition=full_composition,
        adsorbate_definition=adsorbate_definition_local,
        context="run_go",
    )
    params_prepared = (
        _with_surface_in_optimizers(params, surface_config=surface_config)
        if params is not None
        else None
    )
    effective_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    effective_params = _with_system_type_in_optimizer_params(
        params_prepared,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    effective_params = _merge_adsorbate_context_into_params(
        effective_params,
        adsorbate_definition=adsorbate_definition_local,
        adsorbate_fragment_template=adsorbate_fragment_template_local,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    minima = run_scgo_trials(
        full_composition,
        params=effective_params,
        seed=effective_seed,
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
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    system_type_local = _require_system_type(system_type, "run_go_campaign")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    if params is not None:
        _reject_system_definition_in_go_params(params, context="run_go_campaign")
    params_prepared = (
        _with_surface_in_optimizers(params, surface_config=surface_config)
        if params is not None
        else None
    )
    effective_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    effective_params = _with_system_type_in_optimizer_params(
        params_prepared,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    effective_params = _merge_adsorbate_context_into_params(
        effective_params,
        adsorbate_definition=None,
        adsorbate_fragment_template=None,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    compositions_local = _as_composition_list(compositions)
    full_compositions: list[list[str]] = []
    for composition_item in compositions_local:
        adsorbate_definition_local, adsorbate_fragment_template_local, full_comp = (
            build_adsorbate_definition_from_inputs(
                system_type=system_type_local,
                composition=composition_item,
                adsorbates=adsorbates,
                context="run_go_campaign",
            )
        )
        validate_adsorbate_definition(
            system_type=system_type_local,
            composition=full_comp,
            adsorbate_definition=adsorbate_definition_local,
            context="run_go_campaign",
        )
        full_compositions.append(full_comp)
        effective_params["adsorbate_definition"] = adsorbate_definition_local
        effective_params["adsorbate_fragment_template"] = adsorbate_fragment_template_local
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    campaign = run_scgo_campaign_arbitrary_compositions(
        full_compositions,
        params=effective_params,
        seed=effective_seed,
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
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, Any]:
    system_type_local = _require_system_type(system_type, "run_go_ts")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    go_params, ts_params = _materialize_go_ts_params(
        system_type=system_type_local,
        go_params=go_params,
        ts_params=ts_params,
    )
    _reject_system_definition_in_go_params(go_params, context="run_go_ts")
    _reject_system_definition_in_ts_params(ts_params, context="run_go_ts")
    effective_seed = resolve_workflow_seed(
        seed_kw=seed, go_params=go_params, ts_params=ts_params
    )
    go_prepared = _with_surface_in_optimizers(go_params, surface_config=surface_config)
    core_comp = _as_composition(composition)
    adsorbate_definition_local, adsorbate_fragment_template_local, comp = (
        build_adsorbate_definition_from_inputs(
            system_type=system_type_local,
            composition=core_comp,
            adsorbates=adsorbates,
            context="run_go_ts",
        )
    )
    validate_adsorbate_definition(
        system_type=system_type_local,
        composition=comp,
        adsorbate_definition=adsorbate_definition_local,
        context="run_go_ts",
    )
    _validate_go_ts_surface_config(
        go_prepared,
        system_type=system_type_local,
        surface_config=surface_config,
        adsorbate_composition=comp,
    )
    go_local = _with_system_type_in_optimizer_params(
        go_prepared,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    go_local = _merge_adsorbate_context_into_params(
        go_local,
        adsorbate_definition=adsorbate_definition_local,
        adsorbate_fragment_template=adsorbate_fragment_template_local,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    ts_kwargs_local = _coerce_ts_for_runner(
        ts_params,
        fn_name="run_go_ts",
        system_type=system_type_local,
        surface_config=surface_config,
    )
    if output_dir is None:
        out_path = _default_go_ts_output_path(
            comp,
            go_params=go_params,
            output_stem=output_stem,
            output_root=output_root,
        )
    else:
        out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    summary = run_scgo_go_ts_pipeline(
        comp,
        go_params=go_local,
        ts_kwargs=ts_kwargs_local,
        seed=effective_seed,
        verbosity=verbosity,
        output_dir=out_path,
    )
    if log_summary:
        log_go_ts_summary(_LOGGER, summary, wall_time_s=perf_counter() - t0)
    return summary


def run_go_ts_campaign(
    compositions: Iterable[CompositionInput],
    *,
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
    log_summary: bool = True,
) -> dict[str, dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_go_ts_campaign")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    go_params, ts_params = _materialize_go_ts_params(
        system_type=system_type_local,
        go_params=go_params,
        ts_params=ts_params,
    )
    _reject_system_definition_in_go_params(go_params, context="run_go_ts_campaign")
    _reject_system_definition_in_ts_params(ts_params, context="run_go_ts_campaign")
    effective_seed = resolve_workflow_seed(
        seed_kw=seed, go_params=go_params, ts_params=ts_params
    )
    go_prepared = _with_surface_in_optimizers(go_params, surface_config=surface_config)
    compositions_list = _as_composition_list(compositions)
    full_compositions: list[list[str]] = []
    for core_comp in compositions_list:
        adsorbate_definition_local, adsorbate_fragment_template_local, full_comp = (
            build_adsorbate_definition_from_inputs(
                system_type=system_type_local,
                composition=core_comp,
                adsorbates=adsorbates,
                context="run_go_ts_campaign",
            )
        )
        validate_adsorbate_definition(
            system_type=system_type_local,
            composition=full_comp,
            adsorbate_definition=adsorbate_definition_local,
            context="run_go_ts_campaign",
        )
        full_compositions.append(full_comp)
        _validate_go_ts_surface_config(
            go_prepared,
            system_type=system_type_local,
            surface_config=surface_config,
            adsorbate_composition=full_comp,
        )
    go_local = _with_system_type_in_optimizer_params(
        go_prepared,
        system_type=system_type_local,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    go_local = _merge_adsorbate_context_into_params(
        go_local,
        adsorbate_definition=adsorbate_definition_local,
        adsorbate_fragment_template=adsorbate_fragment_template_local,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    ts_kwargs_local = _coerce_ts_for_runner(
        ts_params,
        fn_name="run_go_ts_campaign",
        system_type=system_type_local,
        surface_config=surface_config,
    )
    if output_dir is None:
        parent = _default_go_ts_output_path(
            full_compositions[0],
            go_params=go_params,
            output_stem=output_stem or "go_ts_campaign",
            output_root=output_root,
        )
    else:
        parent = _resolved_path(output_dir)
    out: dict[str, dict[str, Any]] = {}
    t0 = perf_counter()
    for composition in full_compositions:
        formula = get_cluster_formula(composition)
        root = parent / f"{formula}_campaign"
        out[formula] = run_scgo_go_ts_pipeline(
            composition,
            go_params=go_local,
            ts_kwargs=ts_kwargs_local,
            seed=effective_seed,
            verbosity=verbosity,
            output_dir=root,
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
    ts_params: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> list[dict[str, Any]]:
    system_type_local = _require_system_type(system_type, "run_ts_search")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    core_composition = _as_composition(composition)
    adsorbate_definition_local, _adsorbate_fragment_template_local, composition_local = (
        build_adsorbate_definition_from_inputs(
            system_type=system_type_local,
            composition=core_composition,
            adsorbates=adsorbates,
            context="run_ts_search",
        )
    )
    validate_adsorbate_definition(
        system_type=system_type_local,
        composition=composition_local,
        adsorbate_definition=adsorbate_definition_local,
        context="run_ts_search",
    )
    ts_params = _materialize_ts_params(
        system_type=system_type_local,
        ts_params=ts_params,
    )
    _reject_system_definition_in_ts_params(ts_params, context="run_ts_search")
    effective_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_params)
    merged_kwargs = _coerce_ts_for_runner(
        ts_params,
        fn_name="run_ts_search",
        system_type=system_type_local,
        surface_config=surface_config,
    )
    if params is not None:
        merged_kwargs["params"] = params
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    results = _ts_search(
        composition_local,
        output_dir=out_path,
        seed=effective_seed,
        verbosity=verbosity,
        adsorbate_definition=adsorbate_definition_local,
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
    ts_params: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    system_type_local = _require_system_type(system_type, "run_ts_campaign")
    validate_system_type_settings(
        system_type=system_type_local, surface_config=surface_config
    )
    ts_params = _materialize_ts_params(
        system_type=system_type_local,
        ts_params=ts_params,
    )
    _reject_system_definition_in_ts_params(ts_params, context="run_ts_campaign")
    effective_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_params)
    ts_kwargs_local = _coerce_ts_for_runner(
        ts_params,
        fn_name="run_ts_campaign",
        system_type=system_type_local,
        surface_config=surface_config,
    )
    compositions_local = _as_composition_list(compositions)
    full_compositions: list[list[str]] = []
    adsorbate_definition_local: AdsorbateDefinition | None = None
    for core_comp in compositions_local:
        adsorbate_definition_local, _adsorbate_fragment_template_local, full_comp = (
            build_adsorbate_definition_from_inputs(
                system_type=system_type_local,
                composition=core_comp,
                adsorbates=adsorbates,
                context="run_ts_campaign",
            )
        )
        validate_adsorbate_definition(
            system_type=system_type_local,
            composition=full_comp,
            adsorbate_definition=adsorbate_definition_local,
            context="run_ts_campaign",
        )
        full_compositions.append(full_comp)
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    ts_kwargs_merged = dict(ts_kwargs_local)
    if adsorbate_definition_local is not None:
        ts_kwargs_merged["adsorbate_definition"] = adsorbate_definition_local
    campaign = _ts_campaign(
        full_compositions,
        output_dir=out_path,
        params=params,
        seed=effective_seed,
        verbosity=verbosity,
        ts_kwargs=ts_kwargs_merged,
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
    "resolve_workflow_seed",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_ts_campaign",
    "run_ts_search",
]
