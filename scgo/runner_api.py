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

from collections.abc import Iterable
try:
    from typing import TypeAlias
except ImportError:
    from typing import TypeAlias
from typing import Any

from ase import Atoms

from scgo.run_minima import parse_composition_arg
from scgo.utils.logging import get_logger

CompositionInput: TypeAlias = str | list[str] | Atoms
_ALGO_KEYS = ("simple", "bh", "ga")
_LOGGER = get_logger(__name__)


def _as_composition(composition: CompositionInput) -> list[str]:
    if isinstance(composition, Atoms):
        return list(composition.get_chemical_symbols())
    elif isinstance(composition, str):
        return parse_composition_arg(composition)
    elif isinstance(composition, list):
        if not composition:
            raise ValueError("composition list must not be empty")
        return [str(s) for s in composition]
    else:
        raise TypeError(
            f"composition must be str, list[str], or Atoms, got {type(composition).__name__}"
        )


def _as_composition_list(items: Iterable[CompositionInput]) -> list[list[str]]:
    out = [_as_composition(x) for x in items]
    if not out:
        raise ValueError("compositions iterable must not be empty")
    return out


def _resolved_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path is not None else None


def _require_system_type(system_type: SystemType | None, fn_name: str) -> SystemType:
    if system_type is None:
        raise ValueError(f"system_type is required for {fn_name}.")
    return system_type


def _effective_write_timing_json(
    write_timing_json: bool, profile_ga: bool | None
) -> bool:
    return bool(profile_ga) if profile_ga is not None else write_timing_json


def _prepare_run_context(
    composition: CompositionInput,
    *,
    system_type: SystemType | None,
    surface_config: SurfaceSystemConfig | None,
    params: dict[str, Any] | None,
    adsorbates: AdsorbatesInput | None,
    cluster_adsorbate_config: Any | None = None,
    context: str,
) -> tuple[
    SystemType,
    dict[str, Any] | None,
    AdsorbateDefinition | None,
    Atoms | None,
    list[str],
]:
    st = _require_system_type(system_type, context)
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if params is not None:
        _reject_system_keys(params, context=context, kind="go")
    comp = _as_composition(composition)
    ads_def, ads_template, full_comp = build_adsorbate_definition_from_inputs(
        system_type=st, composition=comp, adsorbates=adsorbates, context=context
    )
    validate_adsorbate_definition(
        system_type=st,
        composition=full_comp,
        adsorbate_definition=ads_def,
        context=context,
    )
    params_prep = params or {}
    if params:
        params_prep = _with_surface_in_optimizers(params, surface_config=surface_config)
    if params_prep:
        params_prep = _with_adsorbate_in_optimizers(
            params_prep,
            adsorbate_definition=ads_def,
            adsorbate_fragment_template=ads_template,
            cluster_adsorbate_config=cluster_adsorbate_config,
        )
    return st, params_prep, ads_def, ads_template, full_comp


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


def _merge_adsorbate_context_into_params(
    base: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Attach adsorbate/surface init context for :func:`run_scgo_trials` / GA."""
    out = copy.deepcopy(base) if base is not None else {}
    out.update({k: v for k, v in kwargs.items() if v is not None})
    return out


def _with_system_type_in_optimizer_params(
    params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    write_timing_json: bool = False,
    profile_ga: bool | None = None,
) -> dict[str, Any]:
    effective = _effective_write_timing_json(write_timing_json, profile_ga)
    out = dict(params or {})
    op = out.setdefault("optimizer_params", {})
    for algo in _ALGO_KEYS:
        cfg = op.setdefault(algo, {})
        cfg["system_type"] = system_type
        if algo in ("ga", "bh"):
            cfg.setdefault("write_timing_json", effective)
    return out


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
    _reject_system_keys(ts_params, context=fn_name, kind="ts")
    return coerce_ts_params_to_runner_kwargs(
        ts_params, system_type=system_type, surface_config=surface_config
    )


def _default_ts_params_from_go(
    *,
    system_type: SystemType,
    go_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build TS defaults from preset helpers, aligned with GO calculator settings."""
    p = go_params or {}
    return get_ts_search_params(
        calculator=str(p.get("calculator", "MACE")),
        calculator_kwargs=dict(p.get("calculator_kwargs") or {}),
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


def _with_adsorbate_in_optimizers(
    go_params: dict[str, Any],
    *,
    adsorbate_definition: Any | None = None,
    adsorbate_fragment_template: Any | None = None,
    cluster_adsorbate_config: Any | None = None,
) -> dict[str, Any]:
    """Copy ``go_params``; fan out explicit run adsorbate params to optimizer slots."""
    out = copy.deepcopy(go_params)

    # If any adsorbate param is set, distribute to all optimizer slots
    if (
        adsorbate_definition is not None
        or adsorbate_fragment_template is not None
        or cluster_adsorbate_config is not None
    ):
        op = out.setdefault("optimizer_params", {})
        for key in _ALGO_KEYS:
            if key not in op:
                continue
            slot = op[key]
            if not isinstance(slot, dict):
                raise ValueError(
                    f"optimizer_params['{key}'] must be a dict when using adsorbate parameters"
                )
            if adsorbate_definition is not None:
                ex = slot.get("adsorbate_definition")
                if ex is None:
                    slot["adsorbate_definition"] = adsorbate_definition
                # Don't check for match, multiple definitions may be equivalent
            if adsorbate_fragment_template is not None:
                ex = slot.get("adsorbate_fragment_template")
                if ex is None:
                    slot["adsorbate_fragment_template"] = adsorbate_fragment_template
            if cluster_adsorbate_config is not None:
                ex = slot.get("cluster_adsorbate_config")
                if ex is None:
                    slot["cluster_adsorbate_config"] = cluster_adsorbate_config
    return out


def _reject_system_keys(
    params: dict[str, Any], *, context: str, kind: str = "go"
) -> None:
    for key in ("system_type", "surface_config"):
        if params.get(key) is not None:
            raise ValueError(
                f"{context} does not allow top-level {kind}_params['{key}']. "
                "Use the run function argument instead."
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
    st, params_prep, ads_def, ads_temp, comp = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        params=params,
        adsorbates=adsorbates,
        cluster_adsorbate_config=cluster_adsorbate_config,
        context="run_go",
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    eff_params = _with_system_type_in_optimizer_params(
        params_prep,
        system_type=st,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    eff_params = _merge_adsorbate_context_into_params(
        eff_params,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    minima = run_scgo_trials(
        comp,
        params=eff_params,
        seed=eff_seed,
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
    st = _require_system_type(system_type, "run_go_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if params is not None:
        _reject_system_keys(params, context="run_go_campaign")
    params_prep = (
        _with_surface_in_optimizers(params, surface_config=surface_config)
        if params
        else None
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    eff_params = _with_system_type_in_optimizer_params(
        params_prep,
        system_type=st,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    eff_params = _merge_adsorbate_context_into_params(
        eff_params, cluster_adsorbate_config=cluster_adsorbate_config
    )

    full_compositions: list[list[str]] = []
    for composition_item in _as_composition_list(compositions):
        ads_def, ads_temp, full_comp = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=composition_item,
            adsorbates=adsorbates,
            context="run_go_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full_comp,
            adsorbate_definition=ads_def,
            context="run_go_campaign",
        )
        full_compositions.append(full_comp)
        eff_params["adsorbate_definition"] = ads_def
        eff_params["adsorbate_fragment_template"] = ads_temp

    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    campaign = run_scgo_campaign_arbitrary_compositions(
        full_compositions,
        params=eff_params,
        seed=eff_seed,
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
    st = _require_system_type(system_type, "run_go_ts")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    go_mat, ts_mat = _materialize_go_ts_params(
        system_type=st, go_params=go_params, ts_params=ts_params
    )
    _reject_system_keys(go_mat, context="run_go_ts")
    _reject_system_keys(ts_mat, context="run_go_ts", kind="ts")
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=go_mat, ts_params=ts_mat)
    go_prep = _with_surface_in_optimizers(go_mat, surface_config=surface_config)
    core_comp = _as_composition(composition)
    ads_def, ads_temp, comp = build_adsorbate_definition_from_inputs(
        system_type=st,
        composition=core_comp,
        adsorbates=adsorbates,
        context="run_go_ts",
    )
    validate_adsorbate_definition(
        system_type=st,
        composition=comp,
        adsorbate_definition=ads_def,
        context="run_go_ts",
    )
    _validate_go_ts_surface_config(
        go_prep,
        system_type=st,
        surface_config=surface_config,
        adsorbate_composition=comp,
    )

    go_local = _with_system_type_in_optimizer_params(
        go_prep,
        system_type=st,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    go_local = _merge_adsorbate_context_into_params(
        go_local,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name="run_go_ts", system_type=st, surface_config=surface_config
    )
    out_path = _resolved_path(output_dir) or _default_go_ts_output_path(
        comp, go_params=go_mat, output_stem=output_stem, output_root=output_root
    )
    t0 = perf_counter()
    summary = run_scgo_go_ts_pipeline(
        comp,
        go_params=go_local,
        ts_kwargs=ts_kwargs,
        seed=eff_seed,
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
    st = _require_system_type(system_type, "run_go_ts_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    go_mat, ts_mat = _materialize_go_ts_params(
        system_type=st, go_params=go_params, ts_params=ts_params
    )
    _reject_system_keys(go_mat, context="run_go_ts_campaign")
    _reject_system_keys(ts_mat, context="run_go_ts_campaign", kind="ts")
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=go_mat, ts_params=ts_mat)
    go_prep = _with_surface_in_optimizers(go_mat, surface_config=surface_config)

    full_compositions: list[list[str]] = []
    ads_def, ads_temp = None, None
    for core_comp in _as_composition_list(compositions):
        ads_def, ads_temp, full_comp = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=core_comp,
            adsorbates=adsorbates,
            context="run_go_ts_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full_comp,
            adsorbate_definition=ads_def,
            context="run_go_ts_campaign",
        )
        full_compositions.append(full_comp)
        _validate_go_ts_surface_config(
            go_prep,
            system_type=st,
            surface_config=surface_config,
            adsorbate_composition=full_comp,
        )

    go_local = _with_system_type_in_optimizer_params(
        go_prep,
        system_type=st,
        write_timing_json=write_timing_json,
        profile_ga=profile_ga,
    )
    go_local = _merge_adsorbate_context_into_params(
        go_local,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat,
        fn_name="run_go_ts_campaign",
        system_type=st,
        surface_config=surface_config,
    )
    parent = _resolved_path(output_dir) or _default_go_ts_output_path(
        full_compositions[0],
        go_params=go_mat,
        output_stem=output_stem or "go_ts_campaign",
        output_root=output_root,
    )
    out: dict[str, dict[str, Any]] = {}
    t0 = perf_counter()
    for comp in full_compositions:
        formula = get_cluster_formula(comp)
        out[formula] = run_scgo_go_ts_pipeline(
            comp,
            go_params=go_local,
            ts_kwargs=ts_kwargs,
            seed=eff_seed,
            verbosity=verbosity,
            output_dir=parent / f"{formula}_campaign",
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
    st, _, ads_def, _, comp = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        params=None,
        adsorbates=adsorbates,
        context="run_ts_search",
    )
    ts_mat = _materialize_ts_params(system_type=st, ts_params=ts_params)
    _reject_system_keys(ts_mat, context="run_ts_search", kind="ts")
    eff_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_mat)
    merged = _coerce_ts_for_runner(
        ts_mat, fn_name="run_ts_search", system_type=st, surface_config=surface_config
    )
    if params is not None:
        merged["params"] = params
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    results = _ts_search(
        comp,
        output_dir=out_path,
        seed=eff_seed,
        verbosity=verbosity,
        adsorbate_definition=ads_def,
        **merged,
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
    st = _require_system_type(system_type, "run_ts_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    ts_mat = _materialize_ts_params(system_type=st, ts_params=ts_params)
    _reject_system_keys(ts_mat, context="run_ts_campaign", kind="ts")
    eff_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_mat)
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name="run_ts_campaign", system_type=st, surface_config=surface_config
    )

    full_compositions: list[list[str]] = []
    ads_def: AdsorbateDefinition | None = None
    for core in _as_composition_list(compositions):
        ads_def, _, full = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=core,
            adsorbates=adsorbates,
            context="run_ts_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full,
            adsorbate_definition=ads_def,
            context="run_ts_campaign",
        )
        full_compositions.append(full)
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    if ads_def:
        ts_kwargs["adsorbate_definition"] = ads_def
    campaign = _ts_campaign(
        full_compositions,
        output_dir=out_path,
        params=params,
        seed=eff_seed,
        verbosity=verbosity,
        ts_kwargs=ts_kwargs,
    )
    if log_summary:
        total = sum(len(v) for v in campaign.values())
        ok = sum(
            1 for rl in campaign.values() for r in rl if r.get("status") == "success"
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
