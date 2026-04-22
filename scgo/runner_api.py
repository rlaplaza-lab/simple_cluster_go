"""High-level SCGO workflows (GO, TS, GO+TS, campaigns).

``CompositionInput`` is a formula string, ``list[str]`` of symbols, or ``Atoms``
(only symbols are used for GO). TS options beyond ``params`` / ``output_dir`` go
in ``ts_kwargs`` (same keys as ``scgo.ts_search.transition_state_run``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from ase import Atoms

from scgo.param_presets import TsSearchRegime, build_one_element_go_ts_bundle
from scgo.run_minima import (
    parse_composition_arg,
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_go_ts_pipeline,
    run_scgo_trials,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.ts_search.transition_state_run import (
    run_transition_state_campaign as _ts_campaign,
    run_transition_state_search as _ts_search,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.validation import validate_composition

type CompositionInput = str | list[str] | Atoms


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


def _require_nonempty_symbol(s: str, name: str) -> None:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_size_range(min_atoms: int, max_atoms: int) -> None:
    if min_atoms < 1:
        raise ValueError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be >= min_atoms")


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
) -> list[tuple[float, Atoms]]:
    return run_scgo_trials(
        _as_composition(composition),
        params=params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
        calculator_for_global_optimization=calculator_for_global_optimization,
    )


def run_go_campaign(
    compositions: Iterable[CompositionInput],
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    return _go_campaign(
        _as_composition_list(compositions),
        params=params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


def run_go_element_scan(
    element: str,
    min_atoms: int,
    max_atoms: int,
    *,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    _require_nonempty_symbol(element, "element")
    _require_size_range(min_atoms, max_atoms)
    compositions = [[element] * n for n in range(min_atoms, max_atoms + 1)]
    return _go_campaign(
        compositions,
        params=params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


def run_go_binary_scan(
    element1: str,
    element2: str,
    min_atoms: int,
    max_atoms: int,
    *,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, list[tuple[float, Atoms]]]:
    _require_nonempty_symbol(element1, "element1")
    _require_nonempty_symbol(element2, "element2")
    _require_size_range(min_atoms, max_atoms)
    compositions = [
        [element1] * i + [element2] * (n - i)
        for n in range(min_atoms, max_atoms + 1)
        for i in range(n + 1)
    ]
    return _go_campaign(
        compositions,
        params=params,
        seed=seed,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        output_dir=output_dir,
    )


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
) -> dict[str, Any]:
    return run_scgo_go_ts_pipeline(
        _as_composition(composition),
        ga_params=ga_params,
        ts_kwargs=ts_kwargs,
        seed=seed,
        verbosity=verbosity,
        output_dir=output_dir,
        ts_composition=ts_composition,
        infer_ts_composition_from_minima=infer_ts_composition_from_minima,
    )


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
) -> dict[str, dict[str, Any]]:
    parent = _resolved_path(output_dir)
    out: dict[str, dict[str, Any]] = {}
    for composition in _as_composition_list(compositions):
        formula = get_cluster_formula(composition)
        root = parent / f"{formula}_campaign" if parent else None
        out[formula] = run_scgo_go_ts_pipeline(
            composition,
            ga_params=ga_params,
            ts_kwargs=ts_kwargs,
            seed=seed,
            verbosity=verbosity,
            output_dir=root,
            ts_composition=ts_composition,
            infer_ts_composition_from_minima=infer_ts_composition_from_minima,
        )
    return out


def run_ts_search(
    composition: CompositionInput,
    *,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    ts_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    merged_kwargs = dict(ts_kwargs or {})
    # Keep wrapper-level explicit args authoritative unless they are omitted.
    if "params" not in merged_kwargs:
        merged_kwargs["params"] = params
    if "surface_config" not in merged_kwargs:
        merged_kwargs["surface_config"] = surface_config
    return _ts_search(
        _as_composition(composition),
        output_dir=output_dir,
        seed=seed,
        verbosity=verbosity,
        **merged_kwargs,
    )


def run_ts_campaign(
    compositions: Iterable[CompositionInput],
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    ts_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return _ts_campaign(
        _as_composition_list(compositions),
        output_dir=output_dir,
        params=params,
        seed=seed,
        verbosity=verbosity,
        ts_kwargs=ts_kwargs,
    )


def run_go_ts_with_mlip_preset(
    composition: CompositionInput,
    *,
    default_output_parent: Path,
    default_output_subdir: str,
    backend: str,
    seed: int,
    niter: int,
    population_size: int,
    max_pairs: int,
    regime: TsSearchRegime = "gas",
    model_name: str | None = None,
    uma_task: str = "oc25",
    surface_config: SurfaceSystemConfig | None = None,
    output_dir: str | Path | None = None,
    verbosity: int = 1,
    infer_ts_composition_from_minima: bool = False,
    ga_n_jobs_population_init: int | None = None,
    ga_batch_size: int | None = None,
) -> dict[str, Any]:
    """Run GO then TS using MACE/TorchSim or UMA GA+TS presets (any adsorbate composition)."""
    bn = normalize_backend(backend)
    bundle = build_one_element_go_ts_bundle(
        backend=bn,
        seed=seed,
        model_name=model_name,
        uma_task=uma_task,
        niter=niter,
        population_size=population_size,
        max_pairs=max_pairs,
        regime=regime,
        surface_config=surface_config,
        ga_n_jobs_population_init=ga_n_jobs_population_init,
        ga_batch_size=ga_batch_size,
    )
    return run_go_ts(
        composition,
        ga_params=bundle["ga_params"],
        ts_kwargs=bundle["ts_kwargs"],
        seed=seed,
        verbosity=verbosity,
        output_dir=resolve_runner_output_dir(
            default_output_parent=default_output_parent,
            default_output_subdir=default_output_subdir,
            backend=bn,
            output_dir=output_dir,
        ),
        infer_ts_composition_from_minima=infer_ts_composition_from_minima,
    )


def run_go_ts_one_element(
    element: str,
    n_atoms: int,
    *,
    default_output_parent: Path,
    default_output_subdir: str,
    backend: str,
    seed: int,
    niter: int,
    population_size: int,
    max_pairs: int,
    regime: TsSearchRegime = "gas",
    model_name: str | None = None,
    uma_task: str = "oc25",
    surface_config: SurfaceSystemConfig | None = None,
    output_dir: str | Path | None = None,
    verbosity: int = 1,
    infer_ts_composition_from_minima: bool = False,
    ga_n_jobs_population_init: int | None = None,
    ga_batch_size: int | None = None,
) -> dict[str, Any]:
    _require_nonempty_symbol(element, "element")
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    return run_go_ts_with_mlip_preset(
        [element] * n_atoms,
        default_output_parent=default_output_parent,
        default_output_subdir=default_output_subdir,
        backend=backend,
        seed=seed,
        niter=niter,
        population_size=population_size,
        max_pairs=max_pairs,
        regime=regime,
        model_name=model_name,
        uma_task=uma_task,
        surface_config=surface_config,
        output_dir=output_dir,
        verbosity=verbosity,
        infer_ts_composition_from_minima=infer_ts_composition_from_minima,
        ga_n_jobs_population_init=ga_n_jobs_population_init,
        ga_batch_size=ga_batch_size,
    )


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
    "run_go_binary_scan",
    "run_go_campaign",
    "run_go_element_scan",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_go_ts_one_element",
    "run_go_ts_with_mlip_preset",
    "run_ts_campaign",
    "run_ts_search",
]
