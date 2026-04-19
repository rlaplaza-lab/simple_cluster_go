"""Shared CLI and parameter helpers for the example runner scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scgo.param_presets import (
    get_torchsim_ga_params,
    get_ts_run_kwargs,
    get_ts_search_params,
    get_ts_search_params_uma,
    get_uma_ga_benchmark_params,
)
from scgo.surface.config import SurfaceSystemConfig


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_subdir: str,
    default_output_parent: Path,
    default_niter: int,
    default_population_size: int,
    default_max_pairs: int,
) -> None:
    """Register CLI arguments shared by all example runner scripts."""
    parser.add_argument("--backend", choices=("mace", "uma"), default="mace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override calculator model_name (default UMA: uma-s-1p2; MACE: preset default).",
    )
    parser.add_argument(
        "--uma-task",
        default="oc25",
        help="UMA task_name (only used when --backend uma).",
    )
    parser.add_argument("--niter", type=int, default=default_niter)
    parser.add_argument("--population-size", type=int, default=default_population_size)
    parser.add_argument("--max-pairs", type=int, default=default_max_pairs)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            f"Output root. Default: {default_output_parent}/{default_output_subdir}_<backend>/"
        ),
    )


def make_ga_params(
    args: argparse.Namespace,
    *,
    surface_config: SurfaceSystemConfig | None = None,
) -> dict[str, Any]:
    """Build GA params for either the UMA or MACE backend."""
    if args.backend == "uma":
        model_name = args.model_name or "uma-s-1p2"
        params = get_uma_ga_benchmark_params(
            args.seed, model_name=model_name, task_name=args.uma_task
        )
    else:
        params = get_torchsim_ga_params(seed=args.seed)
        params["calculator"] = "MACE"
        if args.model_name is not None:
            params["calculator_kwargs"]["model_name"] = args.model_name

    ga = params["optimizer_params"]["ga"]
    ga["niter"] = args.niter
    ga["population_size"] = args.population_size
    if surface_config is not None:
        ga["surface_config"] = surface_config
    return params


def make_ts_kwargs(
    args: argparse.Namespace,
    *,
    regime: str,
    surface_config: SurfaceSystemConfig | None = None,
) -> dict[str, Any]:
    """Build kwargs forwarded to ``run_transition_state_search`` per backend."""
    if args.backend == "uma":
        model_name = args.model_name or "uma-s-1p2"
        ts_params = get_ts_search_params_uma(
            regime=regime,
            surface_config=surface_config,
            model_name=model_name,
            task_name=args.uma_task,
        )
    else:
        ts_params = get_ts_search_params(
            regime=regime,
            surface_config=surface_config,
        )
        if args.model_name is not None:
            ts_params["calculator_kwargs"]["model_name"] = args.model_name

    ts_kwargs = get_ts_run_kwargs(ts_params)
    ts_kwargs["max_pairs"] = args.max_pairs
    return ts_kwargs


def resolve_output_root(
    args: argparse.Namespace,
    *,
    default_output_parent: Path,
    default_output_subdir: str,
) -> Path:
    """Pick the output root directory based on CLI args (creating it if needed)."""
    if args.output_root is None:
        output_root = default_output_parent / f"{default_output_subdir}_{args.backend}"
    else:
        output_root = args.output_root
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
