#!/usr/bin/env python3
"""Example runner: Pt5 on a graphite slab, global optimization + TS search.

Mirrors the surface benchmark setup in
``benchmark/benchmark_Pt_surface_graphite.py`` but narrowed to a single
composition (Pt5) and chained with
:func:`scgo.ts_search.run_transition_state_search` to find transition states
connecting the recovered minima. The top slab layer is allowed to relax during
both global optimization and NEB, matching the benchmark.

Supports both MACE (TorchSim GA) and UMA (FairChem) backends; install the
matching ``scgo[mace]`` or ``scgo[uma]`` extra in its own environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from ase import Atoms
from ase.build import graphene

from scgo.param_presets import (
    get_torchsim_ga_params,
    get_ts_run_kwargs,
    get_ts_search_params,
    get_ts_search_params_uma,
    get_uma_ga_benchmark_params,
)
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.runner_surface import read_full_composition_from_first_xyz
from scgo.surface.config import SurfaceSystemConfig
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_SLAB_VACUUM = 12.0
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_graphite"


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_niter: int,
    default_population_size: int,
    default_max_pairs: int,
) -> None:
    parser.add_argument("--backend", choices=("mace", "uma"), default="mace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--uma-task", default="oc25")
    parser.add_argument("--niter", type=int, default=default_niter)
    parser.add_argument("--population-size", type=int, default=default_population_size)
    parser.add_argument("--max-pairs", type=int, default=default_max_pairs)
    parser.add_argument("--output-root", type=Path, default=None)


def _make_ga_params(
    args: argparse.Namespace,
    *,
    surface_config: SurfaceSystemConfig,
) -> dict[str, Any]:
    if args.backend == "uma":
        model_name = args.model_name or "uma-s-1p2"
        params = get_uma_ga_benchmark_params(
            args.seed, model_name=model_name, uma_task=args.uma_task
        )
    else:
        params = get_torchsim_ga_params(seed=args.seed)
        params["calculator"] = "MACE"
        if args.model_name is not None:
            params["calculator_kwargs"]["model_name"] = args.model_name

    ga = params["optimizer_params"]["ga"]
    ga["niter"] = args.niter
    ga["population_size"] = args.population_size
    ga["surface_config"] = surface_config
    return params


def _make_ts_kwargs(
    args: argparse.Namespace,
    *,
    surface_config: SurfaceSystemConfig,
) -> dict[str, Any]:
    if args.backend == "uma":
        model_name = args.model_name or "uma-s-1p2"
        ts_params = get_ts_search_params_uma(
            regime="surface",
            surface_config=surface_config,
            model_name=model_name,
            uma_task=args.uma_task,
        )
    else:
        ts_params = get_ts_search_params(
            regime="surface",
            surface_config=surface_config,
        )
        if args.model_name is not None:
            ts_params["calculator_kwargs"]["model_name"] = args.model_name
    ts_kwargs = get_ts_run_kwargs(ts_params)
    ts_kwargs["max_pairs"] = args.max_pairs
    return ts_kwargs


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is None:
        output_root = DEFAULT_OUTPUT_PARENT / f"{DEFAULT_OUTPUT_SUBDIR}_{args.backend}"
    else:
        output_root = args.output_root
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _build_graphite_slab(
    layers: int = DEFAULT_SLAB_LAYERS,
    vacuum: float = DEFAULT_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_SLAB_REPEAT_XY,
) -> Atoms:
    """Build a graphite slab with periodic in-plane boundary conditions."""
    slab = graphene(formula="C2", vacuum=vacuum)
    slab = slab.repeat((repeat_xy, repeat_xy, max(1, layers)))
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)
    return slab


def _make_surface_config(
    slab_layers: int,
    slab_repeat_xy: int,
) -> SurfaceSystemConfig:
    slab = _build_graphite_slab(layers=slab_layers, repeat_xy=slab_repeat_xy)
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.6,
        adsorption_height_max=3.2,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=600,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(
        parser,
        default_niter=6,
        default_population_size=24,
        default_max_pairs=10,
    )
    parser.add_argument("--slab-layers", type=int, default=DEFAULT_SLAB_LAYERS)
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=DEFAULT_SLAB_REPEAT_XY,
        help="In-plane supercell repeats; default is large enough for Pt5 adsorption.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_root = _resolve_output_root(args)

    composition = [ELEMENT] * N_ATOMS
    formula = get_cluster_formula(composition)

    surface_config = _make_surface_config(args.slab_layers, args.slab_repeat_xy)
    ga_params = _make_ga_params(args, surface_config=surface_config)
    # Surface GA needs extra parallel-init knobs the gas-phase preset omits.
    ga = ga_params["optimizer_params"]["ga"]
    ga["n_jobs_population_init"] = -2
    ga["batch_size"] = 4

    ts_kwargs = _make_ts_kwargs(args, surface_config=surface_config)

    t_start = perf_counter()

    logger.info(
        "Running SCGO GO for %s on graphite (backend=%s, seed=%d) under %s",
        formula,
        args.backend,
        args.seed,
        output_root,
    )
    run_scgo_campaign_one_element(
        ELEMENT,
        N_ATOMS,
        N_ATOMS,
        params=ga_params,
        seed=args.seed,
        output_dir=output_root,
    )

    ts_base_dir = output_root / f"{formula}_searches"
    final_minima_dir = ts_base_dir / "final_unique_minima"
    full_composition = read_full_composition_from_first_xyz(final_minima_dir)

    logger.info("Running TS search for %s under %s", formula, ts_base_dir)
    ts_results = run_transition_state_search(
        full_composition,
        base_dir=ts_base_dir,
        seed=args.seed,
        verbosity=1,
        **ts_kwargs,
    )
    total_success = sum(1 for r in ts_results if r.get("status") == "success")
    total_runs = len(ts_results)

    logger.info("Successful NEBs: %d/%d", total_success, total_runs)
    logger.info("Total wall time: %.2f s", perf_counter() - t_start)


if __name__ == "__main__":
    main()
