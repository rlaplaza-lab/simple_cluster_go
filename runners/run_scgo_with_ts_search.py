#!/usr/bin/env python3
"""Unified SCGO + TS runner for MACE/UMA and cluster/surface workflows."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from ase.build import graphene

from scgo.param_presets import (
    get_default_uma_params,
    get_minimal_ga_params,
    get_ts_run_kwargs,
    get_ts_search_params,
    get_ts_search_params_uma,
)
from scgo.run_minima import (
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_campaign_one_element,
)
from scgo.runner_surface import (
    make_surface_config,
    read_full_composition_from_first_xyz,
)
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mace", "uma"), default="mace")
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--with-oh", action="store_true")
    parser.add_argument("--element", default="Cu")
    parser.add_argument("--min-atoms", type=int, default=4)
    parser.add_argument("--max-atoms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=15)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--profile", action="store_true")
    return parser


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    configured_root = os.environ.get("SCGO_OUTPUT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    mode = "surface" if args.surface else "cluster"
    oh = "_with_oh" if args.with_oh else ""
    return Path(f"ts_search_{mode}_{args.backend}{oh}_results").resolve()


def _make_ga_params(
    args: argparse.Namespace, surface_config: Any | None
) -> dict[str, Any]:
    if args.backend == "uma":
        ga_params = get_default_uma_params()
        ga_params["seed"] = args.seed
    else:
        ga_params = get_minimal_ga_params(seed=args.seed)
    ga = ga_params["optimizer_params"]["ga"]
    ga["n_jobs_population_init"] = 1
    ga["previous_search_glob"] = "__no_seed_dbs__/**/*.db"
    if surface_config is not None:
        ga["surface_config"] = surface_config
        ga["batch_size"] = 4
    return ga_params


def _make_ts_kwargs(
    args: argparse.Namespace, surface_config: Any | None
) -> dict[str, Any]:
    regime = "surface" if args.surface else "gas"
    if args.backend == "uma":
        ts_params = get_ts_search_params_uma(
            regime=regime,
            surface_config=surface_config,
        )
    else:
        ts_params = get_ts_search_params(
            regime=regime,
            surface_config=surface_config,
        )
    ts_kwargs = get_ts_run_kwargs(ts_params)
    ts_kwargs["max_pairs"] = args.max_pairs
    return ts_kwargs


def _cluster_with_optional_oh(args: argparse.Namespace) -> list[str]:
    cluster = [args.element] * args.min_atoms
    if args.with_oh:
        cluster.extend(["O", "H"])
    return cluster


def main() -> None:
    args = _build_parser().parse_args()
    output_root = _resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    run_started = perf_counter()
    surface_config = None
    if args.surface:
        slab = graphene(size=(3, 3, 1), vacuum=12.0)
        slab.pbc = True
        surface_config = make_surface_config(slab)

    ga_params = _make_ga_params(args, surface_config)
    ts_kwargs = _make_ts_kwargs(args, surface_config)

    with contextlib.ExitStack():
        minima_started = perf_counter()
        if args.with_oh:
            cluster_with_oh = _cluster_with_optional_oh(args)
            formula = get_cluster_formula(cluster_with_oh)
            formula_search_dir = output_root / f"{formula}_searches"
            final_minima_dir = formula_search_dir / "final_unique_minima"
            minima_exist = final_minima_dir.exists() and any(
                final_minima_dir.glob("*.xyz")
            )
            if not minima_exist:
                run_scgo_campaign_arbitrary_compositions(
                    [cluster_with_oh],
                    params=ga_params,
                    seed=args.seed,
                    output_dir=output_root,
                    verbosity=1,
                )
            else:
                logger.info("Reusing existing minima under %s", final_minima_dir)
            full_composition = (
                read_full_composition_from_first_xyz(final_minima_dir)
                if args.surface
                else cluster_with_oh
            )
            ts_base_dir = formula_search_dir
            ts_results = run_transition_state_search(
                full_composition,
                base_dir=ts_base_dir,
                seed=args.seed,
                verbosity=1,
                **ts_kwargs,
            )
            total_success = sum(1 for r in ts_results if r.get("status") == "success")
            total_runs = len(ts_results)
        else:
            run_scgo_campaign_one_element(
                args.element,
                args.min_atoms,
                args.max_atoms,
                params=ga_params,
                seed=args.seed,
                output_dir=output_root,
            )
            total_success = 0
            total_runs = 0
            for n_atoms in range(args.min_atoms, args.max_atoms + 1):
                composition = [args.element] * n_atoms
                formula = get_cluster_formula(composition)
                formula_search_dir = output_root / f"{formula}_searches"
                if args.surface:
                    final_minima_dir = formula_search_dir / "final_unique_minima"
                    composition = read_full_composition_from_first_xyz(final_minima_dir)
                ts_results = run_transition_state_search(
                    composition,
                    base_dir=formula_search_dir,
                    seed=args.seed,
                    verbosity=1,
                    **ts_kwargs,
                )
                total_success += sum(
                    1 for r in ts_results if r.get("status") == "success"
                )
                total_runs += len(ts_results)

        logger.info(
            "Minima + TS run finished in %.2f s", perf_counter() - minima_started
        )
        logger.info("Successful NEBs: %d/%d", total_success, total_runs)
        logger.info("Total wall time: %.2f s", perf_counter() - run_started)


if __name__ == "__main__":
    main()
