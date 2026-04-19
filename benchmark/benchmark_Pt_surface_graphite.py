#!/usr/bin/env python3
"""Benchmark Pt cluster search on graphite with top-layer slab relaxation.

Mirrors ``benchmark/benchmark_Pt.py`` for a surface system: sweeps Pt cluster
sizes with one seed, runs SCGO over the range, and logs per-size minima from
the campaign. Outputs live under ``benchmark/results/pt_surface_graphite/`` (same
layout as gas-phase benchmarks: ``<Formula>_searches`` per size). See
``benchmark.benchmark_common.PT_SURFACE_GRAPHITE_RESULTS_DIR``.
"""

from __future__ import annotations

import argparse
import time

from ase import Atoms
from ase.build import graphene

from benchmark.benchmark_common import (
    DEFAULT_CLUSTERS,
    PT_SURFACE_GRAPHITE_RESULTS_DIR,
    add_common_benchmark_cli,
    format_ga_profile_lines,
    get_benchmark_params,
    load_latest_ga_profile,
    parse_atom_count,
)
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_SLAB_VACUUM = 12.0
DEFAULT_OUTPUT_ROOT = PT_SURFACE_GRAPHITE_RESULTS_DIR.resolve()

logger = get_logger(__name__)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Pt-on-graphite SCGO benchmark sweep with top-layer slab relaxation."
        )
    )
    add_common_benchmark_cli(parser)
    parser.set_defaults(niter=6, population_size=24)
    parser.add_argument("--slab-layers", type=int, default=DEFAULT_SLAB_LAYERS)
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=DEFAULT_SLAB_REPEAT_XY,
        help="In-plane supercell repeats; default is large enough for Pt4-Pt11 adsorption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    clusters = args.clusters or DEFAULT_CLUSTERS

    surface_config = _make_surface_config(args.slab_layers, args.slab_repeat_xy)

    params = get_benchmark_params(
        seed=args.seed,
        model_name=args.model_name,
        backend=args.backend,
        uma_task_name=args.uma_task,
    )
    params["optimizer_params"]["ga"]["surface_config"] = surface_config
    params["optimizer_params"]["ga"]["n_jobs_population_init"] = -2
    params["optimizer_params"]["ga"]["niter"] = args.niter
    params["optimizer_params"]["ga"]["population_size"] = args.population_size
    params["optimizer_params"]["ga"]["batch_size"] = 4

    t0 = time.perf_counter()

    logger.info("Per-size summary:")
    total_minima = 0
    for formula in clusters:
        n_atoms = parse_atom_count(formula)
        formula = get_cluster_formula(["Pt"] * n_atoms)
        results = run_scgo_campaign_one_element(
            "Pt",
            n_atoms,
            n_atoms,
            params=params,
            seed=args.seed,
            output_dir=output_root,
        )
        minima = results.get(formula, [])
        n_found = len(minima)
        total_minima += n_found
        if n_found == 0:
            logger.info("  %s: 0 minima", formula)
            continue
        best_energy = minima[0][0]
        logger.info("  %s: %d minima, best E=%.6f eV", formula, n_found, best_energy)
        profile = load_latest_ga_profile(output_root, formula)
        if profile:
            for line in format_ga_profile_lines(profile, detailed=True, max_entries=8):
                logger.info("    %s", line)

    logger.info(
        "Finished benchmark. Total minima saved across formulas: %d under %s",
        total_minima,
        output_root,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBenchmark wall time: {elapsed:.1f} s ({args.backend})")


if __name__ == "__main__":
    main()
