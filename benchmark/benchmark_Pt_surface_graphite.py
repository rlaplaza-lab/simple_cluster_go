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

from benchmark.benchmark_common import (
    DEFAULT_CLUSTERS,
    PT_SURFACE_GRAPHITE_RESULTS_DIR,
    add_common_benchmark_cli,
    apply_ga_benchmark_overrides,
    format_ga_profile_lines,
    get_benchmark_params,
    load_latest_ga_profile,
    parse_atom_count,
)
from scgo.runner_api import run_go_element_scan
from scgo.surface.presets import (
    DEFAULT_GRAPHITE_SLAB_LAYERS,
    DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    make_graphite_surface_config,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

DEFAULT_OUTPUT_DIR = PT_SURFACE_GRAPHITE_RESULTS_DIR.resolve()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Pt-on-graphite SCGO benchmark sweep with top-layer slab relaxation."
        )
    )
    add_common_benchmark_cli(parser)
    parser.set_defaults(niter=6, population_size=24)
    parser.add_argument("--slab-layers", type=int, default=DEFAULT_GRAPHITE_SLAB_LAYERS)
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
        help="In-plane supercell repeats; default is large enough for Pt4-Pt11 adsorption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = args.clusters or DEFAULT_CLUSTERS

    surface_config = make_graphite_surface_config(
        slab_layers=args.slab_layers,
        slab_repeat_xy=args.slab_repeat_xy,
    )

    base_params = get_benchmark_params(
        seed=args.seed,
        model_name=args.model_name,
        backend=args.backend,
        uma_task=args.uma_task,
    )
    params = apply_ga_benchmark_overrides(
        base_params,
        niter=args.niter,
        population_size=args.population_size,
        surface_config=surface_config,
        n_jobs_population_init=-2,
        batch_size=4,
    )

    t0 = time.perf_counter()

    logger.info("Per-size summary:")
    total_minima = 0
    for formula in clusters:
        n_atoms = parse_atom_count(formula)
        formula = get_cluster_formula(["Pt"] * n_atoms)
        results = run_go_element_scan(
            "Pt",
            n_atoms,
            n_atoms,
            params=params,
            seed=args.seed,
            output_dir=output_dir,
        )
        minima = results.get(formula, [])
        n_found = len(minima)
        total_minima += n_found
        if n_found == 0:
            logger.info("  %s: 0 minima", formula)
            continue
        best_energy = minima[0][0]
        logger.info("  %s: %d minima, best E=%.6f eV", formula, n_found, best_energy)
        profile = load_latest_ga_profile(output_dir, formula)
        if profile:
            for line in format_ga_profile_lines(profile, detailed=True, max_entries=8):
                logger.info("    %s", line)

    logger.info(
        "Finished benchmark. Total minima saved across formulas: %d under %s",
        total_minima,
        output_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBenchmark wall time: {elapsed:.1f} s ({args.backend})")


if __name__ == "__main__":
    main()
