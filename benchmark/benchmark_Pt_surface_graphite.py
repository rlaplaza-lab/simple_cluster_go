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
import os
import time
from pathlib import Path

from ase import Atoms
from ase.build import graphene

from benchmark.benchmark_common import (
    DEFAULT_CLUSTERS,
    PT_SURFACE_GRAPHITE_RESULTS_DIR,
    format_ga_profile_lines,
    get_benchmark_params,
    load_latest_ga_profile,
)
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

DEFAULT_MIN_ATOMS = 4
DEFAULT_MAX_ATOMS = 11
DEFAULT_SEED = 42
DEFAULT_UMA_TASK = "oc25"
DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_OUTPUT_ROOT = PT_SURFACE_GRAPHITE_RESULTS_DIR.resolve()

logger = get_logger(__name__)


def _build_graphite_slab(
    layers: int = DEFAULT_SLAB_LAYERS,
    vacuum: float = 12.0,
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
    slab_vacuum: float,
    slab_repeat_xy: int,
) -> SurfaceSystemConfig:
    slab = _build_graphite_slab(
        layers=slab_layers,
        vacuum=slab_vacuum,
        repeat_xy=slab_repeat_xy,
    )
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
    parser.add_argument(
        "--backend",
        choices=("mace", "uma"),
        default=os.environ.get("SCGO_BENCHMARK_BACKEND", "mace"),
        help=(
            "mace: TorchSim GA + MACE (GPU-friendly default for ML potentials); "
            "uma: ASE GA + UMA (install scgo[uma] in a separate env)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override calculator model_name (default UMA: uma-s-1p2; MACE: preset default).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--uma-task",
        default=DEFAULT_UMA_TASK,
        help="UMA task_name (only used when --backend uma).",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        metavar="FORMULA",
        help="Optional subset (e.g. Pt4 Pt5). Default: full DEFAULT_CLUSTERS list.",
    )
    parser.add_argument("--niter", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--slab-layers", type=int, default=DEFAULT_SLAB_LAYERS)
    parser.add_argument("--slab-vacuum", type=float, default=12.0)
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=DEFAULT_SLAB_REPEAT_XY,
        help="In-plane supercell repeats; default is large enough for Pt4–Pt11 adsorption.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--profile-top-n",
        type=int,
        default=8,
        help="Number of top timing phases to print from ga_profile.json.",
    )
    parser.add_argument(
        "--profile-compact",
        action="store_true",
        help="Print only compact profiling summary (disable per-phase timing table).",
    )
    return parser.parse_args()


def _parse_pt_atom_count(cluster_formula: str) -> int:
    digits = "".join(filter(str.isdigit, cluster_formula))
    if not digits:
        raise ValueError(f"Cluster formula '{cluster_formula}' missing atom count.")
    return int(digits)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    clusters = args.clusters or DEFAULT_CLUSTERS

    surface_config = _make_surface_config(
        args.slab_layers,
        args.slab_vacuum,
        args.slab_repeat_xy,
    )

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
        n_atoms = _parse_pt_atom_count(formula)
        formula = get_cluster_formula(["Pt"] * n_atoms)
        results = run_scgo_campaign_one_element(
            "Pt",
            n_atoms,
            n_atoms,
            params=params,
            seed=args.seed,
            output_dir=args.output_root,
        )
        minima = results.get(formula, [])
        n_found = len(minima)
        total_minima += n_found
        if n_found == 0:
            logger.info("  %s: 0 minima", formula)
            continue
        best_energy = minima[0][0]
        logger.info("  %s: %d minima, best E=%.6f eV", formula, n_found, best_energy)
        profile = load_latest_ga_profile(args.output_root, formula)
        if profile:
            for line in format_ga_profile_lines(
                profile,
                detailed=not args.profile_compact,
                max_entries=max(1, args.profile_top_n),
            ):
                logger.info("    %s", line)

    logger.info(
        "Finished benchmark. Total minima saved across formulas: %d under %s",
        total_minima,
        args.output_root,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBenchmark wall time: {elapsed:.1f} s ({args.backend})")


if __name__ == "__main__":
    main()
