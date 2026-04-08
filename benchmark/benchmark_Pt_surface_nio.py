#!/usr/bin/env python3
"""Benchmark Pt cluster search on NiO(100) with top-layer slab relaxation.

Mirrors ``benchmark/benchmark_Pt.py`` for a surface system: sweeps Pt cluster
sizes with one seed, runs SCGO over the range, and logs per-size minima from
the campaign. Outputs live under ``benchmark/results/pt_surface_nio/`` (same
layout as gas-phase benchmarks: ``<Formula>_searches`` per size). See
``benchmark.benchmark_common.PT_SURFACE_NIO_RESULTS_DIR``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from ase.build import bulk, surface

from benchmark.benchmark_common import PT_SURFACE_NIO_RESULTS_DIR
from scgo.param_presets import get_torchsim_ga_params
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

DEFAULT_MIN_ATOMS = 4
DEFAULT_MAX_ATOMS = 11
DEFAULT_SEED = 42
DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_OUTPUT_ROOT = PT_SURFACE_NIO_RESULTS_DIR.resolve()

logger = get_logger(__name__)


def _build_nio_100_slab(
    layers: int = DEFAULT_SLAB_LAYERS,
    vacuum: float = 12.0,
    repeat_xy: int = DEFAULT_SLAB_REPEAT_XY,
) -> Atoms:
    """Build a NiO(100) slab with periodic in-plane boundary conditions."""
    nio_bulk = bulk("NiO", "rocksalt", a=4.17)
    slab = surface(nio_bulk, (1, 0, 0), layers=layers, vacuum=vacuum)
    slab = slab.repeat((repeat_xy, repeat_xy, 1))
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)
    return slab


def _make_surface_config(
    slab_layers: int,
    slab_vacuum: float,
    slab_repeat_xy: int,
) -> SurfaceSystemConfig:
    slab = _build_nio_100_slab(
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
            "Run Pt-on-NiO(100) SCGO benchmark sweep with top-layer slab relaxation."
        )
    )
    parser.add_argument("--min-atoms", type=int, default=DEFAULT_MIN_ATOMS)
    parser.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_atoms > args.max_atoms:
        raise ValueError("--min-atoms must be <= --max-atoms")

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    surface_config = _make_surface_config(
        args.slab_layers,
        args.slab_vacuum,
        args.slab_repeat_xy,
    )

    params = get_torchsim_ga_params(seed=args.seed)
    params["optimizer_params"]["ga"]["surface_config"] = surface_config
    params["optimizer_params"]["ga"]["n_jobs_population_init"] = 1
    params["optimizer_params"]["ga"]["niter"] = args.niter
    params["optimizer_params"]["ga"]["population_size"] = args.population_size
    params["optimizer_params"]["ga"]["batch_size"] = 4

    logger.info(
        "Running Pt-on-NiO benchmark: sizes Pt%d..Pt%d, seed=%d",
        args.min_atoms,
        args.max_atoms,
        args.seed,
    )
    logger.info(
        "Slab constraints: fix_all_slab_atoms=%s, n_relax_top_slab_layers=%s",
        surface_config.fix_all_slab_atoms,
        surface_config.n_relax_top_slab_layers,
    )

    results = run_scgo_campaign_one_element(
        "Pt",
        args.min_atoms,
        args.max_atoms,
        params=params,
        seed=args.seed,
        output_dir=args.output_root,
    )

    logger.info("Per-size summary:")
    total_minima = 0
    for n_atoms in range(args.min_atoms, args.max_atoms + 1):
        formula = get_cluster_formula(["Pt"] * n_atoms)
        minima = results.get(formula, [])
        n_found = len(minima)
        total_minima += n_found
        if n_found == 0:
            logger.info("  %s: 0 minima", formula)
            continue
        best_energy = minima[0][0]
        logger.info("  %s: %d minima, best E=%.6f eV", formula, n_found, best_energy)

    logger.info(
        "Finished benchmark. Total minima saved across sizes: %d under %s",
        total_minima,
        args.output_root,
    )


if __name__ == "__main__":
    main()
