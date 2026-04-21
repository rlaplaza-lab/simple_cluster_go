#!/usr/bin/env python3
"""Example runner: Pt5 + 2OH on graphite, global optimization + TS search.

This mirrors ``runners/run_pt5_graphite.py`` but uses an explicit mixed
adsorbate composition (Pt5OHOH). The script runs global optimization first,
then a transition-state search over minima found in the same output tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from ase import Atoms
from ase.build import graphene

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners._common import (  # noqa: E402
    add_common_args,
    make_ga_params,
    make_ts_kwargs,
    resolve_output_root,
)
from scgo.runner_api import run_go_campaign, run_ts_search  # noqa: E402
from scgo.runner_surface import read_full_composition_from_first_xyz  # noqa: E402
from scgo.surface.config import SurfaceSystemConfig  # noqa: E402
from scgo.utils.helpers import get_cluster_formula  # noqa: E402
from scgo.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"]
DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_SLAB_VACUUM = 12.0
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_2oh_graphite"


def _build_graphite_slab(
    layers: int = DEFAULT_SLAB_LAYERS,
    vacuum: float = DEFAULT_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_SLAB_REPEAT_XY,
) -> Atoms:
    slab = graphene(formula="C2", vacuum=vacuum)
    slab = slab.repeat((repeat_xy, repeat_xy, max(1, layers)))
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)
    return slab


def _make_surface_config(slab_layers: int, slab_repeat_xy: int) -> SurfaceSystemConfig:
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
    add_common_args(
        parser,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_niter=6,
        default_population_size=24,
        default_max_pairs=10,
    )
    parser.add_argument("--slab-layers", type=int, default=DEFAULT_SLAB_LAYERS)
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=DEFAULT_SLAB_REPEAT_XY,
        help="In-plane repeats for graphite slab (default supports Pt5+2OH adsorption).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_root = resolve_output_root(
        args,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
    )

    formula = get_cluster_formula(COMPOSITION)
    surface_config = _make_surface_config(args.slab_layers, args.slab_repeat_xy)
    ga_params = make_ga_params(args, surface_config=surface_config)
    ga = ga_params["optimizer_params"]["ga"]
    ga["n_jobs_population_init"] = -2
    ga["batch_size"] = 4

    ts_kwargs = make_ts_kwargs(args, regime="surface", surface_config=surface_config)
    t_start = perf_counter()

    logger.info(
        "Running SCGO GO for %s on graphite (backend=%s, seed=%d) under %s",
        formula,
        args.backend,
        args.seed,
        output_root,
    )
    run_go_campaign(
        [COMPOSITION],
        params=ga_params,
        seed=args.seed,
        output_dir=output_root,
    )

    ts_base_dir = output_root / f"{formula}_searches"
    final_minima_dir = ts_base_dir / "final_unique_minima"
    full_composition = read_full_composition_from_first_xyz(final_minima_dir)

    logger.info("Running TS search for %s under %s", formula, ts_base_dir)
    ts_results = run_ts_search(
        full_composition,
        output_dir=ts_base_dir,
        params=ga_params,
        seed=args.seed,
        verbosity=1,
        surface_config=surface_config,
        ts_kwargs=ts_kwargs,
    )
    total_success = sum(1 for r in ts_results if r.get("status") == "success")
    logger.info("Successful NEBs: %d/%d", total_success, len(ts_results))
    logger.info("Total wall time: %.2f s", perf_counter() - t_start)


if __name__ == "__main__":
    main()
