#!/usr/bin/env python3
"""Pt5 on graphite GO+TS (``run_go_ts_one_element``, ``make_graphite_surface_config``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from scgo.param_presets import pt5_graphite_go_ts_defaults
from scgo.runner_api import log_go_ts_summary, run_go_ts_one_element
from scgo.surface import make_graphite_surface_config
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_graphite"


def main() -> None:
    d0 = pt5_graphite_go_ts_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mace", "uma"), default="mace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--uma-task", default="oc25")
    parser.add_argument("--niter", type=int, default=d0["niter"])
    parser.add_argument("--population-size", type=int, default=d0["population_size"])
    parser.add_argument("--max-pairs", type=int, default=d0["max_pairs"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--slab-layers",
        type=int,
        default=None,
        help="Graphite layers (default: surface preset).",
    )
    parser.add_argument(
        "--slab-repeat-xy",
        type=int,
        default=None,
        help="In-plane supercell repeats (default: surface preset).",
    )
    args = parser.parse_args()
    slab_kw = {}
    if args.slab_layers is not None:
        slab_kw["slab_layers"] = args.slab_layers
    if args.slab_repeat_xy is not None:
        slab_kw["slab_repeat_xy"] = args.slab_repeat_xy
    surface_config = make_graphite_surface_config(**slab_kw)

    t0 = perf_counter()
    summary = run_go_ts_one_element(
        ELEMENT,
        N_ATOMS,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
        backend=args.backend,
        seed=args.seed,
        niter=args.niter,
        population_size=args.population_size,
        max_pairs=args.max_pairs,
        regime="surface",
        model_name=args.model_name,
        uma_task=args.uma_task,
        surface_config=surface_config,
        output_dir=args.output_root,
        verbosity=1,
        infer_ts_composition_from_minima=True,
        ga_n_jobs_population_init=d0["ga_n_jobs_population_init"],
        ga_batch_size=d0["ga_batch_size"],
    )
    log_go_ts_summary(logger, summary, wall_time_s=perf_counter() - t0)


if __name__ == "__main__":
    main()
