#!/usr/bin/env python3
"""Pt5 gas-phase GO then TS (``run_go_ts_one_element``, ``pt5_gas_go_ts_defaults``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from scgo.param_presets import pt5_gas_go_ts_defaults
from scgo.runner_api import log_go_ts_summary, run_go_ts_one_element
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_gas"


def main() -> None:
    d0 = pt5_gas_go_ts_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mace", "uma"), default="mace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--uma-task", default="oc25")
    parser.add_argument("--niter", type=int, default=d0["niter"])
    parser.add_argument("--population-size", type=int, default=d0["population_size"])
    parser.add_argument("--max-pairs", type=int, default=d0["max_pairs"])
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

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
        regime="gas",
        model_name=args.model_name,
        uma_task=args.uma_task,
        output_dir=args.output_root,
        verbosity=1,
    )
    log_go_ts_summary(logger, summary, wall_time_s=perf_counter() - t0)


if __name__ == "__main__":
    main()
