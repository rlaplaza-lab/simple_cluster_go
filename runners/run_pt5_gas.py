#!/usr/bin/env python3
"""Example runner: Pt5 gas-phase global optimization followed by TS search.

Mirrors the gas-phase benchmark setup in
``benchmark/benchmark_Pt.py`` but narrowed to a single composition (Pt5) and
chained with :func:`scgo.ts_search.run_transition_state_search` to find
transition states connecting the recovered minima.

Supports both MACE (TorchSim GA) and UMA (FairChem) backends; install the
matching ``scgo[mace]`` or ``scgo[uma]`` extra in its own environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Allow `python runners/run_pt5_gas.py` from the repo root to import the
# sibling _common module without requiring `python -m runners.run_pt5_gas`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners._common import (  # noqa: E402
    add_common_args,
    make_ga_params,
    make_ts_kwargs,
    resolve_output_root,
)
from scgo.run_minima import run_scgo_campaign_one_element  # noqa: E402
from scgo.ts_search import run_transition_state_search  # noqa: E402
from scgo.utils.helpers import get_cluster_formula  # noqa: E402
from scgo.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_gas"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_niter=10,
        default_population_size=50,
        default_max_pairs=15,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    output_root = resolve_output_root(
        args,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
    )

    composition = [ELEMENT] * N_ATOMS
    formula = get_cluster_formula(composition)

    ga_params = make_ga_params(args)
    ts_kwargs = make_ts_kwargs(args, regime="gas")

    t_start = perf_counter()

    logger.info(
        "Running SCGO GO for %s (backend=%s, seed=%d) under %s",
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
    logger.info("Running TS search for %s under %s", formula, ts_base_dir)
    ts_results = run_transition_state_search(
        composition,
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
