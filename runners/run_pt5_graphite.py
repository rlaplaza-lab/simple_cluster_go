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
import sys
from pathlib import Path
from time import perf_counter

from ase import Atoms
from ase.build import graphene

# Allow `python runners/run_pt5_graphite.py` from the repo root to import the
# sibling _common module without requiring `python -m runners.run_pt5_graphite`.
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
from scgo.runner_surface import read_full_composition_from_first_xyz  # noqa: E402
from scgo.surface.config import SurfaceSystemConfig  # noqa: E402
from scgo.ts_search import run_transition_state_search  # noqa: E402
from scgo.utils.helpers import get_cluster_formula  # noqa: E402
from scgo.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_SLAB_LAYERS = 5
DEFAULT_SLAB_REPEAT_XY = 4
DEFAULT_SLAB_VACUUM = 12.0
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_graphite"


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
        help="In-plane supercell repeats; default is large enough for Pt5 adsorption.",
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

    surface_config = _make_surface_config(args.slab_layers, args.slab_repeat_xy)
    ga_params = make_ga_params(args, surface_config=surface_config)
    # Surface GA needs extra parallel-init knobs the gas-phase preset omits.
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
