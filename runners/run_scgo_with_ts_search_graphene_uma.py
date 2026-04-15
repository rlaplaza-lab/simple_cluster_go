#!/usr/bin/env python3
"""Run SCGO with UMA on a surface slab then perform TS search."""

import os
from pathlib import Path
from time import perf_counter

from ase.build import graphene
from torch.profiler import ProfilerActivity, profile

from scgo.param_presets import (
    get_default_uma_params,
    get_ts_run_kwargs,
    get_ts_search_params_uma,
)
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.runner_surface import (
    make_surface_config,
    read_full_composition_from_first_xyz,
)
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

ELEMENT = "Cu"
MIN_ATOMS = 4
MAX_ATOMS = 4
RANDOM_SEED = 42

logger = get_logger(__name__)


def _resolve_output_root(default_dir_name: str) -> Path:
    configured_root = os.environ.get("SCGO_OUTPUT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(default_dir_name).resolve()


OUTPUT_ROOT = _resolve_output_root("ts_search_graphene_results_uma")


def _make_slab():
    """Build a 3x3 graphene slab — swap this for any ASE Atoms slab."""
    slab = graphene(size=(3, 3, 1), vacuum=12.0)
    slab.pbc = True
    return slab


def main() -> None:
    run_started = perf_counter()
    slab = _make_slab()
    surface_config = make_surface_config(slab)

    ga_params = get_default_uma_params()
    ga_params["seed"] = RANDOM_SEED
    ga_params["optimizer_params"]["ga"]["n_jobs_population_init"] = 1
    ga_params["optimizer_params"]["ga"]["surface_config"] = surface_config
    ga_params["optimizer_params"]["ga"]["batch_size"] = 4
    ga_params["optimizer_params"]["ga"]["previous_search_glob"] = (
        "__no_seed_dbs__/**/*.db"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    go_formula = get_cluster_formula([ELEMENT] * MIN_ATOMS)
    formula_search_dir = OUTPUT_ROOT / f"{go_formula}_searches"
    final_minima_dir = formula_search_dir / "final_unique_minima"
    minima_exist = final_minima_dir.exists() and any(final_minima_dir.glob("*.xyz"))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        minima_started = perf_counter()
        if not minima_exist:
            run_scgo_campaign_one_element(
                ELEMENT,
                MIN_ATOMS,
                MAX_ATOMS,
                params=ga_params,
                seed=RANDOM_SEED,
                output_dir=OUTPUT_ROOT,
            )
        else:
            logger.info("Reusing existing minima under %s", final_minima_dir)
        logger.info(
            "Minima search stage completed in %.2f s",
            perf_counter() - minima_started,
        )

        full_composition = read_full_composition_from_first_xyz(final_minima_dir)
        logger.info(
            "Detected full surface composition for TS search: %s", full_composition
        )

        ts_params = get_ts_search_params_uma(
            regime="surface",
            surface_config=surface_config,
        )
        ts_kwargs = get_ts_run_kwargs(ts_params)
        ts_kwargs["max_pairs"] = 15
        ts_started = perf_counter()
        ts_results = run_transition_state_search(
            full_composition,
            base_dir=formula_search_dir,
            seed=RANDOM_SEED,
            verbosity=1,
            **ts_kwargs,
        )

        n_success = sum(1 for result in ts_results if result.get("status") == "success")
        logger.info(
            "Finished TS search for surface/cluster. Successful NEBs: %d/%d",
            n_success,
            len(ts_results),
        )
        logger.info("TS search stage completed in %.2f s", perf_counter() - ts_started)
        logger.info("Total runner wall time: %.2f s", perf_counter() - run_started)

    print("\n--- Profiler CPU Output ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print("\n--- Profiler CUDA Output ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
