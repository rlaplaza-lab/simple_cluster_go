"""Run SCGO for a cluster composition with UMA and perform TS searches."""

import os
from pathlib import Path
from time import perf_counter

from torch.profiler import ProfilerActivity, profile

from scgo.param_presets import (
    get_default_uma_params,
    get_ts_run_kwargs,
    get_ts_search_params_uma,
)
from scgo.run_minima import run_scgo_campaign_one_element
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger

# Configuration
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


OUTPUT_ROOT = _resolve_output_root("ts_search_results_uma")


def main() -> None:
    run_started = perf_counter()
    ga_params = get_default_uma_params()
    ga_params["seed"] = RANDOM_SEED
    ga_params["optimizer_params"]["ga"]["n_jobs_population_init"] = 1
    ga_params["optimizer_params"]["ga"]["previous_search_glob"] = (
        "__no_seed_dbs__/**/*.db"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        minima_started = perf_counter()
        run_scgo_campaign_one_element(
            ELEMENT,
            MIN_ATOMS,
            MAX_ATOMS,
            params=ga_params,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_ROOT,
        )
        logger.info(
            "Minima search stage completed in %.2f s",
            perf_counter() - minima_started,
        )

        ts_params = get_ts_search_params_uma()
        total_found = 0
        ts_started = perf_counter()

        for n_atoms in range(MIN_ATOMS, MAX_ATOMS + 1):
            composition = [ELEMENT] * n_atoms
            formula_search_dir = (
                OUTPUT_ROOT / f"{get_cluster_formula(composition)}_searches"
            )
            ts_results = run_transition_state_search(
                composition,
                base_dir=formula_search_dir,
                seed=RANDOM_SEED,
                **get_ts_run_kwargs(ts_params),
            )
            total_found += sum(
                1 for result in ts_results if result["status"] == "success"
            )

        logger.info(
            "Finished TS search: %d successful NEB runs under %s",
            total_found,
            OUTPUT_ROOT,
        )
        logger.info("TS search stage completed in %.2f s", perf_counter() - ts_started)
        logger.info("Total runner wall time: %.2f s", perf_counter() - run_started)

    print("\n--- Profiler CPU Output ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print("\n--- Profiler CUDA Output ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
