#!/usr/bin/env python3
"""Pt5+2OH on graphite GO+TS; TS composition inferred from minima."""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts
from scgo.surface import make_graphite_surface_config

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"]
SEED = 42
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_2oh_graphite"

# Same production scale as ``run_pt5_graphite`` (HPC)
NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
GA_N_JOBS_POPULATION_INIT = -2
GA_BATCH_SIZE = 4


def main() -> None:
    surface_config = make_graphite_surface_config()
    go = get_torchsim_ga_params(SEED)
    go["calculator"] = "MACE"
    ga = go["optimizer_params"]["ga"]
    ga["niter"] = NITER
    ga["population_size"] = POPULATION_SIZE
    ga["surface_config"] = surface_config
    ga["n_jobs_population_init"] = GA_N_JOBS_POPULATION_INIT
    ga["batch_size"] = GA_BATCH_SIZE
    ts = get_ts_search_params(
        system_type="surface_cluster_adsorbate",
        surface_config=surface_config,
    )
    ts["max_pairs"] = MAX_PAIRS
    # Prefer closer-energy endpoint pairs first; graphite runs can otherwise spend
    # budget on distant minima that rarely produce interior saddles.
    ts["energy_gap_threshold"] = 1.0
    ts["neb_n_images"] = 7
    ts["neb_steps"] = 800
    run_go_ts(
        COMPOSITION,
        go=go,
        ts=ts,
        seed=SEED,
        output_dir=None,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        verbosity=1,
        infer_ts_composition_from_minima=True,
        system_type="surface_cluster_adsorbate",
    )


if __name__ == "__main__":
    main()
