#!/usr/bin/env python3
"""Pt5+2OH on graphite: GO + TS via ``run_go_ts``.

``system_type= surface_cluster_adsorbate`` with hierarchical-only adsorbate inputs:
core-only ``COMPOSITION`` plus one-or-more adsorbate ASE ``Atoms`` fragments.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts
from scgo.surface import make_graphite_surface_config

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt"]
SEED = 42
SYSTEM_TYPE = "surface_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_2oh_graphite"

NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
GA_BATCH_SIZE = 4
ADSORBATES = [
    Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]),
    Atoms(symbols=["O", "H"], positions=[[2.2, 0.0, 0.0], [2.2, 0.0, 0.96]]),
]

# Use more lenient connectivity factor for Pt+OH systems
CLUSTER_ADSORBATE_CONFIG = ClusterAdsorbateConfig(
    structure_connectivity_factor=1.8,
)


def _build_go_params(surface_config) -> dict:
    go_params = get_torchsim_ga_params(SEED)
    go_params["calculator"] = "MACE"
    go_params["connectivity_factor"] = 1.8  # override default
    go_params["optimizer_params"]["ga"].update(
        niter=NITER,
        population_size=POPULATION_SIZE,
        surface_config=surface_config,
        batch_size=GA_BATCH_SIZE,
    )
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["energy_gap_threshold"] = 1.0
    ts_params["neb_n_images"] = 7
    ts_params["neb_steps"] = 800
    ts_params["connectivity_factor"] = 1.8  # override default
    return ts_params


def main() -> None:
    surface_config = make_graphite_surface_config()
    go_params = _build_go_params(surface_config)
    ts_params = _build_ts_params()
    run_go_ts(
        COMPOSITION,
        go_params=go_params,
        ts_params=ts_params,
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        surface_config=surface_config,
        system_type=SYSTEM_TYPE,
        adsorbates=ADSORBATES,
        cluster_adsorbate_config=CLUSTER_ADSORBATE_CONFIG,
    )


if __name__ == "__main__":
    main()
