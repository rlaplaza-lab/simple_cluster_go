#!/usr/bin/env python3
"""Pt5+OH gas-phase: GO + TS via ``run_go_ts``.

``system_type= gas_cluster_adsorbate`` with hierarchical-only adsorbate inputs:
core-only ``COMPOSITION`` plus one adsorbate fragment as an ASE ``Atoms`` object.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt"]
SEED = 42
SYSTEM_TYPE = "gas_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_oh_gas"

NITER = 8
POPULATION_SIZE = 40
MAX_PAIRS = 12
ADSORBATES = [Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])]

# Use more lenient connectivity factor for Pt+OH systems (default 1.4 may be too strict)
CLUSTER_ADSORBATE_CONFIG = ClusterAdsorbateConfig(
    structure_connectivity_factor=1.8,
)


def main() -> None:
    go_params = get_torchsim_ga_params(SEED)
    go_params["calculator"] = "MACE"
    go_params["connectivity_factor"] = 1.8  # override default
    go_params["optimizer_params"]["ga"].update(
        niter=NITER, population_size=POPULATION_SIZE
    )
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8  # override default
    run_go_ts(
        COMPOSITION,
        go_params=go_params,
        ts_params=ts_params,
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        system_type=SYSTEM_TYPE,
        adsorbates=ADSORBATES,
        cluster_adsorbate_config=CLUSTER_ADSORBATE_CONFIG,
    )


if __name__ == "__main__":
    main()
