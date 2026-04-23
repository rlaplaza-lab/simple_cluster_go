#!/usr/bin/env python3
"""Pt5OH gas-phase cluster+adsorbate GO+TS via ``run_go_ts``."""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H"]
SEED = 42
SYSTEM_TYPE = "gas_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_oh_gas"

NITER = 8
POPULATION_SIZE = 40
MAX_PAIRS = 12
ADSORBATE_DEFINITION = {
    "adsorbate_symbols": ["O", "H"],
    "core_symbols": ["Pt"],
}


def main() -> None:
    go_params = get_torchsim_ga_params(SEED)
    go_params["calculator"] = "MACE"
    go_params["optimizer_params"]["ga"].update(
        niter=NITER, population_size=POPULATION_SIZE
    )
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    run_go_ts(
        COMPOSITION,
        go_params=go_params,
        ts_params=ts_params,
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        system_type=SYSTEM_TYPE,
        adsorbate_definition=ADSORBATE_DEFINITION,
    )


if __name__ == "__main__":
    main()
