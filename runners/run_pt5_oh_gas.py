#!/usr/bin/env python3
"""Pt5+OH gas-phase: GO + TS via ``run_go_ts``.

``system_type= gas_cluster_adsorbate`` with ``deposition_layout=monolithic`` — one
initial cluster for the full mobile composition; no ``adsorbate_fragment_template`` is
required (fragment placement is only used for ``core_then_fragment``).
"""

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
# Monolithic: single gas-phase cluster for Pt5+OH; hierarchical fragment template N/A.
ADSORBATE_DEFINITION = {
    "adsorbate_symbols": ["O", "H"],
    "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
    "deposition_layout": "monolithic",
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
