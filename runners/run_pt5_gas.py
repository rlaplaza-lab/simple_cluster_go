#!/usr/bin/env python3
"""Pt5 gas-phase GO then TS via ``run_go_ts``."""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts

N_ATOMS = 5
ELEMENT = "Pt"
SEED = 42
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_gas"

# HPC / production GO+TS numbers (MACE + TorchSim GA, flat TS for NEB)
NITER = 10
POPULATION_SIZE = 50
MAX_PAIRS = 15


def main() -> None:
    go = get_torchsim_ga_params(SEED)
    go["calculator"] = "MACE"
    ga = go["optimizer_params"]["ga"]
    ga["niter"] = NITER
    ga["population_size"] = POPULATION_SIZE
    ts = get_ts_search_params(system_type="gas_cluster")
    ts["max_pairs"] = MAX_PAIRS
    run_go_ts(
        [ELEMENT] * N_ATOMS,
        go=go,
        ts=ts,
        seed=SEED,
        output_dir=None,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        verbosity=1,
        system_type="gas_cluster",
    )


if __name__ == "__main__":
    main()
