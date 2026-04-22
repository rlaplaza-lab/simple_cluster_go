#!/usr/bin/env python3
"""Pt5 gas-phase GO then TS via canonical ``run_go_ts`` API."""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import build_one_element_go_ts_bundle, pt5_gas_go_ts_defaults
from scgo.runner_api import resolve_runner_output_dir, run_go_ts

N_ATOMS = 5
ELEMENT = "Pt"
BACKEND = "mace"
SEED = 42
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_gas"


def main() -> None:
    d0 = pt5_gas_go_ts_defaults()
    bundle = build_one_element_go_ts_bundle(
        backend=BACKEND,
        seed=SEED,
        niter=d0["niter"],
        population_size=d0["population_size"],
        max_pairs=d0["max_pairs"],
        system_type="gas_cluster",
    )
    run_go_ts(
        [ELEMENT] * N_ATOMS,
        ga_params=bundle["ga_params"],
        ts_kwargs=bundle["ts_kwargs"],
        seed=SEED,
        output_dir=resolve_runner_output_dir(
            default_output_parent=DEFAULT_OUTPUT_PARENT,
            default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
            backend=BACKEND,
            output_dir=None,
        ),
        verbosity=1,
        system_type="gas_cluster",
    )


if __name__ == "__main__":
    main()
