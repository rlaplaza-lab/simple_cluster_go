#!/usr/bin/env python3
"""Pt5+2OH on graphite: GO + TS via ``run_go_ts``.

``system_type= surface_cluster_adsorbate`` with ``deposition_layout=core_then_fragment``:
hierarchical core + rigid OH dimer fragment, then deposition on the preset graphite slab.
"""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts
from scgo.surface import build_default_fragment_template, make_graphite_surface_config

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"]
SEED = 42
SYSTEM_TYPE = "surface_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_2oh_graphite"

NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
GA_BATCH_SIZE = 4
ADSORBATE_DEFINITION = {
    "adsorbate_symbols": ["O", "H", "O", "H"],
    "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
    "deposition_layout": "core_then_fragment",
}


def _build_go_params(surface_config) -> dict:
    go_params = get_torchsim_ga_params(SEED)
    go_params["calculator"] = "MACE"
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
        adsorbate_definition=ADSORBATE_DEFINITION,
        adsorbate_fragment_template=build_default_fragment_template(
            ADSORBATE_DEFINITION["adsorbate_symbols"]
        ),
    )


if __name__ == "__main__":
    main()
