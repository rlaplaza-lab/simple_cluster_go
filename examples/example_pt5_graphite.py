#!/usr/bin/env python3
"""Pt5 on graphite: GO + TS via ``run_go_ts``.

``system_type= surface_cluster`` — supported Pt5 cluster on the preset graphite slab (no
separate adsorbate fragment).
"""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts
from scgo.surface import make_graphite_surface_config

N_ATOMS = 5
ELEMENT = "Pt"
SEED = 42
SYSTEM_TYPE = "surface_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_graphite"

NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
GA_BATCH_SIZE = 4


def _build_go_params(surface_config) -> dict:
    """Load GO preset, then apply surface-specific knobs for this run."""
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
    """Load TS preset, then tweak max pair budget."""
    ts_params = get_ts_search_params(
        system_type=SYSTEM_TYPE,
        seed=SEED,
    )
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8  # override default
    return ts_params


def main() -> None:
    surface_config = make_graphite_surface_config(
        structure_connectivity_factor=1.8,
    )
    go_params = _build_go_params(surface_config)
    ts_params = _build_ts_params()
    run_go_ts(
        [ELEMENT] * N_ATOMS,
        go_params=go_params,
        ts_params=ts_params,
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        surface_config=surface_config,
        system_type=SYSTEM_TYPE,
    )


if __name__ == "__main__":
    main()
