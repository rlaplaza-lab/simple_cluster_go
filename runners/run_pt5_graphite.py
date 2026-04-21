#!/usr/bin/env python3
"""Example runner: Pt5 on a graphite slab, global optimization + TS search.

Mirrors the surface benchmark setup in
``benchmark/benchmark_Pt_surface_graphite.py`` but narrowed to a single
composition (Pt5) and chained with
:mod:`scgo.runner_api` workflows to find transition states
connecting the recovered minima. The top slab layer is allowed to relax during
both global optimization and NEB, matching the benchmark.

Supports both MACE (TorchSim GA) and UMA (FairChem) backends; install the
matching ``scgo[mace]`` or ``scgo[uma]`` extra in its own environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgo import run_go_ts_one_element
from scgo.param_presets import pt5_graphite_go_ts_defaults
from scgo.surface.presets import (
    DEFAULT_GRAPHITE_SLAB_LAYERS,
    DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    make_graphite_surface_config,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_graphite"

# GA/TS knobs and batch settings: :func:`scgo.param_presets.pt5_graphite_go_ts_defaults`
RUN_CONFIG: dict[str, Any] = {
    "backend": "mace",  # "mace" | "uma"
    "seed": 42,
    "model_name": None,  # None -> backend default
    "uma_task": "oc25",  # only used for UMA backend
    "slab_layers": DEFAULT_GRAPHITE_SLAB_LAYERS,
    "slab_repeat_xy": DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    # None -> runners/results/pt5_graphite_<backend>
    "output_dir": None,
}


def main() -> None:
    surface_config = make_graphite_surface_config(
        slab_layers=int(RUN_CONFIG["slab_layers"]),
        slab_repeat_xy=int(RUN_CONFIG["slab_repeat_xy"]),
    )
    summary = run_go_ts_one_element(
        ELEMENT,
        N_ATOMS,
        default_output_parent=DEFAULT_OUTPUT_PARENT,
        default_output_subdir=DEFAULT_OUTPUT_SUBDIR,
        backend=str(RUN_CONFIG["backend"]),
        seed=int(RUN_CONFIG["seed"]),
        model_name=RUN_CONFIG.get("model_name"),
        uma_task=str(RUN_CONFIG.get("uma_task", "oc25")),
        output_dir=RUN_CONFIG.get("output_dir"),
        regime="surface",
        surface_config=surface_config,
        infer_ts_composition_from_minima=True,
        **pt5_graphite_go_ts_defaults(),
    )
    logger.info(
        "Finished %s graphite GO->TS under %s (successful NEBs=%d/%d, wall_time=%.2f s)",
        summary["formula"],
        summary["output_dir"],
        summary["ts_success_count"],
        summary["ts_total_count"],
        summary["wall_time_s"],
    )


if __name__ == "__main__":
    main()
