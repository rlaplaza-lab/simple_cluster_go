#!/usr/bin/env python3
"""Example runner: Pt5 gas-phase global optimization followed by TS search.

Mirrors the gas-phase benchmark setup in
``benchmark/benchmark_Pt.py`` but narrowed to a single composition (Pt5) and
chained with transition-state search (see :mod:`scgo.runner_api`) to find
transition states connecting the recovered minima.

Supports both MACE (TorchSim GA) and UMA (FairChem) backends; install the
matching ``scgo[mace]`` or ``scgo[uma]`` extra in its own environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgo import run_go_ts_one_element
from scgo.param_presets import pt5_gas_go_ts_defaults
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

N_ATOMS = 5
ELEMENT = "Pt"
DEFAULT_OUTPUT_PARENT = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_SUBDIR = "pt5_gas"

# GA iteration / population / TS pair limits: :func:`scgo.param_presets.pt5_gas_go_ts_defaults`
RUN_CONFIG: dict[str, Any] = {
    "backend": "mace",  # "mace" | "uma"
    "seed": 42,
    "model_name": None,  # None -> backend default
    "uma_task": "oc25",  # only used for UMA backend
    # None -> runners/results/pt5_gas_<backend>
    "output_dir": None,
}


def main() -> None:
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
        regime="gas",
        **pt5_gas_go_ts_defaults(),
    )
    logger.info(
        "Finished %s GO->TS under %s (successful NEBs=%d/%d, wall_time=%.2f s)",
        summary["formula"],
        summary["output_dir"],
        summary["ts_success_count"],
        summary["ts_total_count"],
        summary["wall_time_s"],
    )


if __name__ == "__main__":
    main()
