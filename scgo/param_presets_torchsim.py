"""GA presets that require the MACE + TorchSim stack (``scgo[mace]``)."""

from __future__ import annotations

from typing import Any

import torch

from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.param_presets import _get_base_ga_benchmark_params


def get_torchsim_ga_params_impl(seed: int) -> dict[str, Any]:
    """Return GA params using TorchSim relaxer."""
    params = _get_base_ga_benchmark_params(seed)

    fmax_val = params["optimizer_params"]["ga"]["fmax"]
    niter_local = params["optimizer_params"]["ga"]["niter_local_relaxation"]

    params["optimizer_params"]["ga"].update(
        {
            "relaxer": TorchSimBatchRelaxer(
                force_tol=fmax_val,
                optimizer_name="fire",
                mace_model_name="mace_matpes_0",
                seed=seed,
                max_steps=niter_local,
                dtype=torch.float32,
                autobatcher=True,
                expected_max_atoms=600,
            ),
        },
    )

    return params
