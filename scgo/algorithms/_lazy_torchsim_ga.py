"""Single lazy-import entry for :func:`ga_go_torchsim` (optional TorchSim stack)."""

from __future__ import annotations

from typing import Any, Callable

_MSG = (
    "TorchSim GA requires TorchSim. Install with: pip install 'scgo[mace]' "
    "(MACE) or 'scgo[uma]' (UMA) depending on the model family."
)


def get_ga_go_torchsim() -> Callable[..., Any]:
    try:
        from scgo.algorithms.geneticalgorithm_go_torchsim import (
            ga_go_torchsim,
        )
    except ImportError as e:
        raise ImportError(_MSG) from e
    return ga_go_torchsim
