"""Energy calculator interfaces and helpers.

This package contains helper modules for various energy calculators:

- MACE: Machine learning potentials based on MACE models (optional ``[mace]`` extra)
- UMA: FAIRChem UMA checkpoints (optional ``[uma]`` extra)
- ORCA: Quantum chemistry calculations via ORCA
- VASP: DFT calculations via VASP
- TorchSim: GPU-accelerated batch relaxation (requires ``[mace]``)

Note:
    MACE, UMA, and TorchSim symbols load lazily so ``import scgo.calculators``
    works with only the core dependencies. Install ``scgo[mace]`` or
    ``scgo[uma]`` for the corresponding stack (not both in one environment).
"""

from __future__ import annotations

from typing import Any

from .orca_helpers import prepare_orca_calculations, write_orca_inputs
from .vasp_helpers import prepare_vasp_calculations

__all__ = [
    "MACE",
    "MaceUrls",
    "UMA",
    "prepare_orca_calculations",
    "write_orca_inputs",
    "TorchSimBatchRelaxer",
    "MemoryScalerCache",
    "get_global_memory_scaler_cache",
    "prepare_vasp_calculations",
]


def __getattr__(name: str) -> Any:
    if name == "MACE":
        from .mace_helpers import MACE

        return MACE
    if name == "MaceUrls":
        from .mace_helpers import MaceUrls

        return MaceUrls
    if name == "UMA":
        from .uma_helpers import UMA

        return UMA
    if name in (
        "TorchSimBatchRelaxer",
        "MemoryScalerCache",
        "get_global_memory_scaler_cache",
    ):
        from .torchsim_helpers import (
            MemoryScalerCache,
            TorchSimBatchRelaxer,
            get_global_memory_scaler_cache,
        )

        return {
            "TorchSimBatchRelaxer": TorchSimBatchRelaxer,
            "MemoryScalerCache": MemoryScalerCache,
            "get_global_memory_scaler_cache": get_global_memory_scaler_cache,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
