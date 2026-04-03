"""Energy calculator interfaces and helpers.

This package contains helper modules for various energy calculators:

- MACE: Machine learning potentials based on MACE models
- ORCA: Quantum chemistry calculations via ORCA
- VASP: DFT calculations via VASP
- TorchSim: GPU-accelerated batch relaxation framework

Note:
    Most of these are implementation details or used by runner scripts.
    The ``MACE`` calculator class is the primary user-facing calculator.
    Other helpers are primarily for internal use.
"""

from __future__ import annotations

from .mace_helpers import MACE, MaceUrls
from .orca_helpers import prepare_orca_calculations, write_orca_inputs
from .torchsim_helpers import (
    MemoryScalerCache,
    TorchSimBatchRelaxer,
    get_global_memory_scaler_cache,
)
from .vasp_helpers import prepare_vasp_calculations

__all__ = [
    # MACE
    "MACE",
    "MaceUrls",
    # ORCA
    "prepare_orca_calculations",
    "write_orca_inputs",
    # TorchSim
    "TorchSimBatchRelaxer",
    "MemoryScalerCache",
    "get_global_memory_scaler_cache",
    # VASP
    "prepare_vasp_calculations",
]
