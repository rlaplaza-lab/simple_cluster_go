"""When SCGO may use TorchSim batched NEB/GA paths.

TorchSim can drive multiple model families (e.g., MACE, FairChem/UMA) via model
wrappers in ``torch_sim.models``. Policy helpers here validate whether TorchSim
may be used for a given calculator name and whether the required stack is
installed.

Design note:
- If a caller explicitly requests TorchSim (``use_torchsim=True``), we **fail
  fast** when TorchSim (or the required model support) is missing; we do not
  silently fall back to ASE.
"""

from __future__ import annotations

import importlib.util

from ase.calculators.calculator import Calculator


def mace_torchsim_stack_available() -> bool:
    """True if ``torch_sim`` (``scgo[mace]``) is importable."""
    return importlib.util.find_spec("torch_sim") is not None


def calculator_name_supports_torchsim_batched_neb(calculator_name: str) -> bool:
    """True if calculator family can run TorchSim NEB when its stack is installed."""
    name = calculator_name.strip().upper()
    return name in ("MACE", "UMA")


def _require_torchsim() -> None:
    if importlib.util.find_spec("torch_sim") is None:
        raise ImportError(
            "TorchSim was requested but torch_sim is not installed. "
            "Install the appropriate extra (e.g., pip install 'scgo[uma]' or 'scgo[mace]')."
        )


def _require_torchsim_fairchem() -> None:
    _require_torchsim()
    # torch_sim.models.fairchem requires fairchem-core; validate importability.
    if importlib.util.find_spec("fairchem") is None:
        raise ImportError(
            "TorchSim FairChem/UMA support was requested but fairchem-core is not installed. "
            "Install with: pip install 'scgo[uma]'."
        )


def is_uma_like_calculator(calculator: Calculator | None) -> bool:
    """True for UMA / FAIRChem ASE calculators (never use TorchSim batched path)."""
    if calculator is None:
        return False
    cls_name = calculator.__class__.__name__
    return cls_name in ("UMA", "FAIRChemCalculator")


def resolve_ts_torchsim_flags(
    calculator_name: str,
    use_torchsim: bool | None,
    use_parallel_neb: bool | None,
) -> tuple[bool, bool]:
    """Return effective ``(use_torchsim, use_parallel_neb)`` for TS search.

    If TorchSim is **not** requested, returns ``(False, False)``.
    If TorchSim is requested but unavailable/misconfigured, raises ImportError/ValueError.
    """
    if not bool(use_torchsim):
        return False, False

    name = calculator_name.strip().upper()
    if not calculator_name_supports_torchsim_batched_neb(calculator_name):
        raise ValueError(
            f"Calculator {calculator_name!r} does not support TorchSim NEB."
        )
    if name == "UMA":
        _require_torchsim_fairchem()
    else:
        _require_torchsim()

    return True, bool(use_parallel_neb)
