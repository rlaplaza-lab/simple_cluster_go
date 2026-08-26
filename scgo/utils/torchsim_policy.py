"""When SCGO may use TorchSim batched NEB/GA paths.

TorchSim can drive multiple model families (e.g., MACE, FairChem/UMA, UPET) via
model wrappers in ``torch_sim.models`` and ``metatomic_torchsim``. Policy helpers
here validate whether TorchSim may be used for a given calculator name and whether
the required stack is installed.

Design note:
- If a caller explicitly requests TorchSim (``use_torchsim=True``), we **fail fast**
  when TorchSim (or the required model support) is missing; we do not
  silently fall back to ASE.
"""

from __future__ import annotations

import importlib.util

from ase.calculators.calculator import Calculator

from scgo.exceptions import (
    SCGONotImplementedError,
    SCGOValidationError,
)


def calculator_name_supports_torchsim_batched_neb(calculator_name: str) -> bool:
    """True if calculator family can run TorchSim NEB when its stack is installed."""
    name = calculator_name.strip().upper()
    return name in ("MACE", "UMA", "UPET")


def _require_torchsim() -> None:
    """Fail fast when TorchSim is requested but not installed.

    Raises :class:`~scgo.exceptions.SCGONotImplementedError`, matching the
    missing-backend convention in ``scgo.calculators`` helper modules.
    """
    if importlib.util.find_spec("torch_sim") is None:
        raise SCGONotImplementedError(
            "TorchSim was requested but torch_sim is not installed. "
            "Install the appropriate extra (e.g., pip install 'scgo[uma]', "
            "'scgo[mace]', or 'scgo[upet]')."
        )


def _require_torchsim_fairchem() -> None:
    _require_torchsim()
    # torch_sim.models.fairchem requires fairchem-core; validate importability.
    if importlib.util.find_spec("fairchem") is None:
        raise SCGONotImplementedError(
            "TorchSim FairChem/UMA support was requested but fairchem-core is not installed. "
            "Install with: pip install 'scgo[uma]'."
        )


def _require_torchsim_upet() -> None:
    _require_torchsim()
    if importlib.util.find_spec("upet") is None:
        raise SCGONotImplementedError(
            "TorchSim UPET support was requested but upet is not installed. "
            "Install with: pip install 'scgo[upet]'."
        )
    if importlib.util.find_spec("metatomic_torchsim") is None:
        raise SCGONotImplementedError(
            "TorchSim UPET support was requested but metatomic-torchsim is not installed. "
            "Install with: pip install 'scgo[upet]'."
        )


_MLIP_CALCULATOR_CLASS_NAMES = frozenset(
    {"MACECalculator", "MACE", "UMA", "FAIRChemCalculator", "UPET", "UPETCalculator"}
)


def is_ml_calculator(calculator: Calculator) -> bool:
    """Return True when ``calculator`` is a known MLIP ASE calculator class."""
    return calculator.__class__.__name__ in _MLIP_CALCULATOR_CLASS_NAMES


def is_uma_like_calculator(calculator: Calculator | None) -> bool:
    """True for UMA / FAIRChem ASE calculator instances (by class name)."""
    if calculator is None:
        return False
    cls_name = calculator.__class__.__name__
    return cls_name in ("UMA", "FAIRChemCalculator")


def is_upet_like_calculator(calculator: Calculator | None) -> bool:
    """True for UPET ASE calculator instances (by class name)."""
    if calculator is None:
        return False
    cls_name = calculator.__class__.__name__
    return cls_name in ("UPET", "UPETCalculator")


def resolve_ts_torchsim_flags(
    calculator_name: str,
    use_torchsim: bool | None,
    use_parallel_neb: bool | None,
) -> tuple[bool, bool]:
    """Return effective ``(use_torchsim, use_parallel_neb)`` for TS search.

    If TorchSim is **not** requested, returns ``(False, False)``.
    If TorchSim is requested but unavailable/misconfigured, raises ImportError or
    SCGOValidationError.
    When TorchSim is on and ``use_parallel_neb`` is ``None``, parallel NEB defaults
    to ``True`` (matches presets).
    """
    if not bool(use_torchsim):
        return False, False

    name = calculator_name.strip().upper()
    if not calculator_name_supports_torchsim_batched_neb(calculator_name):
        raise SCGOValidationError(
            f"Calculator {calculator_name!r} does not support TorchSim NEB."
        )
    if name == "UMA":
        _require_torchsim_fairchem()
    elif name == "UPET":
        _require_torchsim_upet()
    else:
        _require_torchsim()

    # None → parallel on (preset default); explicit False stays off.
    return True, True if use_parallel_neb is None else bool(use_parallel_neb)
