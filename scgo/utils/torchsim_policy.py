"""When SCGO may use TorchSim batched NEB/GA paths (MACE stack only).

UMA (fairchem) and classical calculators must use ASE NEB with an attached
calculator. ``TorchSimBatchRelaxer`` loads MACE inside TorchSim and cannot
drive UMA checkpoints.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ase.calculators.calculator import Calculator


def mace_torchsim_stack_available() -> bool:
    """True if ``torch_sim`` (``scgo[mace]``) is importable."""
    return importlib.util.find_spec("torch_sim") is not None


def calculator_name_supports_torchsim_batched_neb(calculator_name: str) -> bool:
    """TorchSim NEB/parallel NEB only supports the MACE-backed relaxer today."""
    return calculator_name.strip().upper() == "MACE"


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
    *,
    logger: Any | None = None,
) -> tuple[bool, bool]:
    """Return effective ``(use_torchsim, use_parallel_neb)`` for TS search.

    Coerces incompatible combinations (UMA + TorchSim, missing stack, etc.)
    to ASE NEB. Parallel NEB is disabled whenever TorchSim is disabled.
    """
    us = bool(use_torchsim)
    up = bool(use_parallel_neb)

    if not us:
        return False, False

    reason: str | None = None
    if not calculator_name_supports_torchsim_batched_neb(calculator_name):
        reason = (
            f"Calculator {calculator_name!r} does not use the TorchSim+MACE batched "
            "NEB path; using ASE NEB with the selected calculator."
        )
    elif not mace_torchsim_stack_available():
        reason = (
            "TorchSim batched NEB requested but the MACE stack is not installed "
            "(pip install 'scgo[mace]'); using ASE NEB instead."
        )

    if reason is not None:
        if logger is not None:
            logger.warning(reason)
        return False, False

    return True, up


def coerce_find_transition_state_torchsim(
    *,
    use_torchsim: bool,
    calculator: Calculator | None,
    pair_id: str,
    logger: Any,
) -> bool:
    """If ``calculator`` is UMA-like, force ASE NEB even when ``use_torchsim`` was True."""
    if not use_torchsim:
        return False
    if calculator is None or not is_uma_like_calculator(calculator):
        return use_torchsim
    logger.warning(
        "Pair %s: UMA/FAIRChem calculator cannot use TorchSim NEB; switching to ASE NEB.",
        pair_id,
    )
    return False
