"""Optimizer utilities for ASE optimizer class resolution.

This module provides utility functions for converting optimizer string names
to their corresponding ASE optimizer classes, eliminating duplicate code.
"""

from __future__ import annotations

from ase.optimize import BFGS, FIRE, LBFGS


def get_optimizer_class(optimizer_name: str) -> type:
    """Convert optimizer string name to ASE optimizer class.

    Args:
        optimizer_name: String name of the optimizer ("FIRE", "BFGS", "LBFGS").
            Case-insensitive.

    Returns:
        The corresponding ASE optimizer class.

    Raises:
        ValueError: If optimizer_name is None or not supported.
    """
    if optimizer_name is None:
        raise ValueError("optimizer_name cannot be None")

    # Map optimizer names to classes
    OPTIMIZERS = {
        "FIRE": FIRE,
        "BFGS": BFGS,
        "LBFGS": LBFGS,
    }

    optimizer_upper = optimizer_name.upper()
    if optimizer_upper not in OPTIMIZERS:
        supported = ", ".join(OPTIMIZERS.keys())
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. Supported optimizers: {supported}",
        )

    return OPTIMIZERS[optimizer_upper]
