"""Third-party warning filters applied when SCGO configures logging."""

from __future__ import annotations

import os
import warnings

_FILTERS_INSTALLED = False


def apply_scgo_runtime_warning_filters() -> None:
    """Register one-time filters for known noisy library warnings."""
    global _FILTERS_INSTALLED
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    if _FILTERS_INSTALLED:
        return
    _FILTERS_INSTALLED = True
    warnings.filterwarnings(
        "ignore",
        message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Extending slab periodicity to 3D for VASP compatibility.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*All systems have reached the maximum number of steps: \d+.*",
        category=UserWarning,
    )


__all__ = ["apply_scgo_runtime_warning_filters"]
