"""Warning integration helpers for SCGO runtime logging."""

from __future__ import annotations

import logging

_FILTERS_INSTALLED = False


def apply_scgo_runtime_warning_filters() -> None:
    """Route Python warnings through logging (do not silence them)."""
    global _FILTERS_INSTALLED
    if _FILTERS_INSTALLED:
        return
    _FILTERS_INSTALLED = True
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.propagate = True
    warnings_logger.setLevel(logging.WARNING)


__all__ = ["apply_scgo_runtime_warning_filters"]
