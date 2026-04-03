"""Global optimization to find minima.

This package contains the core workflow for global optimization of atomic
clusters: single-trial execution, multi-trial orchestration with deduplication,
Hessian validation, and result persistence.
"""

from scgo.minima_search.core import run_trials, scgo

__all__ = ["run_trials", "scgo"]
