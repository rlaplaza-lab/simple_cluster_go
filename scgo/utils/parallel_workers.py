"""Resolve ``n_jobs`` style parallelism (-1 / -2 / positive) to a worker count."""

from __future__ import annotations

import os


def resolve_n_jobs_to_workers(n_jobs: int) -> int:
    """Map batch-parallel ``n_jobs`` to a concrete worker count (``>= 1``).

    Semantics match :func:`scgo.initialization.initializers.create_initial_cluster_batch`:

    - ``1``: sequential (callers using ``max_workers == 1`` stay single-threaded).
    - ``> 1``: use that many workers.
    - ``-1``: all logical CPUs.
    - ``-2``: all logical CPUs except one.
    """
    if n_jobs < 1 and n_jobs not in (-1, -2):
        raise ValueError(f"n_jobs must be >= 1, -1, or -2, got {n_jobs}")
    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu
    if n_jobs == -2:
        return max(1, cpu - 1)
    return n_jobs
