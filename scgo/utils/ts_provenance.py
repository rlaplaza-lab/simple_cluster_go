"""Shared provenance fields for TS / NEB JSON outputs.

``schema_version`` tracks the provenance header; it is **3** for current SCGO
releases. On-disk layouts and per-file keys are documented in the repository
README (sections *What to expect on disk* and transition-state outputs).
"""

from __future__ import annotations

import sys

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone as UTC
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

TS_OUTPUT_SCHEMA_VERSION = 3
CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION = 1


def ts_output_provenance(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return flat metadata merged into TS/NEB JSON and GO ``results_summary.json``."""
    meta: dict[str, Any] = {
        "schema_version": TS_OUTPUT_SCHEMA_VERSION,
        "scgo_version": package_version("scgo"),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta


def package_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA OOM error (exception type or message pattern)."""
    import torch.cuda

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    # Fallback to message pattern matching
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg
