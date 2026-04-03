"""Skip GPU-only tests when CUDA is unavailable (shared by tests and conftest)."""

from __future__ import annotations

import pytest
import torch


def require_cuda(reason: str = "CUDA not available") -> None:
    """Call at the start of a GPU-only test; skips if `torch.cuda.is_available()` is false."""
    if not torch.cuda.is_available():
        pytest.skip(reason)
