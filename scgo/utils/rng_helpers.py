"""Random number generator utilities for reproducible optimization."""

from __future__ import annotations

import numpy as np


def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """Convert optional seed to Generator at API boundary.

    Args:
        seed: Optional integer seed. If None, creates unseeded RNG.

    Returns:
        np.random.Generator instance.
    """
    return np.random.default_rng(seed)


def create_child_rng(parent_rng: np.random.Generator) -> np.random.Generator:
    """Create a child RNG from parent for independent random streams.

    Args:
        parent_rng: Parent RNG to derive child from.

    Returns:
        New RNG with seed derived from parent.
    """
    seed = parent_rng.integers(0, 2**63 - 1)
    return np.random.default_rng(seed)


def ensure_rng_or_create(rng: np.random.Generator | None) -> np.random.Generator:
    """Ensure an RNG exists, creating one if None.

    Args:
        rng: Optional RNG. If None, creates a new unseeded RNG.

    Returns:
        np.random.Generator instance (never None).

    Raises:
        TypeError: If ``rng`` is not ``None`` and not an instance of ``np.random.Generator``.
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    raise TypeError("rng must be an instance of numpy.random.Generator or None")
