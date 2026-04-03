"""Combine a metal core with an OH fragment."""

from __future__ import annotations

import numpy as np
from ase import Atoms


def combine_core_adsorbate(core: Atoms, adsorbate: Atoms) -> Atoms:
    """Concatenate ``core`` then ``adsorbate``; inherit cell and PBC from ``core``.

    Args:
        core: Bare cluster (metal only).
        adsorbate: Fragment atoms in world coordinates (e.g. ``OH``, ``O``, ``H2O``).

    Returns:
        New ``Atoms`` with ``len(core) + len(adsorbate)`` atoms.
    """
    if len(adsorbate) == 0:
        return core.copy()
    out = core.copy() + adsorbate.copy()
    out.set_cell(core.get_cell())
    out.set_pbc(core.get_pbc())
    return out


def expand_cubic_cell_to_fit(atoms: Atoms, margin: float) -> None:
    """Set a cubic cell and center so all atoms lie inside with ``margin`` padding.

    Mutates ``atoms`` in place. For ``pbc=False`` isolated clusters.
    """
    pos = atoms.get_positions()
    if len(pos) == 0:
        return
    span = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
    side = max(span + margin, margin)
    atoms.set_cell([side, side, side])
    atoms.center()
