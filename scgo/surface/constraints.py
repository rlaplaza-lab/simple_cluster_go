"""ASE constraints for slab + adsorbate systems."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.validation import validate_surface_config_slab_prefix


def _distinct_layers_along_axis(
    positions: np.ndarray,
    axis: int,
    n_layers: int,
) -> set[int]:
    """Return atom indices belonging to the n_layers lowest distinct layers."""
    coord = positions[:, axis]
    # Round to reduce float noise when atoms share a layer
    rounded = np.round(coord, decimals=6)
    unique_vals = np.sort(np.unique(rounded))
    if len(unique_vals) < n_layers:
        return set(range(len(positions)))
    cutoff = unique_vals[n_layers - 1]
    indices = {i for i in range(len(positions)) if rounded[i] <= cutoff + 1e-9}
    return indices


def _indices_in_top_n_distinct_layers(
    positions: np.ndarray,
    axis: int,
    n_top: int,
) -> set[int]:
    """Return atom indices in the n_top highest distinct coordinate layers."""
    if n_top < 1 or len(positions) == 0:
        return set()
    coord = positions[:, axis]
    rounded = np.round(coord, decimals=6)
    unique_vals = np.sort(np.unique(rounded))
    if len(unique_vals) <= n_top:
        return set(range(len(positions)))
    top_vals = unique_vals[-n_top:]
    top_set = set(top_vals.tolist())
    return {i for i in range(len(positions)) if rounded[i] in top_set}


def attach_slab_constraints(
    atoms: Atoms,
    n_slab: int,
    *,
    fix_all_slab_atoms: bool,
    n_fix_bottom_slab_layers: int | None,
    n_relax_top_slab_layers: int | None = None,
    surface_normal_axis: int,
) -> None:
    """Attach ``FixAtoms`` for slab atoms; clears existing constraints.

    To relax only the top N slab layers (typical surface region), either set
    ``n_relax_top_slab_layers=N`` or fix the bottom ``L - N`` layers via
    ``n_fix_bottom_slab_layers`` (``L`` = distinct slab layers along the normal).

    Args:
        atoms: Combined slab + adsorbate system (slab indices ``0 .. n_slab-1``).
        n_slab: Number of slab atoms.
        fix_all_slab_atoms: Fix every slab atom.
        n_fix_bottom_slab_layers: If ``fix_all_slab_atoms`` is False and this is
            set, fix only bottom layers (among slab atoms only). Mutually
            exclusive with ``n_relax_top_slab_layers`` at the config level.
        n_relax_top_slab_layers: If ``fix_all_slab_atoms`` is False and this is
            set, fix all slab atoms except those in the top N distinct layers.
        surface_normal_axis: Cartesian axis for layer grouping.
    """
    if n_slab > len(atoms):
        raise ValueError(
            f"attach_slab_constraints: n_slab={n_slab} exceeds len(atoms)={len(atoms)}"
        )

    atoms.constraints = []
    if n_slab <= 0:
        return

    if n_relax_top_slab_layers is not None and n_fix_bottom_slab_layers is not None:
        raise ValueError(
            "attach_slab_constraints: use at most one of "
            "n_fix_bottom_slab_layers and n_relax_top_slab_layers"
        )

    slab_positions = atoms.get_positions()[:n_slab]

    if fix_all_slab_atoms:
        fix_idx = list(range(n_slab))
        atoms.set_constraint(FixAtoms(indices=fix_idx))
        return

    if n_relax_top_slab_layers is not None:
        mobile = _indices_in_top_n_distinct_layers(
            np.asarray(slab_positions),
            surface_normal_axis,
            n_relax_top_slab_layers,
        )
        fix_idx = sorted(set(range(n_slab)) - mobile)
        if fix_idx:
            atoms.set_constraint(FixAtoms(indices=fix_idx))
        return

    if n_fix_bottom_slab_layers is None:
        return

    layer_idx = _distinct_layers_along_axis(
        np.asarray(slab_positions),
        surface_normal_axis,
        n_fix_bottom_slab_layers,
    )
    atoms.set_constraint(FixAtoms(indices=sorted(layer_idx)))


def attach_slab_constraints_from_surface_config(
    atoms: Atoms, config: SurfaceSystemConfig
) -> None:
    """Apply the same ``FixAtoms`` policy as global optimization on ``SurfaceSystemConfig``.

    Use this (or pass ``surface_config`` into :func:`run_transition_state_search`) so
    NEB endpoints match the slab freezing used during GA / local relaxation
    (``fix_all_slab_atoms``, layer-relax modes, ``surface_normal_axis``).
    """
    validate_surface_config_slab_prefix(atoms, config)
    attach_slab_constraints(
        atoms,
        len(config.slab),
        fix_all_slab_atoms=config.fix_all_slab_atoms,
        n_fix_bottom_slab_layers=config.n_fix_bottom_slab_layers,
        n_relax_top_slab_layers=config.n_relax_top_slab_layers,
        surface_normal_axis=config.surface_normal_axis,
    )


def surface_slab_constraint_summary(config: SurfaceSystemConfig) -> dict[str, Any]:
    """JSON-safe snapshot of slab fixing (no embedded :class:`~ase.Atoms` slab)."""
    return {
        "n_slab_atoms": len(config.slab),
        "surface_normal_axis": config.surface_normal_axis,
        "fix_all_slab_atoms": config.fix_all_slab_atoms,
        "n_fix_bottom_slab_layers": config.n_fix_bottom_slab_layers,
        "n_relax_top_slab_layers": config.n_relax_top_slab_layers,
    }
