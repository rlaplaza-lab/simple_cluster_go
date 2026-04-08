"""Geometric validation for slab + adsorbate (supported cluster) deposits."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.initialization.geometry_helpers import (
    get_covalent_radius,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)

# Small slack below nominal slab top (numerical / structural roughness).
_BINDING_PENETRATION_TOLERANCE_A = 0.1


def _slab_top_coordinate(slab: Atoms, axis: int) -> float:
    """Max Cartesian coordinate of slab atoms along ``axis`` (vacuum side)."""
    pos = slab.get_positions()
    if len(pos) == 0:
        return 0.0
    return float(np.max(pos[:, axis]))


def validate_supported_cluster_deposit(
    combined: Atoms,
    n_slab: int,
    *,
    surface_normal_axis: int,
    use_mic: bool = False,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    penetration_tolerance: float = _BINDING_PENETRATION_TOLERANCE_A,
) -> tuple[bool, str]:
    """Validate a combined slab+adsorbate structure after placement.

    Uses the same clash/connectivity semantics as gas-phase
    :func:`~scgo.initialization.geometry_helpers.validate_cluster_structure` on
    the adsorbate slice, and requires at least one adsorbate–slab pair within
    the covalent-radius connectivity threshold (aligned with
    ``connectivity_factor``). Optionally uses MIC for cross-set distances when
    ``use_mic`` is True (match :attr:`SurfaceSystemConfig.comparator_use_mic`).

    Args:
        combined: Full system with slab atoms first, then adsorbate.
        n_slab: Number of slab atoms (prefix length).
        surface_normal_axis: Cartesian axis index for the surface normal.
        use_mic: Pass through to ``Atoms.get_distance`` for adsorbate–slab pairs.
        min_distance_factor: Adsorbate self clash scale (initialization default).
        connectivity_factor: Bonding connectivity scale (initialization default).
        penetration_tolerance: Allow adsorbate atoms this far (Å) below the
            nominal slab top along ``surface_normal_axis``.

    Returns:
        ``(True, "")`` if valid, else ``(False, message)``.
    """
    n = len(combined)
    if n_slab < 0 or n_slab > n:
        return False, f"Invalid n_slab={n_slab} for len(combined)={n}"
    if n_slab == n:
        return False, "No adsorbate atoms in combined structure"

    ads = combined[n_slab:]
    ok, err = validate_cluster_structure(
        ads,
        min_distance_factor,
        connectivity_factor,
        check_clashes=True,
        check_connectivity=True,
    )
    if not ok:
        return False, f"Adsorbate validation failed: {err}"

    slab = combined[:n_slab]
    slab_top = _slab_top_coordinate(slab, surface_normal_axis)
    positions = combined.get_positions()
    ads_coords = positions[n_slab:]
    axis_coord = ads_coords[:, surface_normal_axis]
    if bool(np.any(axis_coord < slab_top - penetration_tolerance)):
        min_c = float(np.min(axis_coord))
        return (
            False,
            "Adsorbate penetrates below nominal slab top along surface normal "
            f"(min coord={min_c:.3f} Å, slab_top={slab_top:.3f} Å)",
        )

    symbols = combined.get_chemical_symbols()
    touches = False
    min_cross = float("inf")
    for i in range(n_slab, n):
        r_i = get_covalent_radius(symbols[i])
        for j in range(n_slab):
            r_j = get_covalent_radius(symbols[j])
            threshold = (r_i + r_j) * connectivity_factor
            d = float(combined.get_distance(i, j, mic=use_mic))
            min_cross = min(min_cross, d)
            if d <= threshold:
                touches = True
                break
        if touches:
            break

    if not touches:
        return (
            False,
            "No adsorbate–slab pair within connectivity distance "
            f"(min cross-set distance={min_cross:.3f} Å, "
            f"connectivity_factor={connectivity_factor})",
        )

    return True, ""
