"""Place gas-phase cluster seeds onto a slab for GA initialization."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.initialization.geometry_helpers import _generate_rotation_matrix
from scgo.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy.random import Generator

    from scgo.surface.config import SurfaceSystemConfig

logger = get_logger(__name__)


def slab_surface_extreme(slab: Atoms, axis: int, *, upper: bool = True) -> float:
    """Return max (or min) Cartesian coordinate of slab atoms along ``axis``."""
    pos = slab.get_positions()
    if len(pos) == 0:
        return 0.0
    return float(np.max(pos[:, axis]) if upper else np.min(pos[:, axis]))


def _in_plane_translation(slab: Atoms, axis: int, rng: Generator) -> np.ndarray:
    """Random fractional shift along the two cell directions not dominated by ``axis``.

    For ``axis == 2``, uses ``cell[0]`` and ``cell[1]``. Uses ``[0, 1)`` fractions.
    """
    cell = slab.get_cell()
    u, v = rng.random(), rng.random()
    if axis == 0:
        shift = u * cell[1] + v * cell[2]
    elif axis == 1:
        shift = u * cell[0] + v * cell[2]
    else:
        shift = u * cell[0] + v * cell[1]
    return np.asarray(shift, dtype=float)


def combine_slab_adsorbate(slab: Atoms, adsorbate: Atoms) -> Atoms:
    """Concatenate slab and adsorbate; adsorbate cell/pbc are replaced by slab's."""
    ads = adsorbate.copy()
    ads.set_cell(slab.get_cell())
    ads.set_pbc(slab.get_pbc())
    return slab.copy() + ads


def create_deposited_cluster(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    rng: Generator,
    config: SurfaceSystemConfig,
    previous_search_glob: str = "**/*.db",
) -> Atoms | None:
    """One adsorbate+slab structure, or None if placement fails.

    Uses :func:`scgo.initialization.create_initial_cluster` for the gas-phase
    seed, then rotates and translates it above the slab with random in-plane
    offset and height in ``[adsorption_height_min, adsorption_height_max]``.
    """
    from scgo.initialization import create_initial_cluster

    axis = config.surface_normal_axis
    slab_top = slab_surface_extreme(slab, axis, upper=True)

    for _ in range(config.max_placement_attempts):
        cluster = create_initial_cluster(
            list(composition),
            vacuum=config.cluster_init_vacuum,
            rng=rng,
            previous_search_glob=previous_search_glob,
            mode=config.init_mode,
        )
        nums = cluster.get_atomic_numbers()
        pos = cluster.get_positions().copy()

        pos -= np.mean(pos, axis=0)
        axis_rot = rng.standard_normal(3)
        axis_rot /= np.linalg.norm(axis_rot)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        rmat = _generate_rotation_matrix(axis_rot, angle)
        pos = pos @ rmat.T

        pos += _in_plane_translation(slab, axis, rng)

        h = float(
            rng.uniform(config.adsorption_height_min, config.adsorption_height_max)
        )
        cluster_min = float(np.min(pos[:, axis]))
        pos[:, axis] += slab_top + h - cluster_min

        ads = Atoms(
            numbers=nums,
            positions=pos,
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
        )

        if atoms_too_close(ads, blmin, use_tags=False):
            continue
        if atoms_too_close_two_sets(ads, slab, blmin):
            continue

        return combine_slab_adsorbate(slab, ads)

    logger.warning(
        "create_deposited_cluster: exceeded max_placement_attempts=%s",
        config.max_placement_attempts,
    )
    return None


def create_deposited_cluster_batch(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    n_structures: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    *,
    previous_search_glob: str = "**/*.db",
    n_jobs: int = 1,
) -> list[Atoms]:
    """Generate multiple deposited structures (sequential or threaded)."""
    if n_structures <= 0:
        return []

    max_attempts = max(n_structures * 50, config.max_placement_attempts)

    if n_jobs == 1:
        out: list[Atoms] = []
        attempts = 0
        while len(out) < n_structures and attempts < max_attempts:
            attempts += 1
            child_rng = np.random.default_rng(
                rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            struct = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
            )
            if struct is not None:
                out.append(struct)
        if len(out) < n_structures:
            raise RuntimeError(
                f"Could only generate {len(out)} of {n_structures} deposited structures; "
                "try widening height range or increasing max_placement_attempts."
            )
        return out

    # Parallel: each worker retries with fresh RNG draws (same spirit as sequential).
    per_worker_limit = max(config.max_placement_attempts, 50)

    def _one() -> Atoms:
        for _ in range(per_worker_limit):
            child_rng = np.random.default_rng(
                rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            s = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
            )
            if s is not None:
                return s
        raise RuntimeError("Parallel deposition failed for one structure")

    workers = min(n_structures, n_jobs if n_jobs > 0 else 1)
    results: list[Atoms] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one) for _ in range(n_structures)]
        results.extend(f.result() for f in as_completed(futures))
    if len(results) < n_structures:
        raise RuntimeError("Parallel batch returned too few structures")
    return results
