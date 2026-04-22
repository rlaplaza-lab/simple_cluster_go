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
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers

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


def _reorder_cluster_to_composition(
    cluster: Atoms, composition: Sequence[str]
) -> Atoms:
    """Reorder generated cluster atoms to match requested symbol sequence."""
    desired = list(composition)
    current = cluster.get_chemical_symbols()
    if current == desired:
        return cluster

    by_symbol: dict[str, list[int]] = {}
    for idx, sym in enumerate(current):
        by_symbol.setdefault(sym, []).append(idx)

    selection: list[int] = []
    for sym in desired:
        matching = by_symbol.get(sym)
        if not matching:
            raise ValueError(
                "Generated cluster symbols do not match requested composition."
            )
        selection.append(matching.pop(0))
    return cluster[selection].copy()


def _random_rotation_matrix(rng: Generator) -> np.ndarray:
    """Return a uniformly random 3D rotation matrix."""
    rotation_axis = rng.standard_normal(3)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return _generate_rotation_matrix(rotation_axis, rotation_angle)


def _place_cluster_above_slab(
    cluster_positions: np.ndarray,
    slab: Atoms,
    slab_top: float,
    axis: int,
    rng: Generator,
    config: SurfaceSystemConfig,
) -> np.ndarray:
    """Rotate/translate centered cluster positions into a deposited position."""
    rotated_positions = cluster_positions @ _random_rotation_matrix(rng).T
    translated_positions = rotated_positions + _in_plane_translation(slab, axis, rng)
    sampled_height = float(
        rng.uniform(config.adsorption_height_min, config.adsorption_height_max)
    )
    cluster_min = float(np.min(translated_positions[:, axis]))
    translated_positions[:, axis] += slab_top + sampled_height - cluster_min
    return translated_positions


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
        cluster_seed = create_initial_cluster(
            list(composition),
            vacuum=config.cluster_init_vacuum,
            rng=rng,
            previous_search_glob=previous_search_glob,
            mode=config.init_mode,
        )
        cluster_seed = _reorder_cluster_to_composition(cluster_seed, composition)
        atomic_numbers = cluster_seed.get_atomic_numbers()
        cluster_positions = cluster_seed.get_positions().copy()

        cluster_positions -= np.mean(cluster_positions, axis=0)
        deposited_positions = _place_cluster_above_slab(
            cluster_positions=cluster_positions,
            slab=slab,
            slab_top=slab_top,
            axis=axis,
            rng=rng,
            config=config,
        )

        adsorbate = Atoms(
            numbers=atomic_numbers,
            positions=deposited_positions,
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
        )

        if atoms_too_close(adsorbate, blmin, use_tags=False):
            continue
        if atoms_too_close_two_sets(adsorbate, slab, blmin):
            continue

        return combine_slab_adsorbate(slab, adsorbate)

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

    # Parallel: precompute deterministic per-task seeds on the main thread.
    per_worker_limit = max(config.max_placement_attempts, 50)
    task_seeds = [
        int(rng.integers(0, 2**63 - 1, dtype=np.int64)) for _ in range(n_structures)
    ]

    def _build_structure_with_seed(task_seed: int) -> Atoms:
        task_rng = np.random.default_rng(task_seed)
        for _ in range(per_worker_limit):
            child_rng = np.random.default_rng(
                task_rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            structure = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
            )
            if structure is not None:
                return structure
        raise RuntimeError(
            "Could not generate deposited structure in parallel worker; "
            "try widening height range or increasing max_placement_attempts."
        )

    workers = min(n_structures, resolve_n_jobs_to_workers(n_jobs))
    ordered_results: list[Atoms | None] = [None] * n_structures
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="scgo_deposit"
    ) as ex:
        futures = {
            ex.submit(_build_structure_with_seed, seed): idx
            for idx, seed in enumerate(task_seeds)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    if any(result is None for result in ordered_results):
        raise RuntimeError("Parallel batch returned too few structures")
    return [result for result in ordered_results if result is not None]
