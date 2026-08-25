"""Structural comparison tools for atomic clusters.

This module provides comparators for determining if two cluster structures are
geometrically equivalent, based on sorted interatomic distance analysis as
described in Vilhelmsen and Hammer, PRL 108, 126101 (2012).

Structures may optionally be compared in a **block-aware** mode: the atoms are
partitioned into role blocks (``mobile_slab`` / ``deposit`` / ``adsorbate``)
whose intra-block fingerprints are combined with configurable weights, and
cross-block element-pair distance lists capture binding geometry (e.g. an
adsorbate registry on top of relaxed slab layers) that pure intra-element
fingerprints cannot see.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from scipy.spatial.distance import cdist, pdist

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_CROSS_WEIGHT,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.metadata.atoms import get_tag

_SORTED_DIST_FP_INFO_KEY = "_scgo_sorted_dist_fp"
_SORTED_DIST_FP_ATTR_KEY = "_scgo_sorted_dist_fp_cache"
_BLOCK_FP_SLOT_PREFIX = "blocks"

BLOCK_ROLES: tuple[str, ...] = ("mobile_slab", "deposit", "adsorbate")
"""Canonical role names for block-aware structure comparison."""


def _validate_block_role(role: str) -> str:
    role_s = str(role)
    if role_s not in BLOCK_ROLES:
        raise SCGOValidationError(
            f"Unknown comparator block role {role!r}; expected one of "
            f"{list(BLOCK_ROLES)}."
        )
    return role_s


@dataclass(frozen=True)
class ComparatorBlock:
    """A named group of atom indices for block-aware comparison."""

    role: str
    """One of :data:`BLOCK_ROLES`."""

    indices: tuple[int, ...]
    """Strictly increasing atom indices into the full Atoms object."""

    def __post_init__(self) -> None:
        _validate_block_role(self.role)
        idx = tuple(int(i) for i in self.indices)
        # Pair each index with its successor; strict=False is intentional so a
        # single-element tuple yields no pairs.
        if (
            not idx
            or any(b <= a for a, b in zip(idx, idx[1:], strict=False))
            or idx[0] < 0
        ):
            raise SCGOValidationError(
                f"ComparatorBlock indices must be non-empty and strictly "
                f"increasing non-negative integers, got {self.indices!r}."
            )
        object.__setattr__(self, "indices", idx)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ComparatorBlocks:
    """Ordered, disjoint role blocks used by block-aware comparison."""

    blocks: tuple[ComparatorBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise SCGOValidationError("ComparatorBlocks requires at least one block.")
        roles = [b.role for b in self.blocks]
        if len(set(roles)) != len(roles):
            raise SCGOValidationError(
                f"ComparatorBlocks roles must be unique, got {roles!r}."
            )
        seen: set[int] = set()
        for block in self.blocks:
            overlap = seen.intersection(block.indices)
            if overlap:
                raise SCGOValidationError(
                    f"ComparatorBlocks overlap at indices {sorted(overlap)[:5]}..."
                )
            seen.update(block.indices)

    @classmethod
    def from_ranges(cls, ranges: Sequence[tuple[str, int, int]]) -> ComparatorBlocks:
        """Build contiguous blocks from ``(role, start, stop)`` triples."""
        return cls(
            blocks=tuple(
                ComparatorBlock(role=str(role), indices=tuple(range(start, stop)))
                for role, start, stop in ranges
            )
        )

    def signature(self) -> str:
        """Stable string identity for fingerprint cache slotting."""
        parts = []
        for block in self.blocks:
            idx = block.indices
            if list(idx) == list(range(idx[0], idx[-1] + 1)):
                span = f"{idx[0]}-{idx[-1]}"
            else:
                span = ",".join(str(i) for i in idx)
            parts.append(f"{block.role}:{span}")
        return ";".join(parts)


def _block_unit_sort_key(key: tuple) -> tuple:
    """Canonical ordering for fingerprint unit keys (stable descriptors)."""
    kind_rank = 0 if key[0] == "intra" else 1
    return (kind_rank, *key[1:])


def iter_ordered_units(
    units: dict[tuple, np.ndarray],
) -> list[tuple[tuple, np.ndarray]]:
    """Return fingerprint units in canonical order (intra before cross)."""
    return sorted(units.items(), key=lambda item: _block_unit_sort_key(item[0]))


def _sorted_dist_cache_slot(n_top: int, mic: bool) -> str:
    """Stable, serializable cache slot key for ``atoms.info`` storage."""
    return f"{int(n_top)}|{int(bool(mic))}"


def _get_sorted_dist_cache_store(atoms: Atoms) -> dict[str, dict]:
    """Return mutable in-memory cache store attached to the Atoms object.

    The heavy fingerprint data lives on a private attribute instead of
    ``atoms.info`` so exports (e.g. extxyz) do not carry non-serializable
    nested dictionaries. A lightweight marker in ``atoms.info`` preserves
    debuggability and backwards intent.
    """
    existing = getattr(atoms, _SORTED_DIST_FP_ATTR_KEY, None)
    if not isinstance(existing, dict):
        existing = {}
        setattr(atoms, _SORTED_DIST_FP_ATTR_KEY, existing)
    # Keep a tiny, hashable marker in atoms.info for visibility.
    atoms.info[_SORTED_DIST_FP_INFO_KEY] = "cached"
    return existing


def _array_dirty_token(arr: np.ndarray) -> tuple:
    """Cheap identity/content token for an ndarray (pointer, shape, dtype, sum)."""
    checksum = float(arr.flat[0] + arr.flat[-1] + float(arr.sum())) if arr.size else 0.0
    return (arr.__array_interface__["data"][0], arr.shape, arr.dtype.str, checksum)


def _atoms_geometry_dirty_token(atoms: Atoms, *, mic: bool) -> tuple:
    """Fast dirty check for positions/numbers/(cell) without hashing bytes."""
    pos = atoms.arrays["positions"]
    numbers = atoms.arrays["numbers"]
    token: tuple = (
        _array_dirty_token(pos),
        _array_dirty_token(numbers),
        bool(mic),
    )
    if mic or np.any(atoms.pbc):
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        pbc = tuple(bool(x) for x in atoms.pbc)
        token = (*token, _array_dirty_token(cell), pbc)
    return token


def _sorted_dist_content_key(atoms: Atoms, *, mic: bool) -> tuple:
    """Build a content key that invalidates when geometry/composition changes."""
    positions = np.ascontiguousarray(atoms.arrays["positions"], dtype=np.float64)
    numbers = np.ascontiguousarray(atoms.arrays["numbers"], dtype=np.int32)
    key: tuple = (hash(positions.tobytes()), hash(numbers.tobytes()), bool(mic))
    if mic or np.any(atoms.pbc):
        cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
        pbc = tuple(bool(x) for x in atoms.pbc)
        key = (*key, hash(cell.tobytes()), pbc)
    return key


def _compute_sorted_dist_list(
    atoms: Atoms,
    mic: bool,
    *,
    n_top: int = 0,
) -> dict[int, np.ndarray]:
    """Compute per-element sorted distance fingerprints without using the cache.

    When ``n_top > 0`` and ``n_top < len(atoms)`` only the trailing ``n_top``
    atoms are fingerprinted (using NumPy index views, not an Atoms slice).
    """
    all_pos = atoms.arrays["positions"]
    all_numbers = atoms.arrays["numbers"]

    if 0 < n_top < len(atoms):
        positions_arr = all_pos[-n_top:]
        numbers = all_numbers[-n_top:]
    else:
        positions_arr = all_pos
        numbers = all_numbers

    unique_types = set(numbers)
    pair_cor: dict[int, np.ndarray] = {}
    # Honor ``mic`` literally so GA uniqueness and Pure stay coherent when
    # callers pass mic=False on periodic cells (e.g. comparator_use_mic=False).
    use_mic_path = bool(mic)

    all_d: np.ndarray | None = None
    if use_mic_path:
        # Compute MIC distances only for the relevant subset while preserving
        # the original cell/PBC.
        # get_distances returns (D_vec, D_scalar); we only need D_scalar.
        _, all_d = get_distances(
            positions_arr,
            positions_arr,
            cell=atoms.cell.array,
            pbc=atoms.pbc,
        )
        # Mirror the upper-triangle behaviour: zero the lower triangle so
        # triu_indices picks only unique pairs (same as get_all_distances path).
        all_d = np.triu(all_d, k=1)

    for n in unique_types:
        i_un = np.flatnonzero(numbers == n)
        if i_un.size == 0:
            continue

        if not use_mic_path:
            d = pdist(positions_arr[i_un]).tolist()
        else:
            if all_d is None:
                raise TypeError(
                    "all_d distance matrix is required when use_mic_path=True"
                )
            sub = all_d[np.ix_(i_un, i_un)]
            d = sub[np.triu_indices(len(i_un), k=1)].tolist()

        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor


def get_sorted_dist_list(
    atoms: Atoms,
    mic: bool = False,
    *,
    n_top: int = 0,
) -> dict[int, np.ndarray]:
    """Calculates a dictionary of sorted interatomic distances for an Atoms object.

    This utility method is used to generate a structural fingerprint of a cluster
    by calculating all interatomic distances for each element type and sorting them.

    Results are cached on ``atoms.info`` under ``_scgo_sorted_dist_fp`` and
    invalidated when positions, numbers, or (for MIC) cell/PBC change. Cache hits
    use a cheap dirty token (array pointer/shape/sum) and skip byte hashing.

    The cache is keyed by ``(n_top, mic)`` so different subsets of the same
    structure can coexist in the cache without evicting each other.

    Args:
        atoms: The Atoms object for which to calculate the distances.
        mic: Whether to use the minimum image convention for periodic systems.
            Defaults to False.
        n_top: Number of trailing atoms to fingerprint.  When ``0`` (default)
            or ``>= len(atoms)`` all atoms are used.  The fingerprint is stored
            on the original ``atoms.info``; no ``Atoms`` slice is created.

    Returns:
        A dictionary where keys are atomic numbers (integers) and values are
        sorted 1D numpy arrays of interatomic distances for that element type.
    """
    mic_b = bool(mic)
    n_top_i = int(n_top)
    if n_top_i <= 0 or n_top_i >= len(atoms):
        # Canonicalize "full-structure" requests so cache keys are stable.
        n_top_i = 0
    cache_key = _sorted_dist_cache_slot(n_top_i, mic_b)

    # Cache entries are keyed by slot so different n_top values don't collide.
    fp_store = _get_sorted_dist_cache_store(atoms)
    cached = fp_store.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("pair_cor"), dict):
        dirty = _atoms_geometry_dirty_token(atoms, mic=mic_b)
        if cached.get("dirty_token") == dirty:
            return cached["pair_cor"]
        content_key = _sorted_dist_content_key(atoms, mic=mic_b)
        if cached.get("content_key") == content_key:
            cached["dirty_token"] = dirty
            return cached["pair_cor"]
    else:
        content_key = _sorted_dist_content_key(atoms, mic=mic_b)

    pair_cor = _compute_sorted_dist_list(atoms, mic=mic_b, n_top=n_top_i)
    fp_store[cache_key] = {
        "content_key": content_key,
        "dirty_token": _atoms_geometry_dirty_token(atoms, mic=mic_b),
        "pair_cor": pair_cor,
    }
    return pair_cor


def _sorted_subset_distances(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    *,
    same_set: bool,
    cell: np.ndarray | None,
    pbc: tuple[bool, ...] | None,
    mic: bool,
) -> np.ndarray:
    """Return sorted pairwise distances within one subset or between two."""
    if mic:
        if same_set:
            _, d = get_distances(pos_i, pos_i, cell=cell, pbc=pbc)
            d = np.triu(d, k=1)
            flat = d[np.triu_indices(len(pos_i), k=1)]
        else:
            _, d = get_distances(pos_i, pos_j, cell=cell, pbc=pbc)
            flat = d.ravel()
    elif same_set:
        flat = pdist(pos_i)
    else:
        flat = cdist(pos_i, pos_j).ravel()
    flat = np.asarray(flat, dtype=np.float64)
    return np.sort(flat)


def _compute_block_distance_units(
    atoms: Atoms,
    mic: bool,
    blocks: ComparatorBlocks,
) -> dict[tuple, np.ndarray]:
    """Compute per-unit sorted distance fingerprints for block-aware comparison.

    Units are keyed ``("intra", block_pos, atomic_number)`` for sorted distances
    among same-element pairs inside one block, and
    ``("cross", i, j, z_i, z_j)`` (``i < j`` block positions) for sorted
    distances between element subsets of two different blocks. Cross units make
    binding geometry (e.g. adsorbate registry on relaxed slab layers) visible
    to the comparator.
    """
    positions = atoms.arrays["positions"]
    numbers = atoms.arrays["numbers"]
    cell = atoms.cell.array if (mic or np.any(atoms.pbc)) else None
    pbc = tuple(bool(x) for x in atoms.pbc)

    sub_positions = [positions[np.asarray(b.indices)] for b in blocks.blocks]
    sub_numbers = [numbers[np.asarray(b.indices)] for b in blocks.blocks]

    units: dict[tuple, np.ndarray] = {}
    for bi in range(len(blocks.blocks)):
        for z in np.unique(sub_numbers[bi]):
            idx = np.flatnonzero(sub_numbers[bi] == z)
            if idx.size < 2:
                continue
            units[("intra", bi, int(z))] = _sorted_subset_distances(
                sub_positions[bi][idx],
                sub_positions[bi][idx],
                same_set=True,
                cell=cell,
                pbc=pbc,
                mic=mic,
            )
    for bi in range(len(blocks.blocks)):
        for bj in range(bi + 1, len(blocks.blocks)):
            for zi in np.unique(sub_numbers[bi]):
                for zj in np.unique(sub_numbers[bj]):
                    idx_i = np.flatnonzero(sub_numbers[bi] == zi)
                    idx_j = np.flatnonzero(sub_numbers[bj] == zj)
                    if idx_i.size == 0 or idx_j.size == 0:
                        continue
                    units[("cross", bi, bj, int(zi), int(zj))] = (
                        _sorted_subset_distances(
                            sub_positions[bi][idx_i],
                            sub_positions[bj][idx_j],
                            same_set=False,
                            cell=cell,
                            pbc=pbc,
                            mic=mic,
                        )
                    )
    return units


def _block_units_cache_slot(blocks: ComparatorBlocks, mic: bool) -> str:
    return f"{_BLOCK_FP_SLOT_PREFIX}:{blocks.signature()}|{int(bool(mic))}"


def get_block_distance_units(
    atoms: Atoms,
    mic: bool,
    blocks: ComparatorBlocks,
) -> dict[tuple, np.ndarray]:
    """Compute (and cache) block-aware distance fingerprint units for ``atoms``.

    Results are cached alongside the legacy fingerprints on ``atoms``, keyed by
    the block signature and ``mic`` so different partitions can coexist.
    """
    mic_b = bool(mic)
    slot = _block_units_cache_slot(blocks, mic_b)
    fp_store = _get_sorted_dist_cache_store(atoms)
    cached = fp_store.get(slot)
    if isinstance(cached, dict) and isinstance(cached.get("units"), dict):
        dirty = _atoms_geometry_dirty_token(atoms, mic=mic_b)
        if cached.get("dirty_token") == dirty:
            return cached["units"]
        content_key = _sorted_dist_content_key(atoms, mic=mic_b)
        if cached.get("content_key") == content_key:
            cached["dirty_token"] = dirty
            return cached["units"]
    else:
        content_key = _sorted_dist_content_key(atoms, mic=mic_b)
    units = _compute_block_distance_units(atoms, mic_b, blocks)
    fp_store[slot] = {
        "content_key": content_key,
        "dirty_token": _atoms_geometry_dirty_token(atoms, mic=mic_b),
        "units": units,
    }
    return units


def get_mobile_atom_indices(atoms: Atoms) -> np.ndarray:
    """Return indices for atoms not constrained by ``FixAtoms``.

    If no fixed atoms are present (or all atoms are fixed), this falls back to
    all atom indices to preserve historical comparison behavior.
    """
    n_atoms = len(atoms)
    fixed_mask = np.zeros(n_atoms, dtype=bool)
    for constraint in getattr(atoms, "constraints", ()):
        if isinstance(constraint, FixAtoms):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            fixed_mask[idx] = True

    if not np.any(fixed_mask):
        return np.arange(n_atoms, dtype=int)

    mobile = np.flatnonzero(~fixed_mask).astype(int, copy=False)
    if mobile.size == 0:
        return np.arange(n_atoms, dtype=int)
    return mobile


def _resolve_n_slab_metadata(a1: Atoms, a2: Atoms) -> int | None:
    """Read a shared ``n_slab_atoms`` partition from structure tags, if present.

    Returns the integer slab count when both structures agree on a positive
    ``n_slab_atoms`` tag, otherwise ``None`` (caller falls back to constraints).
    """
    n1 = get_tag(a1, "n_slab_atoms", None)
    n2 = get_tag(a2, "n_slab_atoms", None)
    if n1 is None and n2 is None:
        return None
    try:
        n1_i = int(n1) if n1 is not None else None
        n2_i = int(n2) if n2 is not None else None
    except (TypeError, ValueError):
        return None
    if n1_i is not None and n2_i is not None and n1_i != n2_i:
        return None
    resolved = n1_i if n1_i is not None else n2_i
    if resolved is None or resolved <= 0:
        return None
    return resolved


def get_shared_mobile_atom_indices(
    a1: Atoms,
    a2: Atoms,
    *,
    n_slab: int | None = None,
) -> np.ndarray:
    """Return index set suitable for comparing two structures.

    When ``n_slab`` is set (e.g. from :class:`~scgo.surface.config.SurfaceSystemConfig`
    at TS time), indices ``n_slab:`` are used on both structures. This is the
    authoritative partition for surface workflows and does not require
    ``FixAtoms`` or stored ``n_slab_atoms`` metadata on loaded minima.

    When ``n_slab`` is ``None``, a stored ``n_slab_atoms`` tag (from
    ``key_value_pairs`` metadata on either structure) provides the same
    surface/adsorbate partition when ``FixAtoms`` constraints are absent.
    Otherwise the intersection of mobile (non-``FixAtoms``) indices is used.
    Raises if the chosen set is empty.
    """
    if len(a1) != len(a2):
        raise SCGOValidationError(
            f"The two configurations must have the same number of atoms: {len(a1)} vs {len(a2)}",
        )

    if n_slab is None:
        n_slab = _resolve_n_slab_metadata(a1, a2)

    if n_slab is not None:
        n_slab_i = int(n_slab)
        if n_slab_i < 0 or n_slab_i >= len(a1):
            raise SCGOValidationError(
                f"n_slab={n_slab_i} invalid for structure comparison (len={len(a1)})."
            )
        mobile = np.arange(n_slab_i, len(a1), dtype=int)
        if mobile.size == 0:
            raise SCGOValidationError(
                "No mobile atoms after applying surface n_slab partition."
            )
        return mobile

    idx1 = get_mobile_atom_indices(a1)
    idx2 = get_mobile_atom_indices(a2)
    shared = np.intersect1d(idx1, idx2, assume_unique=False)
    if shared.size == 0:
        raise SCGOValidationError("No shared mobile atoms across endpoints.")
    return shared.astype(int, copy=False)


class PureInteratomicDistanceComparator:
    """A structural comparator based on sorted interatomic distances.

    This class implements the comparison criteria described in
    L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012),
    but without considering energy differences. It is used to determine if two
    cluster geometries are structurally equivalent.

    Args:
        n_top: The number of atoms from the top of the Atoms object to include
            in the comparison. If None or 0, all atoms are used. Defaults to None.
        tol: The tolerance for the cumulative structural difference (eq. 2 in
            the reference paper). Defaults to `DEFAULT_COMPARATOR_TOL`.
        pair_cor_max: The tolerance for the maximum single interatomic distance
            difference (eq. 3 in the reference paper). Defaults to `DEFAULT_PAIR_COR_MAX`.
        dE: A placeholder for API consistency with other ASE comparators; it is
            not used in this implementation. Defaults to `DEFAULT_ENERGY_TOLERANCE`.
        mic: Whether to use the minimum image convention when calculating
            distances. Defaults to False. Honored literally even when the cell
            has PBC (does not auto-enable MIC from ``atoms.pbc``). Set True for
            adsorbates on periodic slabs via
            :func:`scgo.system_types.resolve_structure_mic`.
        blocks: Optional :class:`ComparatorBlocks` partition enabling weighted,
            block-aware comparison (see :meth:`get_differences`). When ``None``
            the legacy trailing-``n_top`` window comparison is used.
        component_weights: Optional per-role weights applied to intra-block
            cumulative differences; missing roles default to 1.0. A weight of
            ``0`` fully excludes a block (including its cross terms).
        cross_weight: Base weight for cross-block cumulative differences; each
            cross unit is scaled by ``sqrt(w_i * w_j)`` of its endpoint role
            weights (ignored without ``blocks``).
    """

    def __init__(
        self,
        n_top: int | None = None,
        tol: float = DEFAULT_COMPARATOR_TOL,
        pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
        dE: float = DEFAULT_ENERGY_TOLERANCE,
        mic: bool = False,
        blocks: ComparatorBlocks | None = None,
        component_weights: Mapping[str, float] | None = None,
        cross_weight: float = 1.0,
    ):
        self.tol = tol
        self.pair_cor_max = pair_cor_max
        self.dE = dE  # Not used, but kept for API consistency
        self.n_top = n_top or 0
        self.mic = mic
        self.blocks = blocks
        self.cross_weight = float(cross_weight)
        self.component_weights: dict[str, float] = {}
        if self.blocks is not None:
            present_roles = {block.role for block in self.blocks.blocks}
            for role, weight in (component_weights or {}).items():
                _validate_block_role(role)
                weight_f = float(weight)
                if not np.isfinite(weight_f) or weight_f < 0.0:
                    raise SCGOValidationError(
                        f"Comparator weight for role {role!r} must be a "
                        f"non-negative finite float, got {weight!r}."
                    )
                if role not in present_roles:
                    raise SCGOValidationError(
                        f"Comparator weight given for role {role!r} which is "
                        f"not among the provided blocks {sorted(present_roles)}."
                    )
                self.component_weights[role] = weight_f
        elif component_weights:
            raise SCGOValidationError(
                "component_weights requires blocks to be set on the comparator."
            )

    def _role_weight(self, role: str) -> float:
        if not self.component_weights:
            return 1.0
        return float(self.component_weights.get(role, 1.0))

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """Determines if two structures are structurally similar.

        This method calculates the structural differences using `get_differences`
        and returns True if both the cumulative and maximum differences are
        below their respective tolerances.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            True if the structures are considered similar, False otherwise.
            Structures with different compositions are never similar (False).
        """
        cum_diff, max_diff = self.get_differences(a1, a2)

        return cum_diff < self.tol and max_diff < self.pair_cor_max

    def get_differences(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Compute cumulative and maximum structural differences between two Atoms.

        Differences are based on their sorted interatomic distances.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            A tuple containing (cumulative_difference, max_difference).

        Raises:
            SCGOValidationError: If the two Atoms objects do not have the same
                number of atoms.
        """
        if len(a1) != len(a2):
            raise SCGOValidationError(
                "The two configurations must have the same number of atoms",
            )

        if self.blocks is not None:
            return self.__compare_block_structure__(a1, a2)
        return self.__compare_structure__(a1, a2)

    def __compare_block_structure__(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Block-aware structural comparison.

        Each intra-block unit contributes its normalized cumulative difference
        scaled by its element's atom fraction within the block (the legacy
        formula restricted to the block) times the role weight. Each
        cross-block unit carries an effective weight ``cross_weight *
        sqrt(w_i * w_j)`` of its endpoint role weights, so a block with weight
        ``0`` is fully excluded — both directly and through cross terms — while
        uniform weights reduce to ``cross_weight``. Both contributions are
        further scaled by the unit's size share (element atom fraction within
        the block, or element-pair share of the block-pair distances).
        ``max_diff`` is the maximum single-distance difference across all
        non-excluded units.

        Structures whose per-block element counts differ are reported as a
        maximal non-match ``(inf, inf)``.
        """
        n = len(a1)
        for block in self.blocks.blocks:
            if block.indices[-1] >= n:
                raise SCGOValidationError(
                    f"ComparatorBlock {block.role!r} index {block.indices[-1]} "
                    f"out of range for structures of length {n}."
                )
        idx_arrays = [np.asarray(b.indices) for b in self.blocks.blocks]
        for i_arr in idx_arrays:
            if Counter(a1.numbers[i_arr]) != Counter(a2.numbers[i_arr]):
                return (float("inf"), float("inf"))

        units1 = get_block_distance_units(a1, mic=self.mic, blocks=self.blocks)
        units2 = get_block_distance_units(a2, mic=self.mic, blocks=self.blocks)

        total_cum_diff = 0.0
        max_diff = 0.0
        for key, c1 in iter_ordered_units(units1):
            c2 = units2.get(key)
            if c2 is None or len(c1) != len(c2):
                raise SCGOValidationError(
                    "Mismatch in number of distances being compared."
                )
            if len(c1) == 0:
                continue

            d = np.abs(c1 - c2)

            if key[0] == "intra":
                _, bi, elem = key
                weight = self._role_weight(self.blocks.blocks[bi].role)
                frac = float(np.sum(a1.numbers[idx_arrays[bi]] == elem)) / len(
                    idx_arrays[bi]
                )
            else:
                _, bi, bj, _zi, _zj = key
                weight = self.cross_weight * math.sqrt(
                    self._role_weight(self.blocks.blocks[bi].role)
                    * self._role_weight(self.blocks.blocks[bj].role)
                )
                n_pairs_block = len(idx_arrays[bi]) * len(idx_arrays[bj])
                frac = len(c1) / float(n_pairs_block)

            if weight <= 0.0:
                continue

            denom = float(np.sum(c1))
            normalized = float(np.sum(d)) / denom if denom > 1e-10 else 0.0
            total_cum_diff += weight * normalized * frac
            max_diff = max(max_diff, float(np.max(d)))

        return (total_cum_diff, max_diff)

    def __compare_structure__(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Private method to perform the core structural comparison.

        Uses the trailing ``n_top`` atoms when ``n_top > 0``, obtained via
        NumPy index views without copying the ``Atoms`` object so fingerprints
        are cached on the original ``atoms.info``.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            A tuple containing the cumulative difference and the maximum difference.
            Structures with different compositions (element *counts* included)
            are reported as a maximal non-match ``(inf, inf)`` rather than
            raising, so callers such as :meth:`looks_like` simply return False.
        """
        n_top = self.n_top
        n = len(a1)

        # Determine the compared subset's atom numbers (NumPy view, no copy).
        all_nums1 = a1.arrays["numbers"]
        all_nums2 = a2.arrays["numbers"]
        if 0 < n_top < n:
            sub_nums1 = all_nums1[-n_top:]
            sub_nums2 = all_nums2[-n_top:]
        else:
            sub_nums1 = all_nums1
            sub_nums2 = all_nums2

        if Counter(sub_nums1) != Counter(sub_nums2):
            # Different compositions can never "look like" each other; report a
            # maximal difference instead of raising so population dedup and
            # diversity scoring keep working on mixed-composition pools.
            return (float("inf"), float("inf"))

        # Fingerprints are cached on the originals keyed by (n_top, mic).
        p1 = get_sorted_dist_list(a1, mic=self.mic, n_top=n_top)
        p2 = get_sorted_dist_list(a2, mic=self.mic, n_top=n_top)
        numbers = sub_nums1
        total_cum_diff = 0.0
        max_diff = 0.0

        for elem in p1:
            c1 = p1[elem]
            c2 = p2[elem]

            if len(c1) != len(c2):
                # This should not happen if compositions are the same
                raise SCGOValidationError(
                    "Mismatch in number of distances being compared."
                )

            if len(c1) == 0:
                continue

            total_dist_sum = np.sum(c1)
            if total_dist_sum <= 1e-10:  # Use epsilon for floating-point comparison
                continue

            d = np.abs(c1 - c2)
            cum_diff_for_type = np.sum(d)
            max_diff_for_type = np.max(d)

            max_diff = max(max_diff, max_diff_for_type)

            num_atoms_of_type = float(np.sum(numbers == elem))  # Vectorized operation
            total_cum_diff += (
                cum_diff_for_type
                / total_dist_sum
                * num_atoms_of_type
                / float(len(numbers))
            )

        return (total_cum_diff, max_diff)


@dataclass(frozen=True)
class UniquenessSettings:
    """Geometry tolerances plus optional block-aware weighting knobs."""

    comparator_tol: float = DEFAULT_COMPARATOR_TOL
    comparator_pair_cor_max: float = DEFAULT_PAIR_COR_MAX
    component_weights: Mapping[str, float] | None = None
    cross_weight: float = 1.0


def uniqueness_settings_from_mapping(
    params: Mapping[str, Any] | None,
) -> UniquenessSettings:
    params = params or {}
    tol = params.get("comparator_tol")
    pair_cor = params.get("comparator_pair_cor_max")
    weights_raw = params.get("comparator_component_weights")
    weights: dict[str, float] | None = None
    if weights_raw is not None:
        weights = {str(role): float(w) for role, w in dict(weights_raw).items()}
        for role, weight in weights.items():
            _validate_block_role(role)
            weight_f = weights[role]
            if not np.isfinite(weight_f) or weight_f < 0.0:
                raise SCGOValidationError(
                    f"comparator_component_weights[{role!r}] must be a "
                    f"non-negative finite float, got {weight!r}."
                )
    cross_raw = params.get("comparator_cross_weight")
    cross_weight = DEFAULT_CROSS_WEIGHT if cross_raw is None else float(cross_raw)
    return UniquenessSettings(
        comparator_tol=DEFAULT_COMPARATOR_TOL if tol is None else float(tol),
        comparator_pair_cor_max=DEFAULT_PAIR_COR_MAX
        if pair_cor is None
        else float(pair_cor),
        component_weights=weights,
        cross_weight=cross_weight,
    )


def create_geometry_comparator(
    *,
    n_top: int,
    mic: bool = False,
    settings: UniquenessSettings | None = None,
    blocks: ComparatorBlocks | None = None,
) -> PureInteratomicDistanceComparator:
    resolved = settings if settings is not None else UniquenessSettings()
    return PureInteratomicDistanceComparator(
        n_top=n_top,
        tol=resolved.comparator_tol,
        pair_cor_max=resolved.comparator_pair_cor_max,
        mic=mic,
        blocks=blocks,
        component_weights=resolved.component_weights,
        cross_weight=resolved.cross_weight,
    )


class EnergyAndStructureComparator:
    """Energy AND geometry ``looks_like`` for GA population dedup."""

    def __init__(
        self,
        energy_tolerance: float,
        structure_comparator: PureInteratomicDistanceComparator,
    ) -> None:
        self.energy_tolerance = float(energy_tolerance)
        self.structure_comparator = structure_comparator

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        e1 = get_tag(a1, "raw_score", default=None)
        e2 = get_tag(a2, "raw_score", default=None)
        if e1 is None or e2 is None:
            raise SCGOValidationError(
                "EnergyAndStructureComparator requires raw_score on both candidates."
            )
        if abs(float(e1) - float(e2)) > self.energy_tolerance:
            return False
        return self.structure_comparator.looks_like(a1, a2)
