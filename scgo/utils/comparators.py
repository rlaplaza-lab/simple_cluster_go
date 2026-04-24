"""Structural comparison tools for atomic clusters.

This module provides comparators for determining if two cluster structures are
geometrically equivalent, based on sorted interatomic distance analysis as
described in Vilhelmsen and Hammer, PRL 108, 126101 (2012).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)


def get_sorted_dist_list(atoms: Atoms, mic: bool = False) -> dict[int, np.ndarray]:
    """Calculates a dictionary of sorted interatomic distances for an Atoms object.

    This utility method is used to generate a structural fingerprint of a cluster
    by calculating all interatomic distances for each element type and sorting them.

    Args:
        atoms: The Atoms object for which to calculate the distances.
        mic: Whether to use the minimum image convention for periodic systems.
            Defaults to False.

    Returns:
        A dictionary where keys are atomic numbers (integers) and values are
        sorted 1D numpy arrays of interatomic distances for that element type.
    """
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = {}

    for n in unique_types:
        # Use enumerate for better performance and readability
        i_un = [i for i, atom in enumerate(atoms) if atom.number == n]

        if not i_un:
            continue

        # For non-periodic systems, use vectorized NumPy operations for better performance
        if not mic and not np.any(atoms.get_pbc()):
            from scipy.spatial.distance import pdist

            positions = atoms.get_positions()[i_un]
            d = pdist(positions).tolist()
        else:
            # For periodic systems or when mic=True, use original method
            # as get_distance handles PBC correctly
            d = [
                atoms.get_distance(n1, n2, mic)
                for i, n1 in enumerate(i_un)
                for n2 in i_un[i + 1 :]
            ]

        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor


def _n_slab_from_metadata(atoms: Atoms) -> int | None:
    """Best-effort extraction of slab-prefix length from atoms metadata."""
    meta = atoms.info.get("metadata", {}) if isinstance(atoms.info, dict) else {}
    n_slab = meta.get("n_slab_atoms")
    if n_slab is None:
        n_slab = (
            atoms.info.get("n_slab_atoms") if isinstance(atoms.info, dict) else None
        )
    try:
        n_slab_i = int(n_slab)
    except (TypeError, ValueError):
        return None
    if 0 < n_slab_i < len(atoms):
        return n_slab_i
    return None


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


def get_shared_mobile_atom_indices(
    a1: Atoms,
    a2: Atoms,
) -> np.ndarray:
    """Return index set suitable for comparing two structures.

    Uses the intersection of mobile (non-``FixAtoms``) indices from both
    structures. Raises if that intersection is empty.
    """
    if len(a1) != len(a2):
        raise ValueError(
            f"The two configurations must have the same number of atoms: {len(a1)} vs {len(a2)}",
        )

    idx1 = get_mobile_atom_indices(a1)
    idx2 = get_mobile_atom_indices(a2)
    shared = np.intersect1d(idx1, idx2, assume_unique=False)
    if shared.size == 0:
        raise ValueError("No shared mobile atoms across endpoints.")
    # Surface minima may occasionally lose constraints on load. When both
    # endpoints expose slab-prefix metadata, compare only adsorbate+cluster atoms.
    # This avoids slab motion dominating TS pair similarity.
    n_slab_1 = _n_slab_from_metadata(a1)
    n_slab_2 = _n_slab_from_metadata(a2)
    if (
        n_slab_1 is not None
        and n_slab_2 is not None
        and n_slab_1 == n_slab_2
        and shared.size == len(a1)
    ):
        ads_idx = np.arange(n_slab_1, len(a1), dtype=int)
        if ads_idx.size > 0:
            return ads_idx
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
            distances. Defaults to False. Set True for adsorbates on periodic
            slabs when using :func:`scgo.algorithms.ga_common.create_structure_comparator`.
    """

    def __init__(
        self,
        n_top: int | None = None,
        tol: float = DEFAULT_COMPARATOR_TOL,
        pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
        dE: float = DEFAULT_ENERGY_TOLERANCE,
        mic: bool = False,
    ):
        self.tol = tol
        self.pair_cor_max = pair_cor_max
        self.dE = dE  # Not used, but kept for API consistency
        self.n_top = n_top or 0
        self.mic = mic

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
        """
        cum_diff, max_diff = self.get_differences(a1, a2)

        return cum_diff < self.tol and max_diff < self.pair_cor_max

    def get_differences(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Calculates the cumulative and maximum structural differences between two
        Atoms objects based on their sorted interatomic distances.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            A tuple containing (cumulative_difference, max_difference).

        Raises:
            ValueError: If the two Atoms objects do not have the same number of atoms.
        """
        if len(a1) != len(a2):
            raise ValueError(
                "The two configurations must have the same number of atoms",
            )

        # If n_top is defined, only compare the specified number of atoms
        a1top = a1[-self.n_top :] if self.n_top > 0 else a1
        a2top = a2[-self.n_top :] if self.n_top > 0 else a2
        return self.__compare_structure__(a1top, a2top)

    def __compare_structure__(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Private method to perform the core structural comparison.

        Args:
            a1: The first Atoms object (or subset).
            a2: The second Atoms object (or subset).

        Returns:
            A tuple containing the cumulative difference and the maximum difference.
        """
        if set(a1.numbers) != set(a2.numbers):
            raise ValueError("The two configurations must have the same composition")

        p1 = get_sorted_dist_list(a1, mic=self.mic)
        p2 = get_sorted_dist_list(a2, mic=self.mic)
        numbers = a1.numbers
        total_cum_diff = 0.0
        max_diff = 0.0

        for n in p1:
            c1 = p1[n]
            c2 = p2[n]

            if len(c1) != len(c2):
                # This should not happen if compositions are the same
                raise ValueError("Mismatch in number of distances being compared.")

            if len(c1) == 0:
                continue

            total_dist_sum = np.sum(c1)
            if total_dist_sum <= 1e-10:  # Use epsilon for floating-point comparison
                continue

            d = np.abs(c1 - c2)
            cum_diff_for_type = np.sum(d)
            max_diff_for_type = np.max(d)

            max_diff = max(max_diff, max_diff_for_type)

            num_atoms_of_type = float(np.sum(numbers == n))  # Vectorized operation
            total_cum_diff += (
                cum_diff_for_type
                / total_dist_sum
                * num_atoms_of_type
                / float(len(numbers))
            )

        return (total_cum_diff, max_diff)
