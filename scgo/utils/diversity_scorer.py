"""Efficient diversity scoring using vectorized operations.

This module provides efficient calculation of structural diversity scores
using vectorized NumPy operations, avoiding O(N²) pairwise comparisons.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_block_distance_units,
    get_sorted_dist_list,
    iter_ordered_units,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


class DiversityScorer:
    """Efficient scorer for diversity-based fitness calculation.

    Pre-computes and caches reference descriptors for fast diversity
    calculations using vectorized NumPy operations. Supports periodic
    updates by adding new reference structures during optimization.

    Attributes:
        comparator: Comparator used for descriptor extraction.
        reference_structures: List of reference Atoms objects.
        _ref_descriptors: Cached descriptor matrix (N_refs, descriptor_length).
    """

    def __init__(
        self,
        reference_structures: list[Atoms],
        comparator: PureInteratomicDistanceComparator,
    ):
        """Initialize scorer with reference structures.

        Args:
            reference_structures: Reference structures to compare against.
            comparator: Comparator for descriptor extraction (uses mic setting).
        """
        self.comparator = comparator
        self.reference_structures = list(reference_structures)
        self._ref_descriptors = self._compute_descriptors(reference_structures)
        self._weight_vector_cache: dict[int, np.ndarray] = {}

    def _compute_descriptors(
        self,
        structures: list[Atoms],
    ) -> np.ndarray | None:
        """Convert structures to descriptor matrix.

        Args:
            structures: List of Atoms objects to convert.

        Returns:
            (N_refs, descriptor_length) array, or None if the input is empty or
            the descriptors are ragged (inhomogeneous lengths). Ragged input is
            never handed to ``np.array``, which would raise ``ValueError``.
        """
        if not structures:
            return None

        descriptors = [self._atoms_to_descriptor(s) for s in structures]

        lengths = [len(d) for d in descriptors]
        if len(set(lengths)) > 1:
            logger.warning(
                f"Inconsistent descriptor lengths: {sorted(set(lengths))}. "
                f"May indicate different compositions; skipping vectorized "
                f"descriptor matrix and falling back to pairwise scoring"
            )
            return None

        return np.array(descriptors)

    def _atoms_to_descriptor(self, atoms: Atoms) -> np.ndarray:
        """Convert Atoms to flat descriptor vector.

        Legacy mode flattens sorted interatomic distances ordered by atomic
        number. Block-aware mode concatenates the comparator's distance units
        in canonical order (intra-block before cross-block).

        Args:
            atoms: Atoms object to convert.

        Returns:
            1D numpy array of sorted interatomic distances.
        """
        if self.comparator.blocks is not None:
            units = get_block_distance_units(
                atoms,
                mic=self.comparator.mic,
                blocks=self.comparator.blocks,
            )
            parts = [np.asarray(u, dtype=float) for _, u in iter_ordered_units(units)]
        else:
            dist_dict = get_sorted_dist_list(
                atoms, mic=self.comparator.mic, n_top=self.comparator.n_top
            )
            parts = [dist_dict[atomic_num] for atomic_num in sorted(dist_dict.keys())]

        descriptor = np.concatenate(parts) if parts else np.array([])
        return descriptor

    def _slice_weight_vector(self, descriptor_length: int) -> np.ndarray:
        """Per-element weights aligned with one block-aware descriptor row.

        Intra-block slices carry their role weight; cross-block slices carry
        ``cross_weight * sqrt(w_i * w_j)`` of their endpoint role weights
        (mirroring :meth:`PureInteratomicDistanceComparator.
        __compare_block_structure__`). Legacy descriptors get all ones. The
        vector is derived from the first reference structure's fingerprint
        layout and cached per descriptor length.
        """
        comparator = self.comparator
        if comparator.blocks is None:
            return np.ones(descriptor_length, dtype=float)

        cached = self._weight_vector_cache.get(descriptor_length)
        if cached is not None:
            return cached

        if not self.reference_structures:
            return np.ones(descriptor_length, dtype=float)
        units = get_block_distance_units(
            self.reference_structures[0],
            mic=comparator.mic,
            blocks=comparator.blocks,
        )
        weights = np.empty(descriptor_length, dtype=float)
        offset = 0
        for key, unit in iter_ordered_units(units):
            length = len(unit)
            if key[0] == "intra":
                role = comparator.blocks.blocks[key[1]].role
                weights[offset : offset + length] = comparator._role_weight(role)
            else:
                role_i = comparator.blocks.blocks[key[1]].role
                role_j = comparator.blocks.blocks[key[2]].role
                weights[offset : offset + length] = (
                    comparator.cross_weight
                    * (
                        comparator._role_weight(role_i)
                        * comparator._role_weight(role_j)
                    )
                    ** 0.5
                )
            offset += length
        if offset != descriptor_length:
            logger.warning(
                "Diversity weight/layout mismatch (%d vs %d); using uniform weights",
                offset,
                descriptor_length,
            )
            weights = np.ones(descriptor_length, dtype=float)
        self._weight_vector_cache[descriptor_length] = weights
        return weights

    def score(self, atoms: Atoms) -> float:
        """Compute average dissimilarity to all references.

        Calculates average dissimilarity using vectorized operations:
        avg(cum_diff + 0.5*max_diff) over all references.

        Args:
            atoms: Structure to score.

        Returns:
            Average dissimilarity (higher = more diverse). Returns 0.0 if no references.
        """
        if self._ref_descriptors is None or len(self._ref_descriptors) == 0:
            if self.reference_structures:
                # No usable descriptor matrix (e.g. ragged references): score
                # pairwise instead of failing.
                return self._score_pairwise(atoms)
            return 0.0

        candidate_desc = self._atoms_to_descriptor(atoms)

        if len(candidate_desc) != self._ref_descriptors.shape[1]:
            logger.warning(
                f"Descriptor length mismatch: candidate {len(candidate_desc)} vs "
                f"references {self._ref_descriptors.shape[1]}. "
                f"May indicate different compositions"
            )
            return self._score_pairwise(atoms)

        differences = np.abs(candidate_desc - self._ref_descriptors)
        slice_weights = self._slice_weight_vector(len(candidate_desc))
        cum_diffs = (slice_weights * differences).sum(axis=1)
        max_diffs = np.max(differences, axis=1)
        combined_dissimilarities = cum_diffs + 0.5 * max_diffs
        avg_dissimilarity = np.mean(combined_dissimilarities)

        return float(avg_dissimilarity)

    def _score_pairwise(self, atoms: Atoms) -> float:
        """Fallback pairwise scoring for mismatched descriptors.

        Args:
            atoms: Structure to score.

        Returns:
            Average dissimilarity computed pairwise.
        """
        if not self.reference_structures:
            return 0.0

        dissimilarities = []
        for ref in self.reference_structures:
            try:
                cum_diff, max_diff = self.comparator.get_differences(atoms, ref)
            except (ValueError, RuntimeError, SCGOValidationError) as exc:
                get_logger(__name__).debug(
                    "Skipping comparator pair in diversity fallback: %s", exc
                )
                continue
            dissimilarity = cum_diff + 0.5 * max_diff
            if not np.isfinite(dissimilarity):
                # Different composition (comparator reports an infinite
                # difference); it carries no usable diversity information.
                continue
            dissimilarities.append(dissimilarity)

        if not dissimilarities:
            return 0.0

        return float(np.mean(dissimilarities))

    def add_reference(self, atoms: Atoms) -> None:
        """Add an Atoms object to the reference set and update descriptors."""
        self.reference_structures.append(atoms)
        new_desc = self._atoms_to_descriptor(atoms)

        if self._ref_descriptors is None:
            # Either the reference set was empty or its descriptors were ragged;
            # recompute (which returns None again if still ragged).
            self._ref_descriptors = self._compute_descriptors(self.reference_structures)
        else:
            # Verify length matches
            if len(new_desc) != self._ref_descriptors.shape[1]:
                logger.warning(
                    f"New reference descriptor length {len(new_desc)} doesn't match "
                    f"existing {self._ref_descriptors.shape[1]}. "
                    f"Recomputing all descriptors"
                )
                # Recompute all descriptors
                self._ref_descriptors = self._compute_descriptors(
                    self.reference_structures
                )
            else:
                self._ref_descriptors = np.vstack([self._ref_descriptors, new_desc])

    def __len__(self) -> int:
        """Return number of reference structures.

        Returns:
            Number of reference structures.
        """
        return len(self.reference_structures)
