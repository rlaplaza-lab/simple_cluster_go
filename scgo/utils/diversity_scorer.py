"""Efficient diversity scoring using vectorized operations.

This module provides efficient calculation of structural diversity scores
using vectorized NumPy operations, avoiding O(N²) pairwise comparisons.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_sorted_dist_list,
)
from scgo.utils.logging import get_logger


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

    def _compute_descriptors(
        self,
        structures: list[Atoms],
    ) -> np.ndarray | None:
        """Convert structures to descriptor matrix.

        Args:
            structures: List of Atoms objects to convert.

        Returns:
            (N_refs, descriptor_length) array or None if empty.
        """
        if not structures:
            return None

        descriptors = [self._atoms_to_descriptor(s) for s in structures]

        lengths = [len(d) for d in descriptors]
        if len(set(lengths)) > 1:
            logger = get_logger(__name__)
            logger.warning(
                f"Inconsistent descriptor lengths: {set(lengths)}. "
                f"May indicate different compositions."
            )

        return np.array(descriptors)

    def _atoms_to_descriptor(self, atoms: Atoms) -> np.ndarray:
        """Convert Atoms to flat descriptor vector.

        Flattens sorted interatomic distances into a consistent vector format
        for efficient vectorized operations. Order: by atomic number (ascending),
        then by sorted distances.

        Args:
            atoms: Atoms object to convert.

        Returns:
            1D numpy array of sorted interatomic distances.
        """
        dist_dict = get_sorted_dist_list(atoms, mic=self.comparator.mic)

        descriptor_parts = [
            dist_dict[atomic_num] for atomic_num in sorted(dist_dict.keys())
        ]

        if descriptor_parts:
            descriptor = np.concatenate(descriptor_parts)
        else:
            descriptor = np.array([])

        return descriptor

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
            return 0.0

        candidate_desc = self._atoms_to_descriptor(atoms)

        if len(candidate_desc) != self._ref_descriptors.shape[1]:
            logger = get_logger(__name__)
            logger.warning(
                f"Descriptor length mismatch: candidate {len(candidate_desc)} vs "
                f"references {self._ref_descriptors.shape[1]}. "
                f"May indicate different compositions."
            )
            return self._score_pairwise(atoms)

        differences = np.abs(candidate_desc - self._ref_descriptors)
        cum_diffs = np.sum(differences, axis=1)
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
                dissimilarities.append(cum_diff + 0.5 * max_diff)
            except (ValueError, RuntimeError):
                continue

        if not dissimilarities:
            return 0.0

        return float(np.mean(dissimilarities))

    def add_reference(self, atoms: Atoms) -> None:
        """Add an Atoms object to the reference set and update descriptors."""
        self.reference_structures.append(atoms)
        new_desc = self._atoms_to_descriptor(atoms)

        if self._ref_descriptors is None:
            self._ref_descriptors = np.array([new_desc])
        else:
            # Verify length matches
            if len(new_desc) != self._ref_descriptors.shape[1]:
                logger = get_logger(__name__)
                logger.warning(
                    f"New reference descriptor length {len(new_desc)} doesn't match "
                    f"existing {self._ref_descriptors.shape[1]}. "
                    f"Recomputing all descriptors."
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
