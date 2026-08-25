"""Mutations that permute atom types between groups."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
)

from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)
from scgo.ase_ga_patches.mutations._common import _ensure_rng
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.system_types import SystemType, get_system_policy

__all__ = ["CustomPermutationMutation", "PermutationMutation"]


class PermutationMutation(OffspringCreator):
    """Mutation that permutes a percentage of the atom types in the cluster.

    Parameters
    ----------
    n_top: Number of atoms optimized by the GA.

    probability: Fraction of the groups to be permuted; the number of
        pair swaps performed is ceil(n_groups * probability / 2).

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Permutations will then happen
        at the molecular level, i.e. swapping the center-of-
        positions of two moieties while preserving their
        internal geometries.

    blmin: Dictionary defining the minimum distance between atoms
        after the permutation. If equal to None (the default),
        no such check is performed.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    verbose: bool
        If True, print verbose output.

    """

    def __init__(self, n_top, system_type: SystemType, probability=0.33, test_dist_to_slab=True,
                 use_tags=False, target_tags=None, blmin=None, rng=None, verbose=False):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.probability = probability
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.blmin = blmin
        self.system_type = system_type
        self._policy = get_system_policy(system_type)

        self.descriptor = "PermutationMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: permutation")

    def mutate(self, atoms):
        """Does the actual mutation.

        Candidate swap pairs are shuffled once and applied incrementally:
        a swap that introduces a steric clash is skipped (reverted) rather
        than triggering a full redraw of the swap set. The mutation fails
        only when no pair can be applied without a clash.
        """
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[len(atoms) - N:]
        if self.use_tags:
            gather_atoms_by_tag(
                atoms,
                skip_tag=periodic_sheet_tag_to_skip(self.system_type),
            )
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        symbols = atoms.get_chemical_symbols()

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
            if len(unique_tags) == 0:
                return None

        n = len(unique_tags)
        swaps = int(np.ceil(n * self.probability / 2.))

        sym = []
        for tag in unique_tags:
            indices = np.where(tags == tag)[0]
            s = "".join([symbols[j] for j in indices])
            sym.append(s)

        # Permutations with one atom type are not valid - return None
        if len(np.unique(sym)) <= 1:
            return None

        # Pre-compute valid swap pairs: indices (i, j) where sym[i] != sym[j].
        valid_pairs = [(i, j) for i in range(len(unique_tags)) for j in range(i + 1, len(unique_tags))
                       if sym[i] != sym[j]]

        pos = pos_ref.copy()

        if self.blmin is None:
            # No steric model to validate against; apply the target number of
            # swaps unconditionally.
            n_swaps = min(swaps, len(valid_pairs))
            if n_swaps < 1:
                return None
            pair_indices = self.rng.choice(
                len(valid_pairs), size=n_swaps, replace=False
            )
            for pi_idx in np.atleast_1d(pair_indices):
                i, j = valid_pairs[int(pi_idx)]
                ind1 = np.where(tags == unique_tags[i])
                ind2 = np.where(tags == unique_tags[j])
                cop1 = np.mean(pos[ind1], axis=0)
                cop2 = np.mean(pos[ind2], axis=0)
                pos[ind1] += cop2 - cop1
                pos[ind2] += cop1 - cop2
        else:
            # Incremental application over a single shuffled pass of the
            # candidate pairs; clashing swaps are reverted and skipped.
            n_swaps = min(swaps, len(valid_pairs))
            applied = 0
            for pi_idx in self.rng.permutation(len(valid_pairs)):
                if applied >= n_swaps:
                    break
                i, j = valid_pairs[int(pi_idx)]
                ind1 = np.where(tags == unique_tags[i])
                ind2 = np.where(tags == unique_tags[j])
                cop1 = np.mean(pos[ind1], axis=0)
                cop2 = np.mean(pos[ind2], axis=0)
                shift1 = cop2 - cop1
                shift2 = cop1 - cop2
                pos[ind1] += shift1
                pos[ind2] += shift2

                top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
                too_close = atoms_too_close(
                    top, self.blmin, use_tags=self.use_tags)
                if not too_close and self.test_dist_to_slab:
                    too_close = atoms_too_close_two_sets(top, slab, self.blmin)
                if too_close:
                    pos[ind1] -= shift1
                    pos[ind2] -= shift2
                    continue
                applied += 1

            if applied < 1:
                return None

        mutant = slab + Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
        # Apply centering only for gas-phase systems
        if not self._policy.uses_surface:
            mutant.center()
        return mutant


class CustomPermutationMutation(PermutationMutation):
    """PermutationMutation preset with a higher default swap probability and
    slab-distance testing disabled.
    """

    def __init__(
        self,
        n_top,
        system_type: SystemType,
        probability=0.4,
        test_dist_to_slab=False,
        use_tags=False,
        blmin=None,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        super().__init__(
            n_top,
            probability=probability,
            test_dist_to_slab=test_dist_to_slab,
            use_tags=use_tags,
            blmin=blmin,
            system_type=system_type,
            rng=rng,
            verbose=verbose,
        )

# fmt: on
