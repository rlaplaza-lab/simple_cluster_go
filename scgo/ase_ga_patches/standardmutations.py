# fmt: off

from __future__ import annotations

"""A collection of mutations that can be used."""
from math import acos, cos, pi, sin

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
    get_rotation_matrix,
)

from scgo.ase_ga_patches._vector_utils import (
    append_unique_unit_vector as _append_unique_unit_vector,
)
from scgo.ase_ga_patches._vector_utils import random_unit_vector as _random_unit_vector
from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng


def _get_blmin_distance(blmin, atomic_number_a, atomic_number_b):
    key = (int(atomic_number_a), int(atomic_number_b))
    if key in blmin:
        return blmin[key]
    return blmin[(int(atomic_number_b), int(atomic_number_a))]


class RattleMutation(OffspringCreator):
    """An implementation of the rattle mutation as described in:

    R.L. Johnston Dalton Transactions, Vol. 22,
    No. 22. (2003), pp. 4193-4207

    Parameters
    ----------
    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved.

    rattle_prop: The probability with which each atom is rattled.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    rng: Random number generator
        By default numpy.random.

    verbose: bool
        If True, print verbose output.

    """

    def __init__(self, blmin, n_top, rattle_strength=0.8,
                 rattle_prop=0.4, test_dist_to_slab=True, use_tags=False,
                 verbose=False, rng=None):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = "RattleMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        """Generates a new individual by applying the rattle mutation to a parent.

        Args:
            parents: A list containing the parent Atoms object.

        Returns:
            A tuple containing the new Atoms object and a description of the mutation.

        """
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: rattle"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: rattle"

    def mutate(self, atoms):
        """Applies the rattle mutation to the given Atoms object.

        Args:
            atoms: The Atoms object to be mutated.

        Returns:
            A new Atoms object after applying the rattle mutation, or None if mutation fails.

        """
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        st = 2. * self.rattle_strength

        count = 0
        maxcount = 1000
        too_close = True
        unique_tags = np.unique(tags)
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()

            # Guarantee at least one tag is rattled, then sample the rest.
            guaranteed = self.rng.integers(len(unique_tags))
            for idx, tag in enumerate(unique_tags):
                if idx == guaranteed or self.rng.random() < self.rattle_prop:
                    select = np.where(tags == tag)
                    r = self.rng.random(3)
                    pos[select] += st * (r - 0.5)

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(
                top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        mutant = slab + top
        return mutant


class AnisotropicRattleMutation(OffspringCreator):
    """Rattle mutation with stronger in-plane and weaker normal displacement.

    A random plane is sampled every attempt. Selected atoms (or tag groups) are
    displaced primarily in-plane to encourage exploration of flat/pseudolinear
    regions while still allowing smaller out-of-plane motion.
    """

    def __init__(
        self,
        blmin,
        n_top,
        in_plane_strength=1.0,
        normal_strength=0.2,
        rattle_prop=0.5,
        test_dist_to_slab=True,
        use_tags=False,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.in_plane_strength = in_plane_strength
        self.normal_strength = normal_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = "AnisotropicRattleMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: anisotropic_rattle"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: anisotropic_rattle"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        count = 0
        maxcount = 1000
        too_close = True

        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()

            # Random unit normal defining the dominant exploration plane.
            normal = self.rng.normal(0.0, 1.0, 3)
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                continue
            normal = normal / normal_norm

            # Pick a helper vector not parallel to normal.
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, normal)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])

            # Build orthonormal in-plane basis.
            u = np.cross(normal, helper)
            u_norm = np.linalg.norm(u)
            if u_norm <= 1e-12:
                continue
            u = u / u_norm
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)

            # Guarantee at least one tag is moved, then sample the rest.
            unique_tags_local = np.unique(tags)
            guaranteed = self.rng.integers(len(unique_tags_local))
            for idx, tag in enumerate(unique_tags_local):
                if idx == guaranteed or self.rng.random() < self.rattle_prop:
                    select = np.where(tags == tag)
                    a = self.rng.uniform(-self.in_plane_strength, self.in_plane_strength)
                    b = self.rng.uniform(-self.in_plane_strength, self.in_plane_strength)
                    c = self.rng.uniform(-self.normal_strength, self.normal_strength)
                    pos[select] += a * u + b * v + c * normal

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        return slab + top


class OverlapReliefMutation(OffspringCreator):
    """Resolve steric clashes with bounded geometric sweeps.

    The operator accumulates pairwise displacements for atoms that violate
    ``blmin`` and applies a small exploratory jitter only after the repaired
    geometry is valid.
    """

    def __init__(
        self,
        blmin,
        n_top,
        n_sweeps=4,
        jitter=0.02,
        margin=0.04,
        test_dist_to_slab=True,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.n_sweeps = n_sweeps
        self.jitter = jitter
        self.margin = margin
        self.test_dist_to_slab = test_dist_to_slab

        self.descriptor = "OverlapReliefMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: overlap_relief"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: overlap_relief"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        top = atoms[-N:]
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()
        tags = top.get_tags()

        for _ in range(self.n_sweeps):
            displacements = np.zeros_like(positions)
            moved = False

            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    required = _get_blmin_distance(self.blmin, numbers[i], numbers[j])
                    vector = positions[j] - positions[i]
                    distance = np.linalg.norm(vector)
                    if distance + 1e-12 < required:
                        direction = (
                            vector / distance
                            if distance > 1e-12
                            else _random_unit_vector(self.rng)
                        )
                        shift = 0.5 * (required - distance + self.margin)
                        displacements[i] -= shift * direction
                        displacements[j] += shift * direction
                        moved = True

            if self.test_dist_to_slab and len(slab) > 0:
                slab_positions = slab.get_positions()
                slab_numbers = slab.get_atomic_numbers()
                for i in range(len(positions)):
                    for j in range(len(slab_positions)):
                        required = _get_blmin_distance(
                            self.blmin,
                            numbers[i],
                            slab_numbers[j],
                        )
                        vector = positions[i] - slab_positions[j]
                        distance = np.linalg.norm(vector)
                        if distance + 1e-12 < required:
                            direction = (
                                vector / distance
                                if distance > 1e-12
                                else np.array([0.0, 0.0, 1.0])
                            )
                            displacements[i] += (required - distance + self.margin) * direction
                            moved = True

            positions += displacements
            if not moved:
                break

        repaired_positions = positions.copy()
        for add_jitter in (True, False):
            trial_positions = repaired_positions.copy()
            if add_jitter and self.jitter > 0.0:
                trial_positions += self.rng.normal(
                    0.0,
                    self.jitter,
                    size=trial_positions.shape,
                )

            candidate = Atoms(
                numbers,
                positions=trial_positions,
                cell=cell,
                pbc=pbc,
                tags=tags,
            )
            if len(slab) == 0:
                candidate.center()
            if atoms_too_close(candidate, self.blmin):
                continue
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(slab, candidate, self.blmin)
            ):
                continue
            return slab + candidate

        return None


class PermutationMutation(OffspringCreator):
    """Mutation that permutes a percentage of the atom types in the cluster.

    Parameters
    ----------
    n_top: Number of atoms optimized by the GA.

    probability: The probability with which an atom is permuted.

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
        By default numpy.random.

    verbose: bool
        If True, print verbose output.

    """

    def __init__(self, n_top, probability=0.33, test_dist_to_slab=True,
                 use_tags=False, blmin=None, rng=None, verbose=False):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.probability = probability
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.blmin = blmin

        self.descriptor = "PermutationMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: permutation"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: permutation"

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        if self.use_tags:
            gather_atoms_by_tag(atoms)
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        symbols = atoms.get_chemical_symbols()

        unique_tags = np.unique(tags)
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
        valid_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if sym[i] != sym[j]]

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            for _ in range(swaps):
                pi_idx = self.rng.integers(0, len(valid_pairs))
                i, j = valid_pairs[pi_idx]
                ind1 = np.where(tags == unique_tags[i])
                ind2 = np.where(tags == unique_tags[j])
                cop1 = np.mean(pos[ind1], axis=0)
                cop2 = np.mean(pos[ind2], axis=0)
                pos[ind1] += cop2 - cop1
                pos[ind2] += cop1 - cop2

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self.blmin is None:
                too_close = False
            else:
                too_close = atoms_too_close(
                    top, self.blmin, use_tags=self.use_tags)
                if not too_close and self.test_dist_to_slab:
                    too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        mutant = slab + top
        return mutant


class CustomPermutationMutation(PermutationMutation):
    """PermutationMutation that requires ``rng`` to be a ``numpy.random.Generator`` (or None)."""

    def __init__(
        self,
        n_top,
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
            rng=rng,
            verbose=verbose,
        )


class ShellSwapMutation(OffspringCreator):
    """Swap atom groups between inner and outer radial shells.

    This targets alloy ordering directly by preferring swaps between groups with
    different chemical signatures and large radial separation.
    """

    def __init__(
        self,
        n_top,
        inner_fraction=0.33,
        outer_fraction=0.33,
        test_dist_to_slab=True,
        use_tags=False,
        blmin=None,
        max_pair_trials=12,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.inner_fraction = inner_fraction
        self.outer_fraction = outer_fraction
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.blmin = blmin
        self.max_pair_trials = max_pair_trials

        self.descriptor = "ShellSwapMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: shell_swap"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: shell_swap"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        top = atoms[-N:].copy()
        if self.use_tags:
            gather_atoms_by_tag(top)

        tags = top.get_tags() if self.use_tags else np.arange(N)
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        symbols = top.get_chemical_symbols()
        cell = top.get_cell()
        pbc = top.get_pbc()
        unique_tags = np.unique(tags)

        group_indices = []
        group_symbols = []
        group_centers = []
        for tag in unique_tags:
            indices = np.where(tags == tag)[0]
            group_indices.append(indices)
            group_symbols.append("".join(symbols[idx] for idx in indices))
            group_centers.append(np.mean(positions[indices], axis=0))

        if len(np.unique(group_symbols)) <= 1:
            return None

        centers = np.asarray(group_centers)
        radial_center = np.mean(centers, axis=0)
        radial_distances = np.linalg.norm(centers - radial_center, axis=1)
        order = np.argsort(radial_distances)
        inner_count = max(1, min(len(order) - 1, int(np.ceil(len(order) * self.inner_fraction))))
        outer_count = max(1, min(len(order) - 1, int(np.ceil(len(order) * self.outer_fraction))))
        inner_groups = order[:inner_count]
        outer_groups = order[-outer_count:]

        candidate_pairs = []
        for inner_idx in inner_groups:
            for outer_idx in outer_groups:
                if inner_idx == outer_idx:
                    continue
                if group_symbols[inner_idx] == group_symbols[outer_idx]:
                    continue
                radial_gap = abs(radial_distances[inner_idx] - radial_distances[outer_idx])
                candidate_pairs.append((radial_gap, inner_idx, outer_idx))

        if not candidate_pairs:
            for left in range(len(unique_tags)):
                for right in range(left + 1, len(unique_tags)):
                    if group_symbols[left] == group_symbols[right]:
                        continue
                    radial_gap = abs(radial_distances[left] - radial_distances[right])
                    candidate_pairs.append((radial_gap, left, right))

        if not candidate_pairs:
            return None

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        n_trials = min(self.max_pair_trials, len(candidate_pairs))
        pair_order = self.rng.permutation(n_trials) if n_trials > 1 else np.array([0])

        for pair_idx in pair_order:
            _, left, right = candidate_pairs[pair_idx]
            new_positions = positions.copy()
            left_indices = group_indices[left]
            right_indices = group_indices[right]
            left_center = np.mean(new_positions[left_indices], axis=0)
            right_center = np.mean(new_positions[right_indices], axis=0)
            new_positions[left_indices] += right_center - left_center
            new_positions[right_indices] += left_center - right_center

            candidate = Atoms(
                numbers,
                positions=new_positions,
                cell=cell,
                pbc=pbc,
                tags=tags,
            )
            if self.blmin is None:
                return slab + candidate
            if atoms_too_close(candidate, self.blmin, use_tags=self.use_tags):
                continue
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(candidate, slab, self.blmin)
            ):
                continue
            return slab + candidate

        return None


class MirrorMutation(OffspringCreator):
    """A mirror mutation, as described in
    TO BE PUBLISHED.

    This mutation mirrors half of the cluster in a
    randomly oriented cutting plane discarding the other half.

    Parameters
    ----------
    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    reflect: Defines if the mirrored half is also reflected
        perpendicular to the mirroring plane.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, n_top, reflect=True, rng=None,
                 verbose=False, max_tries=1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.max_tries = max_tries
        self.reflect = reflect

        self.descriptor = "MirrorMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: mirror"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: mirror"

    def mutate(self, atoms):
        """Do the mutation of the atoms input."""
        reflect = self.reflect
        tc = True
        slab = atoms[0:len(atoms) - self.n_top]
        top = atoms[len(atoms) - self.n_top: len(atoms)]
        num = top.numbers
        unique_types = list(set(num))
        nu = {u: sum(num == u) for u in unique_types}
        n_tries = self.max_tries
        counter = 0
        changed = False

        while tc and counter < n_tries:
            counter += 1
            cand = top.copy()
            pos = cand.get_positions()

            cm = np.average(top.get_positions(), axis=0)

            # Uniform random direction on the sphere (correct area element).
            theta = acos(1.0 - 2.0 * self.rng.random())
            phi = 2. * pi * self.rng.random()
            n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
            n = np.array(n)

            # Calculate all atoms signed distance to the cutting plane
            D = []
            for (i, p) in enumerate(pos):
                d = np.dot(p - cm, n)
                D.append((i, d))

            # Sort the atoms by their signed distance
            D.sort(key=lambda x: x[1])
            nu_taken = {}

            # Select half of the atoms needed for a full cluster
            p_use = []
            n_use = []
            for (i, _d) in D:
                if num[i] not in nu_taken:
                    nu_taken[num[i]] = 0
                if nu_taken[num[i]] < nu[num[i]] / 2.:
                    p_use.append(pos[i])
                    n_use.append(num[i])
                    nu_taken[num[i]] += 1

            # calculate the mirrored position and add these.
            pn = []
            for p in p_use:
                pt = p - 2. * np.dot(p - cm, n) * n
                if reflect:
                    pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
                pn.append(pt)

            n_use.extend(n_use)
            p_use.extend(pn)

            # In the case of an uneven number of
            # atoms we need to add one extra
            for n in nu:
                if nu[n] % 2 == 0:
                    continue
                while n_use.count(n) > nu[n]:
                    for i in range(int(len(n_use) / 2), len(n_use)):
                        if n_use[i] == n:
                            del p_use[i]
                            del n_use[i]
                            break
                assert n_use.count(n) == nu[n]

            # Make sure we have the correct number of atoms
            # and rearrange the atoms so they are in the right order
            for i in range(len(n_use)):
                if num[i] == n_use[i]:
                    continue
                for j in range(i + 1, len(n_use)):
                    if n_use[j] == num[i]:
                        tn = n_use[i]
                        tp = p_use[i]
                        n_use[i] = n_use[j]
                        p_use[i] = p_use[j]
                        p_use[j] = tp
                        n_use[j] = tn

            # Finally we check that nothing is too close in the end product.
            cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())

            tc = atoms_too_close(cand, self.blmin)
            if tc:
                continue
            tc = atoms_too_close_two_sets(slab, cand, self.blmin)

            if not changed and counter > n_tries // 2:
                reflect = not reflect
                changed = True

            tot = slab + cand

        if counter == n_tries:
            return None
        return tot


class RotationalMutation(OffspringCreator):
    """Mutates a candidate by applying random rotations
    to multi-atom moieties in the structure (atoms with
    the same tag are considered part of one such moiety).

    Only performs whole-molecule rotations, no internal
    rotations.

    For more information, see also:

      * `Zhu Q., Oganov A.R., Glass C.W., Stokes H.T,
        Acta Cryst. (2012), B68, 215-226.`__

        __ https://dx.doi.org/10.1107/S0108768112017466

    Parameters
    ----------
    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    n_top: int or None
        The number of atoms to optimize (None = include all).

    fraction: float
        Fraction of the moieties to be rotated.

    tags: None or list of integers
        Specifies, respectively, whether all moieties or only those
        with matching tags are eligible for rotation.

    min_angle: float
        Minimal angle (in radians) for each rotation;
        should lie in the interval [0, pi].

    test_dist_to_slab: boolean
        Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, n_top=None, fraction=0.33, tags=None,
                 min_angle=1.57, test_dist_to_slab=True, rng=None,
                 verbose=False, max_inner_attempts=10000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.fraction = fraction
        self.tags = tags
        self.min_angle = min_angle
        self.test_dist_to_slab = test_dist_to_slab
        self.max_inner_attempts = max_inner_attempts
        self.descriptor = "RotationalMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: rotational"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: rotational"

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]

        mutant = atoms.copy()
        gather_atoms_by_tag(mutant)
        pos = mutant.get_positions()
        tags = mutant.get_tags()
        eligible_tags = tags if self.tags is None else self.tags

        indices = {}
        for tag in np.unique(tags):
            hits = np.where(tags == tag)[0]
            if len(hits) > 1 and tag in eligible_tags:
                indices[tag] = hits

        n_rot = int(np.ceil(len(indices) * self.fraction))
        chosen_tags = self.rng.choice(list(indices.keys()), size=n_rot,
                                      replace=False)

        too_close = True
        count = 0
        maxcount = self.max_inner_attempts
        while too_close and count < maxcount:
            newpos = np.copy(pos)
            for tag in chosen_tags:
                p = np.copy(newpos[indices[tag]])
                cop = np.mean(p, axis=0)

                if len(p) == 2:
                    # Generate axis guaranteed not collinear with bond:
                    # cross(bond, random) is always perpendicular to bond.
                    line = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                    rvec = self.rng.standard_normal(3)
                    axis = np.cross(line, rvec)
                    norm = np.linalg.norm(axis)
                    if norm < 1e-12:
                        # Extremely rare: rvec exactly parallel to line.
                        # Pick a deterministic fallback perpendicular.
                        alt = np.array([1.0, 0.0, 0.0])
                        if abs(np.dot(line, alt)) > 0.9:
                            alt = np.array([0.0, 1.0, 0.0])
                        axis = np.cross(line, alt)
                        norm = np.linalg.norm(axis)
                    axis /= norm
                else:
                    axis = self.rng.standard_normal(3)
                    axis /= np.linalg.norm(axis)

                angle = self.min_angle + (np.pi - self.min_angle) * self.rng.random()

                m = get_rotation_matrix(axis, angle)
                newpos[indices[tag]] = np.dot(m, (p - cop).T).T + cop

            mutant.set_positions(newpos)
            # Only center gas-phase clusters; surface adsorbates must keep
            # their positions relative to the slab.
            if len(slab) == 0:
                mutant.center()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=True)
            count += 1

            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

        mutant = None if count == maxcount else slab + mutant

        return mutant


class FlatteningMutation(OffspringCreator):
    """A mutation that flattens the nanoparticle by projecting the coordinates
    to a plane that cuts the structure in a random angle.
    Atoms are then perturbed perpendicular to the plane within a given thickness.

    Parameters
    ----------
    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    thickness_factor: Factor to multiply with the average blmin to determine
        the thickness of the slab for projection.

    test_dist_to_slab: Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, n_top, thickness_factor=0.5,
                 test_dist_to_slab=True, rng=None, verbose=False,
                 max_inner_attempts=5000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.thickness_factor = thickness_factor
        self.test_dist_to_slab = test_dist_to_slab
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0

        self.descriptor = "FlatteningMutation"
        self.min_inputs = 1

    def _candidate_normals(self, positions, center_of_mass, slab):
        centered = positions - center_of_mass
        candidates = []
        outward = None
        max_candidates = max(1, min(int(self.max_inner_attempts), 6))

        if len(slab) > 0:
            outward = center_of_mass - np.mean(slab.get_positions(), axis=0)
            _append_unique_unit_vector(candidates, outward)

        if len(centered) > 1:
            covariance = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)
            axes = [eigenvectors[:, index] for index in order]

            for axis in axes:
                oriented_axis = axis
                if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                    oriented_axis = -oriented_axis
                _append_unique_unit_vector(candidates, oriented_axis)

            if len(axes) >= 2:
                blends = [axes[0] + axes[-1]]
                if len(axes) >= 3:
                    blends.append(axes[1] + axes[-1])
                else:
                    blends.append(axes[0] + axes[1])
                for axis in blends:
                    oriented_axis = axis
                    if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                        oriented_axis = -oriented_axis
                    _append_unique_unit_vector(candidates, oriented_axis)

            radial_norms = np.linalg.norm(centered, axis=1)
            if len(radial_norms) > 0:
                radial_axis = centered[int(np.argmax(radial_norms))]
                if outward is not None and np.dot(radial_axis, outward) < 0.0:
                    radial_axis = -radial_axis
                _append_unique_unit_vector(candidates, radial_axis)
        else:
            _append_unique_unit_vector(candidates, np.array([0.0, 0.0, 1.0]))

        attempts = 0
        while len(candidates) < max_candidates and attempts < 100:
            axis = _random_unit_vector(self.rng)
            if outward is not None and np.dot(axis, outward) < 0.0:
                axis = -axis
            _append_unique_unit_vector(candidates, axis)
            attempts += 1

        return candidates[:max_candidates]

    def _resolve_normal_offsets(
        self,
        projected_positions,
        target_offsets,
        atomic_numbers,
        clearance_margin,
    ):
        from scipy.spatial.distance import pdist, squareform

        n_atoms = len(projected_positions)
        if n_atoms <= 1:
            return np.zeros(n_atoms)

        order = np.argsort(target_offsets)
        ordered_targets = np.asarray(target_offsets, dtype=float)[order].copy()
        ordered_positions = projected_positions[order]
        ordered_numbers = atomic_numbers[order]

        # Vectorize lateral distance calculations
        lateral_distances = squareform(pdist(ordered_positions))

        # Vectorize blmin lookup
        blmin_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                blmin_matrix[i, j] = _get_blmin_distance(self.blmin, ordered_numbers[i], ordered_numbers[j])

        required_distances = blmin_matrix + clearance_margin

        # Calculate required offsets only for upper triangle
        required_offsets = np.zeros((n_atoms, n_atoms), dtype=float)
        mask = lateral_distances + 1e-12 < required_distances
        sq_diff = required_distances**2 - lateral_distances**2
        required_offsets[mask] = np.sqrt(np.maximum(sq_diff[mask], 0.0))

        # Sequential solve optimized with vectorized lookback
        solved_offsets = ordered_targets
        for i in range(1, n_atoms):
            solved_offsets[i] = max(
                solved_offsets[i],
                np.max(solved_offsets[:i] + required_offsets[:i, i])
            )

        solved_offsets -= np.mean(solved_offsets)
        offsets = np.empty(n_atoms, dtype=float)
        offsets[order] = solved_offsets
        return offsets

    def _build_flatten_candidate(
        self,
        positions,
        center_of_mass,
        normal,
        atomic_numbers,
        desired_thickness,
        avg_blmin,
    ):
        centered = positions - center_of_mass
        original_offsets = np.dot(centered, normal)
        projected_positions = positions - original_offsets[:, np.newaxis] * normal
        current_span = float(np.ptp(original_offsets))

        if current_span <= 1e-12:
            target_offsets = np.zeros(len(positions), dtype=float)
        else:
            compression = min(1.0, desired_thickness / current_span)
            target_offsets = (original_offsets - np.mean(original_offsets)) * compression

        clearance_margin = max(1e-3, 1e-3 * avg_blmin)
        resolved_offsets = self._resolve_normal_offsets(
            projected_positions,
            target_offsets,
            atomic_numbers,
            clearance_margin,
        )
        candidate_positions = projected_positions + resolved_offsets[:, np.newaxis] * normal

        original_span = max(current_span, 1e-12)
        flattened_span = float(np.ptp(resolved_offsets))
        flatten_ratio = flattened_span / original_span
        rms_displacement = (
            np.linalg.norm(candidate_positions - positions)
            / max(1, len(positions)) ** 0.5
        )
        score = flatten_ratio + 0.15 * (rms_displacement / max(avg_blmin, 1e-12))
        return score, candidate_positions

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: flattening"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]

        return self.finalize_individual(indi), "mutation: flattening"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]

        mutant = top.copy()
        pos = mutant.get_positions()
        atomic_numbers = mutant.get_atomic_numbers()
        cm = np.average(pos, axis=0)

        avg_blmin = np.mean(list(self.blmin.values()))
        desired_thickness = max(0.05 * avg_blmin, avg_blmin * self.thickness_factor)

        candidate_positions = [
            self._build_flatten_candidate(
                pos,
                cm,
                normal,
                atomic_numbers,
                desired_thickness,
                avg_blmin,
            )
            for normal in self._candidate_normals(pos, cm, slab)
        ]

        candidate_positions.sort(key=lambda item: item[0])
        self.last_attempt_count = 0
        for _score, new_positions in candidate_positions:
            self.last_attempt_count += 1
            mutant.set_positions(new_positions)
            # Only center gas-phase clusters; surface adsorbates must keep
            # their positions relative to the slab.
            if len(slab) == 0:
                mutant.center()

            too_close = atoms_too_close(mutant, self.blmin)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

            if not too_close:
                return slab + mutant

        if len(candidate_positions) == 0:
            self.last_attempt_count = 0
            return None

        return None


class BreathingMutation(OffspringCreator):
    """Uniformly scales all atom positions relative to the centre of mass.

    Each attempt samples a random scale factor in ``[scale_min, scale_max]``
    and accepts if no pair of atoms violates *blmin*.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of atoms optimised by the GA.
    scale_min, scale_max : float
        Bounds for the uniform scale-factor distribution.
    test_dist_to_slab : bool
        Also check distances to slab atoms.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Maximum number of random scale attempts per call.
    """

    def __init__(self, blmin, n_top, scale_min=0.9, scale_max=1.1,
                 test_dist_to_slab=True, rng=None, verbose=False,
                 max_inner_attempts=1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.test_dist_to_slab = test_dist_to_slab
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.descriptor = "BreathingMutation"
        self.min_inputs = 1

    def _minimum_feasible_scale(self, positions, atomic_numbers):
        from scipy.spatial.distance import pdist

        n_atoms = len(positions)
        if n_atoms <= 1:
            return self.scale_min

        # Compute pairwise distances
        distances = pdist(positions)
        if np.any(distances <= 1e-12):
            return np.inf

        # Compute pairwise blmin requirements
        blmin_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                blmin_matrix[i, j] = _get_blmin_distance(self.blmin, atomic_numbers[i], atomic_numbers[j])

        # We only need the condensed upper triangle for blmin
        required_condensed = blmin_matrix[np.triu_indices(n_atoms, k=1)]

        # Calculate minimum required scale to avoid clashes
        lower_bound = np.max(required_condensed / distances)
        return max(self.scale_min, lower_bound)

    def _candidate_scales(self, positions, atomic_numbers, slab):
        feasible_lower = self._minimum_feasible_scale(positions, atomic_numbers)
        if feasible_lower > self.scale_max + 1e-12:
            return []

        feasible_lower = max(self.scale_min, feasible_lower)
        interval_width = max(0.0, self.scale_max - feasible_lower)
        max_candidates = max(1, min(int(self.max_inner_attempts), 5))
        candidates = []
        tol = 1e-9
        allow_unit_scale = interval_width <= tol

        def append_candidate(scale):
            scale = float(scale)
            if scale < feasible_lower - tol or scale > self.scale_max + tol:
                return
            if not allow_unit_scale and abs(scale - 1.0) <= tol:
                return
            for existing in candidates:
                if abs(scale - existing) <= 1e-6:
                    return
            candidates.append(scale)

        contraction_width = max(0.0, 1.0 - feasible_lower)
        expansion_width = max(0.0, self.scale_max - 1.0)
        contraction_candidates = []
        expansion_candidates = []

        if contraction_width > tol:
            contraction_candidates = [
                1.0 - 0.5 * contraction_width,
                feasible_lower,
            ]
        if expansion_width > tol:
            expansion_candidates = [
                1.0 + 0.5 * expansion_width,
                self.scale_max,
            ]

        ordered_groups = []
        if len(slab) > 0:
            ordered_groups = [contraction_candidates, expansion_candidates]
        elif expansion_width >= contraction_width:
            ordered_groups = [expansion_candidates, contraction_candidates]
        else:
            ordered_groups = [contraction_candidates, expansion_candidates]

        for group in ordered_groups:
            for scale in group:
                append_candidate(scale)

        if contraction_candidates and expansion_candidates:
            append_candidate(0.5 * (feasible_lower + self.scale_max))
        elif allow_unit_scale:
            append_candidate(1.0)

        return candidates[:max_candidates]

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: breathing"
        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]
        return self.finalize_individual(indi), "mutation: breathing"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]
        pos = top.get_positions()
        cm = np.average(pos, axis=0)
        num = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()

        self.last_attempt_count = 0
        for scale in self._candidate_scales(pos, num, slab):
            self.last_attempt_count += 1
            s = scale
            new_pos = cm + s * (pos - cm)
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc)
            if atoms_too_close(cand, self.blmin):
                continue
            if self.test_dist_to_slab and len(slab) > 0 and atoms_too_close_two_sets(slab, cand, self.blmin):
                    continue
            return slab + cand
        return None


class InPlaneSlideMutation(OffspringCreator):
    """Randomly translates adsorbate atoms parallel to the slab surface.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of adsorbate atoms optimised by the GA.
    surface_normal_axis : int
        Cartesian axis index (0, 1, or 2) normal to the surface.
    max_displacement : float
        Maximum displacement magnitude (Å) per in-plane direction.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Maximum number of random displacement attempts per call.
    """

    def __init__(self, blmin, n_top, surface_normal_axis=2,
                 max_displacement=2.0, rng=None, verbose=False,
                 max_inner_attempts=1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.surface_normal_axis = surface_normal_axis
        self.max_displacement = max_displacement
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.descriptor = "InPlaneSlideMutation"
        self.min_inputs = 1

    def _candidate_shift_vectors(self, slab, positions, in_plane):
        if self.max_displacement <= 1e-12:
            return []

        max_candidates = max(1, min(int(self.max_inner_attempts), 12))
        positions_2d = positions[:, in_plane]
        center_2d = np.mean(positions_2d, axis=0)
        directions = []
        primary_direction = None

        if len(slab) > 0:
            slab_2d = slab.get_positions()[:, in_plane]
            delta = center_2d - slab_2d
            distance_sq = np.sum(delta * delta, axis=1)
            nearest = np.argsort(distance_sq)[:min(8, len(distance_sq))]
            if len(nearest) > 0:
                weights = 1.0 / np.maximum(distance_sq[nearest], 1e-3)
                repulsion = np.sum(delta[nearest] * weights[:, np.newaxis], axis=0)
                _append_unique_unit_vector(directions, repulsion)
                if len(directions) > 0:
                    primary_direction = directions[0]

        centered = positions_2d - center_2d
        if len(centered) > 1:
            covariance = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            for index in np.argsort(eigenvalues)[::-1]:
                _append_unique_unit_vector(directions, eigenvectors[:, index])

        _append_unique_unit_vector(directions, np.array([1.0, 0.0]))
        _append_unique_unit_vector(directions, np.array([0.0, 1.0]))
        _append_unique_unit_vector(directions, np.array([1.0, 1.0]))
        _append_unique_unit_vector(directions, np.array([1.0, -1.0]))

        if primary_direction is None and len(directions) > 0:
            primary_direction = directions[0]

        ordered_directions = []
        if primary_direction is not None:
            ordered_directions.append(primary_direction)
            perpendicular = np.array([-primary_direction[1], primary_direction[0]])
            _append_unique_unit_vector(ordered_directions, primary_direction)
            _append_unique_unit_vector(ordered_directions, perpendicular)
            _append_unique_unit_vector(ordered_directions, -perpendicular)
            _append_unique_unit_vector(ordered_directions, -primary_direction)

        for direction in directions:
            _append_unique_unit_vector(ordered_directions, direction)
            _append_unique_unit_vector(ordered_directions, -direction)

        magnitudes = [
            0.5 * self.max_displacement,
            self.max_displacement,
            0.25 * self.max_displacement,
        ]
        candidate_shifts = []
        for direction in ordered_directions:
            for magnitude in magnitudes:
                if magnitude <= 1e-12:
                    continue
                shift_2d = magnitude * direction
                shift = np.zeros(3, dtype=float)
                shift[in_plane[0]] = shift_2d[0]
                shift[in_plane[1]] = shift_2d[1]
                duplicate = False
                for existing in candidate_shifts:
                    if np.linalg.norm(existing - shift) <= 1e-8:
                        duplicate = True
                        break
                if not duplicate:
                    candidate_shifts.append(shift)
                if len(candidate_shifts) >= max_candidates:
                    return candidate_shifts

        return candidate_shifts[:max_candidates]

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: in_plane_slide"
        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid")]
        return self.finalize_individual(indi), "mutation: in_plane_slide"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]
        pos = top.get_positions().copy()
        num = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()

        in_plane = [i for i in range(3) if i != self.surface_normal_axis]

        self.last_attempt_count = 0
        for shift in self._candidate_shift_vectors(slab, pos, in_plane):
            self.last_attempt_count += 1
            new_pos = pos.copy()
            new_pos += shift
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc)
            if atoms_too_close(cand, self.blmin):
                continue
            if len(slab) > 0 and atoms_too_close_two_sets(slab, cand, self.blmin):
                    continue
            return slab + cand
        return None


# fmt: on
