# fmt: off

from __future__ import annotations

"""A collection of mutations that can be used."""
from math import cos, pi, sin

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
    get_rotation_matrix,
)

from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng


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
        indi.info["data"]["parents"] = [f.info["confid"]]

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
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            ok = False
            for tag in np.unique(tags):
                select = np.where(tags == tag)
                if self.rng.random() < self.rattle_prop:
                    ok = True
                    r = self.rng.random(3)
                    pos[select] += st * (r - 0.5)

            if not ok:
                # Nothing got rattled
                continue

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
        indi.info["data"]["parents"] = [f.info["confid"]]

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

            moved = False
            for tag in np.unique(tags):
                select = np.where(tags == tag)
                if self.rng.random() < self.rattle_prop:
                    moved = True
                    a = self.rng.uniform(-self.in_plane_strength, self.in_plane_strength)
                    b = self.rng.uniform(-self.in_plane_strength, self.in_plane_strength)
                    c = self.rng.uniform(-self.normal_strength, self.normal_strength)
                    pos[select] += a * u + b * v + c * normal

            if not moved:
                continue

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        return slab + top


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
        indi.info["data"]["parents"] = [f.info["confid"]]

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

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            for _ in range(swaps):
                i = j = 0
                while sym[i] == sym[j]:
                    i = self.rng.integers(0, n)
                    j = self.rng.integers(0, n)
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
    """PermutationMutation variant compatible with numpy.random.Generator.

    Passing a ``Generator`` instance allows reproducible sampling without
    falling back to the legacy global ``numpy.random`` module.
    """

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

    def __init__(self, blmin, n_top, reflect=False, rng=None,
                 verbose=False, max_tries: int = 1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.reflect = reflect
        self.max_tries = max_tries

        self.descriptor = "MirrorMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: mirror"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

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

            # first select a randomly oriented cutting plane
            theta = pi * self.rng.random()
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
            for z in nu:
                if nu[z] % 2 == 0:
                    continue
                while n_use.count(z) > nu[z]:
                    for i in range(int(len(n_use) / 2), len(n_use)):
                        if n_use[i] == z:
                            del p_use[i]
                            del n_use[i]
                            break
                assert n_use.count(z) == nu[z]

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
                 verbose=False, max_inner_attempts: int = 10000):
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
        indi.info["data"]["parents"] = [f.info["confid"]]

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
                    line = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                    while True:
                        axis = self.rng.random(3)
                        axis /= np.linalg.norm(axis)
                        a = np.arccos(np.dot(axis, line))
                        if np.pi / 4 < a < np.pi * 3 / 4:
                            break
                else:
                    axis = self.rng.random(3)
                    axis /= np.linalg.norm(axis)

                angle = self.min_angle
                angle += 2 * (np.pi - self.min_angle) * self.rng.random()

                m = get_rotation_matrix(axis, angle)
                newpos[indices[tag]] = np.dot(m, (p - cop).T).T + cop

            mutant.set_positions(newpos)
            # For clusters, just center without wrapping
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
                 max_inner_attempts: int = 5000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.thickness_factor = thickness_factor
        self.test_dist_to_slab = test_dist_to_slab
        self.max_inner_attempts = max_inner_attempts

        self.descriptor = "FlatteningMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: flattening"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]

        return self.finalize_individual(indi), "mutation: flattening"

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]

        mutant = top.copy()
        pos = mutant.get_positions()
        cm = np.average(pos, axis=0)

        # Calculate average blmin for thickness
        avg_blmin = np.mean(list(self.blmin.values()))
        thickness = avg_blmin * self.thickness_factor

        count = 0
        maxcount = self.max_inner_attempts
        too_close = True
        while too_close and count < maxcount:
            count += 1
            # Generate a random normal vector for the plane
            theta = self.rng.random() * np.pi
            phi = self.rng.random() * 2 * np.pi
            n = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
            n = n / np.linalg.norm(n) # Ensure unit vector

            new_pos = np.zeros_like(pos)
            for i in range(len(pos)):
                # Vector from CM to atom
                v = pos[i] - cm
                # Projection onto the plane (vector from CM)
                projected_v = v - np.dot(v, n) * n
                # Add random perturbation perpendicular to the plane
                perturbation_magnitude = self.rng.uniform(-thickness / 2, thickness / 2)
                new_pos[i] = cm + projected_v + perturbation_magnitude * n

            mutant.set_positions(new_pos)
            # For clusters, just center without wrapping
            mutant.center()

            too_close = atoms_too_close(mutant, self.blmin)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

        if count == maxcount:
            return None

        return slab + mutant


# fmt: on
