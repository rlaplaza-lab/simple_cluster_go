"""Rattle-style mutations that perturb atomic positions locally."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _ensure_rng,
    _preserves_mobile_connectivity,
    _resolve_op_connectivity_factor,
    _reanchor_mobile_to_slab,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.system_types import SystemType, get_system_policy

__all__ = ["AnisotropicRattleMutation", "RattleMutation"]


class RattleMutation(OffspringCreator):
    """An implementation of the rattle mutation as described in:

    R.L. Johnston Dalton Transactions, Vol. 22,
    No. 22. (2003), pp. 4193-4207

    Parameters
    ----------
    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved; each
        Cartesian component is displaced uniformly in
        [-rattle_strength, rattle_strength).

    rattle_prop: The probability with which each atom (or each tag group,
        when use_tags is True) is rattled. One group is always rattled
        per attempt, regardless of this probability.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    system_type: System type (e.g., "gas_cluster", "surface_cluster",
        "gas_cluster_adsorbate", "surface_cluster_adsorbate").
        Used to ensure physical validity of mutations.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    verbose: bool
        If True, print verbose output.

    """

    def __init__(self, blmin, n_top, system_type: SystemType, rattle_strength=0.8,
                 rattle_prop=0.4, test_dist_to_slab=True, use_tags=False,
                 target_tags=None, verbose=False, rng=None,
                 surface_normal_axis=2, patience=60):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.system_type = system_type
        self.surface_normal_axis = surface_normal_axis
        # Consecutive tolerated failures once the strength schedule has annealed
        # to its lowest tier; hopeless (very dense) parents bail out early.
        self.patience = max(0, int(patience))
        self._policy = get_system_policy(system_type)

        self.descriptor = "RattleMutation"
        self.min_inputs = 1
        self.last_attempt_count = 0

    def get_new_individual(self, parents):
        """Generates a new individual by applying the rattle mutation to a parent.

        Args:
            parents: A list containing the parent Atoms object.

        Returns:
            A tuple containing the new Atoms object and a description of the mutation.

        """
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: rattle")

    def mutate(self, atoms):
        """Applies the rattle mutation to the given Atoms object.

        Args:
            atoms: The Atoms object to be mutated.

        Returns:
            A new Atoms object after applying the rattle mutation, or None if mutation fails.

        """
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[len(atoms) - N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
        if len(unique_tags) == 0:
            # Nothing to rattle (empty mobile region or no matching tag).
            return None

        maxcount = 1000
        strengths = (
            self.rattle_strength,
            0.5 * self.rattle_strength,
            0.25 * self.rattle_strength,
        )
        use_mic = bool(self._policy.uses_surface)
        parent_mobile = atoms
        self.last_attempt_count = 0
        failures_at_lowest_tier = 0
        for count in range(maxcount):
            tier = min(count * len(strengths) // maxcount, len(strengths) - 1)
            strength = strengths[tier]
            st = 2. * strength
            pos = pos_ref.copy()

            # Guarantee at least one tag is rattled, then sample the rest.
            guaranteed = self.rng.integers(len(unique_tags))
            for idx, tag in enumerate(unique_tags):
                if idx == guaranteed or self.rng.random() < self.rattle_prop:
                    select = np.where(tags == tag)
                    r = self.rng.random(3)
                    pos[select] += st * (r - 0.5)

            self.last_attempt_count += 1
            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                top = _reanchor_mobile_to_slab(
                    atoms, top, slab, self.surface_normal_axis)
            too_close = atoms_too_close(
                top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)
            if too_close:
                if tier == len(strengths) - 1 and self.patience > 0:
                    failures_at_lowest_tier += 1
                    if failures_at_lowest_tier >= self.patience:
                        return None
                continue
            if not _preserves_mobile_connectivity(
                parent_mobile,
                top,
                use_mic=use_mic,
                connectivity_factor=_resolve_op_connectivity_factor(self),
            ):
                if tier == len(strengths) - 1 and self.patience > 0:
                    failures_at_lowest_tier += 1
                    if failures_at_lowest_tier >= self.patience:
                        return None
                continue
            mutant = slab + top
            if not self._policy.uses_surface:
                mutant.center()
            return mutant

        return None


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
        system_type: SystemType,
        in_plane_strength=1.0,
        normal_strength=0.2,
        rattle_prop=0.5,
        test_dist_to_slab=True,
        use_tags=False,
        target_tags=None,
        rng=None,
        verbose=False,
        surface_normal_axis=2,
        patience=60,
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
        self.target_tags = target_tags
        self.system_type = system_type
        self.surface_normal_axis = surface_normal_axis
        # Consecutive tolerated failures once the scale schedule has annealed
        # to its lowest tier; hopeless (very dense) parents bail out early.
        self.patience = max(0, int(patience))
        self._policy = get_system_policy(system_type)

        self.descriptor = "AnisotropicRattleMutation"
        self.min_inputs = 1
        self.last_attempt_count = 0

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: anisotropic_rattle")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        atoms = atoms[len(atoms) - N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
        if len(unique_tags) == 0:
            # Nothing to rattle (empty mobile region or no matching tag).
            return None

        maxcount = 1000
        scales = (1.0, 0.5, 0.25)
        use_mic = bool(self._policy.uses_surface)
        parent_mobile = atoms
        count = 0
        self.last_attempt_count = 0
        failures_at_lowest_tier = 0
        while count < maxcount:
            tier = min(count * len(scales) // maxcount, len(scales) - 1)
            scale = scales[tier]
            in_plane = self.in_plane_strength * scale
            normal_s = self.normal_strength * scale
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
            unique_tags_local = unique_tags
            guaranteed = self.rng.integers(len(unique_tags_local))
            for idx, tag in enumerate(unique_tags_local):
                if idx == guaranteed or self.rng.random() < self.rattle_prop:
                    select = np.where(tags == tag)
                    a = self.rng.uniform(-in_plane, in_plane)
                    b = self.rng.uniform(-in_plane, in_plane)
                    c = self.rng.uniform(-normal_s, normal_s)
                    pos[select] += a * u + b * v + c * normal

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                top = _reanchor_mobile_to_slab(
                    atoms, top, slab, self.surface_normal_axis)
            count += 1
            self.last_attempt_count += 1
            too_close = atoms_too_close(top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)
            failed = too_close
            if not failed and not _preserves_mobile_connectivity(
                parent_mobile,
                top,
                use_mic=use_mic,
                connectivity_factor=_resolve_op_connectivity_factor(self),
            ):
                failed = True
            if failed:
                if tier == len(scales) - 1 and self.patience > 0:
                    failures_at_lowest_tier += 1
                    if failures_at_lowest_tier >= self.patience:
                        return None
                continue
            result = slab + top
            if not self._policy.uses_surface:
                result.center()
            return result

        return None

# fmt: on
