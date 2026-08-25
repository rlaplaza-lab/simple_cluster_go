"""Bounded-retry regression tests for GA crossover and stochastic mutations.

Complements ``test_retry_reduction.py``: these pin the *failure-path* bounds
introduced to stop hopeless parents from burning full retry budgets inside a
single parallel offspring job (batch walltime is bounded by the slowest job):

- CutAndSplicePairing fixed-cell branch stops after a few jittered passes
  instead of re-walking identical cut configurations up to maxcount times.
- RattleMutation / AnisotropicRattleMutation bail out early once the strength
  schedule has annealed to its lowest tier and keeps failing.
- PermutationMutation applies swap pairs incrementally (skipping clashing
  pairs) instead of redrawing the whole swap set on any clash.
"""

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import atoms_too_close

from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.ase_ga_patches.mutations import (
    AnisotropicRattleMutation,
    PermutationMutation,
    RattleMutation,
)


def _blmin(atoms, ratio=0.7):
    from ase_ga.utilities import closest_distances_generator, get_all_atom_types

    all_types = get_all_atom_types(atoms, range(len(atoms)))
    return closest_distances_generator(all_types, ratio_of_covalent_radii=ratio)


def _impossible_blmin(atoms, floor=50.0):
    """blmin that no rearrangement of ``atoms`` can ever satisfy."""
    numbers = sorted(set(atoms.get_atomic_numbers()))
    return {(z1, z2): floor for z1 in numbers for z2 in numbers}


# ---------------------------------------------------------------------------
# 1. Crossover: fixed-cell failure path terminates after few jittered passes
# ---------------------------------------------------------------------------
class TestCutAndSpliceFixedCellBound:
    def test_hopeless_parents_return_none_quickly(self, au2pt2_atoms):
        atoms1 = au2pt2_atoms.copy()
        atoms2 = au2pt2_atoms.copy()
        pairing = CutAndSplicePairing(
            slab=Atoms(cell=atoms1.get_cell(), pbc=atoms1.get_pbc()),
            n_top=len(atoms1),
            blmin=_impossible_blmin(atoms1),
            rng=np.random.default_rng(3),
            fixed_cell_max_passes=3,
        )
        child = pairing.cross(atoms1, atoms2)
        assert child is None
        # 3 passes x at most 12 ranked cuts per pass.
        assert pairing.last_attempt_count <= 3 * 12
        assert pairing.last_attempt_count > 0

    def test_default_passes_are_small(self):
        pairing = CutAndSplicePairing(
            slab=Atoms(),
            n_top=4,
            blmin={(78, 78): 2.0},
            rng=np.random.default_rng(0),
        )
        assert pairing.fixed_cell_max_passes == 3

    def test_jittered_configs_keep_normals_and_re_rank(self, au2pt2_atoms):
        atoms1 = au2pt2_atoms.copy()
        atoms2 = au2pt2_atoms.copy()
        atoms2.positions += 0.3
        cell = np.eye(3) * 20.0
        pairing = CutAndSplicePairing(
            slab=Atoms(cell=cell, pbc=False),
            n_top=len(atoms1),
            blmin=_blmin(atoms1),
            rng=np.random.default_rng(11),
        )
        base = pairing._candidate_cut_configurations(atoms1, atoms2, cell)
        jittered = pairing._jittered_cut_configurations(base, atoms1, atoms2, cell)

        assert len(jittered) == len(base)

        def group_by_normal(configs):
            grouped: dict[tuple, list[tuple[float, np.ndarray]]] = {}
            for score, point, normal in configs:
                key = tuple(np.round(normal, 10))
                grouped.setdefault(key, []).append(
                    (float(score), np.asarray(point).ravel())
                )
            return grouped

        base_by_normal = group_by_normal(base)
        jittered_by_normal = group_by_normal(jittered)
        assert set(base_by_normal) == set(jittered_by_normal)

        # Cut points are offset along their normals for at least some cuts,
        # and the re-ranked walk order stays ascending by balance score.
        some_moved = False
        for key, base_entries in base_by_normal.items():
            jitted_entries = jittered_by_normal[key]
            assert len(base_entries) == len(jitted_entries)
            base_pts = {tuple(np.round(p, 12)) for _, p in base_entries}
            jit_pts = {tuple(np.round(p, 12)) for _, p in jitted_entries}
            if not jit_pts.issubset(base_pts):
                some_moved = True
        assert some_moved
        scores = [score for score, _, _ in jittered]
        assert scores == sorted(scores)
        assert all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# 2. Rattle / anisotropic rattle: lowest-tier consecutive-failure exit
# ---------------------------------------------------------------------------
class TestRattleLowestTierPatience:
    def test_hopeless_dense_parent_exits_before_maxcount(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        # Collapse the tetrahedron so no rattle of strength 0.05 can satisfy
        # a 5 A minimum-interatomic-distance gate.
        atoms.set_positions(np.tile([0.0, 0.0, 0.0], (len(atoms), 1)))
        mut = RattleMutation(
            {(78, 78): 5.0},
            len(atoms),
            system_type="gas_cluster",
            rattle_strength=0.05,
            test_dist_to_slab=False,
            rng=np.random.default_rng(5),
        )
        assert mut.mutate(atoms) is None
        # Lowest tier starts at ~maxcount/3; patience must stop the loop well
        # before the absolute cap of 1000 attempts.
        assert mut.last_attempt_count < 1000

    def test_patience_zero_restores_full_budget(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        atoms.set_positions(np.tile([0.0, 0.0, 0.0], (len(atoms), 1)))
        mut = RattleMutation(
            {(78, 78): 5.0},
            len(atoms),
            system_type="gas_cluster",
            rattle_strength=0.05,
            test_dist_to_slab=False,
            patience=0,
            rng=np.random.default_rng(5),
        )
        assert mut.mutate(atoms) is None
        assert mut.last_attempt_count == 1000

    def test_easy_parent_still_succeeds_with_default_patience(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        mut = RattleMutation(
            _blmin(atoms),
            len(atoms),
            system_type="gas_cluster",
            rattle_strength=0.3,
            rng=np.random.default_rng(7),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert mut.last_attempt_count <= 10


class TestAnisotropicRattleLowestTierPatience:
    def test_hopeless_dense_parent_exits_before_maxcount(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        atoms.set_positions(np.tile([0.0, 0.0, 0.0], (len(atoms), 1)))
        mut = AnisotropicRattleMutation(
            {(78, 78): 5.0},
            len(atoms),
            system_type="gas_cluster",
            in_plane_strength=0.05,
            normal_strength=0.02,
            test_dist_to_slab=False,
            rng=np.random.default_rng(9),
        )
        assert mut.mutate(atoms) is None
        assert mut.last_attempt_count < 1000

    def test_easy_parent_still_succeeds_with_default_patience(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        mut = AnisotropicRattleMutation(
            _blmin(atoms),
            len(atoms),
            system_type="gas_cluster",
            in_plane_strength=0.3,
            normal_strength=0.05,
            test_dist_to_slab=False,
            rng=np.random.default_rng(12),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert mut.last_attempt_count <= 10


# ---------------------------------------------------------------------------
# 3. Permutation: incremental clash-skipping application
# ---------------------------------------------------------------------------
class TestPermutationIncrementalApplication:
    def test_all_clashing_pairs_return_none_without_redraw_spins(self):
        atoms = Atoms(
            "Pt5Au",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [0, 2.5, 0],
                [2.5, 2.5, 0],
                [1.25, 1.25, 2.5],
                [3.75, 1.25, 2.5],
            ],
        )
        atoms.center(vacuum=10.0)
        impossible = {(z1, z2): 100.0 for z1 in (78, 79) for z2 in (78, 79)}
        mut = PermutationMutation(
            len(atoms),
            system_type="gas_cluster",
            probability=1.0,
            blmin=impossible,
            rng=np.random.default_rng(8),
        )
        assert mut.mutate(atoms) is None

    def test_partial_application_is_clash_free_and_changed(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        blmin = _blmin(atoms)
        successes = 0
        for seed in range(10):
            mut = PermutationMutation(
                len(atoms),
                system_type="gas_cluster",
                probability=1.0,
                blmin=blmin,
                rng=np.random.default_rng(seed),
            )
            result = mut.mutate(atoms)
            if result is None:
                continue
            successes += 1
            assert not atoms_too_close(result, blmin)
            assert not np.allclose(result.get_positions(), atoms.get_positions())
            # Stoichiometry preserved per site.
            assert sorted(result.get_chemical_symbols()) == sorted(
                atoms.get_chemical_symbols()
            )
        assert successes == 10

    def test_same_seed_is_reproducible(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        results = []
        for _ in range(2):
            mut = PermutationMutation(
                len(atoms),
                system_type="gas_cluster",
                probability=1.0,
                blmin=_blmin(atoms),
                rng=np.random.default_rng(123),
            )
            results.append(mut.mutate(atoms))
        assert results[0] is not None and results[1] is not None
        assert np.allclose(results[0].get_positions(), results[1].get_positions())

    def test_no_valid_pairs_returns_none(self, pt4_tetrahedron):
        mut = PermutationMutation(
            len(pt4_tetrahedron),
            system_type="gas_cluster",
            probability=1.0,
            rng=np.random.default_rng(1),
        )
        assert mut.mutate(pt4_tetrahedron) is None


@pytest.mark.parametrize("patience", [0, 60])
def test_rattle_patience_attribute_round_trips(pt4_tetrahedron, patience):
    mut = RattleMutation(
        _blmin(pt4_tetrahedron),
        len(pt4_tetrahedron),
        system_type="gas_cluster",
        patience=patience,
        rng=np.random.default_rng(0),
    )
    assert mut.patience == patience
