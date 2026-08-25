import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.mutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    FlatteningMutation,
    InPlaneSlideMutation,
    MirrorMutation,
    OverlapReliefMutation,
    PermutationMutation,
    RattleMutation,
    RotationalMutation,
    ShellSwapMutation,
)

PERMISSIVE_BLMIN = {(z1, z2): 0.5 for z1 in (1, 8, 78, 79) for z2 in (1, 8, 78, 79)}


def _tagged_core_adsorbate():
    """Core (tag 0) plus a frozen four-atom adsorbate (tag 1)."""
    atoms = Atoms(
        symbols=["Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [1.3, 2.25, 0.0],
            [1.3, 0.75, 2.12],
            [1.3, 0.75, 4.30],
            [2.3, 0.75, 4.60],
            [1.3, 1.75, 4.60],
            [1.3, 0.75, 5.30],
        ],
    )
    atoms.set_tags([0, 0, 0, 0, 1, 1, 1, 1])
    atoms.center(vacuum=8.0)
    return atoms


def test_permutation_mutation_returns_none_when_single_species(pt3_atoms, rng):
    mut = PermutationMutation(
        n_top=3, probability=0.5, system_type="gas_cluster", rng=rng
    )
    assert mut.mutate(pt3_atoms.copy()) is None


def test_breathing_mutation_succeeds_on_loose_pt3(pt3_atoms, rng):
    blmin = closest_distances_generator(
        get_all_atom_types(pt3_atoms, range(3)),
        ratio_of_covalent_radii=0.7,
    )
    mut = BreathingMutation(
        blmin,
        3,
        scale_min=0.94,
        scale_max=1.06,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
        max_inner_attempts=3000,
    )
    out = mut.mutate(pt3_atoms.copy())
    assert out is not None, "Mutation must return a result"
    assert len(out) == len(pt3_atoms), "Atom count must be preserved"
    assert out.get_chemical_symbols() == pt3_atoms.get_chemical_symbols()

    import numpy as np

    displacement = np.linalg.norm(out.get_positions() - pt3_atoms.get_positions())
    assert displacement > 1e-6, f"Mutation must displace atoms, got {displacement}"
    assert mut.last_attempt_count <= 5, "Should complete within max attempts"


def test_overlap_relief_mutation_repairs_dense_pt4(rng):
    atoms = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.2],
        ],
    )
    atoms.center(vacuum=8.0)
    blmin = {(78, 78): 2.0}

    mut = OverlapReliefMutation(
        blmin,
        len(atoms),
        n_sweeps=4,
        jitter=0.01,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(atoms.copy())

    assert out is not None
    assert not atoms_too_close(out, blmin)


def test_shell_swap_mutation_moves_minority_species_outward(rng):
    atoms = Atoms(
        ["Au", "Pt", "Pt", "Pt"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [-2.3, 0.0, 0.0],
            [0.0, 2.3, 0.0],
        ],
    )
    atoms.center(vacuum=8.0)
    blmin = {(78, 78): 0.1, (78, 79): 0.1, (79, 79): 0.1}

    def mean_species_radius(atoms_obj, symbol):
        positions = atoms_obj.get_positions()
        center = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions - center, axis=1)
        indices = [
            idx
            for idx, sym in enumerate(atoms_obj.get_chemical_symbols())
            if sym == symbol
        ]
        return float(np.mean(radii[indices]))

    mut = ShellSwapMutation(
        len(atoms),
        blmin=blmin,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(atoms.copy())

    assert out is not None
    assert mean_species_radius(out, "Au") > mean_species_radius(atoms, "Au")


def test_in_plane_slide_mutation_succeeds_on_slab_adsorbate(rng):
    slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z_slab = float(np.max(slab.positions[:, 2]))
    cell = slab.get_cell()
    # Wide in-plane separation; high z reduces slab–adsorbate clash sensitivity.
    ads = Atoms(
        "Pt2",
        positions=[
            [0.15 * cell[0, 0], 0.2 * cell[1, 1], z_slab + 4.0],
            [0.65 * cell[0, 0], 0.55 * cell[1, 1], z_slab + 4.0],
        ],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    full = slab + ads
    idx_top = range(n_slab, len(full))
    blmin = closest_distances_generator(
        get_all_atom_types(full, idx_top),
        ratio_of_covalent_radii=0.7,
    )
    assert not atoms_too_close_two_sets(slab, ads, blmin)

    mut = InPlaneSlideMutation(
        blmin,
        2,
        surface_normal_axis=2,
        system_type="surface_cluster",
        rng=rng,
        max_inner_attempts=8000,
    )
    out = mut.mutate(full)
    assert out is not None
    assert len(out) == len(full)
    assert out.get_atomic_numbers().tolist() == full.get_atomic_numbers().tolist()
    assert mut.last_attempt_count <= 12


def test_factory_registers_overlap_relief_and_shell_swap(rng):
    composition = ["Au", "Pt", "Pt", "Pt"]
    atoms = Atoms("AuPt3")
    blmin = closest_distances_generator(
        get_all_atom_types(atoms, range(len(atoms))),
        ratio_of_covalent_radii=0.7,
    )

    operators, name_map = create_mutation_operators(
        composition,
        len(composition),
        blmin,
        rng=rng,
        use_adaptive=True,
        system_type="gas_cluster",
    )

    assert "overlap_relief" in name_map
    assert "shell_swap" in name_map
    assert operators[name_map["overlap_relief"]].descriptor == "OverlapReliefMutation"
    assert operators[name_map["shell_swap"]].descriptor == "ShellSwapMutation"


def test_anisotropic_rattle_mutation_runs_on_small_cluster(pt3_atoms, rng):
    blmin = {(78, 78): 0.5}
    mut = AnisotropicRattleMutation(
        blmin=blmin,
        n_top=3,
        in_plane_strength=0.2,
        normal_strength=0.05,
        rattle_prop=1.0,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    mutated = mut.mutate(pt3_atoms.copy())
    assert mutated is not None, "Mutation must return a result"
    assert len(mutated) == len(pt3_atoms), "Atom count must be preserved"

    import numpy as np

    displacement = np.linalg.norm(mutated.get_positions() - pt3_atoms.get_positions())
    assert displacement > 1e-6, f"Mutation must displace atoms, got {displacement}"


def test_breathing_mutation_moves_only_targeted_tag_group(rng):
    """G1: ``target_tags`` must gate which atoms the breathing mutation scales."""
    atoms = _tagged_core_adsorbate()
    core = slice(0, 4)
    ads = slice(4, 8)

    def build(target_tags):
        return BreathingMutation(
            PERMISSIVE_BLMIN,
            len(atoms),
            scale_min=0.94,
            scale_max=1.06,
            test_dist_to_slab=False,
            target_tags=target_tags,
            system_type="gas_cluster_adsorbate",
            rng=rng,
        )

    out_core = build([0]).mutate(atoms.copy())
    assert out_core is not None
    # Breathing never re-centres, so the frozen adsorbate must be untouched.
    assert np.array_equal(out_core.get_positions()[ads], atoms.get_positions()[ads])
    assert (
        np.linalg.norm(out_core.get_positions()[core] - atoms.get_positions()[core])
        > 1e-6
    )

    out_ads = build([1]).mutate(atoms.copy())
    assert out_ads is not None
    assert np.array_equal(out_ads.get_positions()[core], atoms.get_positions()[core])
    assert (
        np.linalg.norm(out_ads.get_positions()[ads] - atoms.get_positions()[ads]) > 1e-6
    )


def test_flattening_mutation_moves_only_targeted_tag_group(rng):
    """G1: ``target_tags`` must gate which atoms the flattening mutation projects."""
    atoms = _tagged_core_adsorbate()
    core = slice(0, 4)
    ads = slice(4, 8)

    def build(target_tags):
        return FlatteningMutation(
            PERMISSIVE_BLMIN,
            len(atoms),
            test_dist_to_slab=False,
            target_tags=target_tags,
            system_type="gas_cluster_adsorbate",
            rng=rng,
        )

    out_core = build([0]).mutate(atoms.copy())
    assert out_core is not None
    disp = out_core.get_positions() - atoms.get_positions()
    # Gas-phase flattening re-centres the mutant, so the frozen adsorbate may
    # only be shifted rigidly (identical displacement for every tag-1 atom).
    assert np.allclose(disp[ads] - disp[ads][0], 0.0, atol=1e-10)
    assert not np.allclose(disp[core] - disp[core][0], 0.0, atol=1e-6)

    out_ads = build([1]).mutate(atoms.copy())
    assert out_ads is not None
    disp_ads = out_ads.get_positions() - atoms.get_positions()
    assert np.allclose(disp_ads[core] - disp_ads[core][0], 0.0, atol=1e-10)
    assert not np.allclose(disp_ads[ads] - disp_ads[ads][0], 0.0, atol=1e-6)


def test_overlap_relief_final_check_honours_use_tags(rng):
    """G3: the final sweep must skip same-tag pairs when ``use_tags`` is set."""
    fragment = Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    fragment.set_tags([1, 1])
    fragment.center(vacuum=8.0)
    blmin = {(8, 8): 1.2}

    mut = OverlapReliefMutation(
        blmin,
        len(fragment),
        jitter=0.01,
        test_dist_to_slab=False,
        use_tags=True,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(fragment.copy())

    assert out is not None
    assert len(out) == len(fragment)
    # The rigid fragment keeps its (sub-blmin) internal distance.
    internal = np.linalg.norm(out.get_positions()[1] - out.get_positions()[0])
    assert internal == pytest.approx(0.9, abs=1e-6)


def _fail_until_call(n_calls):
    """Report "too close" for every attempt but the ``n_calls``-th one."""
    state = {"calls": 0}

    def fake_atoms_too_close(*_args, **_kwargs):
        state["calls"] += 1
        return state["calls"] < n_calls

    return fake_atoms_too_close


def test_rattle_accepts_late_success_before_patience(monkeypatch, pt3_atoms, rng):
    """A valid geometry found late in the anneal must still be accepted."""
    from scgo.ase_ga_patches.mutations import rattle as rattle_module

    # Attempt 300 is still above the lowest strength tier (~334), so the
    # lowest-tier patience budget has not started counting yet.
    monkeypatch.setattr(rattle_module, "atoms_too_close", _fail_until_call(300))
    mut = RattleMutation(
        {(78, 78): 0.5},
        3,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(pt3_atoms.copy())
    assert out is not None
    assert len(out) == len(pt3_atoms)


def test_rattle_returns_none_when_never_valid(monkeypatch, pt3_atoms, rng):
    from scgo.ase_ga_patches.mutations import rattle as rattle_module

    monkeypatch.setattr(rattle_module, "atoms_too_close", lambda *a, **k: True)
    mut = RattleMutation(
        {(78, 78): 0.5},
        3,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    assert mut.mutate(pt3_atoms.copy()) is None
    # Lowest-tier consecutive-failure patience stops the loop well before the
    # absolute cap of 1000 attempts.
    assert mut.last_attempt_count < 1000


def test_anisotropic_rattle_accepts_late_success_before_patience(
    monkeypatch, pt3_atoms, rng
):
    from scgo.ase_ga_patches.mutations import rattle as rattle_module

    monkeypatch.setattr(rattle_module, "atoms_too_close", _fail_until_call(300))
    mut = AnisotropicRattleMutation(
        {(78, 78): 0.5},
        3,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(pt3_atoms.copy())
    assert out is not None
    assert len(out) == len(pt3_atoms)


def test_permutation_accepts_success_on_final_pair(monkeypatch, au2pt2_atoms, rng):
    """Incremental application keeps trying candidate pairs after clashes."""
    from scgo.ase_ga_patches.mutations import permutation as permutation_module

    # Au2Pt2 yields four valid swap pairs; the first three clash and the
    # last checked pair succeeds, independent of the shuffle order.
    monkeypatch.setattr(permutation_module, "atoms_too_close", _fail_until_call(4))
    mut = PermutationMutation(
        n_top=4,
        probability=0.5,
        test_dist_to_slab=False,
        blmin=PERMISSIVE_BLMIN,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(au2pt2_atoms.copy())
    assert out is not None
    assert len(out) == len(au2pt2_atoms)


def test_permutation_returns_none_when_never_valid(monkeypatch, au2pt2_atoms, rng):
    from scgo.ase_ga_patches.mutations import permutation as permutation_module

    monkeypatch.setattr(permutation_module, "atoms_too_close", lambda *a, **k: True)
    mut = PermutationMutation(
        n_top=4,
        probability=0.5,
        test_dist_to_slab=False,
        blmin=PERMISSIVE_BLMIN,
        system_type="gas_cluster",
        rng=rng,
    )
    assert mut.mutate(au2pt2_atoms.copy()) is None


def _make_zero_top_operator(name, rng):
    blmin = PERMISSIVE_BLMIN
    system_type = "gas_cluster"
    if name == "anisotropic_rattle":
        return AnisotropicRattleMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "breathing":
        return BreathingMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "flattening":
        return FlatteningMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "in_plane_slide":
        return InPlaneSlideMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "overlap_relief":
        return OverlapReliefMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "permutation":
        return PermutationMutation(0, blmin=blmin, system_type=system_type, rng=rng)
    if name == "rattle":
        return RattleMutation(blmin, 0, system_type=system_type, rng=rng)
    if name == "rotational":
        return RotationalMutation(blmin, n_top=0, system_type=system_type, rng=rng)
    if name == "shell_swap":
        return ShellSwapMutation(0, blmin=blmin, system_type=system_type, rng=rng)
    raise AssertionError(f"unknown operator {name}")


@pytest.mark.parametrize(
    "name",
    [
        "anisotropic_rattle",
        "breathing",
        "flattening",
        "in_plane_slide",
        "overlap_relief",
        "permutation",
        "rattle",
        "rotational",
        "shell_swap",
    ],
)
def test_operators_never_duplicate_atoms_with_zero_n_top(name, au2pt2_atoms, rng):
    """G5: ``n_top=0`` must not make ``atoms[-N:]`` return the whole structure."""
    mut = _make_zero_top_operator(name, rng)
    out = mut.mutate(au2pt2_atoms.copy())
    assert out is None or len(out) <= len(au2pt2_atoms)


def test_rotational_mutation_skips_whole_gas_phase_cluster(rng):
    """G6: rotating an untagged gas-phase cluster is a no-op, so bail out."""
    cluster = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [1.3, 2.25, 0.0],
            [1.3, 0.75, 2.12],
        ],
    )
    cluster.center(vacuum=8.0)
    mut = RotationalMutation(
        {(78, 78): 0.5},
        n_top=len(cluster),
        system_type="gas_cluster",
        rng=rng,
    )
    assert mut.mutate(cluster.copy()) is None


def test_rotational_mutation_rotates_cluster_on_slab(rng):
    """G6: with a slab present, rotating the whole mobile cluster is a real move."""
    slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z_top = float(np.max(slab.positions[:, 2]))
    center = np.mean(slab.positions[:, :2], axis=0)
    cluster = Atoms(
        "Pt4",
        positions=[
            [center[0], center[1], z_top + 3.0],
            [center[0] + 2.6, center[1], z_top + 3.0],
            [center[0] + 1.3, center[1] + 2.25, z_top + 3.0],
            [center[0] + 1.3, center[1] + 0.75, z_top + 5.1],
        ],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    full = slab + cluster

    mut = RotationalMutation(
        {(78, 78): 0.5},
        n_top=len(cluster),
        system_type="surface_cluster",
        rng=rng,
    )
    out = mut.mutate(full.copy())

    assert out is not None
    assert len(out) == len(full)
    assert np.allclose(
        out.get_positions()[:n_slab], full.get_positions()[:n_slab], atol=1e-10
    )
    assert (
        np.linalg.norm(out.get_positions()[n_slab:] - full.get_positions()[n_slab:])
        > 1e-6
    )


def test_rotational_mutation_still_rotates_tagged_gas_phase_core(rng):
    """G6: a tagged core inside a gas-phase cluster remains rotatable."""
    atoms = _tagged_core_adsorbate()
    mut = RotationalMutation(
        PERMISSIVE_BLMIN,
        n_top=len(atoms),
        target_tags=[0],
        system_type="gas_cluster_adsorbate",
        rng=rng,
    )
    out = mut.mutate(atoms.copy())

    assert out is not None
    displacement = out.get_positions() - atoms.get_positions()
    # More than a rigid translation of the whole structure.
    assert not np.allclose(displacement - displacement[0], 0.0, atol=1e-6)


def test_mirror_mutation_skips_whole_gas_phase_cluster(rng):
    """G6: mirroring an untagged gas-phase cluster is an isometry, so bail out."""
    cluster = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [1.3, 2.25, 0.0],
            [1.3, 0.75, 2.12],
        ],
    )
    cluster.center(vacuum=8.0)
    mut = MirrorMutation(
        {(78, 78): 0.5},
        n_top=len(cluster),
        system_type="gas_cluster",
        rng=rng,
    )
    assert mut.mutate(cluster.copy()) is None


def test_mirror_mutation_mirrors_cluster_on_slab(rng):
    """G6: with a slab present, mirroring the whole mobile cluster is a real move."""
    slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z_top = float(np.max(slab.positions[:, 2]))
    center = np.mean(slab.positions[:, :2], axis=0)
    cluster = Atoms(
        "Pt4",
        positions=[
            [center[0], center[1], z_top + 3.0],
            [center[0] + 2.6, center[1], z_top + 3.0],
            [center[0] + 1.3, center[1] + 2.25, z_top + 3.0],
            [center[0] + 1.3, center[1] + 0.75, z_top + 5.1],
        ],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    full = slab + cluster
    parent_low = float(np.min(cluster.positions[:, 2]))

    mut = MirrorMutation(
        {(78, 78): 0.5},
        n_top=len(cluster),
        system_type="surface_cluster",
        rng=rng,
    )
    out = mut.mutate(full.copy())

    assert out is not None
    assert len(out) == len(full)
    assert np.allclose(
        out.get_positions()[:n_slab], full.get_positions()[:n_slab], atol=1e-10
    )
    mobile = out.get_positions()[n_slab:]
    assert np.linalg.norm(mobile - full.get_positions()[n_slab:]) > 1e-6
    assert np.isclose(float(np.min(mobile[:, 2])), parent_low, atol=1e-8)


def test_mirror_mutation_still_mirrors_tagged_gas_phase_core(rng):
    """G6: a tagged core inside a gas-phase cluster remains mirrorable."""
    atoms = _tagged_core_adsorbate()
    mut = MirrorMutation(
        PERMISSIVE_BLMIN,
        n_top=len(atoms),
        target_tags=[0],
        system_type="gas_cluster_adsorbate",
        rng=rng,
    )
    out = mut.mutate(atoms.copy())

    assert out is not None
    displacement = out.get_positions() - atoms.get_positions()
    assert not np.allclose(displacement - displacement[0], 0.0, atol=1e-6)
    assert np.allclose(out.get_positions()[4:], atoms.get_positions()[4:], atol=1e-8)
