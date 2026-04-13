import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.standardmutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    InPlaneSlideMutation,
    OverlapReliefMutation,
    PermutationMutation,
    ShellSwapMutation,
)


def test_permutation_mutation_returns_none_when_single_species(pt3_atoms, rng):
    mut = PermutationMutation(n_top=3, probability=0.5, rng=rng)
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
        rng=rng,
    )
    out = mut.mutate(atoms.copy())

    assert out is not None
    assert mean_species_radius(out, "Au") > mean_species_radius(atoms, "Au")


def test_in_plane_slide_mutation_succeeds_on_slab_adsorbate():
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
        rng=np.random.default_rng(0),
        max_inner_attempts=8000,
    )
    out = mut.mutate(full)
    assert out is not None
    assert len(out) == len(full)
    assert out.get_atomic_numbers().tolist() == full.get_atomic_numbers().tolist()
    assert mut.last_attempt_count <= 12


def test_factory_registers_overlap_relief_and_shell_swap():
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
        rng=np.random.default_rng(7),
        use_adaptive=True,
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
        rng=rng,
    )
    mutated = mut.mutate(pt3_atoms.copy())
    assert mutated is not None, "Mutation must return a result"
    assert len(mutated) == len(pt3_atoms), "Atom count must be preserved"

    import numpy as np

    displacement = np.linalg.norm(mutated.get_positions() - pt3_atoms.get_positions())
    assert displacement > 1e-6, f"Mutation must displace atoms, got {displacement}"
