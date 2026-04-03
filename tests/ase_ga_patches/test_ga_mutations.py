from scgo.ase_ga_patches.standardmutations import (
    AnisotropicRattleMutation,
    PermutationMutation,
)


def test_permutation_mutation_returns_none_when_single_species(pt3_atoms, rng):
    mut = PermutationMutation(n_top=3, probability=0.5, rng=rng)
    assert mut.mutate(pt3_atoms.copy()) is None


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
    assert mutated is not None
    assert len(mutated) == len(pt3_atoms)
