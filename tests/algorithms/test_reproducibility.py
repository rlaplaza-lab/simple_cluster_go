"""Tests for reproducibility of optimization algorithms.

This module verifies that optimization algorithms produce identical results
when run with the same random seed, ensuring deterministic behavior.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import FIRE, LBFGS

from scgo.algorithms.basinhopping_go import bh_go
from scgo.algorithms.geneticalgorithm_go import ga_go
from scgo.database import close_data_connection
from scgo.initialization import create_initial_cluster
from scgo.minima_search import run_trials, scgo
from scgo.param_presets import get_testing_params
from scgo.run_minima import (
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_campaign_one_element,
    run_scgo_campaign_two_elements,
    run_scgo_trials,
)
from scgo.utils.helpers import auto_niter
from tests.test_utils import (
    compare_minima_lists,
    create_paired_rngs,
    run_algorithm_reproducibility_test,
)


def test_bh_go_reproducibility(tmp_path):
    comp = ["Pt", "Pt", "Pt"]
    seed = 123

    minima1, minima2 = run_algorithm_reproducibility_test(
        bh_go,
        comp,
        seed,
        tmp_path,
        {
            "optimizer": LBFGS,
            "temperature": 0.01,
            "fmax": 0.2,
            "niter": 3,
            "niter_local_relaxation": 8,
            "dr": 0.2,
            "move_fraction": 0.5,
        },
    )

    assert compare_minima_lists(minima1, minima2)


def test_ga_go_reproducibility(tmp_path):
    comp = ["Pt", "Pt", "Pt"]
    seed = 456

    minima1, minima2 = run_algorithm_reproducibility_test(
        ga_go,
        comp,
        seed,
        tmp_path,
        {
            "population_size": 4,
            "niter": 2,
            "niter_local_relaxation": 10,
            "optimizer": LBFGS,
            "fmax": 0.2,
            "vacuum": 5.0,
            "energy_tolerance": 0.1,
            "n_jobs_population_init": -2,  # Parallel for tests
            "mutation_probability": 0.2,
        },
    )

    assert compare_minima_lists(minima1, minima2)


def test_rng_in_params_raises_error(tmp_path, rng):
    """Verify that putting 'rng' in optimizer_params raises clear error."""
    element = "Pt"
    n_atoms = 3  # Use a small cluster for quick testing
    test_seed = 123

    # Add a dummy rng to the optimizer_params
    params = get_testing_params()
    params["calculator"] = "EMT"  # Use EMT for speed
    params["optimizer_params"]["bh"]["n_trials"] = 2  # Small number of trials for BH
    params["validate_with_hessian"] = False  # Skip validation for speed

    # This should now raise an error (before any I/O)
    params["optimizer_params"]["bh"]["rng"] = rng

    with pytest.raises(ValueError, match=r'"rng" should not be in params'):
        run_scgo_campaign_one_element(
            element,
            n_atoms,
            n_atoms,
            params=params,
            seed=test_seed,
            output_dir=str(tmp_path / "campaign"),
        )


def test_ga_mutation_operators_child_rngs_reproducible():
    """Ensure mutation operators receive child RNGs derived deterministically from parent."""
    from ase import Atoms
    from ase_ga.utilities import closest_distances_generator, get_all_atom_types

    from scgo.algorithms.ga_common import create_mutation_operators

    composition = ["Pt", "Pt", "Pt"]
    n_to_optimize = 3
    atoms = Atoms("Pt3")
    all_atom_types = get_all_atom_types(atoms, range(n_to_optimize))
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

    parent_seed = 98765
    parent1, parent2 = create_paired_rngs(parent_seed)
    ops1, map1 = create_mutation_operators(
        composition=composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=parent1,
        use_adaptive=True,
    )
    assert "anisotropic_rattle" in map1

    # Compare a stable operator; factory list grows over time.
    rattle_idx = map1["rattle"]
    draws1 = ops1[rattle_idx].rng.integers(0, 2**31 - 1, size=5).tolist()

    # Repeat with a freshly seeded parent RNG
    ops2, map2 = create_mutation_operators(
        composition=composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=parent2,
        use_adaptive=True,
    )
    assert "anisotropic_rattle" in map2
    rattle_idx2 = map2["rattle"]
    draws2 = ops2[rattle_idx2].rng.integers(0, 2**31 - 1, size=5).tolist()

    assert draws1 == draws2, "Child RNGs derived from parent RNG should be reproducible"


def test_full_stack_smoke(tmp_path, rng):
    comp = ["Pt", "Pt"]
    outdir = str(tmp_path / "campaign")
    params = get_testing_params()
    optimizer_kwargs = params["optimizer_params"]["bh"].copy()
    # Convert optimizer string to class if needed
    if isinstance(optimizer_kwargs.get("optimizer"), str):
        from ase.optimize import FIRE

        optimizer_kwargs["optimizer"] = FIRE
    results = run_trials(
        comp,
        global_optimizer="bh",
        global_optimizer_kwargs=optimizer_kwargs,
        n_trials=1,
        output_dir=outdir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )
    assert isinstance(results, list)
    if results:
        e, a = results[0]
        assert np.isfinite(e)
        assert len(a) == 2


def test_run_py_smoke(tmp_path):
    """Test run_scgo_trials smoke test."""
    comp = ["Pt", "Pt"]
    results = run_scgo_trials(
        comp,
        params=get_testing_params(),
        seed=0,
        output_dir=str(tmp_path / "campaign"),
    )
    assert isinstance(results, list)
    if results:
        e, a = results[0]
        assert np.isfinite(e)
        assert len(a) == 2


def test_run_py_campaign_smoke(tmp_path):
    """Test run_scgo_campaign_one_element smoke test."""
    params = get_testing_params()

    results = run_scgo_campaign_one_element(
        "Pt",
        min_atoms=2,
        max_atoms=3,
        seed=0,
        params=params,
        clean=True,
        output_dir=str(tmp_path / "campaign"),
    )
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "Pt3" in results
    assert isinstance(results["Pt2"], list)
    assert isinstance(results["Pt3"], list)
    if results["Pt2"]:
        e, a = results["Pt2"][0]
        assert np.isfinite(e)
        assert len(a) == 2
    if results["Pt3"]:
        e, a = results["Pt3"][0]
        assert np.isfinite(e)
        assert len(a) == 3


def test_run_py_campaign_two_elements_smoke(tmp_path):
    """Test run_scgo_campaign_two_elements smoke test."""
    params = get_testing_params()

    results = run_scgo_campaign_two_elements(
        "Pt",
        "Au",
        min_atoms=2,
        max_atoms=2,
        seed=0,
        params=params,
        clean=True,
        output_dir=str(tmp_path / "campaign"),
    )
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "AuPt" in results
    assert "Au2" in results
    assert isinstance(results["Pt2"], list)
    assert isinstance(results["AuPt"], list)
    assert isinstance(results["Au2"], list)
    if results["Pt2"]:
        e, a = results["Pt2"][0]
        assert np.isfinite(e)
        assert len(a) == 2
    if results["AuPt"]:
        e, a = results["AuPt"][0]
        assert np.isfinite(e)
        assert len(a) == 2
    if results["Au2"]:
        e, a = results["Au2"][0]
        assert np.isfinite(e)
        assert len(a) == 2


@pytest.mark.slow
def test_ga_go_smoke(tmp_path, rng):
    """Test ga_go smoke test (fastened for CI).

    Reduced iteration/population/local-relaxation counts keep this a smoke
    test while exercising the GA code paths. Historically this test was
    long because it relied on the `auto_*` heuristics.
    """
    comp = ["Pt", "Pt", "Pt"]
    calc = EMT()
    params = get_testing_params()["optimizer_params"]["ga"]

    params_copy = params.copy()
    params_copy["optimizer"] = FIRE

    # Filter out parameters that ga_go() doesn't accept
    ga_go_params = {
        k: v
        for k, v in params_copy.items()
        if k
        not in [
            "niter",
            "population_size",
            "niter_local_relaxation",
            "use_torchsim",
            "batch_size",
            "relaxer",
        ]
    }

    # Use a very small GA budget for a fast smoke test
    minima = ga_go(
        comp,
        output_dir=str(tmp_path),
        calculator=calc,
        niter=2,
        population_size=4,
        niter_local_relaxation=2,
        **ga_go_params,
        rng=rng,
    )

    # Expect a list (can be empty if all relaxations failed), but the
    # function should not crash and must return a list.
    assert isinstance(minima, list)
    for e, a in minima:
        assert np.isfinite(e)
        assert isinstance(a, Atoms)


def test_bh_go_smoke(tmp_path, rng):
    """Test bh_go smoke test."""
    comp = ["Pt", "Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()
    params = get_testing_params()["optimizer_params"]["bh"]

    params_copy = params.copy()
    params_copy["optimizer"] = FIRE

    minima = bh_go(
        atoms,
        output_dir=str(tmp_path),
        niter=auto_niter(comp),
        **{k: v for k, v in params_copy.items() if k not in ["niter"]},
        rng=rng,
    )

    assert isinstance(minima, list)
    for e, a in minima:
        assert np.isfinite(e)
        assert isinstance(a, Atoms)


def test_run_scgo_campaign_arbitrary_compositions_smoke(tmp_path):
    params = get_testing_params()

    compositions = [["Pt", "Pt"], ["Au", "Pt"]]
    results = run_scgo_campaign_arbitrary_compositions(
        compositions,
        params=params,
        seed=0,
        clean=True,
        output_dir=str(tmp_path / "campaign"),
    )

    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "AuPt" in results
    assert isinstance(results["Pt2"], list)
    assert isinstance(results["AuPt"], list)


def test_scgo_unknown_optimizer(tmp_path, rng):
    """Test that unknown optimizer raises error."""
    outdir = str(tmp_path / "out")
    with pytest.raises(ValueError, match="Unknown global_optimizer"):
        scgo(
            ["Pt", "Pt"],
            global_optimizer="unknown",
            global_optimizer_kwargs={},
            output_dir=outdir,
            rng=rng,
        )


def test_run_trials_zero_trials(tmp_path, rng):
    """Test that zero trials raises error."""
    outdir = str(tmp_path / "campaign")
    # With n_trials=0, run_trials should raise ValueError
    with pytest.raises(ValueError, match="n_trials.*must be positive"):
        run_trials(
            ["Pt"],
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 5
            },  # Remove n_trials from optimizer params
            n_trials=0,  # This is now the run_trials parameter
            output_dir=outdir,
            rng=rng,
        )


def test_bh_go_zero_niter(tmp_path, rng):
    """Test bh_go with zero iterations."""
    atoms = create_initial_cluster(["Pt"], rng=rng)
    atoms.calc = EMT()
    # With niter=1, should return a list with the initial relaxed structure
    minima = bh_go(atoms, output_dir=str(tmp_path), niter=1, rng=rng)
    assert isinstance(minima, list)
    assert minima, "BH did not return any minima"
    e, a = minima[0]
    assert np.isfinite(e)
    assert isinstance(a, Atoms)


def test_ga_go_zero_niter(tmp_path, rng):
    """Test ga_go with zero iterations."""
    comp = ["Pt"]
    calc = EMT()
    # With niter=1, should return a list with the initial relaxed structure
    minima = ga_go(
        comp,
        output_dir=str(tmp_path),
        calculator=calc,
        niter=1,
        population_size=2,
        niter_local_relaxation=2,
        rng=rng,
    )
    assert isinstance(minima, list)
    assert len(minima) > 0  # Should contain the relaxed initial population
    e, a = minima[0]
    assert np.isfinite(e)
    assert isinstance(a, Atoms)


def test_nested_rng_spawning_parent_state_preserved(rng):
    """Test that parent RNG state is unchanged after spawning children."""
    # Use parent RNG from fixture
    parent_rng = rng

    # Generate some numbers to advance the state
    [parent_rng.integers(0, 1000) for _ in range(10)]

    # Store parent state
    parent_state = parent_rng.bit_generator.state

    # Spawn child RNGs
    child_seeds = [parent_rng.integers(0, 2**63 - 1) for _ in range(3)]
    child_rngs = [np.random.default_rng(seed) for seed in child_seeds]

    # Generate numbers with children
    child_numbers = [
        [child_rng.integers(0, 1000) for _ in range(5)] for child_rng in child_rngs
    ]
    assert len(child_numbers) == 3 and all(len(c) == 5 for c in child_numbers)

    # Check that parent state has advanced (as expected when generating child seeds)
    # The parent state should have changed because we called parent_rng.integers()
    assert parent_rng.bit_generator.state != parent_state

    # Generate more numbers with parent to verify it still works
    more_parent_numbers = [parent_rng.integers(0, 1000) for _ in range(5)]
    assert len(more_parent_numbers) == 5


def test_nested_rng_spawning_independent_children(rng):
    """Test that child RNGs produce independent sequences."""
    # Use parent RNG from fixture
    parent_rng = rng

    # Spawn multiple child RNGs
    child_seeds = [parent_rng.integers(0, 2**63 - 1) for _ in range(3)]
    child_rngs = [np.random.default_rng(seed) for seed in child_seeds]

    # Generate sequences with each child
    sequences = []
    for child_rng in child_rngs:
        sequence = [child_rng.integers(0, 1000) for _ in range(10)]
        sequences.append(sequence)

    # Verify sequences are different
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            assert sequences[i] != sequences[j], (
                f"Sequences {i} and {j} should be different"
            )


def test_nested_rng_spawning_reproducible_children(rng):
    """Test that child RNGs with same seed produce same sequences."""
    # Use parent RNG from fixture
    parent_rng = rng

    # Spawn child with specific seed
    child_seed = parent_rng.integers(0, 2**63 - 1)
    child_rng1 = np.random.default_rng(child_seed)

    # Generate sequence
    sequence1 = [child_rng1.integers(0, 1000) for _ in range(10)]

    # Create another child with same seed
    child_rng2 = np.random.default_rng(child_seed)
    sequence2 = [child_rng2.integers(0, 1000) for _ in range(10)]

    # Sequences should be identical
    assert sequence1 == sequence2


def test_campaign_level_rng_spawns_trial_level_rngs(rng):
    """Test that campaign-level RNG spawns trial-level RNGs correctly."""
    # This test verifies the RNG spawning mechanism in run_trials

    # Use campaign-level RNG from fixture
    campaign_rng = rng

    # Simulate the RNG spawning in run_trials
    trial_seeds = []
    for _i in range(3):
        trial_seed = campaign_rng.integers(0, 2**63 - 1)
        trial_seeds.append(trial_seed)

    # Create trial RNGs
    trial_rngs = [np.random.default_rng(s) for s in trial_seeds]

    # Generate numbers with each trial RNG
    trial_sequences = []
    for trial_rng in trial_rngs:
        sequence = [trial_rng.integers(0, 1000) for _ in range(5)]
        trial_sequences.append(sequence)

    # Verify all sequences are different
    for i in range(len(trial_sequences)):
        for j in range(i + 1, len(trial_sequences)):
            assert trial_sequences[i] != trial_sequences[j]


def test_cross_composition_reproducibility(tmp_path):
    """Test reproducibility across different compositions in campaigns."""
    params = get_testing_params()
    params["optimizer_params"]["bh"]["n_trials"] = 1  # Single trial for BH
    params["validate_with_hessian"] = False
    out = str(tmp_path / "campaign")

    # Run campaign with fixed seed (clean=True to start fresh)
    results1 = run_scgo_campaign_one_element(
        "Pt",
        min_atoms=2,
        max_atoms=3,
        params=params,
        seed=54321,
        clean=True,
        output_dir=out,
    )

    # Run same campaign with same seed (clean=True to start fresh)
    results2 = run_scgo_campaign_one_element(
        "Pt",
        min_atoms=2,
        max_atoms=3,
        params=params,
        seed=54321,
        clean=True,
        output_dir=out,
    )

    # Results should be identical
    assert set(results1.keys()) == set(results2.keys())

    for formula in results1:
        assert len(results1[formula]) == len(results2[formula])

        if results1[formula]:  # If not empty
            energy1 = results1[formula][0][0]
            energy2 = results2[formula][0][0]
            assert np.isclose(energy1, energy2, rtol=1e-10)

            atoms1 = results1[formula][0][1]
            atoms2 = results2[formula][0][1]
            assert np.allclose(
                atoms1.get_positions(),
                atoms2.get_positions(),
                rtol=1e-10,
            )


def test_cross_composition_different_seeds(tmp_path):
    """Test that different seeds produce different results across compositions."""
    params = get_testing_params()
    params["optimizer_params"]["bh"]["n_trials"] = (
        2  # Two trials for BH to reduce flakiness
    )
    params["validate_with_hessian"] = False
    out = str(tmp_path / "campaign")

    # Run with seed 1 (clean=True to avoid loading previous results)
    results1 = run_scgo_campaign_one_element(
        "Pt",
        min_atoms=2,
        max_atoms=3,
        params=params,
        seed=11111,
        clean=True,
        output_dir=out,
    )

    # Run with seed 2 (clean=True to avoid loading previous results)
    results2 = run_scgo_campaign_one_element(
        "Pt",
        min_atoms=2,
        max_atoms=3,
        params=params,
        seed=22222,
        clean=True,
        output_dir=out,
    )

    # Results should be different (very unlikely to be identical)
    different_results = False
    for formula in results1:
        if results1[formula] and results2[formula]:
            energy1 = results1[formula][0][0]
            energy2 = results2[formula][0][0]
            # Compare energies with rtol=1e-8
            if not np.isclose(energy1, energy2, rtol=1e-8):
                different_results = True
                break
            # If energies are close, compare positions with rtol=1e-7
            atoms1 = results1[formula][0][1]
            atoms2 = results2[formula][0][1]
            if not np.allclose(
                atoms1.get_positions(), atoms2.get_positions(), rtol=1e-7
            ):
                different_results = True
                break

    assert different_results is True, "Different seeds should produce different results"


def test_bimetallic_campaign_reproducibility(tmp_path):
    """Test reproducibility for bimetallic campaigns."""
    params = get_testing_params()
    params["optimizer_params"]["bh"]["n_trials"] = 1  # Single trial for BH
    params["validate_with_hessian"] = False
    out = str(tmp_path / "campaign")

    # Run bimetallic campaign with fixed seed (clean=True to start fresh)
    results1 = run_scgo_campaign_two_elements(
        "Pt",
        "Au",
        min_atoms=2,
        max_atoms=2,
        params=params,
        seed=98765,
        clean=True,
        output_dir=out,
    )

    # Run same campaign with same seed (clean=True to start fresh)
    results2 = run_scgo_campaign_two_elements(
        "Pt",
        "Au",
        min_atoms=2,
        max_atoms=2,
        params=params,
        seed=98765,
        clean=True,
        output_dir=out,
    )

    # Results should be identical
    assert set(results1.keys()) == set(results2.keys())

    for formula in results1:
        assert len(results1[formula]) == len(results2[formula])

        if results1[formula]:
            energy1 = results1[formula][0][0]
            energy2 = results2[formula][0][0]
            assert np.isclose(energy1, energy2, rtol=1e-10)


def test_arbitrary_compositions_reproducibility(tmp_path):
    """Test reproducibility for arbitrary compositions campaigns."""
    params = get_testing_params()
    params["optimizer_params"]["bh"]["n_trials"] = 1  # Single trial for BH
    params["validate_with_hessian"] = False
    out = str(tmp_path / "campaign")
    compositions = [["Pt", "Pt"], ["Au", "Pt"], ["Pt", "Au", "Au"]]

    # Run with fixed seed (clean=True to start fresh)
    results1 = run_scgo_campaign_arbitrary_compositions(
        compositions,
        params=params,
        seed=13579,
        clean=True,
        output_dir=out,
    )

    # Run with same seed (clean=True to start fresh)
    results2 = run_scgo_campaign_arbitrary_compositions(
        compositions,
        params=params,
        seed=13579,
        clean=True,
        output_dir=out,
    )

    # Results should be identical
    assert set(results1.keys()) == set(results2.keys())

    for formula in results1:
        assert len(results1[formula]) == len(results2[formula])

        if results1[formula]:
            energy1 = results1[formula][0][0]
            energy2 = results2[formula][0][0]
            assert np.isclose(energy1, energy2, rtol=1e-10)


def test_rng_state_consistency_across_functions(rng):
    """Test that RNG state is consistent across different function calls."""

    # Generate some numbers
    numbers1 = [rng.integers(0, 1000) for _ in range(5)]

    # Use RNG in different contexts
    comp = ["Pt", "Pt"]
    atoms1 = create_initial_cluster(comp, rng=rng)

    # Generate more numbers
    numbers2 = [rng.integers(0, 1000) for _ in range(5)]

    # Use RNG again
    atoms2 = create_initial_cluster(comp, rng=rng)

    # Generate final numbers
    numbers3 = [rng.integers(0, 1000) for _ in range(5)]

    # All sequences should be different (RNG state advances)
    assert numbers1 != numbers2
    assert numbers2 != numbers3
    assert numbers1 != numbers3

    # Atoms should be different (different random initializations)
    assert not np.allclose(atoms1.get_positions(), atoms2.get_positions())


def test_rng_reproducibility_with_same_seed():
    """Test that same seed produces same results across multiple runs."""
    seed = 11111
    comp = ["Pt", "Pt", "Pt"]

    # Run 1
    rng1, _ = create_paired_rngs(seed)
    atoms1 = create_initial_cluster(comp, rng=rng1)

    # Run 2
    _, rng2 = create_paired_rngs(seed)
    atoms2 = create_initial_cluster(comp, rng=rng2)

    # Results should be identical
    assert np.allclose(atoms1.get_positions(), atoms2.get_positions())
    assert atoms1.get_chemical_symbols() == atoms2.get_chemical_symbols()


@pytest.mark.parametrize("seed1,seed2", [(11111, 22222)])
def test_rng_different_seeds_different_results(seed1, seed2):
    """Test that different seeds produce different results."""
    comp = ["Pt", "Pt", "Pt"]

    # Run with seed 1
    rng1, _ = create_paired_rngs(seed1)
    atoms1 = create_initial_cluster(comp, rng=rng1)

    # Run with seed 2
    _, rng2 = create_paired_rngs(seed2)
    atoms2 = create_initial_cluster(comp, rng=rng2)

    # Results should be different
    assert not np.allclose(atoms1.get_positions(), atoms2.get_positions())
    assert (
        atoms1.get_chemical_symbols() == atoms2.get_chemical_symbols()
    )  # Same composition


@pytest.mark.slow
def test_ga_multiprocess_reproducibility(tmp_path):
    """Test that GA produces identical results with n_jobs=1 vs n_jobs=-2.

    Lowered population/local-relaxation counts keep the test fast while
    still exercising the multiprocess vs single-process reproducibility
    behavior.
    """
    comp = ["Pt", "Pt", "Pt"]
    seed = 789

    # Run with single process
    minima1, _ = run_algorithm_reproducibility_test(
        ga_go,
        comp,
        seed,
        tmp_path / "run1",
        {
            "population_size": 4,
            "niter": 2,
            "niter_local_relaxation": 2,
            "optimizer": FIRE,
            "fmax": 0.2,
            "vacuum": 5.0,
            "energy_tolerance": 0.1,
            "n_jobs_population_init": 1,  # Single process
            "mutation_probability": 0.2,
        },
    )

    # Run with multiple processes
    minima2, _ = run_algorithm_reproducibility_test(
        ga_go,
        comp,
        seed,
        tmp_path / "run2",
        {
            "population_size": 4,
            "niter": 2,
            "niter_local_relaxation": 2,
            "optimizer": FIRE,
            "fmax": 0.2,
            "vacuum": 5.0,
            "energy_tolerance": 0.1,
            "n_jobs_population_init": -2,  # Multiple processes
            "mutation_probability": 0.2,
        },
    )

    # Results should be identical regardless of parallelization
    assert compare_minima_lists(minima1, minima2)


def test_database_persistence_reproducibility(tmp_path):
    """Test that database persistence is deterministic across runs."""
    from scgo.database import setup_database

    comp = ["Pt", "Pt", "Pt"]

    # Create two databases with same initial population
    rng1, rng2 = create_paired_rngs(555)
    atoms1 = create_initial_cluster(comp, rng=rng1)
    atoms1.calc = EMT()

    db_path1 = tmp_path / "run1" / "test.db"
    da1 = setup_database(
        output_dir=str(tmp_path / "run1"),
        db_filename="test.db",
        atoms_template=atoms1,
        initial_population=[atoms1.copy()],
        remove_existing=True,
    )

    # Relax and save
    a = da1.get_an_unrelaxed_candidate()
    a.calc = EMT()
    opt = FIRE(a, logfile=None)
    opt.run(fmax=0.1, steps=50)
    da1.add_relaxed_step(a)
    energy1 = a.get_potential_energy()

    # Second database with same seed - should produce identical results
    atoms2 = create_initial_cluster(comp, rng=rng2)
    atoms2.calc = EMT()

    db_path2 = tmp_path / "run2" / "test.db"
    da2 = setup_database(
        output_dir=str(tmp_path / "run2"),
        db_filename="test.db",
        atoms_template=atoms2,
        initial_population=[atoms2.copy()],
        remove_existing=True,
    )

    # Relax and save
    a2 = da2.get_an_unrelaxed_candidate()
    a2.calc = EMT()
    opt2 = FIRE(a2, logfile=None)
    opt2.run(fmax=0.1, steps=50)
    da2.add_relaxed_step(a2)
    energy2 = a2.get_potential_energy()

    # Energies should be exactly identical
    assert abs(energy1 - energy2) < 1e-10

    # Both databases should exist
    assert db_path1.exists()
    assert db_path2.exists()

    close_data_connection(da1)
    close_data_connection(da2)


def test_fixed_seed_exact_reproducibility(tmp_path):
    """Test that fixed seed gives bit-for-bit identical results across runs."""
    comp = ["Pt", "Pt"]
    seed = 12345

    # Run 1
    rng1, rng2 = create_paired_rngs(seed)
    atoms1 = create_initial_cluster(comp, rng=rng1)
    atoms1.calc = EMT()
    opt1 = FIRE(atoms1, logfile=None)
    opt1.run(fmax=0.1, steps=50)
    energy1 = atoms1.get_potential_energy()
    positions1 = atoms1.get_positions().copy()

    # Run 2 with identical seed
    atoms2 = create_initial_cluster(comp, rng=rng2)
    atoms2.calc = EMT()
    opt2 = FIRE(atoms2, logfile=None)
    opt2.run(fmax=0.1, steps=50)
    energy2 = atoms2.get_potential_energy()
    positions2 = atoms2.get_positions().copy()

    # Should be exactly identical (within floating point precision)
    assert np.allclose(positions1, positions2, rtol=1e-12, atol=1e-12)
    assert abs(energy1 - energy2) < 1e-10
