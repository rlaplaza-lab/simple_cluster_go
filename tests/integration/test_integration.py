"""Integration tests for end-to-end SCGO workflows.

These tests verify complete optimization campaigns from initialization through
output generation, ensuring all components work together correctly.
"""

import os
import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read

from scgo.minima_search import run_trials
from scgo.param_presets import get_testing_params
from scgo.run_minima import (
    run_scgo_campaign_one_element,
    run_scgo_campaign_two_elements,
    run_scgo_trials,
)



@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "optimizer,opt_kwargs",
    [
        (
            "bh",
            {
                "niter": 2,
                "dr": 0.2,
                "niter_local_relaxation": 2,
                "temperature": 0.01,
            },
        ),
        (
            "ga",
            {
                "niter": 1,
                "population_size": 2,
                "niter_local_relaxation": 2,
                "mutation_probability": 0.3,
                "vacuum": 8.0,
                "n_jobs_population_init": 1,  # Serial for tests to avoid fork issues
            },
        ),
    ],
)
def test_full_optimizer_workflow(tmp_path, rng, optimizer, opt_kwargs):
    """Test complete optimization workflow from initialization to output files for any optimizer."""
    comp = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / f"{optimizer}_campaign")

    # Run a minimal campaign
    results = run_trials(
        composition=comp,
        global_optimizer=optimizer,
        global_optimizer_kwargs=opt_kwargs,
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify results structure
    assert isinstance(results, list)
    if results:
        for energy, atoms in results:
            assert np.isfinite(energy)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == comp

    assert os.path.exists(output_dir)

    from scgo.utils.run_tracking import get_run_directories
    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]

    trial1_dir = os.path.join(run_dir, "trial_1")
    assert os.path.exists(trial1_dir)
    assert not os.path.exists(os.path.join(run_dir, "trial_2"))
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    # Verify optimizer-specific files
    db_name = f"{optimizer}_go.db"
    trial1_db = os.path.join(trial1_dir, db_name)
    assert os.path.exists(trial1_db)

    # Verify database integrity
    import sqlite3
    for db_path in [trial1_db]:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            assert "systems" in tables
            cols = [r[1] for r in cursor.execute("PRAGMA table_info(systems)").fetchall()]
            
            # Check for run_id persistence
            cursor.execute(f"SELECT {'metadata, ' if 'metadata' in cols else ''}key_value_pairs FROM systems LIMIT 5")
            rows = cursor.fetchall()
            found_runid = False
            for row in rows:
                import json
                meta = json.loads(row[0]) if len(row) > 1 and row[0] else {}
                kv = json.loads(row[-1]) if row[-1] else {}
                if meta.get("run_id") or kv.get("run_id"):
                    found_runid = True
                    break
            assert found_runid is True, f"No run_id persisted in {optimizer} DB rows"

    # Verify GA-specific logs
    if optimizer == "ga":
        trial1_log = os.path.join(trial1_dir, "population.log")
        assert os.path.exists(trial1_log)

    # Verify XYZ files
    if results:
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        xyz_files = list(Path(xyz_dir).glob("*.xyz"))
        if xyz_files:
            atoms_from_file = read(str(xyz_files[0]))
            assert len(atoms_from_file) == 3
            assert "provenance" in atoms_from_file.info
            assert "trial" in atoms_from_file.info["provenance"]

def test_multi_trial_campaign(tmp_path, rng):
    """Test multi-trial campaign with unique minima filtering."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "multi_trial")

    # Run 2 BH trials (handled by run_trials now, not internally by bh_go)
    results = run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "dr": 0.3,
            "niter_local_relaxation": 2,
            "temperature": 0.01,
        },
        n_trials=2,  # run_trials handles multiple trials
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify results are sorted by energy (lowest first)
    if len(results) > 1:
        energies = [energy for energy, _ in results]
        assert energies == sorted(energies)

    # Verify that run_trials created separate directories for each trial
    # New structure: run_*/trial_*/
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]

    for i in range(1, 3):
        trial_dir = os.path.join(run_dir, f"trial_{i}")
        assert os.path.exists(trial_dir), (
            f"Expected trial directory {trial_dir} to exist"
        )
        # Check for database file in each trial directory
        db_file = os.path.join(trial_dir, "bh_go.db")
        assert os.path.exists(db_file), f"Expected database file {db_file} to exist"


@pytest.mark.slow
@pytest.mark.integration
def test_campaign_one_element(tmp_path):
    """Test single-element campaign workflow."""
    params = get_testing_params()

    # Run campaign for Pt2 and Pt3
    results = run_scgo_campaign_one_element(
        element="Pt",
        min_atoms=2,
        max_atoms=3,
        params=params,
        seed=456,
        output_dir=str(tmp_path / "campaign"),
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "Pt3" in results

    # Verify each composition has results
    for formula, minima_list in results.items():
        assert isinstance(minima_list, list)
        if minima_list:
            for energy, atoms in minima_list:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "min_atoms,max_atoms,niter,population_size",
    [
        (2, 4, 1, 5),  # Small clusters: Pt2-4 (replaces test_runner_pt2_pt4.py)
        (5, 6, 3, 10),  # Medium clusters: Pt5-6 (replaces test_runner_pt5_pt6.py)
    ],
)
def test_campaign_one_element_varying_cluster_sizes(
    tmp_path, min_atoms, max_atoms, niter, population_size
):
    """Test single-element campaigns with varying cluster sizes and parameters.

    This test replaces the standalone test_runner_pt2_pt4.py and test_runner_pt5_pt6.py
    runners, providing coverage for:
    - Small clusters (Pt2-4) with minimal iterations (quick validation)
    - Medium clusters (Pt5-6) with moderate iterations (broader validation)
    """
    params = get_testing_params()

    # Adjust GA parameters based on cluster size
    params["optimizer_params"]["ga"]["niter"] = niter
    params["optimizer_params"]["ga"]["population_size"] = population_size
    params["optimizer_params"]["ga"]["n_jobs_population_init"] = 1  # Sequential

    # Run campaign for the specified cluster size range
    results = run_scgo_campaign_one_element(
        element="Pt",
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        params=params,
        seed=456,
        output_dir=str(tmp_path / f"campaign_pt{min_atoms}_{max_atoms}"),
    )

    # Verify results structure
    assert isinstance(results, dict)

    # Check expected compositions are present
    expected_formulas = {f"Pt{i}" for i in range(min_atoms, max_atoms + 1)}
    actual_formulas = set(results.keys())
    assert expected_formulas == actual_formulas, (
        f"Expected formulas {expected_formulas}, got {actual_formulas}"
    )

    # Verify each composition has valid results
    total_structures = 0
    for formula, minima_list in results.items():
        assert isinstance(minima_list, list)
        if minima_list:
            total_structures += len(minima_list)
            for energy, atoms in minima_list:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula
                assert len(atoms) >= min_atoms
                assert len(atoms) <= max_atoms

    # At least one formula should have found some minima
    formulas_with_minima = sum(1 for m in results.values() if m)
    assert formulas_with_minima > 0, "No minima found across all compositions"


@pytest.mark.slow
@pytest.mark.integration
def test_campaign_two_elements(tmp_path):
    """Test bimetallic campaign workflow."""
    params = get_testing_params()

    # Run campaign for Au-Pt bimetallic clusters
    results = run_scgo_campaign_two_elements(
        element1="Au",
        element2="Pt",
        min_atoms=2,
        max_atoms=2,
        params=params,
        seed=789,
        output_dir=str(tmp_path / "campaign"),
    )

    # Verify all expected compositions are present
    expected_formulas = ["Au2", "AuPt", "Pt2"]
    for formula in expected_formulas:
        assert formula in results
        assert isinstance(results[formula], list)

        if results[formula]:
            for energy, atoms in results[formula]:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula


@pytest.mark.integration
def test_run_scgo_trials_integration(tmp_path):
    """Test the high-level run_scgo_trials function."""
    comp = ["Pt", "Pt"]
    params = get_testing_params()

    # Use clean=True and output_dir for isolation
    results = run_scgo_trials(
        comp,
        params=params,
        seed=999,
        clean=True,
        output_dir=str(tmp_path / "pt2_searches"),
    )

    # Verify results structure
    assert isinstance(results, list)
    if results:
        for energy, atoms in results:
            assert np.isfinite(energy)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 2
            assert atoms.get_chemical_symbols() == ["Pt", "Pt"]


def test_run_scgo_trials_deterministic_with_same_seed(tmp_path):
    """run_scgo_trials should be deterministic for a fixed seed."""
    comp = ["Pt", "Pt"]
    params = get_testing_params()
    out = str(tmp_path / "pt2_det")

    # Use clean=True and output_dir for isolation; same dir for both runs
    results1 = run_scgo_trials(
        comp, params=deepcopy(params), seed=1234, clean=True, output_dir=out
    )
    results2 = run_scgo_trials(
        comp, params=deepcopy(params), seed=1234, clean=True, output_dir=out
    )

    assert len(results1) == len(results2)
    for (e1, a1), (e2, a2) in zip(results1, results2, strict=False):
        assert np.isclose(e1, e2)
        assert np.allclose(a1.get_positions(), a2.get_positions())


@pytest.mark.integration
def test_output_directory_creation(tmp_path, rng):
    """Test that output directories are created correctly."""
    comp = ["Pt"]
    output_dir = str(tmp_path / "test_output")

    # Run minimal trial
    results = run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 1},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify directory structure (new structure: run_*/trial_*/)
    assert os.path.exists(output_dir)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    assert os.path.exists(os.path.join(run_dir, "trial_1"))
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    # Verify file naming convention
    if results:
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        xyz_files = list(Path(xyz_dir).glob("*.xyz"))
        assert len(xyz_files) > 0

        # Check naming convention: Pt1_minimum_01_run_{run_id}_trial_{trial_id}.xyz
        xyz_file = xyz_files[0]
        assert "minimum_" in xyz_file.name
        assert "run_" in xyz_file.name
        assert "trial_" in xyz_file.name
        assert xyz_file.name.endswith(".xyz")


@pytest.mark.integration
def test_bh_high_energy_strategy(tmp_path, rng):
    """Test Basin Hopping with high_energy fitness strategy."""
    from scgo.param_presets import get_high_energy_params

    composition = ["Pt", "Pt", "Pt"]
    params = get_high_energy_params()
    params["optimizer_params"]["bh"]["niter"] = 5
    params["optimizer_params"]["bh"]["niter_local_relaxation"] = 2

    results = run_scgo_trials(
        composition,
        params=params,
        seed=42,
        verbosity=1,
        output_dir=str(tmp_path / "bh_high_energy"),
    )

    # Should find some structures
    assert isinstance(results, list)

    # Check fitness values are stored (only in newly computed structures)
    # Note: Fitness is only stored during the current run, not loaded from old databases
    if results:
        # At least some results should have fitness information
        has_fitness = sum(1 for _, atoms in results if "fitness" in atoms.info)
        # We expect at least one structure to have fitness info (from current run)
        # But old structures from database won't have it
        if has_fitness > 0:
            for energy, atoms in results:
                if "fitness" in atoms.info:
                    assert "fitness_strategy" in atoms.info
                    assert atoms.info["fitness_strategy"] == "high_energy"
                    # For high_energy, fitness should equal energy
                    assert atoms.info["fitness"] == pytest.approx(energy)


@pytest.mark.slow
@pytest.mark.integration
def test_ga_diversity_strategy(tmp_path, rng):
    """Test Genetic Algorithm with diversity fitness strategy."""
    from scgo.param_presets import get_diversity_params

    ref_dir = tmp_path / "ref"
    div_dir = tmp_path / "div"

    # First, create some reference structures under tmp_path
    comp_ref = ["Pt", "Pt", "Pt"]
    params_ref = get_testing_params()
    params_ref["optimizer_params"]["ga"]["niter"] = 2
    params_ref["optimizer_params"]["ga"]["population_size"] = 5
    params_ref["optimizer_params"]["ga"][
        "n_jobs_population_init"
    ] = -2  # Parallel for tests

    ref_results = run_scgo_trials(
        comp_ref,
        params=params_ref,
        seed=42,
        clean=True,
        output_dir=str(ref_dir),
    )
    assert len(ref_results) > 0

    # Run with diversity strategy using reference DBs from ref_dir
    composition = ["Pt", "Pt", "Pt"]
    ref_glob = str(ref_dir / "**" / "*.db")
    params_div = get_diversity_params(reference_db_glob=ref_glob)
    params_div["optimizer_params"]["ga"]["niter"] = 2
    params_div["optimizer_params"]["ga"]["population_size"] = 5
    params_div["optimizer_params"]["ga"][
        "n_jobs_population_init"
    ] = -2  # Parallel for tests

    results = run_scgo_trials(
        composition,
        params=params_div,
        seed=43,
        verbosity=1,
        clean=False,
        output_dir=str(div_dir),
    )

    # Should find diverse structures
    assert isinstance(results, list)

    # Check fitness values (only in newly computed structures)
    # Note: Fitness is only stored during the current run, not loaded from old databases
    if results:
        # At least some results should have fitness information
        has_fitness = sum(1 for _, atoms in results if "fitness" in atoms.info)
        # We expect at least one structure to have fitness info (from current run)
        if has_fitness > 0:
            for _energy, atoms in results:
                if "fitness" in atoms.info:
                    assert atoms.info["fitness_strategy"] == "diversity"
                    # Diversity fitness should be non-negative
                    assert atoms.info["fitness"] >= 0.0


@pytest.mark.integration
def test_mixed_fitness_strategies(tmp_path, rng):
    """Test using different fitness strategies for BH and GA."""
    composition = ["Pt", "Pt", "Pt"]
    ref_dir = tmp_path / "ref"
    main_dir = tmp_path / "main"

    # First, create some reference structures under tmp_path
    params_ref = get_testing_params()
    params_ref["optimizer_params"]["bh"]["niter"] = 2
    params_ref["optimizer_params"]["bh"]["niter_local_relaxation"] = 2

    ref_results = run_scgo_trials(
        composition,
        params=params_ref,
        seed=41,
        clean=True,
        output_dir=str(ref_dir),
    )
    assert len(ref_results) > 0

    # Run with mixed strategies using the reference databases we just created
    params = get_testing_params()

    # BH uses diversity, GA uses low_energy
    params["optimizer_params"]["bh"]["fitness_strategy"] = "diversity"
    params["optimizer_params"]["bh"]["diversity_reference_db"] = str(
        ref_dir / "**" / "*.db"
    )
    params["optimizer_params"]["bh"]["niter"] = 3

    params["optimizer_params"]["ga"]["fitness_strategy"] = "low_energy"
    params["optimizer_params"]["ga"]["niter"] = 2

    # Should work without errors
    results = run_scgo_trials(
        composition,
        params=params,
        seed=42,
        verbosity=1,
        clean=False,
        output_dir=str(main_dir),
    )
    assert isinstance(results, list)


@pytest.mark.integration
def test_campaign_database_handle_management(tmp_path, rng):
    """Test that database handles are properly closed in multi-composition campaigns.

    This test verifies that:
    1. Multiple compositions can be processed without file descriptor leaks
    2. Database connections are properly closed after each composition
    3. No lingering locks prevent subsequent operations
    """
    import psutil

    # Get initial file descriptor count
    process = psutil.Process()
    initial_fds = process.num_fds()

    # Run a campaign with multiple compositions (Pt2, Pt3, Pt4)
    params = get_testing_params()
    params["optimizer_params"]["ga"]["niter"] = 1
    params["optimizer_params"]["ga"]["population_size"] = 2
    params["optimizer_params"]["bh"]["niter"] = 1

    results = run_scgo_campaign_one_element(
        element="Pt",
        min_atoms=2,
        max_atoms=4,
        params=params,
        seed=42,
        verbosity=0,
        clean=True,
        output_dir=str(tmp_path / "campaign_fd_test"),
    )

    # Force garbage collection
    import gc

    gc.collect()

    # Check file descriptors haven't grown excessively
    # Allow some growth for normal operations but not hundreds of open files
    final_fds = process.num_fds()
    fd_growth = final_fds - initial_fds

    # Torch/MACE model internals (shared libraries, CUDA driver handles, model
    # file caches) legitimately retain FDs across a campaign.  CI environments
    # are particularly heavy because the entire test suite has already loaded
    # many libraries by this point.  Use a relaxed threshold in CI and a tighter
    # one locally so we still catch catastrophic leaks (hundreds of handles).
    fd_limit = 200 if os.environ.get("CI") else 50
    assert fd_growth < fd_limit, (
        f"File descriptor leak detected: {initial_fds} -> {final_fds} ({fd_growth} leaked)"
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "Pt3" in results
    assert "Pt4" in results

    # Verify we can write files without "too many open files" error
    # This would fail if database handles were still open
    test_file = tmp_path / "test_write.txt"
    for i in range(100):
        with open(test_file, "w") as f:
            f.write(f"test {i}\n")

    assert test_file.exists()
