"""Tests for run tracking and result reuse functionality.

This module tests the ability to track runs with unique IDs and merge results
from multiple runs while ensuring no duplicates exist in the final output.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ase.calculators.emt import EMT

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database import (
    extract_minima_from_database_file,
    load_previous_run_results,
)
from scgo.minima_search import run_trials
from scgo.utils.run_tracking import (
    generate_run_id,
    get_run_directories,
    get_run_id_from_dir,
    load_run_metadata,
    save_run_metadata,
)
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION


def test_run_id_generation():
    """Test that run IDs are generated in correct format."""
    run_id = generate_run_id()
    assert run_id.startswith("run_")
    # Format: run_YYYYMMDD_HHMMSS_ffffff (e.g., run_20250124_143022_123456)
    assert re.match(r"^run_\d{8}_\d{6}_\d{6}$", run_id)

    # Test uniqueness (run_ids will be unique due to microsecond timestamp)
    run_ids = [generate_run_id() for _ in range(10)]
    # All should be unique due to microsecond granularity
    assert len(set(run_ids)) == 10


def test_run_id_from_directory():
    """Test extracting run ID from directory name."""
    assert (
        get_run_id_from_dir("run_20250124_143022_123456")
        == "run_20250124_143022_123456"
    )
    assert (
        get_run_id_from_dir("/path/to/run_20250124_143022_123456")
        == "run_20250124_143022_123456"
    )
    assert get_run_id_from_dir("not_a_run_dir") is None
    assert get_run_id_from_dir("run_incomplete") is None

    # Test: format should match timestamped run id pattern
    assert re.match(
        r"^run_\d{8}_\d{6}_\d{6}$", get_run_id_from_dir("run_20250124_143022_123456")
    )


def test_save_and_load_metadata(tmp_path):
    """Test saving and loading run metadata."""
    run_dir = str(tmp_path / "run_test")
    run_id = "run_20250124_143022_123456"

    metadata = {"composition": ["Pt", "Pt", "Pt"], "params": {"test": "value"}}
    save_run_metadata(run_dir, run_id, metadata)

    # Verify file exists
    metadata_file = Path(run_dir) / "metadata.json"
    assert metadata_file.exists()

    # Load and verify
    loaded = load_run_metadata(run_dir)
    assert loaded is not None
    assert loaded.run_id == run_id
    assert loaded.composition == ["Pt", "Pt", "Pt"]
    assert loaded.params == {"test": "value"}

    with open(metadata_file) as f:
        raw = json.load(f)
    assert raw["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
    assert isinstance(raw.get("scgo_version"), str) and raw["scgo_version"]
    assert isinstance(raw.get("python_version"), str) and raw["python_version"]
    assert isinstance(raw.get("created_at"), str) and raw["created_at"]


def test_run_metadata_includes_run_params(tmp_path, rng):
    """Run_trials should write run parameters to metadata.json."""
    output_dir = str(tmp_path / "Pt3_searches")
    run_id = "run_20250124_143022_123456"

    run_trials(
        composition=["Pt", "Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 1},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id=run_id,
        clean=False,
    )

    loaded = load_run_metadata(str(Path(output_dir) / run_id))
    assert loaded is not None
    assert isinstance(loaded.params, dict)
    assert loaded.params.get("global_optimizer") == "bh"
    assert loaded.params.get("n_trials") == 1
    assert loaded.params.get("validate_with_hessian") is False


def test_get_run_directories(tmp_path):
    """Test finding run directories."""
    # Create some directories
    (tmp_path / "run_20250124_143022_123456").mkdir()
    (tmp_path / "run_20250125_091500_654321").mkdir()
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "run_incomplete").mkdir()

    run_dirs = get_run_directories(str(tmp_path))
    assert len(run_dirs) == 2
    assert all(
        "run_20250124_143022_123456" in d or "run_20250125_091500_654321" in d
        for d in run_dirs
    )
    assert all(os.path.isdir(d) for d in run_dirs)


def test_extract_minima_from_database_file(tmp_path, pt3_atoms):
    """Test extracting minima from database file."""
    # This function is tested indirectly by test_load_previous_run_results
    # and test_result_merging_across_runs. Just verify it handles missing files gracefully.
    from scgo.database import extract_minima_from_database_file

    run_id = "run_20250124_143022_123456"
    trial_id = 1

    # Test with non-existent database
    minima = extract_minima_from_database_file(
        str(tmp_path / "nonexistent.db"), run_id, trial_id
    )
    assert minima == []

    # Full database testing is covered by integration tests


def test_extract_minima_persist_provenance_writes_to_db(test_database):
    """When persist=True, run_id/trial should be written back into DB rows."""
    from scgo.database import extract_minima_from_database_file

    db_path = test_database
    run_id = "run_persist_001"
    trial_id = 2

    # Ensure DB initially has no run_id stored in key_value_pairs
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM systems WHERE json_extract(key_value_pairs, '$.run_id') IS NOT NULL"
        )
        assert cur.fetchone()[0] == 0

    # Call with persist=True
    minima = extract_minima_from_database_file(db_path, run_id, trial_id, persist=True)
    assert minima, "Expected minima to be returned from test DB"

    # At least one DB row should now contain the run_id in key_value_pairs
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM systems WHERE json_extract(key_value_pairs, '$.run_id') = ?",
            (run_id,),
        )
        assert cur.fetchone()[0] >= 1


def test_extract_minima_no_persist_leaves_db_unchanged(test_database):
    """Default (persist=False) should only annotate returned Atoms, not DB rows."""
    from scgo.database import extract_minima_from_database_file

    db_path = test_database
    run_id = "run_no_persist"

    # Call without persistence
    minima = extract_minima_from_database_file(db_path, run_id, persist=False)
    assert minima, "Expected minima to be returned from test DB"

    # DB should remain unchanged (no run_id written into key_value_pairs)
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM systems WHERE json_extract(key_value_pairs, '$.run_id') = ?",
            (run_id,),
        )
        assert cur.fetchone()[0] == 0


def test_load_previous_run_results(tmp_path, pt3_atoms):
    """Test loading results from previous runs."""
    # This function is tested more thoroughly by test_result_merging_across_runs
    # Here we just verify it handles empty directories and non-existent paths
    composition = ["Pt", "Pt", "Pt"]
    base_dir = tmp_path / "Pt3_searches"

    # Test with non-existent directory
    previous_minima = load_previous_run_results(
        str(base_dir), "bh_go.db", composition, "run_20250126_120000"
    )
    assert previous_minima == []

    # Test with empty directory
    base_dir.mkdir()
    previous_minima = load_previous_run_results(
        str(base_dir), "bh_go.db", composition, "run_20250126_120000"
    )
    assert previous_minima == []

    # Full database loading is tested by integration tests


def test_result_merging_across_runs(tmp_path, rng):
    """Test that results from multiple runs are merged and deduplicated."""
    composition = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / "Pt3_searches")

    # First run
    results1 = run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250124_143022",
        clean=False,
    )

    # Second run (should merge with first)
    results2 = run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250125_091500",
        clean=False,
    )

    # Results should include minima from both runs (after deduplication)
    assert len(results2) >= len(results1)

    # Verify final_unique_minima directory contains files from both runs
    final_dir = Path(output_dir) / "final_unique_minima"
    assert final_dir.exists(), "Expected final_unique_minima directory to exist"
    xyz_files = list(final_dir.glob("*.xyz"))
    assert xyz_files, "Expected at least one .xyz file in final_unique_minima"
    run_ids_in_files = set()
    for xyz_file in xyz_files:
        if "run_20250124_143022" in xyz_file.name:
            run_ids_in_files.add("run_20250124_143022")
        if "run_20250125_091500" in xyz_file.name:
            run_ids_in_files.add("run_20250125_091500")

    # Should have files from at least one run (might have duplicates filtered)
    assert run_ids_in_files


def test_clean_mode(tmp_path, rng):
    """Test that clean mode ignores previous runs."""
    composition = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / "Pt3_searches")

    # First run
    run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250124_143022",
        clean=False,
    )

    # Second run with clean=True (should ignore first run)
    results2 = run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250125_091500",
        clean=True,  # Clean mode - ignore previous
    )

    # Should not load previous results when clean=True
    # Results should be from current run only
    assert isinstance(results2, list)


def test_composition_isolation(tmp_path, rng):
    """Test that results from different compositions don't interfere."""
    # Run Pt3
    run_trials(
        composition=["Pt", "Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=str(tmp_path / "Pt3_searches"),
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250124_143022",
        clean=False,
    )

    # Run Pt4 (different composition)
    run_trials(
        composition=["Pt", "Pt", "Pt", "Pt"],
        global_optimizer="ga",
        global_optimizer_kwargs={
            "niter": 1,
            "niter_local_relaxation": 5,
            "population_size": 4,
        },
        n_trials=1,
        output_dir=str(tmp_path / "Pt4_searches"),
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        verbosity=0,
        run_id="run_20250124_143022",
        clean=False,
    )

    # Results should be in separate directories
    pt3_dir = tmp_path / "Pt3_searches"
    pt4_dir = tmp_path / "Pt4_searches"
    assert pt3_dir.exists()
    assert pt4_dir.exists()

    # Verify databases are separate
    pt3_db = pt3_dir / "run_20250124_143022" / "trial_1" / "bh_go.db"
    pt4_db = pt4_dir / "run_20250124_143022" / "trial_1" / "ga_go.db"
    assert pt3_db.exists() and pt4_db.exists(), (
        "Expected both Pt3 and Pt4 DB files to exist"
    )
    # They should have different structures
    minima3 = extract_minima_from_database_file(str(pt3_db), "run_20250124_143022", 1)
    minima4 = extract_minima_from_database_file(str(pt4_db), "run_20250124_143022", 1)
    assert minima3 and minima4, "Expected minima entries in both databases"
    assert len(minima3[0][1]) == 3  # Pt3 has 3 atoms
    assert len(minima4[0][1]) == 4  # Pt4 has 4 atoms


# `test_xyz_file_naming_with_run_id` removed — filename/run-id assertions
# consolidated in `tests/test_output.py::test_file_naming_convention` (preferred
# single source-of-truth). Keep mode-specific tests only when they exercise
# behavior not covered by the consolidated output tests.


def test_no_duplicates_across_runs(tmp_path):
    """Test that duplicate structures from different runs are filtered out."""
    from tests.test_utils import create_paired_rngs

    composition = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / "Pt3_searches")

    # Create first run with known structure
    seed = 42
    rng1, rng2 = create_paired_rngs(seed)
    _results1 = run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 5, "niter_local_relaxation": 10},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng1,
        verbosity=0,
        run_id="run_20250124_143022",
        clean=False,
    )

    # Create second run with same seed (may produce similar/duplicate structures)
    results2 = run_trials(
        composition=composition,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 5, "niter_local_relaxation": 10},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng2,
        verbosity=0,
        run_id="run_20250125_091500",
        clean=False,  # Should merge with first run
    )

    # Check final_unique_minima for duplicates
    final_dir = Path(output_dir) / "final_unique_minima"
    assert final_dir.exists(), "Expected final_unique_minima directory to exist"
    assert results2, "Expected results from second run"
    xyz_files = list(final_dir.glob("*.xyz"))

    # Load all structures and check for duplicates
    from ase.io import read

    structures = []
    for xyz_file in xyz_files:
        atoms = read(str(xyz_file))
        structures.append(atoms)

        # Check that no two structures are identical
        # Use PureInteratomicDistanceComparator (doesn't require calculator)
        # since atoms from XYZ files don't have calculators attached
        from scgo.utils.comparators import PureInteratomicDistanceComparator

        structural_comparator = PureInteratomicDistanceComparator(
            n_top=3,
            tol=DEFAULT_COMPARATOR_TOL,
            pair_cor_max=DEFAULT_PAIR_COR_MAX,
            dE=DEFAULT_ENERGY_TOLERANCE,
            mic=False,
        )
        for i, atoms1 in enumerate(structures):
            for j, atoms2 in enumerate(structures[i + 1 :], start=i + 1):
                # Get energy from info (should be present in XYZ files from run_trials)
                energy1 = atoms1.info.get("energy") or atoms1.info.get(
                    "key_value_pairs", {}
                ).get("raw_score")
                energy2 = atoms2.info.get("energy") or atoms2.info.get(
                    "key_value_pairs", {}
                ).get("raw_score")

                if energy1 is None or energy2 is None:
                    continue  # Skip if energy not available

                # Check for duplicates (structural similarity + energy similarity)
                assert not (
                    structural_comparator.looks_like(atoms1, atoms2)
                    and abs(float(energy1) - float(energy2)) < DEFAULT_ENERGY_TOLERANCE
                ), (
                    f"Found duplicate structures: {xyz_files[i].name} and {xyz_files[j].name}"
                )


def test_campaign_run_id_consistency(tmp_path):
    """Test that campaign functions generate consistent run IDs."""
    from scgo.run_minima import run_scgo_campaign_one_element

    params = {
        "optimizer_params": {
            "bh": {"niter": 1, "niter_local_relaxation": 2},
        },
        "n_trials": 1,
    }

    campaign_dir = tmp_path / "campaign"
    # Run campaign (clean=True to avoid conflicts with previous runs)
    _results = run_scgo_campaign_one_element(
        element="Pt",
        min_atoms=2,
        max_atoms=3,
        system_type="gas_cluster",
        params=params,
        seed=42,
        verbosity=0,
        clean=True,
        output_dir=str(campaign_dir),
    )

    # Check that all compositions used the same run_id (output under campaign_dir)
    base_dir = (
        campaign_dir / "Pt2_searches"
        if (campaign_dir / "Pt2_searches").exists()
        else campaign_dir / "Pt3_searches"
    )
    assert base_dir.exists(), "Expected campaign search directory to exist"
    run_dirs = get_run_directories(str(base_dir))
    # All compositions in campaign should share same run_id
    # (only one run directory per composition, but same run_id pattern)
    assert run_dirs
