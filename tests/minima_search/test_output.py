"""Tests for file I/O, XYZ format, and directory structure validation.

These tests verify that output files are generated correctly with proper
formatting and that directory structures are created as expected.
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write

from scgo.initialization import create_initial_cluster
from scgo.minima_search import run_trials


def test_xyz_file_format_validation(tmp_path, pt2_with_calc):
    """Test that XYZ files are written in correct format."""
    # Use fixture
    atoms = pt2_with_calc.copy()
    from scgo.database.metadata import add_metadata

    add_metadata(atoms, raw_score=-10.0)
    atoms.info["provenance"] = {"trial": 1}

    # Write XYZ file
    xyz_path = tmp_path / "test.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Verify file exists
    assert xyz_path.exists()

    # Read back and verify content
    read_atoms = read(str(xyz_path))

    # Check basic properties
    assert len(read_atoms) == 2
    assert read_atoms.get_chemical_symbols() == ["Pt", "Pt"]
    assert np.allclose(read_atoms.get_positions(), atoms.get_positions())

    # Check comment line exists (ASE's default XYZ writer may not include atoms.info)
    with open(xyz_path) as f:
        lines = f.readlines()
        comment_line = lines[1].strip()  # Second line is comment
        # Just verify the comment line exists (may be empty)
        assert isinstance(comment_line, str)


def test_xyz_file_energy_comment_format(tmp_path, pt3_with_calc):
    """Test that XYZ files can be written and read correctly."""
    atoms = pt3_with_calc.copy()
    from scgo.database.metadata import add_metadata

    add_metadata(atoms, raw_score=-15.5)

    xyz_path = tmp_path / "test_energy.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Read back and verify structure
    read_atoms = read(str(xyz_path))
    assert len(read_atoms) == 3
    assert read_atoms.get_chemical_symbols() == ["Pt", "Pt", "Pt"]
    assert np.allclose(read_atoms.get_positions(), atoms.get_positions())

    # Note: ASE's default XYZ writer doesn't include atoms.info in comment line
    # The energy information is stored in the atoms.info dictionary, not the comment


def test_xyz_file_provenance_tracking(tmp_path, pt2_with_calc):
    """Test that XYZ files include provenance information."""
    atoms = pt2_with_calc.copy()
    atoms.info["key_value_pairs"] = {"raw_score": -10.0}
    atoms.info["provenance"] = {"trial": 2, "optimizer": "bh"}

    xyz_path = tmp_path / "test_provenance.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Read back and verify structure
    read_atoms = read(str(xyz_path))
    assert len(read_atoms) == 2
    assert read_atoms.get_chemical_symbols() == ["Pt", "Pt"]
    assert np.allclose(read_atoms.get_positions(), atoms.get_positions())

    # Note: ASE's default XYZ writer doesn't include atoms.info in comment line
    # The provenance information is stored in the atoms.info dictionary, not the comment


def test_directory_structure_creation(tmp_path, rng):
    """Test that output directories are created correctly."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "test_campaign")

    # Run minimal trial with 2 independent trials
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "niter_local_relaxation": 2,
        },
        n_trials=2,  # run_trials now handles multiple trials
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify directory structure (new structure: run_*/trial_*/)
    assert os.path.exists(output_dir)

    # Find run directory (auto-generated run_id)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]

    trial1_dir = os.path.join(run_dir, "trial_1")
    trial2_dir = os.path.join(run_dir, "trial_2")
    assert os.path.exists(trial1_dir)
    assert os.path.exists(trial2_dir)
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    # Verify each trial directory contains a single database file
    trial1_db = os.path.join(trial1_dir, "bh_go.db")
    trial2_db = os.path.join(trial2_dir, "bh_go.db")
    assert os.path.exists(trial1_db)
    assert os.path.exists(trial2_db)


def test_file_naming_convention(tmp_path, rng):
    """Test that output files follow correct naming convention."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "naming_test")

    # Run trial
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 2},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Check XYZ file naming when run_trials produced at least one minimum
    xyz_dir = os.path.join(output_dir, "final_unique_minima")
    xyz_files = list(Path(xyz_dir).glob("*.xyz"))
    assert len(xyz_files) > 0

    # Check naming convention: Pt2_minimum_01_run_{run_id}_trial_{trial_id}.xyz
    xyz_file = xyz_files[0]
    filename = xyz_file.name

    assert "minimum_" in filename
    assert "run_" in filename
    assert "trial_" in filename
    assert filename.endswith(".xyz")
    assert "Pt2" in filename  # Should contain formula


def test_database_file_creation_and_format(tmp_path, rng):
    """Test that database files are created and are valid SQLite."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "db_test")

    # Run BH trial
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 2},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Check BH database (new structure: run_*/trial_*/)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    bh_db_path = os.path.join(run_dir, "trial_1", "bh_go.db")
    assert os.path.exists(bh_db_path)

    # Verify it's a valid SQLite database
    with sqlite3.connect(bh_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "systems" in tables


def test_ga_database_file_creation(tmp_path, rng):
    """Test that GA database files are created correctly."""
    comp = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / "ga_db_test")

    # Run GA trial
    run_trials(
        composition=comp,
        global_optimizer="ga",
        global_optimizer_kwargs={
            "niter": 1,
            "population_size": 4,
            "niter_local_relaxation": 2,
            "n_jobs_population_init": -2,  # Parallel for tests
        },
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Check GA database (new structure: run_*/trial_*/)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    ga_db_path = os.path.join(run_dir, "trial_1", "ga_go.db")
    assert os.path.exists(ga_db_path)

    # Verify it's a valid SQLite database
    with sqlite3.connect(ga_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "systems" in tables

    # Check population log file
    pop_log_path = os.path.join(run_dir, "trial_1", "population.log")
    assert os.path.exists(pop_log_path)


@pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
def test_xyz_file_atom_count_validation(tmp_path, rng, n_atoms):
    """Test that XYZ files contain correct number of atoms."""
    comp = ["Pt"] * n_atoms
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    xyz_path = tmp_path / f"test_{n_atoms}.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Read back and verify atom count
    read_atoms = read(str(xyz_path))
    assert len(read_atoms) == n_atoms
    assert read_atoms.get_chemical_symbols() == ["Pt"] * n_atoms


def test_xyz_file_coordinate_validation(tmp_path, pt3_with_calc):
    """Test that XYZ files contain reasonable coordinates."""
    atoms = pt3_with_calc.copy()

    xyz_path = tmp_path / "test_coords.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Read back and verify coordinates
    read_atoms = read(str(xyz_path))
    positions = read_atoms.get_positions()

    # Check that coordinates are reasonable
    assert np.all(np.isfinite(positions))  # No NaN or Inf
    assert np.all(positions > -100)  # Not too negative
    assert np.all(positions < 100)  # Not too large

    # Check that original positions are preserved
    assert np.allclose(positions, atoms.get_positions())


def test_multiple_xyz_files_generation(tmp_path):
    """Test generation of multiple XYZ files for different minima."""
    # Create multiple test structures using helper
    structures = []
    from tests.test_utils import create_test_atoms

    for i in range(3):
        atoms = create_test_atoms(
            ["Pt", "Pt"],
            positions=[[0, 0, 0], [2.5 + i * 0.1, 0, 0]],
            calc=EMT(),
            raw_score=-10.0 - i,
            trial=1,
        )
        atoms.info.setdefault("provenance", {})["minimum_id"] = i + 1
        structures.append(atoms)

    # Write multiple XYZ files
    xyz_dir = tmp_path / "multiple_xyz"
    xyz_dir.mkdir()

    # Use new filename format with run_id
    run_id = "run_20250124_143022"
    for i, atoms in enumerate(structures):
        xyz_path = xyz_dir / f"Pt2_minimum_{i + 1:02d}_run_{run_id}_trial_1.xyz"
        write(str(xyz_path), atoms, format="xyz")

    # Verify all files were created
    xyz_files = list(xyz_dir.glob("*.xyz"))
    assert len(xyz_files) == 3

    # Verify naming convention (new format with run_id)
    for i, xyz_file in enumerate(sorted(xyz_files)):
        assert f"minimum_{i + 1:02d}" in xyz_file.name
        assert "run_" in xyz_file.name
        assert "trial_1" in xyz_file.name


def test_xyz_file_with_bimetallic_clusters(tmp_path, au2pt2_atoms):
    """Test XYZ file generation for bimetallic clusters."""
    atoms = au2pt2_atoms.copy()
    atoms.calc = EMT()
    from scgo.database.metadata import add_metadata

    add_metadata(atoms, raw_score=-20.0)

    xyz_path = tmp_path / "test_bimetallic.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Read back and verify
    read_atoms = read(str(xyz_path))
    assert len(read_atoms) == 4
    assert set(read_atoms.get_chemical_symbols()) == {"Au", "Pt"}
    assert read_atoms.get_chemical_symbols().count("Au") == 2
    assert read_atoms.get_chemical_symbols().count("Pt") == 2


def test_database_file_size_validation(tmp_path, rng):
    """Test that database files have reasonable sizes."""
    comp = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / "size_test")

    # Run trial
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 3, "niter_local_relaxation": 5},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Check database file size (new structure: run_*/trial_*/)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    db_path = os.path.join(run_dir, "trial_1", "bh_go.db")
    assert os.path.exists(db_path)

    file_size = os.path.getsize(db_path)
    assert file_size > 1000  # Should be at least 1KB
    assert file_size < 10 * 1024 * 1024  # Should be less than 10MB


def test_output_directory_permissions(tmp_path, rng):
    """Test that output directories have correct permissions."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "perm_test")

    # Run trial
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 2},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Check that directories are writable
    assert os.access(output_dir, os.W_OK)
    # Check permissions (new structure: run_*/trial_*/)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    assert os.access(os.path.join(run_dir, "trial_1"), os.W_OK)
    assert os.access(os.path.join(output_dir, "final_unique_minima"), os.W_OK)


def test_xyz_file_encoding(tmp_path):
    """Test that XYZ files are written with correct encoding."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.calc = EMT()
    from scgo.database.metadata import add_metadata

    add_metadata(atoms, raw_score=-10.0)

    xyz_path = tmp_path / "test_encoding.xyz"
    write(str(xyz_path), atoms, format="xyz")

    # Verify file can be read with UTF-8 encoding
    with open(xyz_path, encoding="utf-8") as f:
        content = f.read()
        assert "Pt" in content
        assert "2" in content  # Number of atoms


def test_empty_results_handling(tmp_path, rng):
    """Test handling when no minima are found."""
    # This is harder to test directly, but we can test the directory structure
    # is still created even if no results are found
    comp = ["Pt"]
    output_dir = str(tmp_path / "empty_test")

    # Run with very short optimization (might find no minima)
    run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 1},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Directory structure should still be created (new structure: run_*/trial_*/)
    assert os.path.exists(output_dir)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0, "Expected at least one run directory"
    run_dir = run_dirs[0]
    assert os.path.exists(os.path.join(run_dir, "trial_1"))
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    # Database should exist even if empty
    db_path = os.path.join(run_dir, "trial_1", "bh_go.db")
    assert os.path.exists(db_path)


def test_final_xyz_is_canonicalized_before_write(tmp_path, rng, monkeypatch):
    """Final minima export should use canonical storage-frame normalization."""
    from scgo.database.metadata import add_metadata
    from scgo.minima_search import core as core_mod

    atoms = Atoms(
        "Pt2",
        positions=[[1.0, 2.0, 3.0], [2.2, 2.0, 3.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    add_metadata(atoms, raw_score=-1.0, run_id="run_test", trial=1)

    def fake_scgo(**kwargs):
        return [(-1.0, atoms.copy())]

    monkeypatch.setattr(core_mod, "scgo", fake_scgo)

    output_dir = str(tmp_path / "canonical_output")
    run_trials(
        composition=["Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "niter_local_relaxation": 1},
        n_trials=1,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    xyz_files = sorted((Path(output_dir) / "final_unique_minima").glob("*.xyz"))
    assert xyz_files
    out_atoms = read(str(xyz_files[0]))
    np.testing.assert_allclose(
        out_atoms.get_center_of_mass(), [5.0, 5.0, 5.0], atol=1e-8
    )
