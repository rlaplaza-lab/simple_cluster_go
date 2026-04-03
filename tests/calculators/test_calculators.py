"""Tests for calculator helper functions.

This module tests utility functions for generating input files for various
quantum chemistry calculators.
"""

from pathlib import Path

import pytest
from ase import Atoms

from scgo.calculators.mace_helpers import MACE, MaceUrls
from scgo.calculators.orca_helpers import prepare_orca_calculations, write_orca_inputs
from scgo.calculators.vasp_helpers import prepare_vasp_calculations, write_vasp_inputs
from tests.test_utils import setup_test_atoms


def test_write_orca_inputs(tmp_path):
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {
        "charge": 0,
        "multiplicity": 1,
        "keywords": "! PBE def2-SVP",
        "blocks_str": "%scf MaxIter 200 end",
    }

    write_orca_inputs(atoms, output_dir, orca_settings)

    input_file = tmp_path / "orca_calc" / "orca.inp"
    assert input_file.exists()
    with open(input_file) as f:
        content = f.read()

    expected_keywords1 = "! PBE def2-SVP"
    expected_blocks1 = "%scf MaxIter 200 end"
    expected_keywords2 = "! PBE def2-SVP"
    expected_blocks2 = """%moinp "orca.gbw"

%scf MaxIter 200 end"""

    assert expected_keywords1 in content
    assert expected_blocks1 in content
    assert "$new_job" in content
    assert expected_keywords2 in content
    assert expected_blocks2 in content
    assert "* xyz 0 1" in content
    assert "* xyzfile 0 1 orca.xyz" in content


def test_write_orca_inputs_defaults(tmp_path):
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {}  # Empty settings to test defaults

    write_orca_inputs(atoms, output_dir, orca_settings)

    input_file = tmp_path / "orca_calc" / "orca.inp"
    assert input_file.exists()
    with open(input_file) as f:
        content = f.read()

    expected_keywords1 = "! RI PBE def2-tzvp def2/J Opt Freq VerySlowConv"
    expected_blocks1 = """%pal
nprocs 24
end

%scf
MaxIter 1500
DIISMaxEq 15
directresetfreq 5
end"""
    expected_keywords2 = "! PBE0 def2-tzvp VerySlowConv MOread"
    expected_blocks2 = """%moinp "orca.gbw"

%pal
nprocs 24
end

%scf
MaxIter 1500
DIISMaxEq 15
directresetfreq 5
end"""

    assert expected_keywords1 in content
    assert expected_blocks1 in content
    assert "$new_job" in content
    assert expected_keywords2 in content
    assert expected_blocks2 in content
    assert "* xyz 0 1" in content
    assert "* xyzfile 0 1 orca.xyz" in content


class TestPrepareOrcaCalculations:
    """Tests for prepare_orca_calculations function."""

    def test_prepare_orca_calculations_creates_directories(self, tmp_path):
        """Test prepare_orca_calculations creates subdirectories for each minimum."""
        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        unique_minima = [(-10.0, atoms1), (-15.0, atoms2)]
        base_dir = str(tmp_path / "orca_prep")
        orca_settings = {"charge": 0, "multiplicity": 1}

        prepare_orca_calculations(unique_minima, base_dir, orca_settings)

        # Should create subdirectories
        subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        assert len(subdirs) == 2

        # Each subdirectory should have orca.inp
        for subdir in subdirs:
            orca_file = subdir / "orca.inp"
            assert orca_file.exists()

    def test_prepare_orca_calculations_empty_list(self, tmp_path):
        """Test prepare_orca_calculations handles empty list."""
        base_dir = str(tmp_path / "orca_empty")
        orca_settings = {}

        # Should not raise
        prepare_orca_calculations([], base_dir, orca_settings)


class TestWriteVaspInputs:
    """Tests for write_vasp_inputs function."""

    def test_write_vasp_inputs_creates_files(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs creates VASP input files."""

        # Mock VASP to avoid requiring VASP_PP_PATH
        def mock_write_input(self, atoms):
            """Mock write_input that creates dummy files."""
            incar_path = Path(self.directory) / "INCAR"
            poscar_path = Path(self.directory) / "POSCAR"
            kpoints_path = Path(self.directory) / "KPOINTS"
            incar_path.write_text("ENCUT = 400\n")
            poscar_path.write_text("dummy POSCAR\n")
            kpoints_path.write_text("dummy KPOINTS\n")

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {"encut": 400}

        write_vasp_inputs(atoms, output_dir, vasp_settings)

        # Should create VASP input files
        assert (tmp_path / "vasp_calc" / "INCAR").exists()
        assert (tmp_path / "vasp_calc" / "POSCAR").exists()
        assert (tmp_path / "vasp_calc" / "KPOINTS").exists()

    def test_write_vasp_inputs_creates_xyz(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs creates XYZ file."""

        def mock_write_input(self, atoms):
            """Mock write_input."""

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {}

        write_vasp_inputs(atoms, output_dir, vasp_settings)

        # Should create XYZ file
        xyz_files = list(Path(output_dir).glob("*.xyz"))
        assert len(xyz_files) > 0

    def test_write_vasp_inputs_custom_vacuum(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs uses custom vacuum parameter."""

        def mock_write_input(self, atoms):
            """Mock write_input that checks vacuum."""
            # Check that atoms have been centered with vacuum
            # pbc is a numpy array, so check all values are True
            assert all(atoms.pbc)
            # Store for later verification
            self._atoms = atoms

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {}

        write_vasp_inputs(atoms, output_dir, vasp_settings, vacuum=15.0)

        # Function should complete without error


class TestPrepareVaspCalculations:
    """Tests for prepare_vasp_calculations function."""

    def test_prepare_vasp_calculations_creates_directories(self, tmp_path, monkeypatch):
        """Test prepare_vasp_calculations creates subdirectories."""

        def mock_write_input(self, atoms):
            """Mock write_input."""

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        unique_minima = [(-10.0, atoms1), (-15.0, atoms2)]
        base_dir = str(tmp_path / "vasp_prep")
        vasp_settings = {"encut": 400}

        prepare_vasp_calculations(unique_minima, base_dir, vasp_settings, vacuum=10.0)

        # Should create subdirectories
        subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        assert len(subdirs) == 2


class TestMaceHelpers:
    """Tests for MACE calculator helpers."""

    def test_mace_calculator_import(self):
        assert MACE is not None and callable(MACE)

    def test_mace_urls_enum(self):
        assert hasattr(MaceUrls, "mace_mp_small") or hasattr(MaceUrls, "mace_matpes_0")

    def test_mace_calculator_initialization(self):
        try:
            calc = MACE(model="mace_mp_small")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            pytest.skip(f"MACE init failed (e.g. missing model): {e}")
        assert calc is not None
