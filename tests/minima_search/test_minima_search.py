"""Tests for scgo.minima_search core orchestration."""

import json
import os

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.emt import EMT

import scgo.minima_search.core as main_mod
from scgo.minima_search import run_trials, scgo
from scgo.utils.helpers import ensure_directory_exists
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION
from tests.test_utils import setup_test_atoms


class TestEnsureCalculator:
    """Tests for _ensure_calculator function."""

    def test_ensure_calculator_with_none(self):
        """Test that None calculator returns default EMT calculator."""
        calc = main_mod._ensure_calculator(None)
        assert isinstance(calc, EMT)

    def test_ensure_calculator_with_calculator(self):
        """Test that provided calculator is returned unchanged."""
        provided_calc = EMT()
        calc = main_mod._ensure_calculator(provided_calc)
        assert calc is provided_calc


class TestValidateCalculatorCompatibility:
    """Tests for calculator interface validation."""

    def test_valid_calculator(self):
        """Test validation passes for valid calculator."""
        calc = EMT()
        is_valid, msg = main_mod._validate_calculator_compatibility(calc)
        assert is_valid is True
        assert "compatible" in msg.lower()

    def test_calculator_missing_method(self):
        """Test validation fails for calculator missing required methods."""

        class BadCalculator:
            """Calculator missing get_forces method."""

            def get_potential_energy(self):
                return 0.0

        calc = BadCalculator()
        is_valid, msg = main_mod._validate_calculator_compatibility(calc)
        assert is_valid is False
        assert "missing" in msg.lower()
        assert "get_forces" in msg

    def test_calculator_custom_required_methods(self):
        """Test validation with custom required methods list."""
        calc = EMT()

        # Should pass with custom list that calculator has
        is_valid, msg = main_mod._validate_calculator_compatibility(
            calc, required_methods=["get_potential_energy"]
        )
        assert is_valid is True

        # Should fail with method calculator doesn't have
        is_valid, msg = main_mod._validate_calculator_compatibility(
            calc, required_methods=["nonexistent_method"]
        )
        assert is_valid is False


class TestScgoFunction:
    """Tests for scgo() function - single trial orchestration."""

    def test_scgo_with_bh_optimizer(self, tmp_path, rng):
        """Test scgo() with basin hopping optimizer."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "test_bh")
        optimizer_kwargs = {"niter": 2, "niter_local_relaxation": 3}

        results = scgo(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)
        # Should create output directory
        assert os.path.exists(output_dir)

    def test_scgo_with_ga_optimizer(self, tmp_path, rng):
        """Test scgo() with genetic algorithm optimizer."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "test_ga")
        optimizer_kwargs = {
            "niter": 2,
            "population_size": 3,
            "niter_local_relaxation": 3,
        }

        results = scgo(
            composition=composition,
            global_optimizer="ga",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)
        assert os.path.exists(output_dir)

    def test_scgo_with_simple_optimizer(self, tmp_path, rng):
        """Test scgo() with simple optimizer."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_simple")
        optimizer_kwargs = {"niter": 1}

        results = scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_scgo_unknown_optimizer(self, tmp_path, rng):
        """Test scgo() raises error for unknown optimizer."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_unknown")

        with pytest.raises(ValueError, match="Unknown global_optimizer"):
            scgo(
                composition=composition,
                global_optimizer="unknown",
                global_optimizer_kwargs={},
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )

    def test_scgo_invalid_calculator(self, tmp_path, rng):
        """Test scgo() validates calculator interface requirements."""

        class BadCalculator:
            """Calculator missing required methods."""

        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_bad_calc")

        with pytest.raises(ValueError, match="Calculator validation failed"):
            scgo(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1},
                output_dir=output_dir,
                rng=rng,
                calculator_for_global_optimization=BadCalculator(),
                verbosity=0,
            )

    def test_scgo_creates_output_directory(self, tmp_path, rng):
        """Test scgo() creates output directory if it doesn't exist."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "new_dir" / "subdir")

        scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            output_dir=output_dir,
            rng=rng,
            verbosity=0,
        )

        assert os.path.exists(output_dir)

    def test_scgo_adds_provenance(self, tmp_path, rng):
        """Test scgo() adds provenance metadata to results."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_provenance")
        run_id = "test_run_123"

        results = scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            output_dir=output_dir,
            rng=rng,
            trial_id=5,
            run_id=run_id,
            verbosity=0,
        )

        for _, atoms in results:
            assert "provenance" in atoms.info
            assert atoms.info["provenance"]["trial"] == 5
            assert atoms.info["provenance"]["run_id"] == run_id

    def test_scgo_empty_composition(self, tmp_path, rng):
        """Test scgo() raises error for empty composition."""
        composition = []
        output_dir = str(tmp_path / "test_empty")

        with pytest.raises(ValueError, match="Composition cannot be empty"):
            scgo(
                composition=composition,
                global_optimizer="simple",
                global_optimizer_kwargs={"niter": 1},
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )


class TestRunTrials:
    """Tests for run_trials() function - multi-trial orchestration."""

    def test_run_trials_single_trial(self, tmp_path, rng):
        """Test run_trials() with single trial."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_test")

        results = run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 3},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)
        assert os.path.exists(output_dir)

    def test_run_trials_multiple_trials(self, tmp_path, rng):
        """Test run_trials() with multiple trials."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_multi")

        results = run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={"niter": 2, "niter_local_relaxation": 3},
            n_trials=3,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)
        # Should create trial directories
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        assert len(run_dirs) > 0

    def test_run_trials_zero_trials_raises_error(self, tmp_path, rng):
        """Test run_trials() raises error for zero trials."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_zero")

        with pytest.raises(ValueError, match="n_trials must be positive"):
            run_trials(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1},
                n_trials=0,
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )

    def test_run_trials_negative_trials_raises_error(self, tmp_path, rng):
        """Test run_trials() raises error for negative trials."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_negative")

        with pytest.raises(ValueError, match="n_trials must be positive"):
            run_trials(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1},
                n_trials=-1,
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )

    def test_run_trials_creates_run_directory(self, tmp_path, rng):
        """Test run_trials() creates run-specific directory."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_run_dir")

        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should create run_* directory
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        assert len(run_dirs) == 1

    def test_run_trials_with_run_id(self, tmp_path, rng):
        """Test run_trials() uses provided run_id."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_custom_id")
        custom_run_id = "custom_run_123"

        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            run_id=custom_run_id,
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should create directory with custom run_id
        run_dir = os.path.join(output_dir, custom_run_id)
        assert os.path.exists(run_dir)

    def test_run_trials_clean_mode(self, tmp_path, rng):
        """Test run_trials() with clean=True ignores previous runs."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_clean")

        # First run
        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            validate_with_hessian=False,
            verbosity=0,
        )

        # Second run with clean=True should start fresh
        results = run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            clean=True,
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_run_trials_with_ga(self, tmp_path, rng):
        """Test run_trials() with genetic algorithm."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_ga")

        results = run_trials(
            composition=composition,
            global_optimizer="ga",
            global_optimizer_kwargs={
                "niter": 2,
                "population_size": 3,
                "niter_local_relaxation": 3,
                "n_jobs_population_init": -2,  # Parallel for tests
            },
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_run_trials_no_minima_found(self, tmp_path, rng):
        """Test run_trials() returns empty list when no minima found."""
        composition = ["Pt"]
        output_dir = str(tmp_path / "trials_no_minima")

        # Use very short run that might not find minima
        results = run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1},
            n_trials=1,
            output_dir=output_dir,
            rng=rng,
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should return list (may be empty)
        assert isinstance(results, list)


class TestWriteResultsSummary:
    """Tests for _write_results_summary function."""

    def test_write_results_summary_creates_file(self, tmp_path):
        """Test _write_results_summary creates summary file."""
        output_dir = str(tmp_path / "summary_test")
        ensure_directory_exists(output_dir)

        # Create some dummy results
        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        results = [(-10.0, atoms1), (-15.0, atoms2)]

        sample_params = {"global_optimizer": "bh", "n_trials": 2}
        main_mod._write_results_summary(
            output_dir=output_dir,
            final_minima=results,
            composition_str="Pt5",
            run_id="test_run_123",
            params=sample_params,
        )

        summary_file = os.path.join(output_dir, "results_summary.json")
        assert os.path.exists(summary_file)

        # Verify content
        with open(summary_file) as f:
            summary = json.load(f)

        assert "composition" in summary
        assert summary["composition"] == "Pt5"
        assert "total_unique_minima" in summary
        assert summary["total_unique_minima"] == 2
        assert summary["params"] == sample_params
        assert summary["run_metadata_relpath"] == "test_run_123/metadata.json"
        assert summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert isinstance(summary.get("scgo_version"), str) and summary["scgo_version"]
        assert isinstance(summary.get("python_version"), str)
        assert isinstance(summary.get("created_at"), str)

    def test_write_results_summary_empty_results(self, tmp_path):
        """Test _write_results_summary handles empty results."""
        output_dir = str(tmp_path / "summary_empty")
        ensure_directory_exists(output_dir)

        main_mod._write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str="Pt2",
            run_id="test_run_empty",
            params=None,
        )

        summary_file = os.path.join(output_dir, "results_summary.json")
        assert os.path.exists(summary_file)

        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["total_unique_minima"] == 0
        assert summary["params"] is None
        assert summary["run_metadata_relpath"] == "test_run_empty/metadata.json"
        assert summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert isinstance(summary.get("scgo_version"), str) and summary["scgo_version"]


class _DummyMLCalculator:
    """Minimal MACE-like calculator for testing TorchSim availability logic."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        class _Model:
            def forward(self):  # pragma: no cover - never actually called
                return None

        self.model = _Model()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        # Provide dummy but finite values
        n_atoms = len(atoms)
        self.energy = -1.0
        self.forces = np.zeros((n_atoms, 3))


def _patch_ga_go_fakes(monkeypatch, atoms):
    called = {"torchsim": False, "ase_ga": False}

    def fake_ga_go_torchsim(**kwargs):
        called["torchsim"] = True
        return [(-1.0, atoms.copy())]

    def fake_ga_go(**kwargs):
        called["ase_ga"] = True
        return [(-0.5, atoms.copy())]

    monkeypatch.setattr(main_mod, "ga_go_torchsim", fake_ga_go_torchsim)
    monkeypatch.setattr(main_mod, "ga_go", fake_ga_go)
    return called


def test_select_and_run_ga_uses_torchsim_for_ml_calculator(monkeypatch, rng):
    """ML calculator always uses ga_go_torchsim."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = _DummyMLCalculator()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called = _patch_ga_go_fakes(monkeypatch, atoms)

    results = main_mod._select_and_run_ga(
        composition=["H", "H"],
        output_dir=".",
        optimizer_kwargs={
            "niter": 1,
            "population_size": 2,
        },
        calculator=calc,
        rng=rng,
        verbosity=0,
    )

    assert called["torchsim"] is True
    assert called["ase_ga"] is False
    assert isinstance(results, list)


def test_select_and_run_ga_uses_ase_for_non_ml_calculator(monkeypatch, rng):
    """Non-ML calculator uses ASE GA."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = EMT()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    called = _patch_ga_go_fakes(monkeypatch, atoms)

    results = main_mod._select_and_run_ga(
        composition=["H", "H"],
        output_dir=".",
        optimizer_kwargs={
            "niter": 1,
            "population_size": 2,
        },
        calculator=calc,
        rng=rng,
        verbosity=0,
    )

    assert called["torchsim"] is False
    assert called["ase_ga"] is True
    assert isinstance(results, list)


def test_sanitize_global_optimizer_kwargs_for_metadata_surface_config():
    """surface_config must not embed ASE Atoms in JSON metadata."""
    from ase.build import fcc111

    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.0,
    )
    raw = {"niter": 1, "surface_config": cfg, "relaxer": object()}
    clean = main_mod._sanitize_global_optimizer_kwargs_for_metadata(raw)
    assert "relaxer" not in clean
    assert isinstance(clean["surface_config"], dict)
    assert clean["surface_config"]["present"] is True
    assert clean["surface_config"]["n_slab_atoms"] == len(slab)
    assert clean["surface_config"]["surface_normal_axis"] == 2
    assert clean["surface_config"]["fix_all_slab_atoms"] is True
    assert clean["surface_config"]["n_fix_bottom_slab_layers"] is None
    assert clean["surface_config"]["n_relax_top_slab_layers"] is None
    assert clean["surface_config"]["adsorption_height_min"] == 1.0
    assert clean["surface_config"]["adsorption_height_max"] == 2.0
    assert clean["surface_config"]["comparator_use_mic"] is False
    assert clean["surface_config"]["cluster_init_vacuum"] == 8.0
    assert clean["surface_config"]["init_mode"] == "smart"
    assert clean["surface_config"]["max_placement_attempts"] == 200
