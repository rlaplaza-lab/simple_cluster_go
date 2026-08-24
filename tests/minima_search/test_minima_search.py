"""Tests for scgo.minima_search core orchestration."""

import json
import os

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import read

import scgo.minima_search.core as main_mod
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.metadata.provenance import OUTPUT_JSON_SCHEMA_VERSION
from scgo.minima_search import run_trials, scgo
from scgo.system_types import AdsorbateDefinition
from scgo.utils.helpers import ensure_directory_exists
from tests.helpers import create_test_atoms, setup_test_atoms


class TestRequireCalculator:
    """Tests for _require_calculator function."""

    def test_require_calculator_with_none(self):
        """Test that None calculator raises SCGOValidationError."""
        with pytest.raises(
            SCGOValidationError, match="calculator_for_global_optimization is required"
        ):
            main_mod._require_calculator(None)

    def test_require_calculator_with_calculator(self):
        """Test that provided calculator is returned unchanged."""
        provided_calc = EMT()
        calc = main_mod._require_calculator(provided_calc)
        assert calc is provided_calc


class TestScgoFunction:
    """Tests for scgo() function - single GO run orchestration."""

    def test_scgo_with_bh_optimizer(self, tmp_path, rng):
        """Test scgo() with basin hopping optimizer."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "test_bh")
        optimizer_kwargs = {
            "niter": 2,
            "niter_local_relaxation": 3,
            "system_type": "gas_cluster",
        }

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
            "system_type": "gas_cluster",
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
        optimizer_kwargs = {"niter": 1, "system_type": "gas_cluster"}

        results = scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_scgo_requires_system_type(self, tmp_path, rng):
        with pytest.raises(SCGOValidationError, match="system_type must be set"):
            scgo(
                composition=["Pt", "Pt"],
                global_optimizer="simple",
                global_optimizer_kwargs={"niter": 1},
                output_dir=str(tmp_path / "missing_system_type"),
                rng=rng,
                calculator_for_global_optimization=EMT(),
                verbosity=0,
            )

    def test_scgo_surface_bh_is_supported(self, tmp_path, rng, monkeypatch):
        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        surface_config = main_mod.SurfaceSystemConfig(
            slab=slab,
            adsorption_height_min=1.0,
            adsorption_height_max=2.5,
        )

        captured: dict[str, object] = {}

        def _fake_bh_go(*, atoms, **kwargs):
            captured["atoms"] = atoms
            captured["kwargs"] = kwargs
            return []

        monkeypatch.setattr(main_mod, "bh_go", _fake_bh_go)
        monkeypatch.setitem(main_mod._ALGORITHM_REGISTRY, "bh", _fake_bh_go)

        results = scgo(
            composition=["Pt", "O", "H"],
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 1,
                "niter_local_relaxation": 1,
                "system_type": "surface_cluster_adsorbate",
                "surface_config": surface_config,
                "adsorbate_definition": AdsorbateDefinition(
                    core_symbols=["Pt"],
                    adsorbate_symbols=["O", "H"],
                ),
                "adsorbate_fragment_template": Atoms(
                    symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]
                ),
            },
            output_dir=str(tmp_path / "surface_bh"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []
        assert len(captured["atoms"]) > len(slab)

    def test_scgo_gas_adsorbate_bh_strips_init_only_kwargs(
        self, tmp_path, rng, monkeypatch
    ):
        captured: dict[str, object] = {}

        def _fake_bh_go(*, atoms, **kwargs):
            captured["atoms"] = atoms
            captured["kwargs"] = kwargs
            return []

        monkeypatch.setattr(main_mod, "bh_go", _fake_bh_go)
        monkeypatch.setitem(main_mod._ALGORITHM_REGISTRY, "bh", _fake_bh_go)

        ads_def = AdsorbateDefinition(
            core_symbols=["Pt", "Pt"],
            adsorbate_symbols=["O"],
            adsorbate_fragment_lengths=[1],
        )
        frag = Atoms(symbols=["O"], positions=[[0.0, 0.0, 0.0]])

        results = scgo(
            composition=["Pt", "Pt", "O"],
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 1,
                "niter_local_relaxation": 1,
                "system_type": "gas_cluster_adsorbate",
                "adsorbate_definition": ads_def,
                "adsorbate_fragment_template": frag,
                "vacuum": 12.0,
                "init_mode": "smart",
                "max_hierarchical_attempts": 5,
                "previous_search_glob": "**/*.db",
            },
            output_dir=str(tmp_path / "gas_bh"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []
        assert len(captured["atoms"]) == 3
        bh_kwargs = captured["kwargs"]
        assert bh_kwargs["adsorbate_definition"] == ads_def
        assert "adsorbate_fragment_template" not in bh_kwargs
        assert "vacuum" not in bh_kwargs
        assert "init_mode" not in bh_kwargs
        assert "max_hierarchical_attempts" not in bh_kwargs
        assert "previous_search_glob" not in bh_kwargs

    def test_scgo_gas_adsorbate_empty_core_is_noop(self, tmp_path, rng):
        results = scgo(
            composition=["O", "H"],
            global_optimizer="ga",
            global_optimizer_kwargs={
                "niter": 1,
                "population_size": 2,
                "system_type": "gas_cluster_adsorbate",
                "adsorbate_definition": AdsorbateDefinition(
                    core_symbols=[],
                    adsorbate_symbols=["O", "H"],
                    adsorbate_fragment_lengths=[2],
                ),
                "adsorbate_fragment_template": Atoms(
                    symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]
                ),
            },
            output_dir=str(tmp_path / "gas_empty_core_noop"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []

    def test_scgo_unknown_optimizer(self, tmp_path, rng):
        """Test scgo() raises error for unknown optimizer."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_unknown")

        with pytest.raises(SCGOValidationError, match="Unknown global_optimizer"):
            scgo(
                composition=composition,
                global_optimizer="unknown",
                global_optimizer_kwargs={"system_type": "gas_cluster"},
                output_dir=output_dir,
                rng=rng,
                calculator_for_global_optimization=EMT(),
                verbosity=0,
            )

    def test_scgo_creates_output_directory(self, tmp_path, rng):
        """Test scgo() creates output directory if it doesn't exist."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "new_dir" / "subdir")

        scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
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
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            run_id=run_id,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        for _, atoms in results:
            from scgo.metadata.atoms import get_tag

            assert get_tag(atoms, "run_id") == run_id

    def test_scgo_empty_composition(self, tmp_path, rng):
        """Test scgo() raises error for empty composition."""
        composition = []
        output_dir = str(tmp_path / "test_empty")

        with pytest.raises(SCGOValidationError, match="Composition cannot be empty"):
            scgo(
                composition=composition,
                global_optimizer="simple",
                global_optimizer_kwargs={"niter": 1},
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )


class TestRunTrials:
    """Tests for run_trials() function - single run orchestration."""

    def test_run_trials_single_run(self, tmp_path, rng):
        """Test run_trials() with a single datetime-tagged run."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_test")

        results = run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 2,
                "niter_local_relaxation": 3,
                "system_type": "gas_cluster",
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)
        assert os.path.exists(output_dir)

    def test_run_trials_creates_db_at_run_root(self, tmp_path, rng):
        """Test run_trials() places database directly under run_*/."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_multi")

        run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 2,
                "niter_local_relaxation": 3,
                "system_type": "gas_cluster",
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        assert len(run_dirs) == 1
        run_dir = os.path.join(output_dir, run_dirs[0])
        assert os.path.exists(os.path.join(run_dir, "bh_go.db"))
        assert not os.path.exists(os.path.join(run_dir, "trial_1"))

    def test_run_trials_missing_system_type_raises(self, tmp_path, rng):
        """Test run_trials() requires system_type in global_optimizer_kwargs."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_missing_st")

        with pytest.raises(SCGOValidationError, match="system_type must be set"):
            run_trials(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1},
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
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
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
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            run_id=custom_run_id,
            calculator_for_global_optimization=EMT(),
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
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Second run with clean=True should start fresh
        results = run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            clean=True,
            calculator_for_global_optimization=EMT(),
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
                "system_type": "gas_cluster",
            },
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
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should return list (may be empty)
        assert isinstance(results, list)


def _slab_pt_adsorbate_pair(*, mobile_xy=(0.1, 0.1), wrap_x=False):
    """Build reference and x-wrapped slab+Pt adsorbate minima with surface metadata."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    ref = slab.copy() + Atoms("Pt", positions=[[mobile_xy[0], mobile_xy[1], z0]])
    x_mob = slab.cell[0, 0] - mobile_xy[0] if wrap_x else mobile_xy[0]
    wrapped = slab.copy() + Atoms("Pt", positions=[[x_mob, mobile_xy[1], z0]])
    for atoms in (ref, wrapped):
        atoms.pbc = slab.pbc
        set_tags(
            atoms,
            run_id="run_test",
            system_type="surface_cluster",
            n_slab_atoms=n_slab,
            raw_score=0.0,
        )
    return ref, wrapped, n_slab


class TestRunTrialsSurfaceAlignment:
    """Slab final minima are aligned to the lowest-energy minimum before write."""

    def test_resolve_surface_alignment_defaults(self):
        kwargs = _resolve_surface_alignment_kwargs(
            {"system_type": "surface_cluster", "surface_config": object()}
        )
        assert kwargs is not None
        assert kwargs["enable_cell_remap"] is True
        assert kwargs["enable_lattice_rotation"] is True
        assert kwargs["max_lattice_shift"] == 1

    def test_resolve_surface_alignment_gas_returns_none(self):
        assert _resolve_surface_alignment_kwargs({"system_type": "gas_cluster"}) is None

    def test_resolve_surface_alignment_reads_params(self):
        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
        kwargs = _resolve_surface_alignment_kwargs(
            {
                "system_type": "surface_cluster",
                "surface_config": cfg,
                "neb_surface_cell_remap": False,
                "neb_surface_lattice_rotation": True,
                "neb_surface_max_lattice_shift": 3,
            }
        )
        assert kwargs is not None
        assert kwargs["enable_cell_remap"] is False
        assert kwargs["enable_lattice_rotation"] is True
        assert kwargs["max_lattice_shift"] == 3

    def test_resolve_n_core_mobile_from_metadata_and_definition(self):
        atoms = Atoms("Pt2OH", positions=[[0, 0, 0], [2, 0, 0], [1, 0, 1], [1, 0, 2]])
        set_tags(atoms, n_core_atoms=2)
        assert main_mod._resolve_n_core_mobile_for_alignment(atoms, {}) == 2
        bare = Atoms("Pt2OH", positions=[[0, 0, 0], [2, 0, 0], [1, 0, 1], [1, 0, 2]])
        assert (
            main_mod._resolve_n_core_mobile_for_alignment(
                bare,
                {
                    "adsorbate_definition": AdsorbateDefinition(
                        core_symbols=["Pt", "Pt"],
                        adsorbate_symbols=["O", "H"],
                        adsorbate_fragment_lengths=[2],
                    )
                },
            )
            == 2
        )

    def test_align_slab_forwards_n_core_mobile(self, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_pbc(reactant, product_positions, **kwargs):
            captured.update(kwargs)
            return product_positions

        monkeypatch.setattr(
            "scgo.ts_search.transition_state._align_product_surface_pbc",
            _fake_pbc,
        )
        ref = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]], cell=[5, 5, 10], pbc=True)
        cand = ref.copy()
        main_mod._align_slab_minimum_to_reference(
            ref,
            cand,
            n_slab=1,
            enable_cell_remap=True,
            enable_lattice_rotation=False,
            max_lattice_shift=1,
            n_core_mobile=1,
        )
        assert captured.get("n_core_mobile") == 1

    def test_run_trials_aligns_slab_final_minima_to_best(
        self, tmp_path, rng, monkeypatch
    ):
        ref, wrapped, _n_slab = _slab_pt_adsorbate_pair(wrap_x=True)
        align_calls = 0

        def _fake_scgo(**_kwargs):
            return [(-1.0, ref), (-0.5, wrapped)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        orig_align = main_mod._align_slab_minimum_to_reference

        def _spy_align(reference, candidate, **kwargs):
            nonlocal align_calls
            align_calls += 1
            orig_align(reference, candidate, **kwargs)

        monkeypatch.setattr(main_mod, "_align_slab_minimum_to_reference", _spy_align)

        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
        output_dir = str(tmp_path / "slab_align")

        run_trials(
            composition=["Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={
                "niter": 1,
                "system_type": "surface_cluster",
                "surface_config": cfg,
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert align_calls == 2
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        written = sorted(f for f in os.listdir(xyz_dir) if f.endswith(".xyz"))
        assert len(written) == 2
        best_written = read(os.path.join(xyz_dir, written[0]))
        second_written = read(os.path.join(xyz_dir, written[1]))
        disp = second_written.get_positions() - best_written.get_positions()
        assert abs(float(disp[-1, 0])) < 0.5

    def test_run_trials_forwards_alignment_knobs(self, tmp_path, rng, monkeypatch):
        ref, wrapped, n_slab = _slab_pt_adsorbate_pair(wrap_x=True)
        captured: dict[str, int] = {}

        def _fake_scgo(**_kwargs):
            return [(-1.0, ref), (-0.5, wrapped)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        from scgo.ts_search import transition_state as ts_mod

        orig_pbc = ts_mod._align_product_surface_pbc

        def _spy_pbc(reactant, product_positions, **kwargs):
            captured["max_lattice_shift"] = kwargs.get("max_lattice_shift", -1)
            return orig_pbc(reactant, product_positions, **kwargs)

        monkeypatch.setattr(ts_mod, "_align_product_surface_pbc", _spy_pbc)

        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)

        run_trials(
            composition=["Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={
                "niter": 1,
                "system_type": "surface_cluster",
                "surface_config": cfg,
                "neb_surface_max_lattice_shift": 2,
            },
            output_dir=str(tmp_path / "slab_knobs"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert captured["max_lattice_shift"] == 2
        assert n_slab > 0

    def test_run_trials_gas_skips_slab_alignment(self, tmp_path, rng, monkeypatch):
        atoms = create_test_atoms(["Pt", "Pt"])
        set_tags(atoms, run_id="run_test", system_type="gas_cluster")
        align_calls = 0

        def _fake_scgo(**_kwargs):
            return [(-1.0, atoms)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        def _spy_align(*_args, **_kwargs):
            nonlocal align_calls
            align_calls += 1

        monkeypatch.setattr(main_mod, "_align_slab_minimum_to_reference", _spy_align)

        run_trials(
            composition=["Pt", "Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=str(tmp_path / "gas_no_align"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert align_calls == 0


def _resolve_surface_alignment_kwargs(kwargs):
    return main_mod._resolve_surface_alignment_kwargs(kwargs)


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

        sample_params = {"global_optimizer": "bh"}
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
        assert summary["schema_version"] == OUTPUT_JSON_SCHEMA_VERSION
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
        assert summary["schema_version"] == OUTPUT_JSON_SCHEMA_VERSION
        assert isinstance(summary.get("scgo_version"), str) and summary["scgo_version"]


def test_scgo_ga_delegates_to_ga_go(monkeypatch, rng, tmp_path):
    """Unified GA path in scgo() calls ga_go."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    called = {"ga": False}

    def fake_ga_go(**kwargs):
        called["ga"] = True
        return [(-1.0, atoms.copy())]

    monkeypatch.setattr(main_mod, "ga_go", fake_ga_go)

    results = scgo(
        composition=["H", "H"],
        global_optimizer="ga",
        global_optimizer_kwargs={
            "niter": 1,
            "population_size": 2,
            "system_type": "gas_cluster",
        },
        output_dir=str(tmp_path / "ga_delegate"),
        rng=rng,
        calculator_for_global_optimization=EMT(),
        verbosity=0,
    )

    assert called["ga"] is True
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
    assert clean["surface_config"]["slab_chemical_symbols"] == list(
        slab.get_chemical_symbols()
    )
    assert clean["surface_config"]["surface_normal_axis"] == 2
    assert clean["surface_config"]["fix_all_slab_atoms"] is True
    assert clean["surface_config"]["n_fix_bottom_slab_layers"] is None
    assert clean["surface_config"]["n_relax_top_slab_layers"] is None
    assert clean["surface_config"]["adsorption_height_min"] == 1.0
    assert clean["surface_config"]["adsorption_height_max"] == 2.0
    assert clean["surface_config"]["comparator_use_mic"] is True
    assert clean["surface_config"]["cluster_init_vacuum"] == 8.0
    assert clean["surface_config"]["init_mode"] == "smart"
    assert clean["surface_config"]["max_placement_attempts"] == 200


def test_run_trials_passes_hessian_params_to_is_true_minimum(
    tmp_path, monkeypatch, rng
):
    """Preset Hessian knobs must reach is_true_minimum when validation is enabled."""
    from ase.calculators.emt import EMT

    captured: dict[str, object] = {}

    def _fake_is_true_minimum(*, atoms, calculator, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(main_mod, "is_true_minimum", _fake_is_true_minimum)

    def _fake_scgo(*_args, **_kwargs):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
        atoms.calc = EMT()
        energy = float(atoms.get_potential_energy())
        return [(energy, atoms)]

    monkeypatch.setattr(main_mod, "scgo", _fake_scgo)
    monkeypatch.setattr(
        main_mod, "filter_unique_minima", lambda candidates, *_args, **_kw: candidates
    )

    outdir = str(tmp_path / "searches")
    run_trials(
        composition=["Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "system_type": "gas_cluster",
        },
        output_dir=outdir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=True,
        check_hessian=False,
        fmax_threshold=0.02,
        imag_freq_threshold=25.0,
        rng=rng,
        clean=True,
    )

    assert captured.get("check_hessian") is False
    assert captured.get("fmax_threshold") == 0.02
    assert captured.get("imag_freq_threshold") == 25.0


def test_run_trials_dedupe(tmp_path, monkeypatch, rng):
    captured: dict[str, object] = {}

    def _fake_scgo(*_args, **_kwargs):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
        atoms.calc = EMT()
        return [(float(atoms.get_potential_energy()), atoms)]

    monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

    def _spy_filter(
        candidates,
        energy_tolerance=None,
        *,
        n_top,
        comparator_tol=None,
        comparator_pair_cor_max=None,
        **kwargs,
    ):
        captured["energy_tolerance"] = energy_tolerance
        captured["comparator_tol"] = comparator_tol
        captured["comparator_pair_cor_max"] = comparator_pair_cor_max
        captured["n_top"] = n_top
        return candidates

    monkeypatch.setattr(main_mod, "filter_unique_minima", _spy_filter)

    run_trials(
        composition=["Pt", "Pt"],
        global_optimizer="ga",
        global_optimizer_kwargs={
            "niter": 1,
            "system_type": "gas_cluster",
            "energy_tolerance": 0.05,
            "comparator_tol": 0.02,
            "comparator_pair_cor_max": 0.4,
            "comparator_n_top": 1,
        },
        output_dir=str(tmp_path / "searches"),
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        clean=True,
        search_mobile_count=2,
    )

    assert captured["energy_tolerance"] == pytest.approx(0.05)
    assert captured["comparator_tol"] == pytest.approx(0.02)
    assert captured["comparator_pair_cor_max"] == pytest.approx(0.4)
    assert captured["n_top"] == 1


def test_run_trials_dedupe_n_top_without_comparator_override(
    tmp_path, monkeypatch, rng
):
    captured: dict[str, object] = {}

    def _fake_scgo(*_args, **_kwargs):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
        atoms.calc = EMT()
        return [(float(atoms.get_potential_energy()), atoms)]

    monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

    def _spy_filter(candidates, _energy_tolerance=None, *, n_top, **kwargs):
        captured["n_top"] = n_top
        return candidates

    monkeypatch.setattr(main_mod, "filter_unique_minima", _spy_filter)

    run_trials(
        composition=["Pt", "Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
        output_dir=str(tmp_path / "searches"),
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        clean=True,
        search_mobile_count=3,
    )

    assert captured["n_top"] == 3


def test_run_trials_dedupe_n_top_falls_back_to_composition(tmp_path, monkeypatch, rng):
    captured: dict[str, object] = {}

    def _fake_scgo(*_args, **_kwargs):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
        atoms.calc = EMT()
        return [(float(atoms.get_potential_energy()), atoms)]

    monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

    def _spy_filter(candidates, _energy_tolerance=None, *, n_top, **kwargs):
        captured["n_top"] = n_top
        return candidates

    monkeypatch.setattr(main_mod, "filter_unique_minima", _spy_filter)

    run_trials(
        composition=["Pt", "Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
        output_dir=str(tmp_path / "searches"),
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
        clean=True,
    )

    assert captured["n_top"] == 3


class TestValidateCandidatesParallel:
    """Tests for 4.3: cheap, robust parallel-validation startup.

    The deep-copy picklability probe was removed; instead, parallel startup
    failures (e.g. a non-picklable calculator) are caught and downgraded to a
    clean sequential fallback rather than crashing.
    """

    def _payloads(self, n=4):
        return [(float(i), Atoms("Pt2"), 0.1, False, 0.0) for i in range(n)]

    def test_single_worker_skips_parallel(self, monkeypatch):
        """With <=1 worker the helper defers to the sequential caller path."""
        called = {"submit": False}

        class _Tracked:
            def __init__(self, *a, **k):
                called["submit"] = True

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _Tracked)
        ok, minima = main_mod._validate_candidates_parallel(EMT(), self._payloads(), 1)
        assert ok is False
        assert minima == []
        assert called["submit"] is False

    def test_fallback_on_pickling_error(self, monkeypatch):
        """A non-picklable calculator must fall back, not crash."""
        import pickle

        class _Boom:
            def __init__(self, *a, **k):
                raise pickle.PicklingError("calculator is not picklable")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _Boom)
        ok, minima = main_mod._validate_candidates_parallel(EMT(), self._payloads(), 2)
        assert ok is False
        assert minima == []

    def test_fallback_on_broken_process_pool(self, monkeypatch):
        """A broken worker pool at startup must fall back, not crash."""

        class _Boom:
            def __init__(self, *a, **k):
                from concurrent.futures.process import BrokenProcessPool

                raise BrokenProcessPool("worker died")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _Boom)
        ok, minima = main_mod._validate_candidates_parallel(EMT(), self._payloads(), 2)
        assert ok is False
        assert minima == []

    def test_parallel_success(self, monkeypatch):
        """A working executor returns validated minima with parallel_ok=True."""

        class _FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class _SyncExecutor:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, payload):
                return _FakeFuture(fn(payload))

        monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncExecutor)
        monkeypatch.setattr(
            main_mod, "_validate_minimum_worker", lambda p: (p[0], p[1])
        )

        ok, minima = main_mod._validate_candidates_parallel(EMT(), self._payloads(), 2)
        assert ok is True
        assert len(minima) == 4
        assert {e for e, _ in minima} == {0.0, 1.0, 2.0, 3.0}


def _prepared_slab_search_config():
    """Return a prepared slab-search config plus (n_fixed, n_full)."""
    from scgo.surface.partition import prepare_slab_search_surface_config

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    cfg = main_mod.SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    return cfg, int(part.n_fixed), len(cfg.slab)


def _slab_adsorbate_candidate(cfg, n_fixed: int, n_full: int, *, migrate_sheet: bool):
    """Build a slab + OH candidate; optionally sink part of the mobile sheet."""
    from ase import Atoms

    atoms = cfg.slab.copy()
    pos = atoms.get_positions()
    if migrate_sheet:
        pos[n_fixed : n_fixed + 2, 2] -= 7.5
        atoms.set_positions(pos)
    anchor = pos[n_full - 1]
    atoms += Atoms(
        "OH",
        positions=[
            [anchor[0], anchor[1], anchor[2] + 1.6],
            [anchor[0], anchor[1], anchor[2] + 2.56],
        ],
    )
    return atoms


def test_final_structural_gate_honors_n_slab_deposit():
    """Backstop gate drops a migrated search-mobile sheet (G1) like GA storage.

    With the frozen-prefix ``n_slab_deposit`` supplied, sheet atoms sinking
    below the fixed-stack top violate the penetration boundary and the
    candidate is dropped. Without it (the old behavior) those atoms sit inside
    the unchecked full-slab prefix and the candidate is kept.
    """
    cfg, n_fixed, n_full = _prepared_slab_search_config()

    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    gate_kwargs = {
        "system_type": "surface_adsorbate",
        "adsorbate_definition": ads_def,
    }
    good = (-0.1, _slab_adsorbate_candidate(cfg, n_fixed, n_full, migrate_sheet=False))
    bad = (0.1, _slab_adsorbate_candidate(cfg, n_fixed, n_full, migrate_sheet=True))

    kept_with_deposit = main_mod._gate_structurally_valid_candidates(
        [good, bad],
        "surface_adsorbate",
        cfg,
        n_full,
        gate_kwargs,
        None,
        n_slab_deposit=n_fixed,
    )
    assert sorted(e for e, _ in kept_with_deposit) == [-0.1]

    kept_without_deposit = main_mod._gate_structurally_valid_candidates(
        [good, bad],
        "surface_adsorbate",
        cfg,
        n_full,
        gate_kwargs,
        None,
        n_slab_deposit=None,
    )
    assert sorted(e for e, _ in kept_without_deposit) == [-0.1, 0.1]


def test_bh_surface_gates_accept_conforming_trials(tmp_path, rng):
    """BH smoke on a slab-search surface: conforming trials pass the gates."""
    from ase.calculators.emt import EMT

    from scgo.algorithms.basinhopping_go import bh_go

    cfg, n_fixed, _n_full = _prepared_slab_search_config()
    atoms = cfg.slab.copy()
    atoms.calc = EMT()

    minima = bh_go(
        atoms,
        output_dir=str(tmp_path / "bh_surface"),
        niter=1,
        temperature=0.0,
        dr=0.15,
        niter_local_relaxation=3,
        system_type="surface",
        surface_config=cfg,
        n_slab=n_fixed,
        verbosity=0,
        rng=rng,
    )
    assert len(minima) >= 1
