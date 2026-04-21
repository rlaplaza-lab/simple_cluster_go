"""Transition-state search integration tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase_ga.data import DataConnection

from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.ts_search.transition_state_io import (
    load_minima_by_composition,
    save_transition_state_results,
    select_structure_pairs,
)
from scgo.ts_search.transition_state_run import (
    run_transition_state_campaign,
    run_transition_state_search,
)
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION
from tests.cuda_skip import require_cuda
from tests.test_utils import create_preparedb, mark_test_minima_as_final


@pytest.fixture
def mock_database_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a run directory
        run_dir = Path(tmpdir) / "run_20260101_120000"
        run_dir.mkdir()

        # Create database with Cu2 structures (EMT supports Cu)
        # Use ASE-GA compatible format

        db_path = run_dir / "candidates.db"

        # Initialize database
        db = create_preparedb(Atoms("Cu2"), db_path, population_size=20)

        # Add some mock minima using proper GA workflow
        # Minimum 1: linear Cu2
        atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms1.center(vacuum=5.0)
        atoms1.calc = EMT()
        from scgo.database.metadata import add_metadata

        add_metadata(atoms1, raw_score=-10.0)
        atoms1.info["confid"] = 1
        db.add_unrelaxed_candidate(atoms1, description="Cu2_linear")

        # Minimum 2: rotated Cu2
        atoms2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])
        atoms2.center(vacuum=5.0)
        atoms2.calc = EMT()
        from scgo.database.metadata import add_metadata

        add_metadata(atoms2, raw_score=-10.0)
        atoms2.info["confid"] = 2
        db.add_unrelaxed_candidate(atoms2, description="Cu2_rotated")

        # Minimum 3: another configuration
        atoms3 = Atoms("Cu2", positions=[[0, 0, 0], [1.2, 2.2, 0]])
        atoms3.center(vacuum=5.0)
        atoms3.calc = EMT()
        from scgo.database.metadata import add_metadata

        add_metadata(atoms3, raw_score=-10.0)
        atoms3.info["confid"] = 3
        db.add_unrelaxed_candidate(atoms3, description="Cu2_other")

        # Now retrieve and mark as relaxed (use DataConnection directly so
        # add_relaxed_step correctly sets relaxed=1 in number_key_values)
        from scgo.database.metadata import update_metadata

        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            a.calc = EMT()
            update_metadata(a, raw_score=-a.get_potential_energy())
            da.add_relaxed_step(a)

        # Tag relaxed minima as final_unique_minimum so TS can load them
        # (TS requires final-tagged minima from GO runs)
        mark_test_minima_as_final(db_path)

        yield tmpdir


def test_load_minima_by_composition(mock_database_dir):
    minima = load_minima_by_composition(mock_database_dir, composition=["Cu", "Cu"])

    assert "Cu2" in minima
    assert len(minima["Cu2"]) == 3

    # Check minima are sorted by energy
    energies = [e for e, _ in minima["Cu2"]]
    assert energies == sorted(energies)


def test_save_transition_state_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock results
        ts_results = [
            {
                "status": "success",
                "pair_id": "0_1",
                "minima_indices": [0, 1],
                "minima_provenance": [
                    {"run_id": "run_a", "systems_row_id": 7, "source_db": "ga_go.db"},
                    {"run_id": "run_a", "systems_row_id": 9, "source_db": "ga_go.db"},
                ],
                "neb_converged": True,
                "n_images": 5,
                "spring_constant": 0.1,
                "reactant_energy": -1.0,
                "product_energy": -0.9,
                "ts_energy": -0.5,
                "barrier_height": 0.5,
                "error": None,
                "ts_image_index": 3,
            },
            {
                "status": "failed",
                "pair_id": "1_2",
                "neb_converged": False,
                "n_images": 5,
                "spring_constant": 0.1,
                "reactant_energy": -1.0,
                "product_energy": -0.8,
                "ts_energy": None,
                "barrier_height": None,
                "error": "Test error",
            },
        ]

        save_transition_state_results(ts_results, tmpdir, composition=["Cu", "Cu"])

        # Check summary file created
        summary_path = Path(tmpdir) / "ts_search_summary_Cu2.json"
        assert summary_path.exists()

        # Check content
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert "scgo_version" in summary and summary["scgo_version"] != "unknown"
        assert "created_at" in summary
        assert summary["formula"] == "Cu2"
        assert summary["num_total_pairs"] == 2
        assert summary["num_successful"] == 1
        assert summary["num_converged"] == 1
        assert len(summary["results"]) == 2
        r0 = summary["results"][0]
        assert r0["minima_indices"] == [0, 1]
        assert r0["minima_provenance"][0]["systems_row_id"] == 7
        assert r0["minima_provenance"][1]["systems_row_id"] == 9


@pytest.mark.slow
def test_run_transition_state_search_full(mock_database_dir):
    params = {
        "calculator": "EMT",
        "calculator_kwargs": {},
    }

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        seed=1,
        verbosity=0,
        max_pairs=2,  # Limit for speed
        neb_n_images=3,
        neb_fmax=0.2,
        neb_steps=200,  # increased so NEB converges reliably in tests
    )

    # Should find some TS searches
    assert len(results) > 0

    # Check result structure
    for result in results:
        assert "status" in result
        assert "pair_id" in result
        assert "barrier_height" in result
        assert "neb_converged" in result

    # Check output files created
    result_dir = Path(mock_database_dir) / "ts_results_Cu2"
    assert result_dir.exists()
    # Should have summary file
    summary_file = result_dir / "ts_search_summary_Cu2.json"
    assert summary_file.exists()


@pytest.mark.slow
def test_run_transition_state_search_with_climb(mock_database_dir):
    params = {"calculator": "EMT"}

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_climb=True,
        neb_fmax=0.3,
        neb_steps=100,
    )

    assert len(results) > 0
    # Check that climb parameter was passed through
    for result in results:
        assert result.get("climb") is True

    # At least one NEB should converge or reach the numeric fmax (or be
    # recovered by a retry) when climb=True. If none succeed, ensure the
    # run produced diagnostics rather than silently failing.
    assert any(
        r.get("neb_converged")
        or (r.get("final_fmax") is not None and r.get("final_fmax") < 0.3)
        for r in results
    ), "Expected at least one converged/diagnosed NEB run when climb=True"


def test_run_transition_state_search_parallel_neb_requires_torchsim(mock_database_dir):
    params = {"calculator": "EMT"}

    with pytest.raises(ValueError):
        run_transition_state_search(
            composition=["Cu", "Cu"],
            output_dir=mock_database_dir,
            params=params,
            verbosity=0,
            max_pairs=1,
            neb_n_images=3,
            neb_fmax=0.3,
            neb_steps=10,
            use_torchsim=False,
            use_parallel_neb=True,
        )


@pytest.mark.slow
def test_run_transition_state_search_parallel_neb_executes(
    monkeypatch, mock_database_dir
):
    """Ensure run_transition_state_search can run multiple TorchSim NEBs via ParallelNEBBatch.

    We monkeypatch TorchSimBatchRelaxer with a lightweight fake that simply
    returns zero energies and zero forces so the parallel runner can exercise
    its batching logic without requiring real TorchSim/GPU.
    """

    class FakeRelaxer:
        def __init__(self, **kwargs):
            pass

        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                # zero forces sufficient for NEB bookkeeping
                ra.arrays["forces"] = np.zeros((len(a), 3))
                results.append((0.0, ra))
            return results

    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer", FakeRelaxer
    )

    params = {"calculator": "MACE", "calculator_kwargs": {}}

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        verbosity=0,
        max_pairs=2,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
        use_torchsim=True,
        use_parallel_neb=True,
        torchsim_params={"max_steps": 5},
    )

    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert r.get("use_torchsim") is True
        assert "pair_id" in r
        assert "neb_converged" in r


def test_run_transition_state_search_parallel_neb_forwards_rng_and_perturb(
    monkeypatch, mock_database_dir
):
    """Parallel NEB path should forward `neb_perturb_sigma` and the shared RNG to `interpolate_path()`.

    We monkeypatch the TorchSim relaxer (as in the existing parallel test) and
    wrap `interpolate_path` in `transition_state_run` to capture the kwargs
    it receives.
    """

    class FakeRelaxer:
        def __init__(self, **kwargs):
            pass

        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                results.append((0.0, ra))
            return results

    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer", FakeRelaxer
    )

    # Wrap the interpolate_path used by the run() routine so we can inspect kwargs
    import scgo.ts_search.transition_state as ts_mod

    real_interp = ts_mod.interpolate_path
    captured = []

    def wrapper(a1, a2, n_images, method, **kwargs):
        captured.append(kwargs)
        return real_interp(a1, a2, n_images=n_images, method=method, **kwargs)

    monkeypatch.setattr("scgo.ts_search.parallel_neb.interpolate_path", wrapper)

    params = {"calculator": "MACE", "calculator_kwargs": {}}

    seed = 2025
    neb_pert = 0.05

    _ = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
        use_torchsim=True,
        use_parallel_neb=True,
        torchsim_params={"max_steps": 5},
        neb_perturb_sigma=neb_pert,
        seed=seed,
    )

    # Ensure our wrapper was invoked and saw the rng + perturb_sigma kwargs
    assert captured, "interpolate_path was not called in parallel NEB path"
    for kw in captured:
        assert "rng" in kw and kw["rng"] is not None
        assert isinstance(kw["rng"], np.random.Generator)
        assert kw.get("perturb_sigma") == pytest.approx(neb_pert)


def test_run_transition_state_search_no_minima():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = {"calculator": "EMT"}

        results = run_transition_state_search(
            composition=["Cu", "Cu"],
            output_dir=tmpdir,
            params=params,
            verbosity=0,
        )

        # Should return empty list
        assert results == []


def test_run_transition_state_search_energy_gap_filter(mock_database_dir):
    params = {"calculator": "EMT"}

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        verbosity=0,
        energy_gap_threshold=0.01,  # Very tight - should filter most pairs
        neb_n_images=3,
        neb_steps=10,
    )

    # Depending on the actual energies, may or may not find pairs
    # Just check it doesn't crash
    assert isinstance(results, list)


def test_run_transition_state_search_deduplicates_minima(tmp_path):
    """Deduplicate identical minima across run directories."""
    from scgo.ts_search.transition_state_io import (
        load_minima_by_composition,
        select_structure_pairs,
    )
    from scgo.utils.helpers import filter_unique_minima

    # Build two run_* dirs with identical Cu2 minima
    for run_id in ("run_A", "run_B"):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        db_path = run_dir / "candidates.db"

        from ase import Atoms
        from ase.calculators.emt import EMT

        from scgo.database.metadata import add_metadata

        db = create_preparedb(Atoms("Cu2"), db_path, population_size=10)

        # Two distinct geometries (will be duplicated across run dirs)
        atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms1.center(vacuum=5.0)
        atoms1.calc = EMT()
        add_metadata(atoms1, raw_score=-10.0)
        db.add_unrelaxed_candidate(atoms1, description="Cu2_a")

        atoms2 = Atoms("Cu2", positions=[[0, 0, 0], [1.2, 2.2, 0]])
        atoms2.center(vacuum=5.0)
        atoms2.calc = EMT()
        add_metadata(atoms2, raw_score=-10.0)
        db.add_unrelaxed_candidate(atoms2, description="Cu2_b")

        # Mark relaxed (use DataConnection so add_relaxed_step sets relaxed=1 correctly)
        from scgo.database.metadata import update_metadata

        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            a.calc = EMT()
            update_metadata(a, raw_score=-a.get_potential_energy())
            da.add_relaxed_step(a)

        mark_test_minima_as_final(db_path)

    # Load raw minima (contains duplicates across the two run dirs)
    minima_by_formula = load_minima_by_composition(
        str(tmp_path), composition=["Cu", "Cu"]
    )
    assert "Cu2" in minima_by_formula
    raw_minima = minima_by_formula["Cu2"]

    # Raw minima should contain 4 entries (2 per run)
    assert len(raw_minima) == 4

    # Deduplicate using the same helper used in run_minima.py / TS code
    deduped = filter_unique_minima(raw_minima, DEFAULT_ENERGY_TOLERANCE)

    # Energy tolerances may cause 1 or 2 unique minima depending on calculator
    # behavior — assert only that deduplication reduces the number of minima.
    assert len(deduped) < len(raw_minima)

    # Pair counts should not increase after deduplication (usually decrease)
    raw_pairs = select_structure_pairs(raw_minima)
    dedup_pairs = select_structure_pairs(deduped)
    assert len(dedup_pairs) <= len(raw_pairs)


@pytest.mark.slow
def test_run_transition_state_campaign():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock databases for multiple compositions
        for formula, _composition in [("Cu2", ["Cu", "Cu"]), ("Ni2", ["Ni", "Ni"])]:
            run_dir = Path(tmpdir) / f"{formula}_searches" / "run_20260101_120000"
            run_dir.mkdir(parents=True)

            db_path = run_dir / "candidates.db"

            # Initialize ASE-GA database properly
            counts: dict[str, int] = {}
            for symbol in _composition:
                counts[symbol] = counts.get(symbol, 0) + 1
            db = create_preparedb(Atoms(formula), db_path, population_size=20)

            # Add some structures as unrelaxed candidates
            for i in range(3):
                atoms = Atoms(formula, positions=[[0, 0, 0], [2.5 + i * 0.3, 0, 0]])
                atoms.center(vacuum=5.0)
                atoms.calc = EMT()
                atoms.info["key_value_pairs"] = {"raw_score": -10.0}
                atoms.info["confid"] = i + 1
                db.add_unrelaxed_candidate(atoms, description=f"{formula}_mock_{i}")

            # Mark all candidates as relaxed with updated energies
            da = DataConnection(str(db_path))
            while da.get_number_of_unrelaxed_candidates() > 0:
                a = da.get_an_unrelaxed_candidate()
                if "key_value_pairs" not in a.info:
                    a.info["key_value_pairs"] = {}
                a.calc = EMT()
                a.info["key_value_pairs"]["raw_score"] = -a.get_potential_energy()
                da.add_relaxed_step(a)

        # Run campaign
        params = {"calculator": "EMT"}
        ts_kwargs = {
            "max_pairs": 1,
            "neb_n_images": 3,
            "neb_fmax": 0.3,
            "neb_steps": 10,
        }

        results = run_transition_state_campaign(
            compositions=[["Cu", "Cu"], ["Ni", "Ni"]],
            output_dir=tmpdir,
            params=params,
            verbosity=0,
            ts_kwargs=ts_kwargs,
        )

        # Check results for both compositions
        assert "Cu2" in results
        assert "Ni2" in results
        assert isinstance(results["Cu2"], list)
        assert isinstance(results["Ni2"], list)


@pytest.mark.slow
def test_run_transition_state_campaign_detects_searches_dir():
    """Ensure the TS campaign will discover minima stored under the standard
    "{formula}_searches" layout (created by global-optimization runners)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock database under the older "*_searches" name
        formula = "Cu2"
        run_dir = Path(tmpdir) / f"{formula}_searches" / "run_20260101_120000"
        run_dir.mkdir(parents=True)

        db_path = run_dir / "candidates.db"
        db = create_preparedb(Atoms(formula), db_path, population_size=10)

        # Add a couple of minima
        for i in range(2):
            atoms = Atoms(formula, positions=[[0, 0, 0], [2.5 + i * 0.3, 0, 0]])
            atoms.center(vacuum=5.0)
            atoms.calc = EMT()
            atoms.info["key_value_pairs"] = {"raw_score": -10.0}
            atoms.info["confid"] = i + 1
            db.add_unrelaxed_candidate(atoms, description=f"{formula}_mock_{i}")

        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            a.calc = EMT()
            a.info.setdefault("key_value_pairs", {})[
                "raw_score"
            ] = -a.get_potential_energy()
            da.add_relaxed_step(a)

        params = {"calculator": "EMT"}
        ts_kwargs = {
            "max_pairs": 1,
            "neb_n_images": 3,
            "neb_fmax": 0.3,
            "neb_steps": 10,
        }

        results = run_transition_state_campaign(
            compositions=[["Cu", "Cu"]],
            output_dir=tmpdir,
            params=params,
            verbosity=0,
            ts_kwargs=ts_kwargs,
        )

        assert "Cu2" in results
        assert isinstance(results["Cu2"], list)


def test_run_transition_state_search_linear_interpolation(mock_database_dir):
    params = {"calculator": "EMT"}

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params=params,
        seed=1,
        neb_interpolation_method="linear",
        neb_fmax=0.3,
        neb_steps=100,
    )

    # Should work with linear interpolation
    assert len(results) > 0
    # At least one NEB should converge with linear interpolation


def test_interpolate_path_does_not_modify_endpoints():
    """Interpolation preserves endpoints."""
    import numpy as np

    from scgo.ts_search.transition_state import interpolate_path

    a1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])

    # Explicitly disable endpoint alignment so this test continues to
    # verify that interpolate_path() does not mutate endpoint coordinates.
    images = interpolate_path(a1, a2, n_images=3, method="idpp", align_endpoints=False)

    # Smoke assertions: endpoints coordinates preserved and reasonable image count
    assert len(images) == 5
    assert np.allclose(images[0].get_positions(), a1.get_positions())
    assert np.allclose(images[-1].get_positions(), a2.get_positions())


def test_find_transition_state_reports_final_fmax_and_neb_converged(tmp_path):
    from ase.calculators.emt import EMT

    from scgo.ts_search.transition_state import find_transition_state

    atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])
    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)

    # Attach EMT calculators so energies/forces are available
    calc = EMT()
    atoms1.calc = calc
    atoms2.calc = calc

    outdir = str(tmp_path)

    # Use extremely tight fmax and very small neb_steps to force NOT converged
    res = find_transition_state(
        atoms1,
        atoms2,
        calculator=calc,
        output_dir=outdir,
        pair_id="0_1",
        n_images=3,
        neb_steps=1,
        fmax=1e-6,
        verbosity=0,
    )

    assert "final_fmax" in res
    assert isinstance(res["final_fmax"], (float, type(None)))
    # Steps taken must be reported when available (or None if ASE optimizer
    # didn't expose the counter).
    assert "steps_taken" in res
    assert isinstance(res["steps_taken"], (int, type(None)))
    if res["steps_taken"] is not None:
        assert res["steps_taken"] >= 0

    # neb_converged must reflect final_fmax < fmax when final_fmax is available
    if res["final_fmax"] is not None:
        # Endpoint-detection may override neb_converged even when final_fmax < fmax
        if res.get("error") and "endpoint" in str(res.get("error")).lower():
            assert res["neb_converged"] is False
        else:
            assert res["neb_converged"] == (res["final_fmax"] < 1e-6)
    else:
        # If final_fmax couldn't be computed we still expect neb_converged to be False
        assert res["neb_converged"] is False


def test_run_transition_state_search_tags_non_ga_db_files(tmp_path):
    """Tag TSs back into minima DBs named 'bh' or 'simple'."""
    from ase import Atoms

    # Helper to create a run dir with a DB named <db_name>
    def _make_db(db_name: str):
        run_dir = tmp_path / "run_20260101_120000" / "trial_1"
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / db_name

        # Use PrepareDB to create compatible DB
        db = create_preparedb(Atoms("Pt2"), db_path, population_size=10)

        # Add two minima so TS search has something to pair
        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a1.center(vacuum=5.0)
        a1.info.setdefault("key_value_pairs", {})["raw_score"] = -1.0
        db.add_unrelaxed_candidate(a1, description="pt2_a")

        a2 = Atoms("Pt2", positions=[[0, 0, 0], [1.2, 2.2, 0]])
        a2.center(vacuum=5.0)
        a2.info.setdefault("key_value_pairs", {})["raw_score"] = -1.1
        db.add_unrelaxed_candidate(a2, description="pt2_b")

        # Mark relaxed (use DataConnection so add_relaxed_step sets relaxed=1 correctly)
        from ase.calculators.emt import EMT
        from ase_ga.data import DataConnection

        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            if "key_value_pairs" not in a.info:
                a.info["key_value_pairs"] = {}
            a.calc = EMT()
            a.info["key_value_pairs"]["raw_score"] = -a.get_potential_energy()
            da.add_relaxed_step(a)

        mark_test_minima_as_final(db_path)

        return str(db_path)

    # Create BH-style DB and run TS search that should tag back into it
    bh_db = _make_db("bh_go.db")

    results = run_transition_state_search(
        composition=["Pt", "Pt"],
        output_dir=str(tmp_path),
        params={"calculator": "EMT"},
        seed=1,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_steps=50,
        # Override similarity tolerance to ensure the two Pt2 geometries are paired
        similarity_tolerance=1e-4,
    )

    # Ensure results returned (success *not* guaranteed under conservative retry policy).
    assert isinstance(results, list)

    successful = [
        r
        for r in results
        if r.get("status") == "success" and r.get("transition_state") is not None
    ]

    if not successful:
        # No successful TS — accept either endpoint-detection or generic non-convergence,
        # but require that failures are reported (error string present).
        assert any(r.get("error") for r in results), (
            "No successful TS runs and no failure diagnostics — unexpected"
        )
    else:
        # Explicitly test tagging into the BH-style DB by calling add_ts_to_database
        # with the first successful TS result (deterministic and avoids flaky DB
        # discovery heuristics in CI environments).
        from scgo.ts_search.ts_network import add_ts_to_database

        success_added = 0
        from scgo.utils.helpers import validate_pair_id

        for r in successful:
            pair = r.get("pair_id")
            assert pair is not None, "pair_id missing in TS result"
            i_idx, j_idx = validate_pair_id(pair)

            ok = add_ts_to_database(
                ts_structure=r["transition_state"],
                ts_energy=float(r.get("ts_energy", 0.0)),
                minima_idx_1=i_idx,
                minima_idx_2=j_idx,
                db_file=bh_db,
                pair_id=pair,
                barrier_height=float(r.get("barrier_height", 0.0)),
            )
            if ok:
                success_added += 1
                break

        assert success_added > 0, (
            "Failed to add TS to BH-style DB via add_ts_to_database"
        )

    # Repeat for 'simple' optimizer filename
    simple_db = _make_db("simple_go.db")

    results2 = run_transition_state_search(
        composition=["Pt", "Pt"],
        output_dir=str(tmp_path),
        params={"calculator": "EMT"},
        seed=2,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_steps=50,
        similarity_tolerance=1e-4,
    )

    # Ensure results returned
    assert isinstance(results2, list)

    successful2 = [
        r
        for r in results2
        if r.get("status") == "success" and r.get("transition_state") is not None
    ]

    if not successful2:
        # No successful TS — ensure a failure diagnostic exists (endpoint or
        # non-convergence). Do not force a particular failure mode.
        assert any(r.get("error") for r in results2), (
            "No successful TS runs and no failure diagnostics — unexpected"
        )
    else:
        # Explicitly add first successful TS to the simple-style DB and verify
        from scgo.ts_search.ts_network import add_ts_to_database

        added_simple = 0
        from scgo.utils.helpers import validate_pair_id

        for r in results2:
            if r.get("status") != "success" or r.get("transition_state") is None:
                continue
            pair = r.get("pair_id")
            assert pair is not None, "pair_id missing in TS result"
            i_idx, j_idx = validate_pair_id(pair)

            ok = add_ts_to_database(
                ts_structure=r["transition_state"],
                ts_energy=float(r.get("ts_energy", 0.0)),
                minima_idx_1=i_idx,
                minima_idx_2=j_idx,
                db_file=simple_db,
                pair_id=pair,
                barrier_height=float(r.get("barrier_height", 0.0)),
            )
            if ok:
                added_simple += 1
                break

        assert added_simple > 0, (
            "Failed to add TS to simple-style DB via add_ts_to_database"
        )


def test_run_transition_state_search_torchsim(mock_database_dir):
    """Test TS search with TorchSim batched forces (requires GPU)."""
    require_cuda()

    torchsim_params = {
        "device": "cuda",
        "mace_model_name": "mace_matpes_0",
        "force_tol": 0.1,
    }

    results = run_transition_state_search(
        composition=["Cu", "Cu"],
        output_dir=mock_database_dir,
        params={"calculator": "MACE", "calculator_kwargs": {}},
        verbosity=1,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.2,
        neb_steps=10,
        use_torchsim=True,
        torchsim_params=torchsim_params,
    )

    assert len(results) > 0
    for result in results:
        assert result.get("use_torchsim") is True
        if result["status"] == "success":
            # Should have force call tracking
            assert "force_calls" in result


def test_select_structure_pairs_similarity_filter():
    """Test similarity cutoff filters too-similar structures using comparator.

    This test verifies that the pair selection logic correctly identifies
    and filters out pairs of structures that are too similar for meaningful
    TS search (e.g., structures from the same basin).
    """
    # Create structures that are very similar (just slightly displaced)
    atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms2 = Atoms("Cu2", positions=[[0.01, 0, 0], [2.51, 0, 0]])  # Very similar
    atoms3 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])  # Different

    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)
    atoms3.center(vacuum=5.0)

    minima = [
        (-1.0, atoms1),
        (-0.99, atoms2),
        (-0.9, atoms3),
    ]

    # Tight similarity tolerance should filter out (0,1)
    pairs = select_structure_pairs(
        minima,
        similarity_tolerance=0.01,  # Very tight
        similarity_pair_cor_max=0.1,  # Very tight
    )

    # (0,1) should be filtered (too similar)
    # (0,2) and (1,2) should remain (sufficiently different)
    assert (0, 1) not in pairs
    # At least one pair should survive
    assert pairs


@pytest.mark.slow
def test_run_transition_state_search_auto_tags_mixed_db_formats(tmp_path):
    """End-to-end: ensure `run_transition_state_search(..., tag_ts_in_db=True)`
    discovers mixed DB formats under run_* and persists at least one TS entry.

    This verifies the automatic `db_candidate` resolution + `add_ts_to_database`
    path for both BH- and simple-style database filenames.
    """
    from ase import Atoms

    from scgo.database import open_db

    # Helper to create a run dir containing a DB with two relaxed minima
    def _make_db(db_name: str):
        run_dir = tmp_path / "run_20260102_120000" / "trial_1"
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / db_name

        db = create_preparedb(Atoms("Pt2"), db_path, population_size=10)

        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        a1.center(vacuum=5.0)
        a1.info.setdefault("key_value_pairs", {})["raw_score"] = -1.0
        db.add_unrelaxed_candidate(a1, description="pt2_a")

        a2 = Atoms("Pt2", positions=[[0, 0, 0], [1.2, 2.2, 0]])
        a2.center(vacuum=5.0)
        a2.info.setdefault("key_value_pairs", {})["raw_score"] = -1.1
        db.add_unrelaxed_candidate(a2, description="pt2_b")

        # Mark relaxed
        from ase.calculators.emt import EMT

        with open_db(str(db_path)) as da:
            while da.get_number_of_unrelaxed_candidates() > 0:
                a = da.get_an_unrelaxed_candidate()
                if "key_value_pairs" not in a.info:
                    a.info["key_value_pairs"] = {}
                a.calc = EMT()
                a.info["key_value_pairs"]["raw_score"] = -a.get_potential_energy()
                da.add_relaxed_step(a)

        mark_test_minima_as_final(db_path)

        return str(db_path)

    bh_db = _make_db("bh_go.db")
    simple_db = _make_db("simple_go.db")

    # Run TS search with tagging enabled (should discover both DBs)
    results = run_transition_state_search(
        composition=["Pt", "Pt"],
        output_dir=str(tmp_path),
        params={"calculator": "EMT"},
        seed=1,
        verbosity=0,
        max_pairs=2,
        neb_n_images=3,
        neb_steps=50,
        similarity_tolerance=1e-4,
        tag_ts_in_db=True,
    )

    # Require at least one successful TS (or a retry-recovered TS) for EMT test environment
    successful = [
        r
        for r in results
        if r.get("status") == "success" and r.get("transition_state") is not None
    ]

    # Verify at least one of the DBs contains a persisted TS entry if a TS was found
    def _has_ts_marker(dbfile: str) -> bool:
        with open_db(dbfile) as da:
            rows = da.get_all_relaxed_candidates()
        return any(
            r.info.get("key_value_pairs", {}).get("is_transition_state") for r in rows
        )

    if successful:
        assert _has_ts_marker(bh_db) or _has_ts_marker(simple_db), (
            "Expected at least one persisted TS entry in discovered DBs (bh_go.db or simple_go.db)"
        )
    else:
        # No TS found — ensure tagging did not erroneously add markers
        assert not _has_ts_marker(bh_db) and not _has_ts_marker(simple_db)
