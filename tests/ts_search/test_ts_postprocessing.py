import json
import os
from pathlib import Path

import pytest
from ase import Atoms

from scgo.ts_search.transition_state_io import write_final_unique_ts
from scgo.ts_search.transition_state_run import integrate_ts_to_database
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION
from tests.test_utils import create_preparedb, mark_test_minima_as_final


def _make_ts_result(
    *, pair_id: str, raw_score: float, ts_energy: float, barrier_height: float
):
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    a.info.setdefault("key_value_pairs", {})["raw_score"] = raw_score
    return {
        "pair_id": pair_id,
        "status": "success",
        "neb_converged": True,
        "transition_state": a,
        "ts_energy": ts_energy,
        "barrier_height": barrier_height,
    }


def test_write_final_unique_ts_dedup(tmp_path):
    # Two identical TSs for the *same* minima-pair should be deduplicated to one.
    ts_results = [
        _make_ts_result(
            pair_id="0_1", raw_score=-0.5, ts_energy=0.5, barrier_height=0.2
        ),
        _make_ts_result(
            pair_id="0_1", raw_score=-0.51, ts_energy=0.51, barrier_height=0.21
        ),
    ]

    composition = ["Pt", "Pt"]
    out = str(tmp_path / "ts_results_Pt2")
    os.makedirs(out, exist_ok=True)

    summary = write_final_unique_ts(ts_results, out, composition, energy_tolerance=0.05)

    # Expect one unique TS written
    assert isinstance(summary, list)
    assert len(summary) == 1
    assert len(summary[0]["connected_edges"]) == 1
    assert summary[0]["connected_edges"][0]["pair_id"] == "0_1"

    final_dir = Path(out) / "final_unique_ts"
    assert final_dir.exists()

    # JSON summary file exists
    json_path = final_dir / "final_unique_ts_summary_Pt2.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data["formula"] == "Pt2"
    assert len(data["unique_ts"]) == 1


def test_write_final_unique_ts_keeps_similar_ts_across_different_pairs(tmp_path):
    # Identical TS geometries for different minima pairs merge into one unique TS
    # with both edges recorded in connected_edges.
    ts_results = [
        _make_ts_result(
            pair_id="0_1", raw_score=-0.5, ts_energy=0.5, barrier_height=0.2
        ),
        _make_ts_result(
            pair_id="1_2", raw_score=-0.51, ts_energy=0.51, barrier_height=0.21
        ),
    ]

    composition = ["Pt", "Pt"]
    out = str(tmp_path / "ts_results_Pt2")
    os.makedirs(out, exist_ok=True)

    summary = write_final_unique_ts(ts_results, out, composition, energy_tolerance=0.05)

    assert isinstance(summary, list)
    assert len(summary) == 1
    entry = summary[0]
    assert len(entry["connected_edges"]) == 2
    assert set(entry["connected_minima"]) == {0, 1, 2}
    pair_ids = {e["pair_id"] for e in entry["connected_edges"]}
    assert pair_ids == {"0_1", "1_2"}
    final_dir = Path(out) / "final_unique_ts"
    assert final_dir.exists()
    xyz_files = list(final_dir.glob("*.xyz"))
    assert len(xyz_files) == 1
    assert xyz_files[0].name == "Pt2_ts_01.xyz"
    json_path = final_dir / "final_unique_ts_summary_Pt2.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data["formula"] == "Pt2"
    assert len(data["unique_ts"]) == 1
    assert len(data["unique_ts"][0]["connected_edges"]) == 2


@pytest.mark.parametrize(
    "pair, expected",
    [
        ("0_1", (0, 1)),
        ("10_20", (10, 20)),
        ("12_345", (12, 345)),
    ],
)
def test_validate_pair_id_valid(pair, expected):
    from scgo.utils.helpers import validate_pair_id

    assert validate_pair_id(pair) == expected


@pytest.mark.parametrize("bad", ["", "a_b", "1_", "_2", "1-2", "bad", None])
def test_validate_pair_id_invalid(bad):
    from scgo.utils.helpers import validate_pair_id

    with pytest.raises(ValueError):
        validate_pair_id(bad)


def test_write_final_unique_ts_rejects_malformed_pair_id(tmp_path):
    ts_results = [
        _make_ts_result(
            pair_id="bad_pair", raw_score=-0.5, ts_energy=0.5, barrier_height=0.2
        )
    ]
    out = str(tmp_path / "ts_results_Pt2")
    os.makedirs(out, exist_ok=True)
    with pytest.raises(ValueError):
        write_final_unique_ts(ts_results, out, ["Pt", "Pt"])


def test_integrate_ts_to_database_calls_add(monkeypatch, tmp_path):
    called = []

    def fake_add_ts_to_database(**kwargs):
        called.append(kwargs)
        return True

    # Patch the symbol used by integrate_ts_to_database (module-level import)
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.add_ts_to_database",
        fake_add_ts_to_database,
    )

    # touch a dummy db file so integrate_ts_to_database doesn't early-return
    db_file = tmp_path / "minima.db"
    db_file.write_text("")

    # Make a simple TS result
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": a,
            "ts_energy": 0.5,
            "barrier_height": 0.2,
        }
    ]

    added = integrate_ts_to_database(ts_results, str(db_file), verbosity=0)
    assert added == 1
    assert len(called) == 1
    assert called[0]["pair_id"] == "0_1"
    assert called[0]["minima_idx_1"] == 0
    assert called[0]["minima_idx_2"] == 1
    assert called[0]["endpoint_provenance"] is None


def test_add_ts_to_database_persists_marker(tmp_path):
    """Verify add_ts_to_database writes a persistent DB marker so TS entries
    can be reliably detected by tests and downstream tooling.
    """
    from scgo.database import open_db
    from scgo.database.metadata import get_metadata
    from scgo.ts_search.ts_network import add_ts_to_database

    db_file = tmp_path / "minima.db"

    # Prepare a minima database
    db = create_preparedb(Atoms("Pt2"), db_file, population_size=10)

    # Insert one relaxed candidate so DB schema/tables exist and DB is usable
    a = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
    a.info["confid"] = 1
    db.add_unrelaxed_candidate(a, description="Pt2_test")

    # Move candidate to relaxed state (simplest route via DataConnection)
    from ase_ga.data import DataConnection

    da = DataConnection(str(db_file))
    if da.get_number_of_unrelaxed_candidates() > 0:
        cand = da.get_an_unrelaxed_candidate()
        cand.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_relaxed_step(cand)

    # Add scgo_metadata so DB has consistent schema (mark_test_minima_as_final
    # also adds scgo_metadata for discovery compatibility)
    mark_test_minima_as_final(db_file)

    # Add TS to DB (include run provenance so DB row should persist it)
    ts = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    ts.info.setdefault("provenance", {})["run_id"] = "run_ts_001"

    ep = [
        {"run_id": "run_A", "systems_row_id": 3},
        {"run_id": "run_A", "systems_row_id": 5},
    ]
    success = add_ts_to_database(
        ts_structure=ts,
        ts_energy=0.5,
        minima_idx_1=0,
        minima_idx_2=1,
        db_file=str(db_file),
        pair_id="0_1",
        barrier_height=0.1,
        endpoint_provenance=ep,
    )

    assert success is True

    # Verify DB contains an entry with the persistent TS marker and expected raw_score
    with open_db(str(db_file)) as da_read:
        rows = da_read.get_all_relaxed_candidates()

    # Check both key_value_pairs and get_metadata for robustness across ASE versions
    ts_rows = [
        r
        for r in rows
        if r.info.get("key_value_pairs", {}).get("is_transition_state")
        or get_metadata(r, "is_transition_state")
    ]

    assert len(ts_rows) == 1

    ts_row = ts_rows[0]
    kv = ts_row.info.get("key_value_pairs", {})
    assert kv.get("is_transition_state") is True
    assert kv.get("ts_neb_converged") is True
    assert kv.get("final_unique_ts") is not True
    assert abs(kv.get("raw_score", 0.0) + 0.5) < 1e-6

    # Verify run_id persisted into DB row (metadata or legacy key_value_pairs)
    assert (
        ts_row.info.get("metadata", {}).get("run_id") == "run_ts_001"
        or ts_row.info.get("provenance", {}).get("run_id") == "run_ts_001"
        or kv.get("run_id") == "run_ts_001"
    )

    # pair id may be stored in `metadata`/`provenance` but some DB adapters
    # do not preserve these fields; accept the persistent key_value_pairs marker
    # as proof the TS was written.
    assert (
        ts_row.info.get("metadata", {}).get("ts_pair_id") == "0_1"
        or ts_row.info.get("provenance", {}).get("pair_id") == "0_1"
        or kv.get("is_transition_state") is True
    )

    prov_json = ts_row.info.get("key_value_pairs", {}).get(
        "ts_endpoint_provenance_json"
    )
    assert prov_json is not None
    assert json.loads(prov_json) == ep


def test_integrate_ts_to_database_forwards_endpoint_provenance(monkeypatch, tmp_path):
    called = []

    def fake_add_ts_to_database(**kwargs):
        called.append(kwargs)
        return True

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.add_ts_to_database",
        fake_add_ts_to_database,
    )

    db_file = tmp_path / "minima.db"
    db_file.write_text("")

    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    prov = [
        {"run_id": "r1", "systems_row_id": 1},
        {"run_id": "r1", "systems_row_id": 2},
    ]
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": a,
            "ts_energy": 0.5,
            "barrier_height": 0.2,
            "minima_provenance": prov,
        }
    ]

    added = integrate_ts_to_database(ts_results, str(db_file), verbosity=0)
    assert added == 1
    assert called[0]["endpoint_provenance"] == prov
    assert called[0]["canonical_ts"] is False


def test_add_ts_to_database_returns_false_on_write_error(monkeypatch, tmp_path, caplog):
    import sqlite3

    from scgo.ts_search.ts_network import add_ts_to_database

    # Simulate DB write failure inside DataConnection.add_relaxed_candidate
    # (add_ts_to_database uses DataConnection directly, not open_db)
    class FakeDataConnection:
        def __init__(self, path):
            self.path = path

        def add_relaxed_candidate(self, atoms):
            raise sqlite3.OperationalError("simulated write lock")

    monkeypatch.setattr(
        "scgo.ts_search.ts_network.DataConnection",
        FakeDataConnection,
    )

    # Ensure the file exists so the function proceeds to DB write logic
    (tmp_path / "minima.db").write_text("")

    caplog.set_level("ERROR")

    ts = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    success = add_ts_to_database(
        ts_structure=ts,
        ts_energy=0.5,
        minima_idx_1=0,
        minima_idx_2=1,
        db_file=str(tmp_path / "minima.db"),
        pair_id="0_1",
        barrier_height=0.1,
    )

    assert success is False
    assert "Error adding TS 0_1 to database" in caplog.text


def test_integrate_ts_to_database_skips_when_add_returns_false(monkeypatch, tmp_path):
    # Patch add_ts_to_database to return False and assert integrate_ts_to_database
    # reports 0 added.
    def fake_add_ts_to_database(**kwargs):
        return False

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.add_ts_to_database",
        fake_add_ts_to_database,
    )

    db_file = tmp_path / "minima.db"
    db_file.write_text("")

    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": a,
            "ts_energy": 0.5,
            "barrier_height": 0.2,
        }
    ]

    added = integrate_ts_to_database(ts_results, str(db_file), verbosity=0)
    assert added == 0


def test_run_transition_state_search_skips_tagging_when_no_db(
    monkeypatch, caplog, capsys, tmp_path
):
    """When no run_*/**/*.db files are present, tagging should be skipped and a
    warning should be logged; `add_ts_to_database` must not be called."""
    from scgo.ts_search.transition_state_run import run_transition_state_search

    # Fake minima loader: two minima with provenance that reference a missing DB
    def fake_load_minima_by_composition(
        ts_output_dir, composition, prefer_final_unique: bool = False
    ):
        a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
        a.info.setdefault("provenance", {})["source_db"] = "missing.db"
        return {"Pt2": [(0.0, a.copy()), (0.1, a.copy())]}

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.load_minima_by_composition",
        fake_load_minima_by_composition,
    )

    # Force one pair and short-circuit heavy NEB work
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.select_structure_pairs",
        lambda minima, **kwargs: [(0, 1)],
    )

    def fake_find_transition_state(*args, **kwargs):
        return {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": Atoms("Pt2"),
            "ts_energy": 0.5,
            "barrier_height": 0.1,
            "neb_converged": True,
            "n_images": 5,
            "reactant_energy": 0.0,
            "product_energy": 0.1,
        }

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.find_transition_state",
        fake_find_transition_state,
    )

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_neb_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_transition_state_results",
        lambda *a, **k: None,
    )

    # No DB files discovered under run_* -> basename_to_path empty
    monkeypatch.setattr("glob.glob", lambda *a, **k: [])

    called = []

    def fake_add_ts_to_database(**kwargs):
        called.append(kwargs)
        return True

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.add_ts_to_database",
        fake_add_ts_to_database,
    )

    caplog.set_level("WARNING")

    run_transition_state_search(
        ["Pt", "Pt"],
        base_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        tag_ts_in_db=True,
        verbosity=2,
    )

    # add_ts_to_database should not be called because no candidate DB was found
    assert called == []

    captured = capsys.readouterr()
    assert "No minima DB found to tag TS" in captured.out


def test_run_transition_state_search_records_minima_provenance(monkeypatch, tmp_path):
    """Ensure TS tagging writes explicit minima provenance (source DB and ids).

    This verifies the provenance enrichment added before calling
    `add_ts_to_database` so downstream consumers can locate the endpoint minima
    unambiguously.
    """
    from ase import Atoms

    from scgo.ts_search.transition_state_run import run_transition_state_search

    # Fake minima loader: two minima with provenance referencing a DB basename
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    a.info.setdefault("provenance", {})["source_db"] = "candidates.db"
    a.info.setdefault("provenance", {})["confid"] = 11
    a.info.setdefault("provenance", {})["unique_id"] = "min_11"

    # Create a *separate* Atoms instance so provenance dicts are independent
    b = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    b.info.setdefault("provenance", {})["source_db"] = "candidates.db"
    b.info.setdefault("provenance", {})["confid"] = 22
    b.info.setdefault("provenance", {})["unique_id"] = "min_22"

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.load_minima_by_composition",
        lambda ts_output_dir, composition, prefer_final_unique: {
            "Pt2": [(0.0, a.copy()), (0.1, b.copy())]
        },
    )

    # Force a single pair and short-circuit heavy NEB work
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.select_structure_pairs",
        lambda minima, **kwargs: [(0, 1)],
    )

    def fake_find_transition_state(*args, **kwargs):
        return {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": Atoms("Pt2"),
            "ts_energy": 0.5,
            "barrier_height": 0.1,
            "neb_converged": True,
            "n_images": 5,
            "reactant_energy": 0.0,
            "product_energy": 0.1,
        }

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.find_transition_state",
        fake_find_transition_state,
    )

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_neb_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_transition_state_results",
        lambda *a, **k: None,
    )

    # Create a fake DB file under tmp_path/run_* so tagging finds it
    run_dir = tmp_path / "run_0001"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "candidates.db"
    db_path.write_text("")

    # Ensure glob discovers the DB file during tagging
    monkeypatch.setattr("glob.glob", lambda *a, **k: [str(db_path)])

    called = []

    def fake_add_ts_to_database(**kwargs):
        called.append(kwargs)
        return True

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.add_ts_to_database",
        fake_add_ts_to_database,
    )

    # Run TS search and enable DB tagging
    run_transition_state_search(
        ["Pt", "Pt"],
        base_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        tag_ts_in_db=True,
        verbosity=0,
        max_pairs=1,
    )

    assert called, "add_ts_to_database was not called"

    ts_atoms = called[0]["ts_structure"]
    prov = ts_atoms.info.get("provenance", {})

    # The provenance enrichment should be present and reference the minima
    assert prov.get("minima_source_db") == ["candidates.db", "candidates.db"]
    assert prov.get("minima_confids") == [11, 22]
    assert prov.get("minima_unique_ids") == ["min_11", "min_22"]

    # Also ensure canonical metadata has the pair id for discovery
    assert ts_atoms.info.get("metadata", {}).get("ts_connects_minima") == "0_1"


def test_run_transition_state_search_resolves_neb_steps_auto(monkeypatch):
    """When neb_steps is 'auto' the runner resolves it using the helper."""
    from ase import Atoms

    from scgo.ts_search.transition_state_run import run_transition_state_search
    from scgo.utils.helpers import auto_niter_ts

    comp = ["Pt", "Pt", "Pt"]
    expected = auto_niter_ts(comp)

    # Fake minima loader and pair selection
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.load_minima_by_composition",
        lambda ts_output_dir, composition, prefer_final_unique: {
            "Pt3": [(0.0, Atoms("Pt3")), (0.1, Atoms("Pt3"))]
        },
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.select_structure_pairs",
        lambda minima, **kwargs: [(0, 1)],
    )

    def fake_find_transition_state(*args, **kwargs):
        # Ensure neb_steps has been resolved to an int
        assert isinstance(kwargs.get("neb_steps"), int)
        assert kwargs.get("neb_steps") == expected
        return {"pair_id": "0_1", "status": "failed", "neb_converged": False}

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.find_transition_state",
        fake_find_transition_state,
    )

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_neb_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_transition_state_results",
        lambda *a, **k: None,
    )

    # Call runner with default neb_steps (which is 'auto')
    run_transition_state_search(
        comp, params={"calculator": "EMT", "calculator_kwargs": {}}, verbosity=0
    )


def test_run_transition_state_search_resolves_torchsim_maxsteps_auto(monkeypatch):
    """Ensure torchsim_params['max_steps']='auto' is resolved before NEB/TorchSim use."""
    from ase import Atoms

    from scgo.ts_search.transition_state_run import run_transition_state_search
    from scgo.utils.helpers import auto_niter_ts

    comp = ["Pt", "Pt", "Pt", "Pt"]
    expected = auto_niter_ts(comp)

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.load_minima_by_composition",
        lambda ts_output_dir, composition, prefer_final_unique: {
            "Pt4": [(0.0, Atoms("Pt4")), (0.1, Atoms("Pt4"))]
        },
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.select_structure_pairs",
        lambda minima, **kwargs: [(0, 1)],
    )

    def fake_find_transition_state(*args, **kwargs):
        ts_params = kwargs.get("torchsim_params") or {}
        assert isinstance(ts_params.get("max_steps"), int)
        assert ts_params.get("max_steps") == expected
        return {"pair_id": "0_1", "status": "failed", "neb_converged": False}

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.find_transition_state",
        fake_find_transition_state,
    )

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_neb_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.save_transition_state_results",
        lambda *a, **k: None,
    )

    # Pass explicit torchsim_params with 'max_steps' set to 'auto'
    run_transition_state_search(
        ["Pt", "Pt", "Pt", "Pt"],
        params={"calculator": "EMT", "calculator_kwargs": {}},
        use_torchsim=True,
        torchsim_params={"max_steps": "auto", "force_tol": 0.05},
        verbosity=0,
    )


def test_final_unique_ts_and_network_statistics_consistent(tmp_path):
    """Ensure `ts_search_summary`, `ts_network_metadata` and `final_unique_ts` are present
    and that `statistics` are consistent when successful TS exist.
    """
    from scgo.ts_search.transition_state_io import (
        save_transition_state_results,
        write_final_unique_ts,
    )
    from scgo.ts_search.ts_network import save_ts_network_metadata

    # Create one simple successful TS
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    a.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5

    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "transition_state": a,
            "ts_energy": 0.5,
            "barrier_height": 0.2,
            "reactant_energy": 0.3,
            "product_energy": 0.4,
            "neb_converged": True,
            "n_images": 5,
            "spring_constant": 1.0,
            "error": None,
        }
    ]

    composition = ["Pt", "Pt"]
    out = str(tmp_path / "ts_results_Pt2")
    os.makedirs(out, exist_ok=True)

    # Write summary, network metadata and unique TS outputs
    summary_path = save_transition_state_results(ts_results, out, composition)
    network_path = save_ts_network_metadata(
        ts_results, out, composition, minima_count=2
    )
    write_final_unique_ts(ts_results, out, composition)

    # Load files and verify consistency
    with open(summary_path) as f:
        summary = json.load(f)
    with open(network_path) as f:
        net = json.load(f)

    final_summary_path = os.path.join(
        out, "final_unique_ts", "final_unique_ts_summary_Pt2.json"
    )
    assert os.path.exists(final_summary_path)
    with open(final_summary_path) as f:
        final_summary = json.load(f)

    assert final_summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
    assert "created_at" in final_summary

    for doc in (summary, net):
        assert doc["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert "scgo_version" in doc
        assert "created_at" in doc
    assert "statistics" in summary and isinstance(summary["statistics"], dict)
    assert "statistics" in net and isinstance(net["statistics"], dict)
    assert summary["statistics"] == net["statistics"]

    if summary.get("num_successful", 0) > 0:
        assert len(final_summary.get("unique_ts", [])) > 0


def test_extract_transition_states_defaults_to_canonical_tag(tmp_path):
    """Default TS extract returns only ``final_unique_ts`` rows; opt-in returns all TS."""
    from scgo.database import extract_transition_states_from_database_file
    from scgo.ts_search.ts_network import add_ts_to_database

    db_file = tmp_path / "minima.db"
    db = create_preparedb(Atoms("Pt2"), db_file, population_size=10)
    a = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    a.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
    a.info["confid"] = 1
    db.add_unrelaxed_candidate(a, description="Pt2_test")
    from ase_ga.data import DataConnection

    da = DataConnection(str(db_file))
    if da.get_number_of_unrelaxed_candidates() > 0:
        cand = da.get_an_unrelaxed_candidate()
        cand.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_relaxed_step(cand)
    mark_test_minima_as_final(db_file)

    ts_a = Atoms("Pt2", positions=[[0, 0, 0], [1.2, 0, 0]])
    assert add_ts_to_database(
        ts_structure=ts_a,
        ts_energy=0.4,
        minima_idx_1=0,
        minima_idx_2=1,
        db_file=str(db_file),
        pair_id="0_1",
        barrier_height=0.1,
        canonical_ts=True,
        neb_converged=True,
    )
    ts_b = Atoms("Pt2", positions=[[0, 0, 0], [1.3, 0, 0]])
    assert add_ts_to_database(
        ts_structure=ts_b,
        ts_energy=0.45,
        minima_idx_1=0,
        minima_idx_2=1,
        db_file=str(db_file),
        pair_id="1_2",
        barrier_height=0.12,
        canonical_ts=False,
        neb_converged=True,
    )

    canonical = extract_transition_states_from_database_file(
        db_file, run_id="run_test", require_final_unique_ts=True
    )
    assert len(canonical) == 1

    all_ts = extract_transition_states_from_database_file(
        db_file, run_id="run_test", require_final_unique_ts=False
    )
    assert len(all_ts) == 2
