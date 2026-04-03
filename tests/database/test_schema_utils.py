import sqlite3
from pathlib import Path

from scgo.database.discovery import find_databases_simple
from scgo.database.schema import get_scgo_metadata, is_scgo_database
from scgo.database.streaming import iter_database_minima


def _create_dummy_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        # Minimal table to emulate an ASE DB file (no scgo_metadata)
        conn.execute(
            "CREATE TABLE systems (id INTEGER PRIMARY KEY, energy REAL, key_value_pairs TEXT)"
        )
        conn.commit()
    finally:
        conn.close()


def test_get_scgo_metadata_returns_empty_for_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000" / "trial_1"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    assert get_scgo_metadata(db_path) == {}
    assert not is_scgo_database(db_path)


def test_find_databases_skips_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000" / "trial_1"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    found = find_databases_simple(tmp_path, db_pattern="**/*.db")
    # Should skip the non-SCGO DB created above
    assert db_path not in found
    assert found == []


def test_iter_database_minima_skips_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000" / "trial_1"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    items = list(iter_database_minima(db_path))
    assert items == []


def test_setup_database_marks_scgo_db(tmp_path: Path):
    from ase import Atoms

    from scgo.database.connection import close_data_connection
    from scgo.database.helpers import setup_database

    run_dir = tmp_path / "run_000" / "trial_1"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 1)])

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    try:
        db_path = run_dir / "ga_go.db"
        meta = get_scgo_metadata(db_path)
        assert meta.get("created_by") == "scgo"
        assert "schema_version" in meta and int(meta["schema_version"]) >= 1
    finally:
        # Ensure resources cleaned up
        close_data_connection(da)


def test_find_databases_includes_scgo_db(tmp_path: Path):
    from ase import Atoms

    from scgo.database.helpers import setup_database

    run_dir = tmp_path / "run_000" / "trial_1"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 1)])

    # Create a proper SCGO DB
    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    try:
        found = find_databases_simple(tmp_path, db_pattern="**/*.db")
        assert (run_dir / "ga_go.db") in found
    finally:
        from scgo.database.connection import close_data_connection

        close_data_connection(da)
