import pytest

import scgo.database.connection as conn_mod


def test_open_db_raises_runtime_error_when_json1_missing(monkeypatch, tmp_path):
    """open_db propagates RuntimeError when JSON1 extension is unavailable."""
    db_path = tmp_path / "test.db"
    db_path.touch()

    def fake_ensure_sqlite_json1(conn):
        raise RuntimeError("SQLite JSON1 extension is required")

    monkeypatch.setattr(conn_mod, "_ensure_sqlite_json1", fake_ensure_sqlite_json1)

    with (
        pytest.raises(RuntimeError, match="SQLite JSON1 extension is required"),
        conn_mod.open_db(db_path),
    ):
        pass


def test_open_db_succeeds_when_json1_available(tmp_path):
    # Sanity check using a real connection; should not raise
    db_path = tmp_path / "test_ok.db"
    db_path.touch()
    with conn_mod.open_db(db_path) as db, db.c.managed_connection() as conn:
        cur = conn.execute("SELECT 1")
        assert cur.fetchone()[0] == 1


def test_get_connection_checks_json1(monkeypatch, tmp_path):
    """Ensure get_connection performs the JSON1 availability check."""
    db_path = tmp_path / "test_conn.db"
    db_path.touch()

    # Force the JSON1 checker to raise so we can verify get_connection calls it
    def fake_ensure(conn):
        raise RuntimeError("SQLite JSON1 extension is required")

    monkeypatch.setattr(conn_mod, "_ensure_sqlite_json1", fake_ensure)

    with (
        pytest.raises(RuntimeError, match="SQLite JSON1 extension is required"),
        conn_mod.get_connection(db_path),
    ):
        pass
