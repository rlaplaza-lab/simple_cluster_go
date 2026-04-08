"""Minimal DB writer for multiprocessing stress tests (spawn-safe).

Must not import ``scgo`` (``scgo/__init__.py`` eagerly loads MLIP stacks). Uses
``ase_ga`` + ``sqlite3`` only.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3

from ase import Atoms
from ase_ga.data import DataConnection


def _ensure_sqlite_json1(db_path: str) -> None:
    with sqlite3.connect(db_path, timeout=5.0) as tmp_conn:
        cur = tmp_conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
        _ = cur.fetchone()


def _close_da(da: DataConnection) -> None:
    if not hasattr(da, "c") or da.c is None:
        return
    with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
        if getattr(da.c, "connection", None) is not None:
            with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
                da.c.connection.commit()
            with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
                da.c.connection.close()
    with contextlib.suppress(TypeError, AttributeError):
        da.c.__exit__(None, None, None)


def write_to_database(args: tuple[str, int, int]) -> tuple[bool, int]:
    """Write ``n_structures`` rows from worker ``worker_id`` to ``db_path``."""
    db_path, n_structures, worker_id = args
    db_path = str(db_path)
    busy_timeout = 60000
    cache_size_mb = 64

    if os.path.exists(db_path):
        with (
            contextlib.suppress(sqlite3.OperationalError),
            sqlite3.connect(db_path, timeout=busy_timeout / 1000.0) as conn,
        ):
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024};")
            conn.commit()

    _ensure_sqlite_json1(db_path)
    da = DataConnection(db_path)
    conn = getattr(getattr(da, "c", None), "connection", None)
    if conn is not None:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")

    try:
        atoms_list: list[Atoms] = []
        for i in range(n_structures):
            atoms = Atoms(
                ["Pt", "Pt"],
                positions=[[0, 0, 0], [2.5 + i * 0.1, 0, 0]],
            )
            atoms.info.setdefault("key_value_pairs", {})
            atoms.info["key_value_pairs"]["raw_score"] = -10.0 - i * 0.1
            atoms.info.setdefault("data", {})
            atoms.info["data"]["worker_tag"] = f"w{worker_id}"
            atoms_list.append(atoms)

        for atoms in atoms_list:
            da.add_unrelaxed_candidate(
                atoms, description=f"concurrent_stress:w{worker_id}"
            )
            da.add_relaxed_step(atoms)
        return True, worker_id
    finally:
        _close_da(da)
