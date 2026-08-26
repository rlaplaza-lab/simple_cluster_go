"""SQLite ``scgo_metadata`` table — DB identity / schema stamp.

Distinct from structure tags (:mod:`scgo.metadata.atoms`) and from the
output-JSON provenance header (:mod:`scgo.metadata.provenance`).
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from scgo.utils.logging import get_logger

logger = get_logger(__name__)

CURRENT_DB_SCHEMA_VERSION = 2

SCGO_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS scgo_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def _upsert_scgo_metadata_keys(
    conn: sqlite3.Connection, *, schema_version: int
) -> None:
    conn.execute(SCGO_METADATA_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('created_by', 'scgo')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('schema_version', ?)",
        (str(schema_version),),
    )


def get_db_stamp(db_path: str | Path) -> dict[str, str]:
    """Return key/value pairs from the ``scgo_metadata`` table, or {}."""
    try:
        db_file = str(db_path)
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=0.1)
    except (sqlite3.DatabaseError, FileNotFoundError) as exc:
        logger.debug("Could not open scgo_metadata from %s: %s", db_path, exc)
        return {}

    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scgo_metadata'"
        )
        if cur.fetchone() is None:
            return {}
        rows = conn.execute("SELECT key, value FROM scgo_metadata").fetchall()
        return {r[0]: r[1] for r in rows}
    except (sqlite3.DatabaseError, FileNotFoundError) as exc:
        logger.debug("Could not read scgo_metadata from %s: %s", db_path, exc)
        return {}
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()


_scgo_database_cache: dict[str, bool] = {}


def clear_db_stamp_cache() -> None:
    """Clear the :func:`is_scgo_db` memoization cache."""
    _scgo_database_cache.clear()


def is_scgo_db(db_path: str | Path) -> bool:
    """True if ``scgo_metadata.created_by`` is ``scgo``.

    The result is memoized per resolved path; call :func:`clear_db_stamp_cache`
    after stamping or replacing a file outside this module.
    """
    key = str(Path(db_path).resolve())
    cached = _scgo_database_cache.get(key)
    if cached is not None:
        return cached
    meta = get_db_stamp(db_path)
    result = bool(meta) and meta.get("created_by") == "scgo"
    _scgo_database_cache[key] = result
    return result


def stamp_db(db_path: str | Path, *, schema_version: int | None = None) -> None:
    """Write ``scgo_metadata`` so :func:`is_scgo_db` accepts this file.

    Called by :func:`~scgo.setup_database`, and by tests and
    tools that build SQLite files directly. Clears the :func:`is_scgo_db` cache.
    """
    # Local import: scgo.database.connection pulls in the scgo.database package,
    # which imports scgo.database.helpers, which imports this module.
    from scgo.database.connection import _run_sqlite

    ver = schema_version if schema_version is not None else CURRENT_DB_SCHEMA_VERSION
    path = str(db_path)

    def _stamp(conn: sqlite3.Connection) -> None:
        _upsert_scgo_metadata_keys(conn, schema_version=ver)

    _run_sqlite(path, _stamp)
    clear_db_stamp_cache()
