"""Database connection management for SCGO (HPC-oriented)."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from ase_ga.data import DataConnection

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def apply_sqlite_pragmas(
    conn: sqlite3.Connection,
    *,
    wal_mode: bool = False,
    busy_timeout: int = 30000,
    cache_size_mb: int = 64,
) -> None:
    """Apply PRAGMAs appropriate for SQLite databases in SCGO.

    Modes:
      wal_mode=False (HPC default): rollback journal, memory temp, delete on close.
      wal_mode=True: write-ahead-logging, normal sync, autocheckpoint.
    """
    if wal_mode:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024};")
            conn.execute("PRAGMA wal_autocheckpoint=1000;")
    else:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode=DELETE;")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA temp_store=MEMORY;")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024};")


@contextmanager
def get_connection(
    db_path: str | Path,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> Generator[DataConnection, None, None]:
    """Open and yield a DataConnection with optional WAL, timeouts and cache tuning.

    WAL mode is disabled by default for better HPC compatibility.
    Enable it only if running on reliable local filesystems with heavy concurrent access.
    """
    db_path = str(db_path)
    # Configure SQLite before opening DataConnection
    if wal_mode and os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path, timeout=busy_timeout / 1000.0) as conn:
                apply_sqlite_pragmas(
                    conn,
                    wal_mode=True,
                    busy_timeout=busy_timeout,
                    cache_size_mb=cache_size_mb,
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to configure SQLite for {db_path}: {e}")

    da = DataConnection(db_path)

    # Fail fast: ensure JSON1 is available since many DB helpers use json_extract.
    # Raise a clear error here rather than allowing downstream queries to fail
    # with cryptic sqlite OperationalErrors.
    _ensure_sqlite_json1(db_path)

    # Apply busy_timeout to ASE's active connection.
    _apply_busy_timeout(da, busy_timeout)

    conn = getattr(getattr(da, "c", None), "connection", None)
    if conn is not None:
        apply_sqlite_pragmas(
            conn,
            busy_timeout=busy_timeout,
            cache_size_mb=cache_size_mb,
            wal_mode=wal_mode,
        )

    try:
        yield da
    finally:
        # Explicit cleanup to avoid file handle leaks
        close_data_connection(da)


def close_data_connection(da: DataConnection | None, log_errors: bool = True) -> None:
    """Safely close a DataConnection object.

    Handles the fact that ASE's SQLite3Database doesn't have a close()
    method but does support the context manager protocol (__exit__).

    Note:
        ASE database objects may have their internal SQLite connection invalidated
        (set to None) in certain conditions (errors, timeouts, external closes).
        This is a benign state during cleanup and should not produce error messages.

    Args:
        da: DataConnection object to close (can be None)
        log_errors: Whether to log errors at debug level (default True)

    Example:
        >>> da = DataConnection('path/to/db.db')
        >>> try:
        ...     # work with da
        ... finally:
        ...     close_data_connection(da)
    """
    if da is None:
        return

    try:
        if hasattr(da, "c") and da.c is not None and hasattr(da.c, "__exit__"):
            if hasattr(da.c, "connection") and da.c.connection is None:
                return

            # SQLite3Database doesn't have close(), but has __exit__ for cleanup
            da.c.__exit__(None, None, None)
    except (
        sqlite3.OperationalError,
        sqlite3.DatabaseError,
        TypeError,
        AttributeError,
    ) as e:
        if log_errors:
            logger.debug(f"Error closing database connection: {e}")


def _ensure_sqlite_json1(db_path: str) -> None:
    """Ensure the SQLite JSON1 extension is available for this database file.

    Raises RuntimeError with a helpful message if JSON functions (e.g. json_extract)
    are not available on the underlying SQLite build.
    """
    try:
        with sqlite3.connect(db_path, timeout=5.0) as tmp_conn:
            cur = tmp_conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
            _ = cur.fetchone()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            "SQLite JSON1 extension is required but not available. "
            "Please use a Python build or system SQLite with JSON1 support (e.g., install a sqlite3 package with JSON1 enabled)."
        ) from e


def _apply_busy_timeout(da, busy_timeout: int) -> None:
    """Apply PRAGMA busy_timeout to the connection used by DataConnection.

    Ensures that even when ASE has already created a connection, we configure it
    for concurrent access (retry on lock instead of failing immediately).
    """
    conn = getattr(getattr(da, "c", None), "connection", None)
    if conn is not None:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")


@contextmanager
def open_db(
    db_path: str | Path,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> Generator[DataConnection, None, None]:
    """Open and yield a database connection with automatic resource management.

    This is the primary interface for all database access in SCGO.
    It automatically handles connection lifecycle and cleanup.

    Args:
        db_path: Path to database file (str or Path object)
        busy_timeout: SQLite busy timeout in milliseconds (default 30s)
        wal_mode: Enable WAL mode for concurrent access (default False).
            WAL mode improves concurrency on local filesystems but can cause locking
            issues on shared/network filesystems common in HPC environments.
        cache_size_mb: SQLite cache size in MB (default 64)

    Yields:
        DataConnection: Database connection ready for use

    Example:
        >>> with open_db('output/run_xyz/db.db') as da:
        ...     atoms = da.get_an_unrelaxed_candidate()
        ...     # work with atoms
        ...     da.add_relaxed_step(atoms)
    """
    db_path_str = str(db_path)

    # Use the existing get_connection context manager
    # JSON1 check is already performed in get_connection, no need to duplicate
    with get_connection(
        db_path_str,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
    ) as conn:
        yield conn
