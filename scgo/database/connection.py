"""Database connection management for SCGO (HPC-oriented)."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from ase.db.sqlite import SQLite3Database as _ASESQLiteDatabase
from ase_ga.data import DataConnection

from scgo.database.sync import PRESET_AGGRESSIVE, retry_on_lock
from scgo.exceptions import (
    SCGODatabaseError,
    SCGOValidationError,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _force_close_ase_connection(conn: sqlite3.Connection) -> None:
    """Reliably release an ASE-managed SQLite connection.

    On CPython 3.12+ ``sqlite3.Connection.close()`` can be *deferred* when an
    active statement is still associated with the connection. That deferred
    close then surfaces as a ``ResourceWarning: unclosed database`` during
    interpreter/GC teardown (e.g. pytest's ``gc_collect_harder``), even though
    ASE already called ``close()``. Rolling back any open transaction before
    closing forces the underlying handle to be released immediately.
    """
    with contextlib.suppress(sqlite3.Error, AttributeError):
        conn.rollback()
    with contextlib.suppress(sqlite3.Error, AttributeError):
        conn.close()


def _patch_ase_managed_connection() -> None:
    """Wrap ASE's ``SQLite3Database.managed_connection`` so handles always close.

    ASE opens ephemeral SQLite handles inside ``managed_connection`` and relies
    on its own ``__exit__`` to commit and close them. Under heavy GC (pytest's
    ``gc_collect_harder``) that deferred close can emit a ``ResourceWarning``.

    This wrapper manually drives ASE's ``managed_connection`` generator so ASE's
    own commit/close logic runs first, then force-closes the handle. Persistent
    connections owned by ``self.connection`` are left intact for ASE to reuse
    and close via its own context manager.
    """
    import sys

    original = _ASESQLiteDatabase.managed_connection

    @contextmanager
    def _wrapped_managed_connection(self, commit_frequency=5000):
        cm = original(self, commit_frequency)
        conn = cm.__enter__()
        try:
            yield conn
        finally:
            cm.__exit__(*sys.exc_info()[:3])
            if self.connection is None:
                _force_close_ase_connection(conn)

    _ASESQLiteDatabase.managed_connection = _wrapped_managed_connection


_patch_ase_managed_connection()


def _open_ase_db_backend(backend) -> None:
    """Open a persistent ASE DB backend connection for the current scope.

    ASE's ``managed_connection()`` creates ephemeral SQLite handles when
    ``backend.connection`` is ``None``. Entering the backend context keeps a
    single connection for all subsequent operations and allows reliable cleanup
    via :func:`close_data_connection`.
    """
    if backend is None:
        return
    if getattr(backend, "connection", None) is not None:
        return
    if hasattr(backend, "__enter__"):
        backend.__enter__()


def _unwrap_data_connection(da: DataConnection | object) -> DataConnection:
    """Return the underlying :class:`~ase_ga.data.DataConnection` when wrapped."""
    return getattr(da, "_da", da)


def _apply_scgo_sqlite_settings(
    da: DataConnection,
    *,
    busy_timeout: int,
    wal_mode: bool,
    cache_size_mb: int,
    close_after: bool,
) -> None:
    """Open ASE backend, apply SCGO SQLite settings, optionally close afterward."""
    backend = getattr(da, "c", None)
    if backend is None:
        return

    _open_ase_db_backend(backend)
    conn = getattr(backend, "connection", None)
    try:
        if conn is None:
            return
        _ensure_sqlite_json1(conn=conn)
        apply_sqlite_pragmas(
            conn,
            busy_timeout=busy_timeout,
            cache_size_mb=cache_size_mb,
            wal_mode=wal_mode,
        )
    finally:
        if close_after and getattr(backend, "connection", None) is not None:
            with contextlib.suppress(sqlite3.DatabaseError, AttributeError):
                backend.__exit__(None, None, None)


def _apply_pragma(conn: sqlite3.Connection, statement: str) -> None:
    """Execute a PRAGMA statement, logging failures at debug level."""
    try:
        conn.execute(statement)
    except sqlite3.OperationalError as exc:
        logger.debug("SQLite PRAGMA failed (%s): %s", statement, exc)


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
        _apply_pragma(conn, "PRAGMA journal_mode=WAL;")
        _apply_pragma(conn, "PRAGMA synchronous=NORMAL;")
        _apply_pragma(conn, f"PRAGMA busy_timeout={busy_timeout};")
        _apply_pragma(conn, "PRAGMA temp_store=MEMORY;")
        _apply_pragma(conn, f"PRAGMA cache_size=-{cache_size_mb * 1024};")
        _apply_pragma(conn, "PRAGMA wal_autocheckpoint=1000;")
    else:
        _apply_pragma(conn, f"PRAGMA busy_timeout={busy_timeout};")
        _apply_pragma(conn, "PRAGMA journal_mode=DELETE;")
        _apply_pragma(conn, "PRAGMA temp_store=MEMORY;")
        _apply_pragma(conn, f"PRAGMA cache_size=-{cache_size_mb * 1024};")


def open_data_connection_for_setup(
    db_path: str | Path,
    *,
    wal_mode: bool = False,
    busy_timeout: int = 30000,
    cache_size_mb: int = 64,
) -> DataConnection:
    """Open a long-lived :class:`~ase_ga.data.DataConnection` for ``setup_database``.

    Applies SCGO PRAGMAs via :func:`_apply_scgo_sqlite_settings`, which performs
    the first SQLite open. Callers should wrap this in :func:`~scgo.database.sync.database_retry`
    when running on shared or contended filesystems.

    The connection is closed after PRAGMAs are applied so later ``with da.c:``
    (ASE SQLite) can reopen cleanly; file-level settings such as WAL persist.
    """
    da = DataConnection(str(db_path))
    _apply_scgo_sqlite_settings(
        da,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
        close_after=True,
    )
    return da


def _open_data_connection(
    db_path: str,
    *,
    busy_timeout: int,
    wal_mode: bool,
    cache_size_mb: int,
) -> DataConnection:
    """Open a :class:`~ase_ga.data.DataConnection` and apply SCGO SQLite settings."""
    da = DataConnection(db_path)
    _apply_scgo_sqlite_settings(
        da,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
        close_after=False,
    )
    return da


_open_data_connection_with_retry = retry_on_lock(
    config=PRESET_AGGRESSIVE,
    operation_name="open DataConnection",
)(_open_data_connection)


@contextmanager
def get_connection(
    db_path: str | Path,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> Generator[DataConnection, None, None]:
    """Open and yield an ASE :class:`~ase_ga.data.DataConnection` (with cleanup on exit).

    This is the primary context manager for SCGO database access.

    WAL mode is off by default (``DELETE`` journal) for shared/HPC filesystems;
    pass ``wal_mode=True`` on local disks when you need more write concurrency.

    Transient sqlite lock errors during open are retried with the same backoff
    policy used by :func:`~scgo.database.helpers.setup_database`.

    Args:
        db_path: Path to the ``.db`` file.
        busy_timeout: SQLite busy timeout in milliseconds (default 30s).
        wal_mode: If True, apply WAL-related PRAGMAs.
        cache_size_mb: SQLite page cache size hint in MiB.
    """
    db_path = str(db_path)
    da = _open_data_connection_with_retry(
        db_path,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
    )

    try:
        yield da
    finally:
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

    da = _unwrap_data_connection(da)
    backend = getattr(da, "c", None)
    if backend is None:
        return

    conn = getattr(backend, "connection", None)
    if conn is None:
        return

    # Release ASE's persistent handle first (commits + closes via __exit__).
    try:
        backend.__exit__(None, None, None)
    except (sqlite3.DatabaseError, TypeError, AttributeError) as e:
        if log_errors:
            logger.debug("Error closing database connection: %s", e)

    # Force-release any handle that ASE left open. On CPython 3.12+ the
    # deferred close can otherwise surface as ``ResourceWarning`` during GC.
    conn = getattr(backend, "connection", None)
    if conn is not None:
        _force_close_ase_connection(conn)
        backend.connection = None


def _run_sqlite(
    db_path: str | Path,
    callback,
    *,
    timeout: float = 30.0,
    commit: bool = True,
) -> None:
    """Run *callback(conn)* on a short-lived SQLite connection with explicit close."""
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        callback(conn)
        if commit:
            conn.commit()
    finally:
        with contextlib.suppress(sqlite3.DatabaseError):
            conn.close()


def _ensure_sqlite_json1(
    db_path: str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Ensure the SQLite JSON1 extension is available for this database file.

    Raises RuntimeError with a helpful message if JSON functions (e.g. json_extract)
    are not available on the underlying SQLite build.
    """
    try:
        if conn is not None:
            cur = conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
            _ = cur.fetchone()
            return
        if db_path is None:
            raise SCGOValidationError("db_path is required when conn is not provided")

        def _probe(active_conn: sqlite3.Connection) -> None:
            cur = active_conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
            _ = cur.fetchone()

        _run_sqlite(db_path, _probe, timeout=5.0)
    except sqlite3.OperationalError as e:
        raise SCGODatabaseError(
            "SQLite JSON1 extension is required but not available. "
            "Please use a Python build or system SQLite with JSON1 support (e.g., install a sqlite3 package with JSON1 enabled)."
        ) from e
