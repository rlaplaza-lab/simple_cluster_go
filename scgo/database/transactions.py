"""Simple transaction helpers for SCGO databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from ase_ga.data import DataConnection

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


_VALID_ISOLATION_LEVELS = frozenset({"DEFERRED", "IMMEDIATE", "EXCLUSIVE"})


@contextmanager
def database_transaction(
    db: DataConnection,
    isolation_level: str = "DEFERRED",
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a transaction.

    Yields:
        sqlite3.Connection: Raw connection. Commits on success; rolls back on error.
    """
    if not hasattr(db, "c") or db.c is None:
        raise ValueError("Invalid database connection")

    if isolation_level.upper() not in _VALID_ISOLATION_LEVELS:
        raise ValueError(
            f"Invalid isolation level: {isolation_level!r}. "
            f"Must be one of {sorted(_VALID_ISOLATION_LEVELS)}"
        )

    # Use managed_connection() to get actual SQLite connection
    with db.c.managed_connection() as conn:
        try:
            conn.execute(f"BEGIN {isolation_level.upper()}")
            logger.debug(f"Started {isolation_level} transaction")

            yield conn  # Yield connection instead of db

            conn.commit()
            logger.debug("Transaction committed")
        except Exception:
            conn.rollback()
            logger.debug("Transaction rolled back")
            raise
