"""Database health check utilities for SCGO.

Provides tools to diagnose and validate ASE database files.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def check_database_health(db_path: str | Path) -> dict:
    """Check database health and return diagnostic information.

    Checks for:
    - File existence and permissions
    - SQLite corruption
    - Schema validity
    - WAL mode status
    - Database statistics

    Args:
        db_path: Path to database file

    Returns:
        dict: Health check results with keys:
            - 'healthy': bool
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'info': dict of database statistics
    """
    db_path = Path(db_path)
    result = {"healthy": True, "errors": [], "warnings": [], "info": {}}

    # Check file existence
    if not db_path.exists():
        result["healthy"] = False
        result["errors"].append(f"Database file does not exist: {db_path}")
        return result

    # Check file permissions
    if not os.access(db_path, os.R_OK):
        result["healthy"] = False
        result["errors"].append(f"Database file is not readable: {db_path}")
        return result

    # Check SQLite integrity
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            # Run integrity check
            cursor = conn.execute("PRAGMA integrity_check;")
            integrity_result = cursor.fetchone()[0]

            if integrity_result != "ok":
                result["healthy"] = False
                result["errors"].append(f"Integrity check failed: {integrity_result}")

            # Get database size
            cursor = conn.execute("PRAGMA page_count;")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size;")
            page_size = cursor.fetchone()[0]
            db_size_mb = (page_count * page_size) / (1024 * 1024)
            result["info"]["size_mb"] = round(db_size_mb, 2)

            # Check journal mode
            cursor = conn.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            result["info"]["journal_mode"] = journal_mode

            # Get table count
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
            )
            table_count = cursor.fetchone()[0]
            result["info"]["table_count"] = table_count

            # Check for ASE tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='systems';"
            )
            has_systems_table = cursor.fetchone() is not None

            if not has_systems_table:
                result["warnings"].append(
                    "Missing 'systems' table - may not be an ASE database"
                )
            else:
                # Count rows in systems table
                cursor = conn.execute("SELECT COUNT(*) FROM systems;")
                row_count = cursor.fetchone()[0]
                result["info"]["systems_count"] = row_count

    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        result["healthy"] = False
        result["errors"].append(f"Database error: {e}")

    return result


def get_database_statistics(db_path: str | Path) -> dict:
    """Get detailed statistics about a database.

    Args:
        db_path: Path to database file

    Returns:
        dict: Database statistics including:
            - size_mb: Database size in megabytes
            - systems_count: Number of entries in systems table
            - journal_mode: Current journal mode
            - page_size: SQLite page size
            - fragmentation: Estimated fragmentation percentage
    """
    db_path = Path(db_path)
    stats = {}

    if not db_path.exists():
        logger.warning(f"Database does not exist: {db_path}")
        return stats

    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            # Basic size info
            cursor = conn.execute("PRAGMA page_count;")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size;")
            page_size = cursor.fetchone()[0]
            stats["page_count"] = page_count
            stats["page_size"] = page_size
            stats["size_mb"] = round((page_count * page_size) / (1024 * 1024), 2)

            # Journal mode
            cursor = conn.execute("PRAGMA journal_mode;")
            stats["journal_mode"] = cursor.fetchone()[0]

            # Freelist (fragmentation indicator)
            cursor = conn.execute("PRAGMA freelist_count;")
            freelist = cursor.fetchone()[0]
            stats["freelist_count"] = freelist
            if page_count > 0:
                stats["fragmentation_pct"] = round((freelist / page_count) * 100, 2)

            # Count systems
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM systems;")
                stats["systems_count"] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                stats["systems_count"] = None

            # Get table names
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            stats["tables"] = [row[0] for row in cursor.fetchall()]

    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Failed to get statistics for %s: %s", db_path, e)

    return stats
