"""Memory-efficient streaming iterators for large databases.

Provides generators for iterating over database contents without loading
everything into memory at once.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

from ase import Atoms

from scgo.database.connection import get_connection
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.metadata import add_metadata
from scgo.database.schema import is_scgo_database
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)


def _energy_sql_expression() -> str:
    """SQL expression for structure energy (``systems.energy`` and/or GA ``raw_score``)."""
    kv = SYSTEMS_JSON_COLUMN
    return (
        "COALESCE(energy, "
        f"CASE WHEN json_extract({kv}, '$.raw_score') IS NOT NULL "
        f"THEN (-CAST(json_extract({kv}, '$.raw_score') AS REAL)) "
        f"ELSE NULL END)"
    )


def aggregate_relaxed_energy_stats(
    db_path: str | Path,
) -> dict[str, float | int | None]:
    """Aggregate COUNT, MIN, MAX, AVG of ``systems.energy`` for relaxed rows only.

    Uses the same ``json_extract(..., '$.relaxed') = 1`` filter as
    ``count_database_structures`` / ``iter_database_minima``. Does not load
    atomic structures.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {
            "count": 0,
            "min_energy": None,
            "max_energy": None,
            "avg_energy": None,
        }

    if not is_scgo_database(db_path):
        return {
            "count": 0,
            "min_energy": None,
            "max_energy": None,
            "avg_energy": None,
        }

    try:
        with get_connection(str(db_path)) as da, da.c.managed_connection() as conn:
            relaxed_col = SYSTEMS_JSON_COLUMN
            energy_expr = _energy_sql_expression()
            cur = conn.execute(
                f"SELECT COUNT(*), MIN({energy_expr}), MAX({energy_expr}), AVG({energy_expr}) "
                f"FROM systems WHERE json_extract({relaxed_col}, '$.relaxed') = 1"
            )
            row = cur.fetchone()
            if not row:
                return {
                    "count": 0,
                    "min_energy": None,
                    "max_energy": None,
                    "avg_energy": None,
                }
            n, emin, emax, eavg = row[0], row[1], row[2], row[3]
            n = int(n or 0)
            return {
                "count": n,
                "min_energy": float(emin) if emin is not None else None,
                "max_energy": float(emax) if emax is not None else None,
                "avg_energy": float(eavg) if eavg is not None else None,
            }
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError, TypeError) as e:
        logger.debug("aggregate_relaxed_energy_stats failed for %s: %s", db_path, e)
        return {
            "count": 0,
            "min_energy": None,
            "max_energy": None,
            "avg_energy": None,
        }


def _safe_get_atoms(da, row_id):
    """Safely fetch atoms for a given row id.

    Returns (Atoms, None) on success or (None, reason_str) on failure. Logs
    a debug-level traceback and includes a truncated metadata snippet to aid
    debugging without exposing large JSON blobs.
    """
    try:
        return da.get_atoms(row_id), None
    except (
        KeyError,
        IndexError,
        sqlite3.DatabaseError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as e:
        snippet = "<unavailable>"
        try:
            with da.c.managed_connection() as conn:
                r = conn.execute(
                    f"SELECT {SYSTEMS_JSON_COLUMN} FROM systems WHERE id=?",
                    (row_id,),
                ).fetchone()
                if r:
                    snippet = str(r[0] or "")[:200]
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError, TypeError):
            snippet = "<snippet_unavailable>"

        logger.debug(
            "Row fetch failed id=%s: %s; snippet=%s", row_id, e, snippet, exc_info=True
        )
        return None, f"{e!r} | meta_snippet={snippet}"


def _iter_relaxed_minima_from_da(
    da,
    db_path: Path,
    chunk_size: int = 100,
):
    """Yield (energy, atoms_copy) for relaxed rows using chunked id queries."""
    if chunk_size is None or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    with da.c.managed_connection() as conn:
        relaxed_col = SYSTEMS_JSON_COLUMN
        raw_score_col = SYSTEMS_JSON_COLUMN

        try:
            cur = conn.execute(
                f"SELECT COUNT(*) FROM systems WHERE json_extract({relaxed_col}, '$.relaxed') = 1"
            )
            total = int((cur.fetchone() or [0])[0] or 0)
        except (sqlite3.DatabaseError, sqlite3.OperationalError, TypeError, ValueError):
            total = 0

        logger.debug(
            "Streaming %s structures from %s (chunk_size=%s)",
            total,
            db_path,
            chunk_size,
        )

        try:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE json_extract({relaxed_col}, '$.relaxed') = 1 "
                f"ORDER BY CAST(json_extract({raw_score_col}, '$.raw_score') AS REAL) DESC"
            )
        except sqlite3.OperationalError:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE json_extract({relaxed_col}, '$.relaxed') = 1 ORDER BY id"
            )

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            for (row_id,) in rows:
                candidate, reason = _safe_get_atoms(da, row_id)
                if candidate is None:
                    logger.warning("Failed to fetch atoms id=%s: %s", row_id, reason)
                    continue

                energy = extract_energy_from_atoms(candidate)
                if energy is None:
                    logger.log(TRACE, "Skipping candidate id=%s: no energy", row_id)
                    continue

                out = candidate.copy()
                # Stable link back to the SQLite `systems.id` row (for TS ↔ minima traceability).
                try:
                    add_metadata(out, systems_row_id=int(row_id))
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to attach systems_row_id metadata: %s", e)
                yield (energy, out)


def iter_database_minima(
    db_path: str | Path,
    chunk_size: int = 100,
) -> Generator[tuple[float, Atoms], None, None]:
    """Iterate over minima from database in memory-efficient chunks.

    Yields structures one at a time without loading entire database into memory.
    Useful for processing very large databases (1000+ structures).

    Args:
        db_path: Path to database file
        chunk_size: Number of structures to load at once (default 100)

    Yields:
        tuple: (energy, atoms) for each minimum

    Example:
        >>> for energy, atoms in iter_database_minima("large_db.db"):
        ...     process_structure(atoms)
    """
    db_path = Path(db_path)

    if not db_path.exists():
        logger.warning("Database does not exist: %s", db_path)
        return

    if not is_scgo_database(db_path):
        logger.debug("Skipping non-SCGO database: %s", db_path)
        return

    try:
        with get_connection(str(db_path)) as da:
            yield from _iter_relaxed_minima_from_da(da, db_path, chunk_size)
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Error streaming from %s: %s", db_path, e)
        raise


def iter_databases_minima(
    db_paths: list[str | Path],
    max_structures: int | None = None,
) -> Generator[tuple[float, Atoms], None, None]:
    """Iterate over minima from multiple databases.

    Efficiently streams structures from multiple database files without
    loading everything into memory.

    Args:
        db_paths: List of database file paths
        max_structures: Optional limit on total structures to yield

    Yields:
        tuple: (energy, atoms) for each minimum across all databases

    Example:
        >>> db_files = ["db1.db", "db2.db", "db3.db"]
        >>> for energy, atoms in iter_databases_minima(db_files, max_structures=1000):
        ...     process_structure(atoms)
    """
    count = 0

    for db_path in db_paths:
        if max_structures and count >= max_structures:
            logger.debug("Reached max_structures limit (%s)", max_structures)
            break

        for energy, atoms in iter_database_minima(db_path):
            yield (energy, atoms)
            count += 1

            if max_structures and count >= max_structures:
                break

    logger.debug("Streamed %s total structures from %s databases", count, len(db_paths))


def count_database_structures(db_path: str | Path) -> int:
    """Count structures in database without loading them.

    Fast counting operation that doesn't load actual atomic structures.

    Args:
        db_path: Path to database file

    Returns:
        int: Number of relaxed structures in database
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return 0

    if not is_scgo_database(db_path):
        logger.debug("Skipping count for non-SCGO database: %s", db_path)
        return 0

    try:
        with get_connection(str(db_path)) as da, da.c.managed_connection() as conn:
            relaxed_col = SYSTEMS_JSON_COLUMN
            cur = conn.execute(
                f"SELECT COUNT(*) FROM systems WHERE json_extract({relaxed_col}, '$.relaxed') = 1"
            )
            res = cur.fetchone()
            return int((res or [0])[0] or 0)
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Error counting structures in %s: %s", db_path, e)
        return 0
