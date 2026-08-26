"""Memory-efficient streaming iterators for large databases.

Provides generators for iterating over database contents without loading
everything into memory at once.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

from ase import Atoms

from scgo.database.connection import get_connection
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.metadata.db_stamp import is_scgo_db
from scgo.utils.helpers import copy_atoms, extract_energy_from_atoms
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)

# ASE sqlite ``_select`` projects 26 ``systems`` fields plus ``data``.
_ASE_SYSTEMS_COLUMN_COUNT = 26
_ASE_ROW_VALUE_COUNT = _ASE_SYSTEMS_COLUMN_COUNT + 1


def _load_atoms_chunk_via_get_atoms(row_ids: list[int], da) -> list[tuple[int, Atoms]]:
    """Decode one id at a time when bulk sqlite projection is unavailable."""
    out: dict[int, Atoms] = {}
    for row_id in row_ids:
        try:
            atoms = da.get_atoms(row_id)
        except (
            KeyError,
            IndexError,
            sqlite3.DatabaseError,
            ValueError,
            TypeError,
        ) as row_exc:
            logger.warning(
                "Failed to fetch atoms id=%s from chunked stream: %s",
                row_id,
                row_exc,
            )
            continue
        if atoms is not None:
            out[int(row_id)] = atoms
    return [(row_id, out[row_id]) for row_id in row_ids if row_id in out]


def _load_atoms_chunk(row_ids: list[int], da) -> list[tuple[int, Atoms]]:
    """Load atom rows for a chunk of ids through ASE's row decoder.

    ASE's public ``select`` accepts only a single id, so we fetch the chunk with
    one ``WHERE id IN (...)`` using ASE's column projection, then
    ``_convert_tuple_to_row`` + ``toatoms(add_additional_information=True)``.
    """
    if not row_ids:
        return []

    convert = getattr(da.c, "_convert_tuple_to_row", None)
    colnames = getattr(da.c, "columnnames", None)
    if convert is None or colnames is None or len(colnames) < _ASE_ROW_VALUE_COUNT:
        return _load_atoms_chunk_via_get_atoms(row_ids, da)

    columnindex = list(range(_ASE_SYSTEMS_COLUMN_COUNT)) + [_ASE_SYSTEMS_COLUMN_COUNT]
    what = ", ".join("systems." + colnames[i] for i in columnindex)
    placeholders = ",".join("?" * len(row_ids))
    try:
        with da.c.managed_connection() as conn:
            cur = conn.execute(
                f"SELECT {what} FROM systems WHERE id IN ({placeholders})",
                tuple(int(i) for i in row_ids),
            )
            value_rows = cur.fetchall()
    except (
        sqlite3.DatabaseError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        logger.warning(
            "Failed to select atoms chunk ids=%s (%s); falling back to per-id get_atoms",
            row_ids,
            exc,
        )
        return _load_atoms_chunk_via_get_atoms(row_ids, da)

    loaded: dict[int, Atoms] = {}
    for shortvalues in value_rows:
        try:
            values: list[object | None] = [None] * _ASE_ROW_VALUE_COUNT
            for idx, col_i in enumerate(columnindex):
                values[col_i] = shortvalues[idx]
            row = convert(tuple(values))
            atoms = row.toatoms(add_additional_information=True)
        except (
            KeyError,
            IndexError,
            sqlite3.DatabaseError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            row_id = shortvalues[0] if shortvalues else None
            logger.warning(
                "Failed to decode atoms id=%s from chunked stream: %s",
                row_id,
                exc,
            )
            continue
        if atoms is not None:
            loaded[int(row.id)] = atoms

    return [(row_id, loaded[row_id]) for row_id in row_ids if row_id in loaded]


def relaxed_rows_where_clause(
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
) -> str:
    """Build SQL WHERE fragment for relaxed-row streaming filters."""
    col = SYSTEMS_JSON_COLUMN
    clauses = [f"json_extract({col}, '$.relaxed') = 1"]
    if require_final_minimum:
        clauses.append(f"json_extract({col}, '$.final_unique_minimum') = 1")
    if exclude_transition_states:
        clauses.append(f"COALESCE(json_extract({col}, '$.is_transition_state'), 0) = 0")
    if require_transition_state:
        clauses.append(f"json_extract({col}, '$.is_transition_state') = 1")
    if require_final_ts:
        clauses.append(f"json_extract({col}, '$.final_unique_ts') = 1")
    return " AND ".join(clauses)


def iter_relaxed_structures(
    da,
    db_path: Path,
    chunk_size: int = 100,
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
):
    """Yield (energy, atoms_copy) for relaxed rows using chunked id queries."""
    if chunk_size is None or chunk_size <= 0:
        raise SCGOValidationError("chunk_size must be a positive integer")

    where_sql = relaxed_rows_where_clause(
        require_final_minimum=require_final_minimum,
        exclude_transition_states=exclude_transition_states,
        require_transition_state=require_transition_state,
        require_final_ts=require_final_ts,
    )

    with da.c.managed_connection() as conn:
        json_col = SYSTEMS_JSON_COLUMN

        if logger.isEnabledFor(TRACE):
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM systems WHERE {where_sql}")
                total = int((cur.fetchone() or [0])[0] or 0)
            except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
                logger.debug("COUNT query failed for %s: %s", db_path, exc)
                total = 0
            logger.debug(
                "Streaming %s structures from %s (chunk_size=%s)",
                total,
                db_path,
                chunk_size,
            )

        try:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE {where_sql} "
                f"ORDER BY CAST(json_extract({json_col}, '$.raw_score') AS REAL) DESC"
            )
        except sqlite3.OperationalError:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE {where_sql} ORDER BY id"
            )

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            row_ids = [int(row_id) for (row_id,) in rows]
            for row_id, candidate in _load_atoms_chunk(row_ids, da):
                energy = extract_energy_from_atoms(candidate)
                if energy is None:
                    logger.trace("Skipping candidate id=%s: no energy", row_id)
                    continue

                out = copy_atoms(candidate)
                try:
                    set_tags(out, systems_row_id=int(row_id))
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to attach systems_row_id tag: %s", e)
                yield (energy, out)


def iter_database_minima(
    db_path: str | Path,
    chunk_size: int = 100,
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
) -> Generator[tuple[float, Atoms], None, None]:
    """Iterate over minima from database in memory-efficient chunks."""
    db_path = Path(db_path)

    if not db_path.exists():
        logger.warning("Database does not exist: %s", db_path)
        return

    if not is_scgo_db(db_path):
        logger.debug("Skipping non-SCGO database: %s", db_path)
        return

    with get_connection(str(db_path)) as da:
        yield from iter_relaxed_structures(
            da,
            db_path,
            chunk_size,
            require_final_minimum=require_final_minimum,
            exclude_transition_states=exclude_transition_states,
            require_transition_state=require_transition_state,
            require_final_ts=require_final_ts,
        )
