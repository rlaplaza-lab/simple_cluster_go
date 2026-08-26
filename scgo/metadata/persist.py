"""SQL helpers that update structure tags stored in ASE ``key_value_pairs``."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.metadata.atoms import get_tag
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_key_value_pairs_row(row: tuple) -> dict[str, Any]:
    """Parse ``key_value_pairs`` JSON from ``SELECT id, energy, key_value_pairs`` rows."""
    try:
        kv_json = row[2]
        if not kv_json:
            return {}
        return json.loads(kv_json)
    except (json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        logger.debug("Failed to parse key_value_pairs row: %s", exc)
        return {}


def _find_first_relaxed_row(rows: list) -> tuple | None:
    for r in rows:
        if _parse_key_value_pairs_row(r).get("relaxed"):
            return r
    return None


def _match_row_by_stored_final_id(
    conn,
    *,
    kvp: str,
    select_cols: str,
    final_id: str,
) -> tuple | None:
    fid_conditions = [
        f"CAST(json_extract({kvp}, '$.final_id') AS TEXT) = ?",
        f"CAST(json_extract({kvp}, '$.unique_id') AS TEXT) = ?",
        "CAST(unique_id AS TEXT) = ?",
    ]
    fid_params = [final_id, final_id, final_id]
    query = (
        f"SELECT {select_cols} FROM systems WHERE "
        + " OR ".join(fid_conditions)
        + " ORDER BY rowid ASC"
    )
    rows = conn.execute(query, tuple(fid_params)).fetchall()
    if not rows:
        return None
    return _find_first_relaxed_row(rows) or rows[0]


def mark_final_minima_in_db(
    final_minima_info: list[dict],
    base_dir: str | Path,
    db_paths: list[str | Path] | None = None,
) -> dict:
    """Mark final unique minima in database ``systems.key_value_pairs`` JSON rows.

    Rows are matched by ``final_id`` stored at relaxed persist time
    (:func:`scgo.metadata.atoms.ensure_final_id`).

    Args:
        final_minima_info: List of dicts with keys ``atoms`` (Atoms, required),
            ``final_id`` (str, required), ``rank`` (1-based int, optional) and
            ``final_written`` (str filepath or filename, optional). Other keys,
            such as ``energy``, are ignored
        base_dir: Base output directory searched for database files, used only
            when ``db_paths`` is not given
        db_paths: Optional explicit list of database files to search/update

    Returns:
        dict: Summary counts, e.g.
            ``{"dbs_touched": int, "rows_updated": int, "details": {db_path: rows}}``
    """
    # Circular: connection → sync → utils → helpers → metadata.atoms;
    # discovery imports db_stamp / run_dir.
    from scgo.database.connection import get_connection
    from scgo.database.discovery import DatabaseDiscovery
    from scgo.database.sync import retry_transaction

    discovery = DatabaseDiscovery(base_dir)

    total_rows_updated = 0
    dbs_touched: set[str] = set()
    details: dict[str, int] = {}

    updates_by_db: dict[str, list[dict[str, Any]]] = {}
    for info in final_minima_info:
        atoms = info.get("atoms")
        rank = info.get("rank")
        final_written = info.get("final_written")
        final_id = info.get("final_id")

        if atoms is None:
            logger.warning("Missing atoms entry in mark_final_minima_in_db; skipping")
            continue

        if final_id is None:
            logger.warning("Missing final_id in mark_final_minima_in_db; skipping")
            continue

        run_id = get_tag(atoms, "run_id")

        if db_paths:
            db_files = [Path(p) for p in db_paths]
        else:
            db_files = discovery.find_databases(run_id=run_id)

        if not db_files:
            logger.warning(
                "No databases found for run=%s in mark_final_minima_in_db "
                "— check output layout, registry, or pass db_paths",
                run_id,
            )
            continue

        for db_path in db_files:
            db_key = str(db_path)
            updates_by_db.setdefault(db_key, []).append(
                {
                    "run_id": run_id,
                    "rank": rank,
                    "final_written": final_written,
                    "final_id": str(final_id),
                }
            )

    for db_key, db_updates in updates_by_db.items():
        db_path = Path(db_key)
        try:
            with get_connection(db_path) as db:

                def _mark_rows(
                    conn: sqlite3.Connection,
                    updates: list[dict[str, Any]] = db_updates,
                ) -> int:
                    kvp = SYSTEMS_JSON_COLUMN
                    select_cols = f"id, energy, {kvp}"
                    rows_updated_this_db = 0
                    for update in updates:
                        row = _match_row_by_stored_final_id(
                            conn,
                            kvp=kvp,
                            select_cols=select_cols,
                            final_id=update["final_id"],
                        )
                        if row is None:
                            continue

                        row_id, _, kv_col = row

                        try:
                            existing = json.loads(kv_col) if kv_col else {}
                        except (json.JSONDecodeError, TypeError, ValueError):
                            existing = {}

                        run_id = update["run_id"]
                        rank = update["rank"]
                        final_written = update["final_written"]
                        fid = update["final_id"]

                        if run_id is not None:
                            existing["run_id"] = run_id

                        fw_val = (
                            os.path.basename(str(final_written))
                            if final_written is not None
                            else None
                        )
                        final_keys = {
                            "final_unique_minimum": True,
                            "final_rank": int(rank) if rank is not None else None,
                            "final_written": fw_val,
                            "final_id": fid,
                        }
                        existing.update(
                            {k: v for k, v in final_keys.items() if v is not None}
                        )

                        conn.execute(
                            f"UPDATE systems SET {kvp} = ? WHERE id = ?",
                            (json.dumps(existing), row_id),
                        )
                        rows_updated_this_db += 1
                    return rows_updated_this_db

                rows_updated_this_db = retry_transaction(
                    db,
                    _mark_rows,
                    operation_name="mark_final_minima",
                    isolation_level="IMMEDIATE",
                )
                if rows_updated_this_db > 0:
                    total_rows_updated += rows_updated_this_db
                    dbs_touched.add(db_key)
                    details[db_key] = details.get(db_key, 0) + rows_updated_this_db
        except (sqlite3.DatabaseError, OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed marking final minima for %s: %s", db_path, e)
            continue

    return {
        "dbs_touched": len(dbs_touched),
        "rows_updated": total_rows_updated,
        "details": details,
    }
