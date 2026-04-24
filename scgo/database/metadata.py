"""Metadata helper functions for SCGO databases."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ase import Atoms

from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)

# Cache of generations for which we've already emitted a debug-level
# metadata log. Prevents noisy repeated debug messages when many
# candidates in the same generation call `add_metadata`.
_debug_logged_generations: set[int] = set()


def add_metadata(
    atoms: Atoms,
    run_id: str | None = None,
    trial_id: int | None = None,
    generation: int | None = None,
    **extra_metadata: Any,
) -> None:
    """Add metadata to an Atoms object (stored in atoms.info['metadata'])."""
    # Initialize metadata dict if not present
    if "metadata" not in atoms.info:
        atoms.info["metadata"] = {}

    metadata = atoms.info["metadata"]

    # Store standard metadata
    if run_id is not None:
        metadata["run_id"] = run_id
    if trial_id is not None:
        metadata["trial_id"] = trial_id
    if generation is not None:
        metadata["generation"] = generation

    # Store extra metadata
    metadata.update(extra_metadata)

    # ASE DB persists key_value_pairs; ensure raw_score and run_id / trial_id for DB rows
    kv = atoms.info.setdefault("key_value_pairs", {})
    if "raw_score" in extra_metadata:
        kv["raw_score"] = extra_metadata["raw_score"]
    if run_id is not None:
        kv["run_id"] = run_id
    if trial_id is not None:
        kv["trial_id"] = trial_id

    # Provenance for run/trial discovery (used by scgo() results and downstream)
    if run_id is not None or trial_id is not None:
        prov = atoms.info.setdefault("provenance", {})
        if run_id is not None:
            prov["run_id"] = run_id
        if trial_id is not None:
            prov["trial_id"] = trial_id

    # TS search provenance: minima_source_db, minima_confids, minima_unique_ids,
    # ts_endpoint_provenance (per-endpoint dicts linking TS to GO minima rows).
    for key in (
        "minima_source_db",
        "minima_confids",
        "minima_unique_ids",
        "ts_endpoint_provenance",
    ):
        if key in extra_metadata:
            atoms.info.setdefault("provenance", {})[key] = extra_metadata[key]

    # Per-candidate: very verbose trace-level record so callers that want
    # full candidate-level detail can enable trace logging.
    keys = list(metadata.keys())
    # Use numeric TRACE level via logger.log(...) because the `.trace`
    # convenience method may not be installed in all runtime setups.
    logger.log(TRACE, "Added metadata to atoms: %s", keys)

    # Per-generation: emit a single debug-level message the first time
    # `add_metadata` is called for a particular generation to reduce
    # repeated debug noise during GA population processing.
    if generation is not None and generation not in _debug_logged_generations:
        logger.debug("Added metadata to atoms: %s", keys)
        _debug_logged_generations.add(generation)


def get_metadata(atoms: Atoms, key: str, default: Any = None) -> Any:
    """Retrieve a metadata value from an Atoms object, or return default.

    Order: ``metadata`` (canonical), ``provenance`` (TS / discovery), then
    ``key_value_pairs`` (ASE DB persisted fields, e.g. raw_score from GA).
    """
    for src in (
        atoms.info.get("metadata", {}),
        atoms.info.get("provenance", {}),
        atoms.info.get("key_value_pairs", {}),
    ):
        if key in src:
            return src[key]
    return default


def get_all_metadata(atoms: Atoms) -> dict[str, Any]:
    """Return metadata from atoms.info['metadata'] (canonical source)."""
    return dict(atoms.info.get("metadata", {}))


# ---------------------------------------------------------------------------
# Internal helpers: SCGO ``systems`` rows use ``key_value_pairs`` JSON (no
# separate ``metadata`` column in the SQLite schema for DBs we create).
# ---------------------------------------------------------------------------


def _parse_key_value_pairs_row(row: tuple) -> dict[str, Any]:
    """Parse ``key_value_pairs`` JSON from ``SELECT id, energy, key_value_pairs`` rows."""
    try:
        kv_json = row[2]
        if not kv_json:
            return {}
        return json.loads(kv_json)
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        return {}


def _is_row_relaxed_row(row: tuple) -> bool:
    return bool(_parse_key_value_pairs_row(row).get("relaxed"))


def _find_first_relaxed_row(rows: list) -> tuple | None:
    for r in rows:
        if _is_row_relaxed_row(r):
            return r
    return None


def _row_matches_run_trial_row(row: tuple, run_id, trial) -> bool:
    kv = _parse_key_value_pairs_row(row)
    if kv.get("run_id") != run_id:
        return False
    if trial is None:
        return True
    return kv.get("trial_id") == trial


def _extract_row_energy(r) -> float | None:
    """Return ``systems.energy`` (column index 1 in row selects)."""
    try:
        return float(r[1])
    except (TypeError, ValueError, IndexError):
        return None


def _select_best_row(
    rows: list,
    run_id,
    trial,
    energy: float | None,
    tolerance: float,
) -> tuple | None:
    """Select the best matching row from candidates using a priority cascade.

    Priority order:
    1. Row matching run/trial AND marked relaxed
    2. Row matching run/trial (any)
    3. Row closest in energy within tolerance
    """
    matching = [r for r in rows if _row_matches_run_trial_row(r, run_id, trial)]
    relaxed_match = _find_first_relaxed_row(matching)
    if relaxed_match is not None:
        return relaxed_match
    if matching:
        return matching[0]

    if energy is not None:
        best, best_delta = None, None
        for r in rows:
            e_db = _extract_row_energy(r)
            if e_db is None:
                continue
            delta = abs(energy - e_db)
            if (
                best is None
                or delta < best_delta
                or (delta == best_delta and r[0] < best[0])
            ):
                best, best_delta = r, delta
        if best is not None and best_delta <= tolerance:
            return best

    return None


def update_metadata(atoms: Atoms, **updates: Any) -> None:
    """Update ``atoms.info['metadata']``; mirror ``raw_score`` into key_value_pairs (ASE)."""
    if "metadata" not in atoms.info:
        atoms.info["metadata"] = {}

    atoms.info["metadata"].update(updates)

    if "raw_score" in updates:
        if "key_value_pairs" not in atoms.info:
            atoms.info["key_value_pairs"] = {}
        atoms.info["key_value_pairs"]["raw_score"] = updates["raw_score"]


def persist_provenance(
    atoms: Atoms,
    run_id: str | None = None,
    trial_id: int | None = None,
) -> None:
    """Persist run/trial provenance to ``atoms.info`` for discovery.

    Writes ``run_id`` / ``trial_id`` to provenance, ``metadata`` (via :func:`add_metadata`),
    and ``key_value_pairs`` for ASE persistence.
    """
    if run_id is not None or trial_id is not None:
        add_metadata(atoms, run_id=run_id, trial_id=trial_id)
    prov = atoms.info.setdefault("provenance", {})
    if run_id is not None:
        prov["run_id"] = run_id
    if trial_id is not None:
        prov["trial_id"] = trial_id
    if run_id is not None:
        atoms.info.setdefault("key_value_pairs", {})["run_id"] = run_id
    if trial_id is not None:
        atoms.info.setdefault("key_value_pairs", {})["trial_id"] = trial_id


def filter_by_metadata(
    structures: list[Atoms],
    **filters: Any,
) -> list[Atoms]:
    """Return structures whose metadata match all provided filters."""
    return [
        atoms
        for atoms in structures
        if all(get_metadata(atoms, key) == value for key, value in filters.items())
    ]


def mark_final_minima_in_db(
    final_minima_info: list[dict],
    base_dir: str | Path,
    db_paths: list[str | Path] | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """Mark final unique minima in database ``systems.key_value_pairs`` JSON rows.

    Args:
        final_minima_info: List of dicts with keys: 'energy' (float), 'atoms' (Atoms),
            'rank' (1-based int), 'final_written' (str filepath or filename)
        base_dir: Base output directory to search for database files (used by discovery)
        db_paths: Optional explicit list of database files to search/update (skips registry discovery)
        tolerance: Energy tolerance for best-match fallback (default 1e-4)

    Returns:
        dict: summary containing counts, e.g. {"dbs_touched": int, "rows_updated": int, "details": {db_path: rows}}
    """
    from scgo.database.connection import open_db
    from scgo.database.discovery import DatabaseDiscovery
    from scgo.database.sync import retry_transaction

    # Discovery tries the JSON registry first, then falls back to a recursive glob
    # under base_dir (canonical run_*/trial_* layout still matches the glob).
    discovery = DatabaseDiscovery(base_dir)

    # Summary counters
    total_rows_updated = 0
    dbs_touched: set[str] = set()
    details: dict[str, int] = {}

    for info in final_minima_info:
        atoms = info.get("atoms")
        energy = info.get("energy")
        rank = info.get("rank")
        final_written = info.get("final_written")

        if atoms is None:
            logger.warning("mark_final_minima_in_db: missing atoms entry, skipping")
            continue

        # Extract provenance and identifiers from canonical metadata
        run_id = get_metadata(atoms, "run_id")
        trial = get_metadata(atoms, "trial_id")
        confid = (
            get_metadata(atoms, "confid")
            or get_metadata(atoms, "gaid")
            or get_metadata(atoms, "id")
        )
        # Prefer explicit db_paths when provided (allows tagging non-registered DBs)
        if db_paths:
            db_files = [Path(p) for p in db_paths]
        else:
            db_files = discovery.find_databases(run_id=run_id, trial_id=trial)

        if not db_files:
            logger.warning(
                "mark_final_minima_in_db: no databases found for "
                f"run={run_id} trial={trial} — check output layout, registry, or pass db_paths"
            )
            continue

        for db_path in db_files:
            try:
                # Use retry_transaction for robust writes
                with (
                    open_db(db_path) as db,
                    retry_transaction(
                        db,
                        operation_name="mark_final_minima",
                    ) as conn,
                ):
                    kvp = SYSTEMS_JSON_COLUMN
                    select_cols = f"id, energy, {kvp}"

                    # Try exact match by final_id first (highest priority), then confid/gaid/id
                    row = None

                    # Prefer explicit final identifier when provided in final_minima_info
                    final_id = info.get("final_id")
                    if final_id is not None:
                        fid = str(final_id)
                        fid_conditions = [
                            f"CAST(json_extract({kvp}, '$.final_id') AS TEXT) = ?",
                            f"CAST(json_extract({kvp}, '$.unique_id') AS TEXT) = ?",
                            "CAST(unique_id AS TEXT) = ?",
                        ]
                        fid_params = [fid, fid, fid]
                        query = (
                            f"SELECT {select_cols} FROM systems WHERE "
                            + " OR ".join(fid_conditions)
                            + " ORDER BY rowid ASC"
                        )
                        cursor = conn.execute(query, tuple(fid_params))
                        rows = cursor.fetchall()
                        if rows:
                            chosen = _find_first_relaxed_row(rows)
                            row = chosen or rows[0]

                    # If final_id did not produce a match, try confid/gaid/id
                    if row is None and confid is not None:
                        confid_str = str(confid)
                        conditions = [
                            f"CAST(json_extract({kvp}, '$.confid') AS TEXT) = ?",
                            f"CAST(json_extract({kvp}, '$.gaid') AS TEXT) = ?",
                            f"CAST(json_extract({kvp}, '$.id') AS TEXT) = ?",
                            "CAST(unique_id AS TEXT) = ?",
                        ]
                        params = [confid_str, confid_str, confid_str, confid_str]
                        query = (
                            f"SELECT {select_cols} FROM systems WHERE "
                            + " OR ".join(conditions)
                            + " ORDER BY rowid ASC"
                        )
                        cursor = conn.execute(query, tuple(params))
                        rows = cursor.fetchall()
                        if rows:
                            chosen = _find_first_relaxed_row(rows)
                            row = chosen or rows[0]

                    # If no exact match, fallback to best-match by energy within run/trial
                    if row is None and run_id is not None:
                        if trial is not None:
                            query = f"""
                                SELECT {select_cols}
                                FROM systems
                                WHERE json_extract({kvp}, '$.run_id') = ?
                                  AND json_extract({kvp}, '$.trial_id') = ?
                            """
                            params = (run_id, trial)
                        else:
                            query = f"""
                                SELECT {select_cols}
                                FROM systems
                                WHERE json_extract({kvp}, '$.run_id') = ?
                            """
                            params = (run_id,)

                        cursor = conn.execute(query, params)
                        rows = cursor.fetchall()
                        if rows:
                            row = _select_best_row(
                                rows, run_id, trial, energy, tolerance
                            )
                            logger.debug(
                                "mark_final_minima_in_db: candidate row ids for "
                                "run=%s in %s: %s",
                                run_id,
                                db_path,
                                [str(r[0]) for r in rows],
                            )
                            if row is None:
                                continue

                    if row is None:
                        continue

                    row_id, _, kv_col = row

                    try:
                        existing = json.loads(kv_col) if kv_col else {}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing = {}

                    if run_id is not None:
                        existing["run_id"] = run_id
                    if trial is not None:
                        existing.setdefault("trial_id", trial)

                    final_id_val = info.get("final_id")
                    fw_val = (
                        os.path.basename(str(final_written))
                        if final_written is not None
                        else None
                    )
                    final_keys = {
                        "final_unique_minimum": True,
                        "final_rank": int(rank) if rank is not None else None,
                        "final_written": fw_val,
                        "final_id": str(final_id_val)
                        if final_id_val is not None
                        else None,
                    }
                    existing.update(
                        {k: v for k, v in final_keys.items() if v is not None}
                    )

                    conn.execute(
                        f"UPDATE systems SET {kvp} = ? WHERE id = ?",
                        (json.dumps(existing), row_id),
                    )

                    conn.commit()

                    # Track successful update for summary
                    total_rows_updated += 1
                    dbs_touched.add(str(db_path))
                    details[str(db_path)] = details.get(str(db_path), 0) + 1
            except (
                sqlite3.DatabaseError,
                sqlite3.OperationalError,
                OSError,
                json.JSONDecodeError,
                ValueError,
            ) as e:
                logger.warning(f"mark_final_minima_in_db: failed for {db_path}: {e}")
                continue

    return {
        "dbs_touched": len(dbs_touched),
        "rows_updated": total_rows_updated,
        "details": details,
    }
