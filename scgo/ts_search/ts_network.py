"""Transition-state DB integration and connectivity utilities."""

from __future__ import annotations

import contextlib
import glob
import json
import os
import sqlite3
from typing import Any

from ase import Atoms

from scgo.database import get_connection
from scgo.database.sync import PRESET_TS_NETWORK, database_retry
from scgo.metadata.atoms import get_tag, set_tags
from scgo.metadata.provenance import output_json_provenance
from scgo.ts_search.transition_state import minima_provenance_dict
from scgo.ts_search.ts_statistics import compute_ts_statistics
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger
from scgo.utils.path_keys import resolve_run_path_key

logger = get_logger(__name__)


def _stamp_ts_metadata(
    ts_atoms: Atoms,
    *,
    ts_energy: float,
    minima_idx_1: int,
    minima_idx_2: int,
    pair_id: str,
    barrier_height: float,
    neb_converged: bool,
    canonical_ts: bool,
    endpoint_provenance: list[dict[str, Any]] | None,
) -> None:
    """Stamp TS annotations into structure tags (``key_value_pairs``).

    Non-scalars and ASE-ambiguous strings are handled by :func:`set_tags`.
    """
    tags: dict[str, Any] = {
        "source": "ts_search",
        "connects": [minima_idx_1, minima_idx_2],
        "pair_id": pair_id,
        "potential_energy": ts_energy,
        "ts_connects_minima": f"{minima_idx_1}_{minima_idx_2}",
        "ts_pair_id": pair_id,
        "ts_barrier_height": barrier_height,
        "is_transition_state": True,
        "ts_neb_converged": neb_converged,
        "raw_score": -float(ts_energy),
    }
    if canonical_ts:
        tags["final_unique_ts"] = True

    if endpoint_provenance is not None:
        tags["ts_endpoint_provenance_json"] = [dict(p) for p in endpoint_provenance]

    set_tags(ts_atoms, **tags)
    ts_atoms.info.setdefault("data", {})


def add_ts_to_database(
    ts_structure: Atoms,
    ts_energy: float,
    minima_idx_1: int,
    minima_idx_2: int,
    db_file: str,
    pair_id: str,
    barrier_height: float,
    endpoint_provenance: list[dict[str, Any]] | None = None,
    *,
    canonical_ts: bool = False,
    neb_converged: bool = True,
) -> bool:
    """Add a transition state structure to the minima database.

    Args:
        ts_structure: Transition state Atoms object.
        ts_energy: Computed energy of the TS (eV).
        minima_idx_1: Index of first minimum this TS connects.
        minima_idx_2: Index of second minimum this TS connects.
        db_file: Path to minima database file (``*.db``).
        pair_id: Identifier for this pair (e.g., "0_1").
        barrier_height: Barrier height from minimum to TS (eV).
        endpoint_provenance: Optional list of two dicts (one per endpoint minimum)
            with DB/run identifiers, e.g. from ``minima_provenance_dict``, so TS
            rows can be traced back to the exact GO minima used for the NEB pair.
        canonical_ts: If True, tag the row as ``final_unique_ts`` (deduplicated
            converged TS from the standard pipeline). Integrator-only writes
            should leave this False.
        neb_converged: Whether the NEB reached convergence; stored for queries.

    Returns:
        True if successfully added, False otherwise.
    """
    if not os.path.exists(db_file):
        logger.error("Database file not found: %s", db_file)
        return False
    if ts_structure is None:
        logger.error("Cannot add TS %s: transition-state structure is missing", pair_id)
        return False

    if barrier_height < 0:
        logger.warning(
            "TS barrier height is negative (%.4f eV) for pair %s; storing anyway.",
            barrier_height,
            pair_id,
        )

    try:
        with get_connection(db_file) as da:
            ts_atoms = ts_structure.copy()
            ts_atoms.calc = None

            _stamp_ts_metadata(
                ts_atoms,
                ts_energy=ts_energy,
                minima_idx_1=minima_idx_1,
                minima_idx_2=minima_idx_2,
                pair_id=pair_id,
                barrier_height=barrier_height,
                neb_converged=neb_converged,
                canonical_ts=canonical_ts,
                endpoint_provenance=endpoint_provenance,
            )

            run_id_src = get_tag(ts_atoms, "run_id")
            if run_id_src is not None:
                set_tags(ts_atoms, run_id=run_id_src)

            def _add() -> bool:
                ts_db_atoms = ts_atoms.copy()
                with contextlib.suppress(AttributeError, TypeError, RuntimeError):
                    ts_db_atoms.center()
                if "tags" in ts_db_atoms.arrays:
                    del ts_db_atoms.arrays["tags"]
                da.add_relaxed_candidate(ts_db_atoms)
                return True

            database_retry(
                _add,
                config=PRESET_TS_NETWORK,
                exception_types=(sqlite3.OperationalError, OSError),
            )

        logger.debug(
            "Added TS %s (E=%.6f eV) to DB (minima %s-%s)",
            pair_id,
            ts_energy,
            minima_idx_1,
            minima_idx_2,
        )

        return True

    except (sqlite3.DatabaseError, OSError, ValueError):
        logger.exception("Error adding TS %s to database", pair_id)
        return False


def tag_unique_ts_in_databases(
    unique_ts: list[dict[str, Any]],
    minima: list,
    base_dir: str,
) -> int:
    """Persist deduplicated TS entries into discovered ``run_*/*.db`` minima databases.

    Iterates each unique TS edge, picks a candidate database that contains one
    of the two endpoint minima (matched on ``source_db`` basename), augments
    the TS Atoms with GO provenance, and calls :func:`add_ts_to_database`.

    Returns the number of successful row insertions.
    """
    db_files = glob.glob(os.path.join(base_dir, "run_*", "**", "*.db"), recursive=True)
    basename_to_path = {os.path.basename(p): p for p in db_files}
    logger.debug(
        "Tagging: discovered DB basenames for %s: %s",
        base_dir,
        list(basename_to_path.keys()),
    )

    added = 0
    missing_db_pairs: list[str] = []

    def _get_min_id(idx: int, key: str):
        if not (0 <= idx < len(minima)):
            return None
        return get_tag(minima[idx][1], key)

    for item in unique_ts:
        ts_energy = item.get("ts_energy")
        atoms_obj = item.get("_atoms_obj")
        edge_list: list[dict[str, Any]] = list(item.get("connected_edges") or [])
        if not edge_list:
            logger.warning(
                "Skipping unique TS without connected_edges while tagging DB: %s",
                item.get("filename"),
            )
            continue

        for edge in edge_list:
            pair_id = edge.get("pair_id")
            mi = edge.get("minima_indices")
            if pair_id is None or not isinstance(mi, (list, tuple)) or len(mi) != 2:
                continue
            i, j = int(mi[0]), int(mi[1])
            barrier = edge.get("barrier_height")
            neb_conv = edge.get("neb_converged")
            endpoint_prov = edge.get("minima_provenance")
            if ts_energy is None or barrier is None:
                logger.warning(
                    "Skipping TS %s for DB tag due to missing energies: ts=%s barrier=%s",
                    pair_id,
                    ts_energy,
                    barrier,
                )
                continue

            src_db_i = _get_min_id(i, "source_db")
            src_db_j = _get_min_id(j, "source_db")
            db_candidate = basename_to_path.get(src_db_i) or basename_to_path.get(
                src_db_j
            )
            if db_candidate is None:
                missing_db_pairs.append(str(pair_id))
                continue

            try:
                atoms_for_db = atoms_obj.copy() if atoms_obj is not None else None

                if atoms_for_db is not None:
                    run_id_src = _get_min_id(i, "run_id") or _get_min_id(j, "run_id")
                    if run_id_src is not None:
                        set_tags(atoms_for_db, run_id=run_id_src)

                    set_tags(
                        atoms_for_db,
                        connects=[i, j],
                        minima_source_db=[src_db_i, src_db_j],
                        minima_confids=[
                            _get_min_id(i, "confid"),
                            _get_min_id(j, "confid"),
                        ],
                        minima_unique_ids=[
                            _get_min_id(i, "unique_id"),
                            _get_min_id(j, "unique_id"),
                        ],
                        ts_connects_minima=f"{i}_{j}",
                    )

                success = add_ts_to_database(
                    ts_structure=atoms_for_db,
                    ts_energy=float(ts_energy),
                    minima_idx_1=int(i),
                    minima_idx_2=int(j),
                    db_file=db_candidate,
                    pair_id=str(pair_id),
                    barrier_height=float(barrier),
                    endpoint_provenance=endpoint_prov,
                    canonical_ts=True,
                    neb_converged=bool(neb_conv),
                )
                if success:
                    added += 1
                else:
                    logger.warning(
                        "add_ts_to_database returned False for %s -> %s",
                        pair_id,
                        db_candidate,
                    )
            except (sqlite3.DatabaseError, OSError, ValueError):
                logger.exception(
                    "Failed to add TS %s to DB %s",
                    pair_id,
                    db_candidate,
                )

    if missing_db_pairs:
        logger.warning(
            "No minima DB found to tag TS for %d edge(s) under %s. Sample pair_ids=%s",
            len(missing_db_pairs),
            base_dir,
            missing_db_pairs[:5],
        )

    if added:
        logger.info("Added %d unique TS to DB", added)

    return added


def save_ts_network_metadata(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    minima_count: int,
    minima: list | None = None,
    minima_base_dir: str | None = None,
    run_context: dict[str, Any] | None = None,
    path_key: str | None = None,
) -> str:
    """Write ``ts_network_metadata.json`` (edges, barriers, optional provenance).

    ``path_key`` is the component-aware directory identity; when omitted it is
    derived from ``composition`` and ``run_context['system_type']``. For
    slab-target runs (empty ``composition``) ``formula`` falls back to it so it
    is never empty.
    """
    os.makedirs(output_dir, exist_ok=True)

    resolved_path_key = path_key or resolve_run_path_key(
        composition,
        system_type=(run_context or {}).get("system_type"),
        params=run_context,
    )
    formula = get_cluster_formula(composition) or resolved_path_key

    network: dict[str, Any] = output_json_provenance(extra=run_context or {})
    network.update(
        {
            "path_key": resolved_path_key,
            "composition": composition,
            "formula": formula,
            "num_minima": minima_count,
            "ts_connections": [],
            "statistics": {
                "total_ts_found": 0,
                "converged_ts": 0,
                "successful_ts": 0,
                "min_barrier": None,
                "max_barrier": None,
                "avg_barrier": None,
            },
        }
    )

    for result in ts_results:
        if result.get("status") != "success":
            continue

        # Validate required numeric fields are present; skip and warn otherwise.
        reactant_energy = result.get("reactant_energy")
        product_energy = result.get("product_energy")
        ts_energy = result.get("ts_energy")
        barrier_height = result.get("barrier_height")

        if any(
            x is None
            for x in (reactant_energy, product_energy, ts_energy, barrier_height)
        ):
            logger.warning(
                "Skipping malformed successful TS result for pair %s: "
                "reactant=%s product=%s ts=%s barrier=%s",
                result.get("pair_id"),
                reactant_energy,
                product_energy,
                ts_energy,
                barrier_height,
            )
            continue

        parts = str(result["pair_id"]).split("_")
        if len(parts) != 2:
            logger.warning("Bad pair_id %r (expected i_j)", result.get("pair_id"))
            continue
        try:
            min_idx_1 = int(parts[0])
            min_idx_2 = int(parts[1])
        except ValueError as e:
            logger.warning("Could not parse pair_id %r: %s", result.get("pair_id"), e)
            continue

        connection = {
            "pair_id": result["pair_id"],
            "minima_indices": [min_idx_1, min_idx_2],
            "reactant_energy": float(reactant_energy),
            "product_energy": float(product_energy),
            "ts_energy": float(ts_energy),
            "barrier_height": float(barrier_height),
            "barrier_forward": (
                float(result["barrier_forward"])
                if result.get("barrier_forward") is not None
                else None
            ),
            "barrier_reverse": (
                float(result["barrier_reverse"])
                if result.get("barrier_reverse") is not None
                else None
            ),
            "neb_converged": bool(result.get("neb_converged")),
            "n_images": int(result.get("n_images", 0)),
        }
        if minima is not None:
            connection["minima_provenance"] = [
                minima_provenance_dict(minima, min_idx_1),
                minima_provenance_dict(minima, min_idx_2),
            ]

        network["ts_connections"].append(connection)

    network["statistics"] = compute_ts_statistics(network["ts_connections"])

    if minima_base_dir is not None:
        network["minima_base_dir"] = minima_base_dir

    # Save network metadata
    network_path = os.path.join(output_dir, "ts_network_metadata.json")
    with open(network_path, "w") as f:
        json.dump(network, f, indent=2)

    n_conn = len(network["ts_connections"])
    stats = network["statistics"]
    if stats.get("min_barrier") is not None:
        logger.info(
            "Wrote %s (%d edges, barriers %.4f-%.4f eV, mean %.4f eV)",
            network_path,
            n_conn,
            stats["min_barrier"],
            stats["max_barrier"],
            stats["avg_barrier"],
        )
    else:
        logger.info("Wrote %s (%d edges, no barrier stats)", network_path, n_conn)

    return network_path
