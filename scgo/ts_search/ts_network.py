"""Transition-state DB integration and connectivity utilities."""

from __future__ import annotations

import contextlib
import glob
import heapq
import json
import os
import sqlite3
from collections import deque
from typing import Any, cast

from ase import Atoms
from ase_ga.data import DataConnection

from scgo.database.metadata import add_metadata, get_metadata, persist_provenance
from scgo.database.sync import retry_with_backoff
from scgo.ts_search.transition_state import minima_provenance_dict
from scgo.ts_search.ts_statistics import compute_ts_statistics
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger
from scgo.utils.ts_provenance import ts_output_provenance

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
    """Stamp TS annotations into ``info["metadata"]`` and (for ASE DB) ``key_value_pairs``."""
    md_payload: dict[str, Any] = {
        "potential_energy": ts_energy,
        "ts_connects_minima": f"{minima_idx_1}_{minima_idx_2}",
        "ts_pair_id": pair_id,
        "ts_barrier_height": barrier_height,
        "is_transition_state": True,
        "ts_neb_converged": neb_converged,
    }
    if canonical_ts:
        md_payload["final_unique_ts"] = True

    add_metadata(
        ts_atoms,
        source="ts_search",
        connects=[minima_idx_1, minima_idx_2],
        pair_id=pair_id,
        **md_payload,
    )

    if endpoint_provenance is not None:
        provenance_copy = [dict(p) for p in endpoint_provenance]
        add_metadata(ts_atoms, ts_endpoint_provenance=provenance_copy)
        ts_atoms.info.setdefault("key_value_pairs", {})[
            "ts_endpoint_provenance_json"
        ] = json.dumps(provenance_copy)

    ts_atoms.info.setdefault("data", {})
    # Mirror query-critical fields into key_value_pairs for ASE DB persistence.
    kvp = ts_atoms.info.setdefault("key_value_pairs", {})
    kvp["is_transition_state"] = True
    kvp["ts_neb_converged"] = neb_converged
    if canonical_ts:
        kvp["final_unique_ts"] = True
    kvp["raw_score"] = -float(ts_energy)


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
        db_file: Path to minima database file (*.db).
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
        logger.error(f"Database file not found: {db_file}")
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
        da = DataConnection(db_file)
        ts_atoms = ts_structure.copy()
        ts_atoms.calc = None
        if "tags" in ts_atoms.arrays:
            del ts_atoms.arrays["tags"]

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

        run_id_src = get_metadata(ts_atoms, "run_id")
        trial_src = get_metadata(ts_atoms, "trial_id")
        if run_id_src is not None or trial_src is not None:
            persist_provenance(ts_atoms, run_id=run_id_src, trial_id=trial_src)

        def _add() -> bool:
            ts_db_atoms = ts_atoms.copy()
            with contextlib.suppress(AttributeError, TypeError, RuntimeError):
                ts_db_atoms.center()
            if "tags" in ts_db_atoms.arrays:
                del ts_db_atoms.arrays["tags"]
            da.add_relaxed_candidate(ts_db_atoms)
            return True

        retry_with_backoff(
            _add,
            max_retries=5,
            initial_delay=0.05,
            backoff_factor=2.0,
            exception_types=(sqlite3.OperationalError, OSError),
        )

        logger.info(
            "Added TS %s (E=%.6f eV) to DB (minima %s–%s)",
            pair_id,
            ts_energy,
            minima_idx_1,
            minima_idx_2,
        )

        return True

    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError, ValueError):
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
        return get_metadata(minima[idx][1], key)

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
                    trial_src = _get_min_id(i, "trial_id") or _get_min_id(j, "trial_id")
                    if run_id_src is not None or trial_src is not None:
                        add_metadata(
                            atoms_for_db, run_id=run_id_src, trial_id=trial_src
                        )

                    add_metadata(
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
            except (
                sqlite3.DatabaseError,
                sqlite3.OperationalError,
                OSError,
                ValueError,
            ):
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

    return added


def save_ts_network_metadata(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    minima_count: int,
    minima: list | None = None,
    minima_base_dir: str | None = None,
    run_context: dict[str, Any] | None = None,
) -> str:
    """Write ``ts_network_metadata_<formula>.json`` (edges, barriers, optional provenance)."""
    os.makedirs(output_dir, exist_ok=True)

    formula = get_cluster_formula(composition)

    network: dict[str, Any] = ts_output_provenance(extra=run_context or {})
    network.update(
        {
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
    network_path = os.path.join(output_dir, f"ts_network_metadata_{formula}.json")
    with open(network_path, "w") as f:
        json.dump(network, f, indent=2)

    n_conn = len(network["ts_connections"])
    stats = network["statistics"]
    if stats.get("min_barrier") is not None:
        logger.info(
            "Wrote %s (%d edges, barriers %.4f–%.4f eV, mean %.4f eV)",
            network_path,
            n_conn,
            stats["min_barrier"],
            stats["max_barrier"],
            stats["avg_barrier"],
        )
    else:
        logger.info("Wrote %s (%d edges, no barrier stats)", network_path, n_conn)

    return network_path


def build_connectivity_graph(
    network_metadata_file: str,
) -> dict[str, Any]:
    """Build an adjacency representation of the TS network.

    Args:
        network_metadata_file: Path to TS network metadata JSON file.

    Returns:
        Dictionary with adjacency list and metadata for network analysis.
    """
    if not os.path.exists(network_metadata_file):
        logger.warning(f"Network metadata file not found: {network_metadata_file}")
        return {}

    with open(network_metadata_file) as f:
        network = cast(dict[str, Any], json.load(f))

    return _build_graph_from_connections(
        ts_connections=network.get("ts_connections", []),
        num_minima=int(network.get("num_minima", 0)),
        composition=network.get("composition"),
        formula=network.get("formula"),
        statistics=network.get("statistics"),
    )


def build_connectivity_graph_from_final_unique_ts(
    final_unique_ts_summary_file: str,
) -> dict[str, Any]:
    """Build connectivity graph from ``final_unique_ts_summary_*.json``.

    Requires schema entries with ``connected_edges``.
    """
    if not os.path.exists(final_unique_ts_summary_file):
        logger.warning(
            "Final unique TS summary file not found: %s",
            final_unique_ts_summary_file,
        )
        return {}

    with open(final_unique_ts_summary_file) as f:
        summary = json.load(f)

    unique_ts = summary.get("unique_ts", [])
    ts_connections: list[dict[str, Any]] = []
    maxima = -1

    for item in unique_ts:
        connected_edges = item.get("connected_edges")
        if not isinstance(connected_edges, list):
            continue
        for edge in connected_edges:
            minima_indices = edge.get("minima_indices")
            if (
                not isinstance(minima_indices, (list, tuple))
                or len(minima_indices) != 2
            ):
                continue
            idx1, idx2 = int(minima_indices[0]), int(minima_indices[1])
            maxima = max(maxima, idx1, idx2)
            ts_connections.append(
                {
                    "pair_id": edge.get("pair_id"),
                    "minima_indices": [idx1, idx2],
                    "barrier_height": edge.get("barrier_height"),
                    "barrier_forward": edge.get("barrier_forward"),
                    "barrier_reverse": edge.get("barrier_reverse"),
                    "ts_energy": edge.get("ts_energy"),
                    "neb_converged": edge.get("neb_converged"),
                }
            )

    # If minima_count isn't present in this summary type, infer from observed indices.
    num_minima = int(summary.get("num_minima", maxima + 1 if maxima >= 0 else 0))
    return _build_graph_from_connections(
        ts_connections=ts_connections,
        num_minima=num_minima,
        composition=summary.get("composition"),
        formula=summary.get("formula"),
        statistics=summary.get("statistics"),
    )


def _build_graph_from_connections(
    ts_connections: list[dict[str, Any]],
    num_minima: int,
    composition: list[str] | None,
    formula: str | None,
    statistics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build adjacency/edge metadata from a normalized edge list."""
    adjacency_sets: dict[int, set[int]] = {i: set() for i in range(num_minima)}
    edge_metadata: dict[tuple[int, int], dict[str, Any]] = {}

    for connection in ts_connections:
        minima_indices = connection.get("minima_indices")
        if not isinstance(minima_indices, (list, tuple)) or len(minima_indices) != 2:
            continue
        idx1, idx2 = int(minima_indices[0]), int(minima_indices[1])
        if idx1 not in adjacency_sets:
            adjacency_sets[idx1] = set()
        if idx2 not in adjacency_sets:
            adjacency_sets[idx2] = set()
        adjacency_sets[idx1].add(idx2)
        adjacency_sets[idx2].add(idx1)

        edge_key = tuple(sorted((idx1, idx2)))
        edge_metadata[edge_key] = {
            "pair_id": connection.get("pair_id"),
            "barrier": connection.get("barrier_height"),
            "forward": connection.get("barrier_forward"),
            "reverse": connection.get("barrier_reverse"),
            "ts_energy": connection.get("ts_energy"),
            "converged": connection.get("neb_converged"),
        }

    adjacency = {i: sorted(neighs) for i, neighs in adjacency_sets.items()}
    return {
        "num_nodes": len(adjacency),
        "num_edges": len(edge_metadata),
        "adjacency": adjacency,
        "edge_metadata": edge_metadata,
        "composition": composition,
        "formula": formula,
        "statistics": statistics or {},
    }


def get_connected_components(
    graph: dict[str, Any],
) -> list[set[int]]:
    """Find connected components in the TS network.

    Returns list of sets, each set contains minima indices in that component.

    Args:
        graph: Connectivity graph from build_connectivity_graph().

    Returns:
        List of sets, each representing a connected component.
    """
    adjacency = graph.get("adjacency", {})
    visited = set()
    components = []

    def dfs(node: int, component: set[int]) -> None:
        """Depth-first search to find component."""
        visited.add(node)
        component.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in range(graph.get("num_nodes", 0)):
        if node not in visited:
            component: set[int] = set()
            dfs(node, component)
            components.append(component)

    return sorted(components, key=len, reverse=True)


def find_shortest_path(
    graph: dict[str, Any],
    start: int,
    end: int,
) -> list[int] | None:
    """Find shortest path between two minima in TS network.

    Uses breadth-first search to find path with fewest TS hops.

    Args:
        graph: Connectivity graph from build_connectivity_graph().
        start: Index of starting minimum.
        end: Index of ending minimum.

    Returns:
        List of minima indices from start to end, or None if no path exists.
    """
    adjacency: dict[int, list[int]] = graph.get("adjacency", {})

    if start == end:
        return [start]
    if start not in adjacency or end not in adjacency:
        return None

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        for neighbor in adjacency.get(node, []):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None  # No path exists


def find_minimum_barrier_path(
    graph: dict[str, Any],
    start: int,
    end: int,
) -> tuple[list[int], float] | None:
    """Find path between minima with lowest accumulated barriers.

    Args:
        graph: Connectivity graph from build_connectivity_graph().
        start: Index of starting minimum.
        end: Index of ending minimum.

    Returns:
        Tuple of (path, total_barrier) or None if no path exists.
    """
    adjacency: dict[int, list[int]] = graph.get("adjacency", {})
    edge_metadata = graph.get("edge_metadata", {})

    if start == end:
        return [start], 0.0
    if start not in adjacency or end not in adjacency:
        return None

    # Dijkstra's algorithm
    distances = {node: float("inf") for node in adjacency}
    distances[start] = 0.0
    parents = dict.fromkeys(adjacency, None)

    priority_queue = [(0.0, start)]

    while priority_queue:
        current_dist, node = heapq.heappop(priority_queue)

        if current_dist > distances[node]:
            continue

        if node == end:
            # Reconstruct path
            path: list[int] = []
            current: int | None = end
            while current is not None:
                path.append(current)
                current = parents[current]
            return list(reversed(path)), distances[end]

        for neighbor in adjacency.get(node, []):
            edge_key = tuple(sorted([node, neighbor]))
            raw_barrier = edge_metadata.get(edge_key, {}).get("barrier", float("inf"))
            try:
                barrier = float(raw_barrier)
            except (TypeError, ValueError):
                barrier = float("inf")
            if neighbor not in distances:
                distances[neighbor] = float("inf")
                parents[neighbor] = None

            new_dist = current_dist + barrier

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                parents[neighbor] = node
                heapq.heappush(priority_queue, (new_dist, neighbor))

    return None  # No path exists
