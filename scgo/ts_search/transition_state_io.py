"""I/O helpers for transition-state search outputs."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from ase import Atoms
from ase.io import write as ase_write

from scgo.constants import DEFAULT_COMPARATOR_TOL, DEFAULT_ENERGY_TOLERANCE
from scgo.database import (
    extract_minima_from_database_file,
)
from scgo.database.discovery import list_discovered_db_paths_with_run_trial
from scgo.database.metadata import add_metadata, get_metadata
from scgo.surface.validation import validate_stored_slab_adsorbate_metadata
from scgo.ts_search.ts_statistics import compute_ts_statistics
from scgo.utils.helpers import get_cluster_formula, validate_pair_id
from scgo.utils.logging import get_logger
from scgo.utils.ts_provenance import ts_output_provenance

from .transition_state import (
    calculate_structure_similarity,
    minima_provenance_dict,
)


def _relative_db_path(db_file: str, base_dir: str) -> str:
    """Return DB path relative to base dir, or basename when unavailable."""
    try:
        return os.path.relpath(db_file, base_dir)
    except (OSError, ValueError):
        return os.path.basename(db_file)


def load_minima_by_composition(
    base_dir: str,
    composition: list[str] | None = None,
    prefer_final_unique: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Load minima from all runs, optionally filtered by composition.

    Scans base_dir for run_*/ subdirectories containing *.db database files.
    Extracts minima from all databases and groups by chemical formula.

    By default only ``final_unique_minimum`` rows are loaded (canonical GO
    output). When ``prefer_final_unique`` is True, results are also
    deduplicated by ``final_id`` > ``final_written`` > ``final_rank`` when
    those keys exist.

    Args:
        base_dir: Root directory containing run_*/ subdirectories.
        composition: Optional list of atomic symbols to filter results (e.g., ["Pt", "Au"]).
            If provided, only minima matching this composition are returned.
        prefer_final_unique: If True (default), only final-tagged minima; set False
            to load all relaxed non-TS rows.

    Returns:
        Dictionary mapping composition formula strings to lists of (energy, Atoms) tuples,
        each sorted by energy (lowest first). Returns empty dict if no minima found.

    Example:
        >>> minima = load_minima_by_composition("Pt3_ts_searches", ["Pt", "Pt", "Pt"])
        >>> list(minima.keys())
        ['Pt3']
    """
    logger = get_logger(__name__)

    if not os.path.exists(base_dir):
        logger.warning("Output directory does not exist: %s", base_dir)
        return {}

    minima_by_formula: dict[str, list[tuple[float, Atoms]]] = {}

    target_formula = get_cluster_formula(composition) if composition else None

    db_files_with_run_trial = list_discovered_db_paths_with_run_trial(
        base_dir, composition=composition, use_cache=True
    )

    for db_file, run_id, trial_id in db_files_with_run_trial:
        try:
            # When prefer_final_unique=True, require_final=True so we only load
            # GO's canonical final unique minima (DB rows tagged final_unique_minimum).
            minima = extract_minima_from_database_file(
                db_file,
                run_id=run_id,
                trial_id=trial_id,
                require_final=prefer_final_unique,
            )

            if not minima:
                continue

            # Get composition from first structure
            first_atoms = minima[0][1]
            symbols = first_atoms.get_chemical_symbols()
            formula = get_cluster_formula(symbols)

            # Filter by target composition if specified
            if target_formula and formula != target_formula:
                continue

            # Add to results with run_id and trial_id in provenance
            if formula not in minima_by_formula:
                minima_by_formula[formula] = []

            for energy, atoms in minima:
                atoms_copy = atoms.copy()
                add_metadata(
                    atoms_copy,
                    run_id=run_id,
                    trial_id=trial_id,
                    source_db=os.path.basename(db_file),
                    source_db_relpath=_relative_db_path(db_file, base_dir),
                )
                validate_stored_slab_adsorbate_metadata(atoms_copy)
                minima_by_formula[formula].append((energy, atoms_copy))

        except (ValueError, OSError) as e:
            logger.warning(
                "Failed to load minima from %s: %s: %s",
                db_file,
                type(e).__name__,
                e,
            )

    # Sort each formula's minima by energy
    for formula in minima_by_formula:
        minima_by_formula[formula] = sorted(
            minima_by_formula[formula], key=lambda x: x[0]
        )

    # Prefer and deduplicate final-tagged minima by canonical keys.
    if prefer_final_unique:

        def _final_key(atoms: Atoms) -> tuple[str, str] | None:
            md = atoms.info.get("metadata", {}) or {}
            fid = md.get("final_id")
            if fid:
                return ("final_id", str(fid))
            fw = md.get("final_written")
            if fw:
                return ("final_written", str(fw))
            fr = md.get("final_rank")
            if fr is not None:
                return ("final_rank", str(fr))
            return None

        for formula, entries in list(minima_by_formula.items()):
            # Find entries explicitly tagged as final_unique_minimum
            final_entries = [
                e for e in entries if get_metadata(e[1], "final_unique_minimum", False)
            ]
            if not final_entries:
                # No final-tagged minima for this formula; keep original list
                continue

            # Group by canonical final key; unkeyed entries remain as-is.
            grouped: dict[tuple[str, str] | None, list[tuple[float, Atoms]]] = {}
            for energy, atoms in final_entries:
                k = _final_key(atoms)
                grouped.setdefault(k, []).append((energy, atoms))

            deduped: list[tuple[float, Atoms]] = []
            for key, group in grouped.items():
                group.sort(key=lambda x: x[0])
                if key is None:
                    deduped.extend(group)
                else:
                    deduped.append(group[0])
            deduped.sort(key=lambda x: x[0])

            # Replace minima list for this formula with deduplicated finals
            minima_by_formula[formula] = deduped

    return minima_by_formula


def select_structure_pairs(
    minima: list[tuple[float, Atoms]],
    max_pairs: int | None = None,
    energy_gap_threshold: float | None = None,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = 0.1,
    surface_aware: bool = False,
) -> list[tuple[int, int]]:
    """Select pairs of minima for TS calculations.

    Pairs nearby minima in energy space, using permutation-invariant structural
    comparison to avoid pairing very similar structures. When ``max_pairs`` is
    set, candidates are ranked by a physics-guided score (energy gap and
    structural dissimilarity) before taking the top N.

    Args:
        minima: List of (energy, Atoms) tuples, sorted by energy.
        max_pairs: Maximum number of pairs to generate. If None, generates all pairs.
            Default None.
        energy_gap_threshold: Only pair structures with energy gap below this threshold (eV).
            If None, pairs all structures. Default None.
        similarity_tolerance: Cumulative difference tolerance for structure comparison.
            Structures with cumulative difference below this value are considered too similar
            to pair. Default `DEFAULT_COMPARATOR_TOL` (tighter than GA duplicate detection).
        similarity_pair_cor_max: Maximum single distance difference tolerance.
            Default 0.1 Å (tighter than GA to ensure truly distinct structures).
        surface_aware: Use slightly looser scoring scales (slab / periodic systems).

    Returns:
        List of (index1, index2) tuples where index1 < index2, indicating which minima to pair.
    """
    logger = get_logger(__name__)

    if len(minima) < 2:
        logger.info(f"Only {len(minima)} minima, need at least 2 to pair")
        return []

    scored_pairs: list[tuple[float, int, int]] = []
    n_skipped_similar = 0

    def _score_candidate(gap: float, cum_diff: float, max_diff: float) -> float:
        """Return higher-is-better physics-guided priority score.

        Uses a compact blend of:
        - energy-gap proximity to a regime-dependent target window,
        - moderate structural dissimilarity preference,
        - endpoint mismatch penalty to avoid overly-discontinuous paths.
        """
        # Surface systems often tolerate/require slightly larger endpoint deltas.
        gap_center = 0.45 if surface_aware else 0.30
        gap_width = 0.55 if surface_aware else 0.40
        gap_score = math.exp(-(((gap - gap_center) / max(1e-8, gap_width)) ** 2))

        # Prefer distinct structures, but saturate to avoid over-valuing extremes.
        cum_scale = 0.12 if surface_aware else 0.09
        distinct_score = 1.0 - math.exp(-max(0.0, cum_diff) / max(1e-8, cum_scale))

        # Strongly penalize very large single-pair distortions.
        mismatch_scale = 0.45 if surface_aware else 0.35
        mismatch_penalty = math.exp(-max(0.0, max_diff) / max(1e-8, mismatch_scale))

        return 0.5 * gap_score + 0.35 * distinct_score + 0.15 * mismatch_penalty

    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            energy_i, atoms_i = minima[i]
            energy_j, atoms_j = minima[j]

            # Energy gap filter
            gap = abs(energy_j - energy_i)
            if energy_gap_threshold is not None and gap > energy_gap_threshold:
                # Minima are energy-sorted; once the gap is too large, later j
                # for this i can only increase it.
                break

            # Permutation-invariant similarity filter
            try:
                cum_diff, max_diff, are_similar = calculate_structure_similarity(
                    atoms_i,
                    atoms_j,
                    tolerance=similarity_tolerance,
                    pair_cor_max=similarity_pair_cor_max,
                    use_mic=surface_aware,
                )

                if are_similar:
                    n_skipped_similar += 1
                    logger.debug(
                        "Skipping pair (%s, %s): structures too similar "
                        "(cum_diff=%.4f, max_diff=%.3f Å)",
                        i,
                        j,
                        cum_diff,
                        max_diff,
                    )
                    continue
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    f"Failed to calculate similarity for pair ({i}, {j}): {type(e).__name__}: {e}"
                )
                continue

            scored_pairs.append(
                (_score_candidate(gap, float(cum_diff), float(max_diff)), i, j)
            )

    if n_skipped_similar:
        logger.debug(
            "Pair selection: skipped %d too-similar candidate pairs",
            n_skipped_similar,
        )

    if not scored_pairs:
        return []

    # Deterministic ordering: higher score first, then stable index tie-break.
    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranked_pairs = [(i, j) for _score, i, j in scored_pairs]
    if max_pairs is None:
        return ranked_pairs

    return ranked_pairs[:max_pairs]


def save_transition_state_results(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    run_context: dict[str, Any] | None = None,
) -> str:
    """Save all transition state results to a summary JSON file.

    Args:
        ts_results: List of result dictionaries from find_transition_state().
        output_dir: Directory where summary will be saved.
        composition: List of atomic symbols for the composition.

    Returns:
        Path to saved summary file.
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    formula = get_cluster_formula(composition)

    summary = ts_output_provenance(extra=run_context or {})
    summary.update(
        {
            "composition": composition,
            "formula": formula,
            "num_total_pairs": len(ts_results),
            "num_successful": sum(1 for r in ts_results if r["status"] == "success"),
            "num_converged": sum(1 for r in ts_results if r["neb_converged"]),
            "results": [],
        }
    )

    for result in ts_results:
        # Create JSON-serializable result (remove Atoms objects)
        result_json = {
            "pair_id": result["pair_id"],
            "status": result["status"],
            "neb_converged": result["neb_converged"],
            "n_images": result["n_images"],
            "spring_constant": result["spring_constant"],
            "reactant_energy": result["reactant_energy"],
            "product_energy": result["product_energy"],
            "ts_energy": result["ts_energy"],
            "barrier_height": result["barrier_height"],
            "error": result["error"],
        }
        if result.get("minima_indices") is not None:
            result_json["minima_indices"] = result["minima_indices"]
        if result.get("minima_provenance") is not None:
            result_json["minima_provenance"] = result["minima_provenance"]
        if result["status"] == "success":
            result_json["ts_image_index"] = result.get("ts_image_index")

        summary["results"].append(result_json)

    # Keep statistics aligned with ts_network metadata output.
    summary["statistics"] = compute_ts_statistics(ts_results)

    summary_path = os.path.join(output_dir, f"ts_search_summary_{formula}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "TS summary %s (success %s/%s, converged %s/%s)",
        summary_path,
        summary["num_successful"],
        summary["num_total_pairs"],
        summary["num_converged"],
        summary["num_total_pairs"],
    )

    return summary_path


def _cluster_ts_candidates_globally(
    candidates: list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]],
    energy_tolerance: float,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
) -> list[list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]]]:
    """Cluster TS candidates by energy + geometry in one deterministic pass."""
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c[0])
    clusters: list[list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]]] = []
    representatives: list[tuple[float, Atoms]] = []

    for cand in sorted_candidates:
        energy, atoms, *_ = cand
        matched_idx: int | None = None

        for idx, (rep_energy, rep_atoms) in enumerate(representatives):
            if abs(float(energy) - float(rep_energy)) > energy_tolerance:
                continue
            _cum, _maxd, are_similar = calculate_structure_similarity(
                rep_atoms,
                atoms,
                tolerance=similarity_tolerance,
                pair_cor_max=similarity_pair_cor_max,
            )
            if are_similar:
                matched_idx = idx
                break

        if matched_idx is None:
            clusters.append([cand])
            representatives.append((float(energy), atoms))
        else:
            clusters[matched_idx].append(cand)

    return clusters


def write_final_unique_ts(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = 0.1,
    minima: list | None = None,
    minima_base_dir: str | None = None,
    run_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate successful TS geometries globally and write unique `.xyz` files.

    Structures that are the same across different minima pairs (e.g. a
    bifurcation TS) are merged into one file. Each returned dict includes
    ``connected_edges`` listing every ``pair_id`` / ``minima_indices`` that
    produced that geometry.

    Returns a list of dictionaries with keys including:
      - ``connected_edges``, ``connected_minima``
      - ``pair_id``, ``minima_indices`` (first edge)
      - ``ts_energy``, ``barrier_height`` (from lowest-energy cluster member)
      - ``filename``, ``neb_converged``

    This function is best-effort and will not raise on IO errors.
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    formula = get_cluster_formula(composition)

    # Collect successful TS candidates
    candidates: list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]] = []
    for result in ts_results:
        if result.get("status") != "success":
            continue
        if not result.get("neb_converged", False):
            continue
        ts_atoms = result.get("transition_state")
        ts_energy = result.get("ts_energy")
        pair_id = result.get("pair_id")
        if ts_atoms is None or ts_energy is None or pair_id is None:
            continue
        # Parse minima indices from pair_id (strict validation)

        minima_indices = validate_pair_id(pair_id)

        candidates.append(
            (float(ts_energy), ts_atoms.copy(), pair_id, minima_indices, result)
        )

    final_dir = os.path.join(output_dir, "final_unique_ts")
    os.makedirs(final_dir, exist_ok=True)

    summary_list: list[dict[str, Any]] = []

    if not candidates:
        # Write empty summary
        summary_path = os.path.join(
            final_dir, f"final_unique_ts_summary_{formula}.json"
        )
        empty_data: dict[str, Any] = ts_output_provenance(extra=run_context or {})
        empty_data.update({"formula": formula, "unique_ts": []})
        if minima_base_dir is not None:
            empty_data["minima_base_dir"] = minima_base_dir
        with open(summary_path, "w") as f:
            json.dump(empty_data, f, indent=2)
        logger.info(f"No successful TSs to deduplicate for {formula}")
        return []

    clusters = _cluster_ts_candidates_globally(
        candidates,
        energy_tolerance,
        similarity_tolerance,
        similarity_pair_cor_max,
    )

    rank = 0
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda c: c[0])
        seen_pair: set[str] = set()
        connected_edges: list[dict[str, Any]] = []
        for _energy, _atoms, pair_id, minima_indices, result in cluster_sorted:
            if pair_id in seen_pair:
                continue
            seen_pair.add(pair_id)
            edge: dict[str, Any] = {
                "pair_id": pair_id,
                "minima_indices": [int(minima_indices[0]), int(minima_indices[1])],
                "barrier_height": result.get("barrier_height"),
                "neb_converged": bool(result.get("neb_converged", False)),
                "reactant_energy": result.get("reactant_energy"),
                "product_energy": result.get("product_energy"),
                "barrier_forward": result.get("barrier_forward"),
                "barrier_reverse": result.get("barrier_reverse"),
            }
            if minima is not None:
                i, j = minima_indices
                edge["minima_provenance"] = [
                    minima_provenance_dict(minima, i),
                    minima_provenance_dict(minima, j),
                ]
            connected_edges.append(edge)

        connected_edges.sort(
            key=lambda e: (e["minima_indices"][0], e["minima_indices"][1])
        )

        energy, atoms, _pid, _mi, result = min(cluster, key=lambda c: c[0])

        first_edge = connected_edges[0]
        pair_id = str(first_edge["pair_id"])
        minima_indices = [
            int(first_edge["minima_indices"][0]),
            int(first_edge["minima_indices"][1]),
        ]

        connected_minima_sorted = sorted(
            {idx for e in connected_edges for idx in e["minima_indices"]}
        )

        rank += 1
        atoms_clean = atoms.copy()
        atoms_clean.calc = None
        atoms_clean.center()
        if "tags" in atoms_clean.arrays:
            del atoms_clean.arrays["tags"]

        if len(connected_edges) > 1:
            filename = f"{formula}_ts_{rank:02d}.xyz"
        else:
            filename = f"{formula}_ts_{rank:02d}_pair_{first_edge['pair_id']}.xyz"
        filepath = os.path.join(final_dir, filename)
        ase_write(filepath, atoms_clean)

        item: dict[str, Any] = {
            "pair_id": pair_id,
            "ts_energy": float(energy),
            "barrier_height": result.get("barrier_height"),
            "minima_indices": minima_indices,
            "connected_edges": connected_edges,
            "connected_minima": connected_minima_sorted,
            "filename": filepath,
            "neb_converged": bool(result.get("neb_converged", False)),
            "_atoms_obj": atoms,
        }
        if minima is not None:
            i, j = minima_indices
            item["minima_provenance"] = [
                minima_provenance_dict(minima, i),
                minima_provenance_dict(minima, j),
            ]
        summary_list.append(item)

    # Write summary (serialize without Atoms objects)
    serializable_summary = []
    for item in summary_list:
        serial_item = {k: v for k, v in item.items() if k != "_atoms_obj"}
        serializable_summary.append(serial_item)

    summary_path = os.path.join(final_dir, f"final_unique_ts_summary_{formula}.json")
    summary_data: dict[str, Any] = ts_output_provenance(extra=run_context or {})
    summary_data.update({"formula": formula, "unique_ts": serializable_summary})
    if minima_base_dir is not None:
        summary_data["minima_base_dir"] = minima_base_dir
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(
        "Unique TS: %d structures in %s, summary %s",
        len(summary_list),
        final_dir,
        summary_path,
    )

    return summary_list
