"""I/O helpers for transition-state search outputs."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write as ase_write

from scgo.algorithms.ga_common import (
    extract_constraint_index_lists,
    reconstruct_constraints_from_index_lists,
)
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_TS_PAIR_COR_MAX,
)
from scgo.database import (
    extract_minima_from_database_file,
)
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.metadata.atoms import get_tag
from scgo.metadata.provenance import output_json_provenance
from scgo.pair_selection_defaults import pair_selection_param_defaults
from scgo.surface.validation import (
    validate_stored_mobile_partition_metadata,
    validate_stored_slab_adsorbate_metadata,
)
from scgo.ts_search.ts_statistics import compute_ts_statistics
from scgo.utils.comparators import ComparatorBlocks, PureInteratomicDistanceComparator
from scgo.utils.helpers import get_cluster_formula, validate_pair_id
from scgo.utils.logging import get_logger

from .transition_state import (
    _overlay_product_core,
    _permute_atoms_block_to_match,
    calculate_structure_similarity,
    minima_provenance_dict,
)

logger = get_logger(__name__)

# Absolute ceiling for adsorbate pair oversample before IDPP re-rank.
_ADSORBATE_PAIR_OVERSAMPLE_CAP = 50


def adsorbate_pair_select_cap(max_pairs: int) -> int:
    """Return oversample size: ``min(max_pairs * 10, max(max_pairs, 50))``."""
    mp = int(max_pairs)
    return min(mp * 10, max(mp, _ADSORBATE_PAIR_OVERSAMPLE_CAP))


def resolve_ts_pair_select_cap(
    max_pairs: int | None,
    *,
    has_adsorbate: bool,
    max_endpoint_mismatch: float | None,
) -> int | None:
    """Return the ``select_structure_pairs`` cap before NEB.

    ``max_pairs`` is always the final NEB budget. Adsorbate searches with an
    endpoint-mismatch gate oversample the select pool
    (:func:`adsorbate_pair_select_cap`) so an IDPP screen can re-rank
    candidates; bare systems (including surface presets that set
    ``max_endpoint_mismatch``) pass ``max_pairs`` through unchanged.
    """
    if max_pairs is None or int(max_pairs) <= 0:
        return max_pairs
    if has_adsorbate and max_endpoint_mismatch is not None:
        return adsorbate_pair_select_cap(int(max_pairs))
    return int(max_pairs)


def load_minima_by_composition(
    base_dir: str,
    composition: list[str] | None = None,
    prefer_final_unique: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Load minima from all runs, optionally filtered by composition.

    Scans ``base_dir`` for ``run_*/`` subdirectories containing ``*.db`` database
    files. Extracts minima from all databases and groups by chemical formula.

    By default only ``final_unique_minimum`` rows are loaded (canonical GO
    output). Structural deduplication across runs is left to callers (e.g.
    :func:`scgo.utils.helpers.filter_unique_minima` in TS search).

    Args:
        base_dir: Root directory containing ``run_*/`` subdirectories.
        composition: Optional list of atomic symbols to filter results (e.g., ["Pt", "Au"]).
            If provided, only minima matching this composition are returned.
        prefer_final_unique: If True (default), only final-tagged minima; set False
            to load all relaxed non-TS rows.

    Returns:
        Dictionary mapping composition formula strings to lists of (energy, Atoms) tuples,
        each sorted by energy (lowest first). Returns empty dict if no minima found.

    Example:
        >>> minima = load_minima_by_composition("Pt3_searches", ["Pt", "Pt", "Pt"])
        >>> list(minima.keys())
        ['Pt3']
    """
    if not os.path.exists(base_dir):
        logger.warning("Output directory does not exist: %s", base_dir)
        return {}

    minima_by_formula: dict[str, list[tuple[float, Atoms]]] = {}

    target_formula = get_cluster_formula(composition) if composition else None

    db_files_with_run = list_discovered_db_paths_with_run(
        base_dir, composition=composition, use_cache=False
    )

    for db_file, run_id in db_files_with_run:
        if not run_id:
            logger.warning(
                "Skipping database %s: could not resolve run_id from path layout",
                db_file,
            )
            continue
        try:
            try:
                db_relpath = os.path.relpath(db_file, base_dir)
            except (OSError, ValueError):
                db_relpath = os.path.basename(db_file)
            # prefer_final_unique -> require_final so only final_unique_minimum rows load.
            minima = extract_minima_from_database_file(
                db_file,
                run_id=run_id,
                require_final=prefer_final_unique,
                source_db_relpath=db_relpath,
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

            # Add to results with run_id in provenance
            if formula not in minima_by_formula:
                minima_by_formula[formula] = []

            for energy, atoms in minima:
                # Rebuild slab FixAtoms / adsorbate FixBondLengths from the
                # persisted index lists (the TorchSim GA path otherwise writes an
                # unconstrained relaxed row). Additive: a constraint already
                # present on the loaded Atoms (e.g. the native DB round-trip) is
                # never overwritten.
                reconstruct_constraints_from_index_lists(
                    atoms,
                    fix_atoms_indices=get_tag(atoms, "fix_atoms_indices_json"),
                    fix_bond_lengths_pairs=get_tag(
                        atoms, "fix_bond_lengths_pairs_json"
                    ),
                )
                validate_stored_slab_adsorbate_metadata(atoms)
                validate_stored_mobile_partition_metadata(atoms)
                minima_by_formula[formula].append((energy, atoms))

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

    return minima_by_formula


def _core_slice_atoms(atoms: Atoms, *, n_slab: int, n_core: int) -> Atoms:
    """Thin Atoms copy of ``[n_slab:n_slab+n_core]`` (layout: slab | core | adsorbate)."""
    i0 = max(0, int(n_slab))
    i1 = i0 + int(n_core)
    return Atoms(
        numbers=np.asarray(atoms.numbers[i0:i1], dtype=int),
        positions=np.asarray(atoms.get_positions()[i0:i1], dtype=float),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def _mobile_slice_atoms(atoms: Atoms, *, n_slab: int) -> Atoms:
    """Thin Atoms copy of atoms after the frozen slab prefix."""
    i0 = max(0, int(n_slab))
    return Atoms(
        numbers=np.asarray(atoms.numbers[i0:], dtype=int),
        positions=np.asarray(atoms.get_positions()[i0:], dtype=float),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def _pair_mic_context(
    atoms: Atoms, use_mic: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(cell, pbc)`` for MIC pair gates, or ``(None, None)``."""
    if use_mic and bool(np.any(atoms.pbc)):
        return (
            np.asarray(atoms.cell.array, dtype=float),
            np.asarray(atoms.pbc, dtype=bool),
        )
    return None, None


def _adsorbate_max_displacement(
    atoms_i: Atoms,
    atoms_j: Atoms,
    *,
    n_slab: int,
    n_core: int,
    use_mic: bool,
    n_adsorbate: int | None = None,
) -> float:
    """Max adsorbate-atom displacement after overlaying the metal/mobile core.

    Layout is ``[slab | core | adsorbate]``. Overlay (same operator as NEB
    endpoint prep) runs first; then adsorbate atoms are matched so a site hop
    is not mixed with rigid reorientation. Minima are not mutated.

    When ``n_adsorbate`` is set (surface adsorbate on a searchable top slab
    layer), the adsorbate block is the trailing ``n_adsorbate`` atoms and any
    middle mobile-slab block is treated as the overlay core even if
    ``n_core==0``.
    """
    n_slab = max(0, int(n_slab))
    n_core = max(0, int(n_core))
    if len(atoms_j) != len(atoms_i):
        return 0.0
    if n_adsorbate is not None:
        n_ads = max(0, int(n_adsorbate))
        n_middle = len(atoms_i) - n_slab - n_ads
        if n_middle < 0 or n_ads <= 0:
            return 0.0
        # Empty metal-core + searchable top layer: treat the middle block as
        # the overlay core so the hop is OH-only, not top-layer+OH.
        if n_core == 0 and n_middle > 0:
            n_core = n_middle
    else:
        n_ads = len(atoms_i) - n_slab - n_core
    if n_ads <= 0:
        return 0.0

    pos_i = np.asarray(atoms_i.get_positions(), dtype=float)
    mic_cell, mic_pbc = _pair_mic_context(atoms_i, use_mic)

    a0, a1 = n_slab + n_core, n_slab + n_core + n_ads
    pos_j, nums_j = _overlay_product_core(
        atoms_i,
        np.asarray(atoms_j.get_positions(), dtype=float),
        np.asarray(atoms_j.numbers, dtype=int),
        n_slab=n_slab,
        n_core=n_core,
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
    )

    ads_i = Atoms(
        numbers=np.asarray(atoms_i.numbers[a0:a1], dtype=int),
        positions=pos_i[a0:a1],
        cell=atoms_i.cell,
        pbc=atoms_i.pbc,
    )
    ads_j = Atoms(
        numbers=nums_j[a0:a1],
        positions=pos_j[a0:a1],
        cell=atoms_j.cell,
        pbc=atoms_j.pbc,
    )
    matched_ads, _matched_ads_nums = _permute_atoms_block_to_match(
        ads_i,
        ads_j,
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
    )
    pos_j[a0:a1] = matched_ads

    dlt = pos_j[a0:a1] - pos_i[a0:a1]
    if mic_cell is not None and mic_pbc is not None:
        dlt, _ = find_mic(dlt, mic_cell, mic_pbc)
    return float(np.max(np.linalg.norm(dlt, axis=1)))


def _core_rms_displacement(
    atoms_i: Atoms,
    atoms_j: Atoms,
    *,
    n_slab: int,
    n_core: int,
    use_mic: bool,
) -> float:
    """RMS Cartesian displacement of the core after overlaying the product core.

    Minima are not mutated. Gas cores overlay by fingerprint + Kabsch + spatial
    rematch; slab cores stay in the lab frame.
    """
    n_core = max(0, int(n_core))
    if n_core <= 0:
        return 0.0
    n_slab = max(0, int(n_slab))
    i0 = n_slab
    i1 = i0 + n_core
    if i1 > len(atoms_i) or i1 > len(atoms_j):
        return 0.0
    pos_i = np.asarray(atoms_i.get_positions()[i0:i1], dtype=float)
    mic_cell, mic_pbc = _pair_mic_context(atoms_i, use_mic)
    pos_j, _nums_j = _overlay_product_core(
        atoms_i,
        np.asarray(atoms_j.get_positions(), dtype=float),
        np.asarray(atoms_j.numbers, dtype=int),
        n_slab=n_slab,
        n_core=n_core,
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
    )
    dlt = pos_j[i0:i1] - pos_i
    if mic_cell is not None and mic_pbc is not None:
        dlt, _ = find_mic(dlt, mic_cell, mic_pbc)
    return float(np.sqrt(np.mean(np.sum(dlt * dlt, axis=1))))


def select_structure_pairs(
    minima: list[tuple[float, Atoms]],
    max_pairs: int | None = None,
    energy_gap_threshold: float | None = None,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = DEFAULT_TS_PAIR_COR_MAX,
    surface_aware: bool = False,
    *,
    use_mic: bool,
    n_slab: int | None = None,
    max_endpoint_mismatch: float | None = None,
    adsorbate_aware: bool = False,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    pair_core_rms_max: float | None = None,
    pair_score_gap_center: float | None = None,
    pair_score_gap_width: float | None = None,
    pair_score_cum_scale: float | None = None,
    pair_score_mismatch_scale: float | None = None,
    pair_score_core_rms_scale: float | None = None,
    pair_score_w_gap: float | None = None,
    pair_score_w_distinct: float | None = None,
    pair_score_w_mismatch: float | None = None,
    pair_score_w_core: float | None = None,
) -> list[tuple[int, int]]:
    """Select pairs of minima likely connected by a transition state.

    Endpoints should be close in energy and structurally related, with unequal
    component weights (layout ``[slab | core | adsorbate]``):

    - Bare cluster / bare surface (``adsorbate_aware=False``): fingerprint
      mobile atoms; skip near-duplicates; optionally gate large ``max_diff``.
    - Metal core + adsorbate (gas or on a slab, ``n_core_mobile > 0``):
      hard-gate on core similarity (``max_endpoint_mismatch``,
      ``pair_core_rms_max``). Same-core adsorbate site hops are kept even when
      the adsorbate moves a lot. Ranking uses ``pair_score_*`` knobs.
    - Adsorbate-only (``n_core_mobile == 0``): fingerprint / displace the
      mobile adsorbate; do not skip ``are_similar`` (OH fingerprints are often
      vacuous). ``max_endpoint_mismatch`` gates adsorbate Cartesian travel.

    Scoring / gate defaults come from
    :func:`scgo.pair_selection_defaults.pair_selection_param_defaults` and are
    overridable via ``ts_params`` (see docs).

    When ``max_pairs`` is set, survivors are ranked before taking the top N.
    Callers may pass an adsorbate oversample cap here; the TS runner still
    truncates to the user ``max_pairs`` NEB budget afterward (see
    :func:`resolve_ts_pair_select_cap`).

    Args:
        minima: List of (energy, Atoms) tuples, sorted by energy.
        max_pairs: Maximum number of pairs to generate. If None, all survivors.
        energy_gap_threshold: Max endpoint energy gap (eV).
        similarity_tolerance: Bare-path cumulative duplicate tolerance.
        similarity_pair_cor_max: Bare-path / core fingerprint pair-cor max.
        surface_aware: Select surface vs gas default score scales when overrides
            are omitted.
        use_mic: MIC for fingerprints and adsorbate displacement.
        n_slab: Frozen slab prefix; ignored in comparisons.
        max_endpoint_mismatch: Å gate — core fingerprint when a core exists,
            else adsorbate Cartesian max displacement.
        adsorbate_aware: Use adsorbate multi-component pairing rules.
        n_core_mobile: Metal-core atom count; ``0`` / ``None`` = no core block.
        pair_core_rms_max: Hard max core RMS (Å) for adsorbate+core pairs.
        pair_score_gap_center: Preferred energy gap (eV) for ranking.
        pair_score_gap_width: Gaussian width for the energy-gap score (eV).
        pair_score_cum_scale: Scale (Å) for distinctness / adsorbate-hop term.
        pair_score_mismatch_scale: Scale (Å) for fingerprint ``max_diff`` term.
        pair_score_core_rms_scale: Scale (Å) for core-RMS soft score.
        pair_score_w_gap: Ranking weight for the energy-gap term.
        pair_score_w_distinct: Ranking weight for distinctness / adsorbate hop.
        pair_score_w_mismatch: Ranking weight for fingerprint mismatch.
        pair_score_w_core: Ranking weight for core RMS (adsorbate+core only).

    Returns:
        List of ``(index1, index2)`` with ``index1 < index2``.
    """
    mic = bool(use_mic)

    if len(minima) < 2:
        logger.info("Only %d minima, need at least 2 to pair", len(minima))
        return []

    defaults = pair_selection_param_defaults(
        surface_aware=bool(surface_aware),
        adsorbate_aware=bool(adsorbate_aware),
    )

    def _resolve(name: str, value: float | None) -> float | None:
        return defaults[name] if value is None else value

    core_rms_limit = _resolve("pair_core_rms_max", pair_core_rms_max)
    gap_center = float(_resolve("pair_score_gap_center", pair_score_gap_center))
    gap_width = float(_resolve("pair_score_gap_width", pair_score_gap_width))
    cum_scale = float(_resolve("pair_score_cum_scale", pair_score_cum_scale))
    mismatch_scale = float(
        _resolve("pair_score_mismatch_scale", pair_score_mismatch_scale)
    )
    core_rms_scale = float(
        _resolve("pair_score_core_rms_scale", pair_score_core_rms_scale)
    )
    w_gap = float(_resolve("pair_score_w_gap", pair_score_w_gap))
    w_distinct = float(_resolve("pair_score_w_distinct", pair_score_w_distinct))
    w_mismatch = float(_resolve("pair_score_w_mismatch", pair_score_w_mismatch))
    w_core = float(_resolve("pair_score_w_core", pair_score_w_core))

    scored_pairs: list[tuple[float, int, int]] = []
    n_skipped_similar = 0
    n_skipped_mismatch = 0
    n_skipped_energy = 0
    n_skipped_core_rms = 0
    slab_len = int(n_slab) if n_slab is not None else 0
    n_core = int(n_core_mobile) if n_core_mobile is not None else 0
    fingerprint_core = bool(adsorbate_aware) and n_core > 0
    fingerprint_n = n_core if fingerprint_core else max(0, len(minima[0][1]) - slab_len)
    shared_comparator = PureInteratomicDistanceComparator(
        n_top=fingerprint_n,
        tol=similarity_tolerance,
        pair_cor_max=similarity_pair_cor_max,
        mic=mic,
    )

    def _score_candidate(
        gap: float,
        cum_diff: float,
        max_diff: float,
        core_rms: float | None,
        ads_hop: float | None,
    ) -> float:
        """Higher-is-better priority for capped ``max_pairs`` ranking."""
        gap_score = math.exp(-(((gap - gap_center) / max(1e-8, gap_width)) ** 2))
        if adsorbate_aware:
            if ads_hop is not None:
                distinct_score = 1.0 - math.exp(
                    -max(0.0, ads_hop) / max(1e-8, cum_scale)
                )
            else:
                distinct_score = math.exp(-max(0.0, cum_diff) / max(1e-8, cum_scale))
            mismatch_score = math.exp(-max(0.0, max_diff) / max(1e-8, mismatch_scale))
            score = (
                w_gap * gap_score
                + w_distinct * distinct_score
                + w_mismatch * mismatch_score
            )
            if core_rms is not None and w_core > 0.0:
                core_score = math.exp(-max(0.0, core_rms) / max(1e-8, core_rms_scale))
                score += w_core * core_score
            return score

        distinct_score = 1.0 - math.exp(-max(0.0, cum_diff) / max(1e-8, cum_scale))
        mismatch_score = math.exp(-max(0.0, max_diff) / max(1e-8, mismatch_scale))
        return (
            w_gap * gap_score
            + w_distinct * distinct_score
            + w_mismatch * mismatch_score
        )

    def _fingerprint_pair(atoms_i: Atoms, atoms_j: Atoms) -> tuple[float, float, bool]:
        if fingerprint_core:
            a_i = _core_slice_atoms(atoms_i, n_slab=slab_len, n_core=n_core)
            a_j = _core_slice_atoms(atoms_j, n_slab=slab_len, n_core=n_core)
            return calculate_structure_similarity(
                a_i,
                a_j,
                tolerance=similarity_tolerance,
                pair_cor_max=similarity_pair_cor_max,
                use_mic=mic,
                n_slab=None,
                ignore_fixed_atoms=False,
                comparator=shared_comparator,
            )
        if adsorbate_aware and slab_len > 0:
            a_i = _mobile_slice_atoms(atoms_i, n_slab=slab_len)
            a_j = _mobile_slice_atoms(atoms_j, n_slab=slab_len)
            return calculate_structure_similarity(
                a_i,
                a_j,
                tolerance=similarity_tolerance,
                pair_cor_max=similarity_pair_cor_max,
                use_mic=mic,
                n_slab=None,
                ignore_fixed_atoms=False,
                comparator=shared_comparator,
            )
        return calculate_structure_similarity(
            atoms_i,
            atoms_j,
            tolerance=similarity_tolerance,
            pair_cor_max=similarity_pair_cor_max,
            use_mic=mic,
            n_slab=n_slab,
            comparator=shared_comparator,
        )

    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            energy_i, atoms_i = minima[i]
            energy_j, atoms_j = minima[j]

            gap = abs(energy_j - energy_i)
            if energy_gap_threshold is not None and gap > energy_gap_threshold:
                n_skipped_energy += len(minima) - j
                break

            try:
                cum_diff, max_diff, are_similar = _fingerprint_pair(atoms_i, atoms_j)
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "Failed to calculate similarity for pair (%s, %s): %s: %s",
                    i,
                    j,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                continue

            # Bare: skip near-duplicates. Adsorbate: keep similar cores / vacuous
            # OH fingerprints (site hops are the intended pairs).
            if not adsorbate_aware and are_similar:
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

            ads_hop: float | None = None
            if adsorbate_aware:
                ads_hop = _adsorbate_max_displacement(
                    atoms_i,
                    atoms_j,
                    n_slab=slab_len,
                    n_core=n_core,
                    use_mic=mic,
                    n_adsorbate=n_adsorbate_mobile,
                )

            # Core present: gate on core fingerprint. Adsorbate-only: gate on
            # adsorbate Cartesian hop (fingerprint is often vacuous for OH).
            gate_metric = float(max_diff)
            if adsorbate_aware and not fingerprint_core and ads_hop is not None:
                gate_metric = float(ads_hop)
            if max_endpoint_mismatch is not None and gate_metric > float(
                max_endpoint_mismatch
            ):
                n_skipped_mismatch += 1
                logger.debug(
                    "Skipping pair (%s, %s): endpoint mismatch too large "
                    "(metric=%.3f Å > %.3f Å)",
                    i,
                    j,
                    gate_metric,
                    max_endpoint_mismatch,
                )
                continue

            core_rms: float | None = None
            if fingerprint_core:
                core_rms = _core_rms_displacement(
                    atoms_i,
                    atoms_j,
                    n_slab=slab_len,
                    n_core=n_core,
                    use_mic=mic,
                )
                if core_rms_limit is not None and core_rms > float(core_rms_limit):
                    n_skipped_core_rms += 1
                    logger.debug(
                        "Skipping pair (%s, %s): core RMS too large "
                        "(core_rms=%.3f Å > %.3f Å)",
                        i,
                        j,
                        core_rms,
                        core_rms_limit,
                    )
                    continue

            scored_pairs.append(
                (
                    _score_candidate(
                        gap,
                        float(cum_diff),
                        float(max_diff),
                        core_rms,
                        ads_hop,
                    ),
                    i,
                    j,
                )
            )

    if n_skipped_similar:
        logger.debug(
            "Pair selection: skipped %d too-similar candidate pairs",
            n_skipped_similar,
        )
    if n_skipped_mismatch:
        logger.debug(
            "Pair selection: skipped %d high-mismatch candidate pairs",
            n_skipped_mismatch,
        )
    if n_skipped_core_rms:
        logger.debug(
            "Pair selection: skipped %d high core-RMS candidate pairs",
            n_skipped_core_rms,
        )

    if not scored_pairs:
        if adsorbate_aware:
            e0 = float(minima[0][0])
            e1 = float(minima[-1][0])
            logger.warning(
                "Pair selection: no suitable pairs "
                "(energy=%d, mismatch=%d, core_rms=%d); energy span=%.3f eV",
                n_skipped_energy,
                n_skipped_mismatch,
                n_skipped_core_rms,
                abs(e1 - e0),
            )
        return []

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
    run_id: str | None = None,
) -> str:
    """Save all transition state results to ``results_summary.json``.

    Args:
        ts_results: List of result dictionaries from find_transition_state().
        output_dir: TS results root directory where summary will be saved.
        composition: List of atomic symbols for the composition.
        run_context: Optional NEB/search context merged into the summary.
        run_id: Optional run ID for the current TS search invocation.

    Returns:
        Path to saved summary file.
    """
    os.makedirs(output_dir, exist_ok=True)

    formula = get_cluster_formula(composition)

    summary = output_json_provenance(extra=run_context or {})
    summary.update(
        {
            "composition": composition,
            "formula": formula,
            "num_total_pairs": len(ts_results),
            "num_successful": sum(1 for r in ts_results if r["status"] == "success"),
            "num_converged": sum(1 for r in ts_results if r.get("neb_converged")),
            "current_run_id": run_id,
            "run_metadata_relpath": (
                f"{run_id}/metadata.json" if run_id is not None else None
            ),
            "run_timing_relpath": (
                f"{run_id}/timing.json"
                if run_id is not None
                and os.path.isfile(os.path.join(output_dir, run_id, "timing.json"))
                else None
            ),
            "results": [],
        }
    )

    for result in ts_results:
        # Create JSON-serializable result (remove Atoms objects)
        result_json = {
            "pair_id": result["pair_id"],
            "status": result["status"],
            "neb_converged": result.get("neb_converged", False),
            "n_images": result.get("n_images"),
            "spring_constant": result.get("spring_constant"),
            "reactant_energy": result.get("reactant_energy"),
            "product_energy": result.get("product_energy"),
            "ts_energy": result.get("ts_energy"),
            "barrier_height": result.get("barrier_height"),
            "error": result.get("error"),
        }
        # Persist the constraint index lists for the reactant/product/TS Atoms so
        # downstream consumers of results_summary.json can rebuild slab FixAtoms /
        # adsorbate FixBondLengths. Additive: only present roles are recorded and
        # only when they actually carry constraints (no Atoms -> no-op).
        for role, key in (
            ("reactant", "reactant_structure"),
            ("product", "product_structure"),
            ("transition_state", "transition_state"),
        ):
            struct = result.get(key)
            if struct is None:
                continue
            lists = extract_constraint_index_lists(struct)
            if lists["fix_atoms_indices"]:
                result_json[f"{role}_fix_atoms_indices_json"] = lists[
                    "fix_atoms_indices"
                ]
            if lists["fix_bond_lengths_pairs"]:
                result_json[f"{role}_fix_bond_lengths_pairs_json"] = lists[
                    "fix_bond_lengths_pairs"
                ]
        if result.get("minima_indices") is not None:
            result_json["minima_indices"] = result["minima_indices"]
        if result.get("minima_provenance") is not None:
            result_json["minima_provenance"] = result["minima_provenance"]
        if result["status"] == "success":
            result_json["ts_image_index"] = result.get("ts_image_index")

        summary["results"].append(result_json)

    # Keep statistics aligned with ts_network metadata output.
    summary["statistics"] = compute_ts_statistics(ts_results)

    summary_path = os.path.join(output_dir, "results_summary.json")
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
    *,
    use_mic: bool = False,
    n_slab: int | None = None,
    blocks: ComparatorBlocks | None = None,
    component_weights: dict[str, float] | None = None,
    cross_weight: float = 1.0,
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
                use_mic=use_mic,
                n_slab=n_slab,
                blocks=blocks,
                component_weights=component_weights,
                cross_weight=cross_weight,
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
    similarity_pair_cor_max: float = DEFAULT_TS_PAIR_COR_MAX,
    minima: list | None = None,
    minima_base_dir: str | None = None,
    run_context: dict[str, Any] | None = None,
    surface_aware: bool = False,
    n_slab: int | None = None,
    path_key: str | None = None,
    blocks: ComparatorBlocks | None = None,
    component_weights: dict[str, float] | None = None,
    cross_weight: float = 1.0,
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
    os.makedirs(output_dir, exist_ok=True)
    formula = path_key or get_cluster_formula(composition)

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
        summary_path = os.path.join(final_dir, "final_unique_ts_summary.json")
        empty_data: dict[str, Any] = output_json_provenance(extra=run_context or {})
        empty_data.update({"formula": formula, "unique_ts": []})
        if minima_base_dir is not None:
            empty_data["minima_base_dir"] = minima_base_dir
        with open(summary_path, "w") as f:
            json.dump(empty_data, f, indent=2)
        logger.info("No successful TSs to deduplicate for %s", formula)
        return []

    clusters = _cluster_ts_candidates_globally(
        candidates,
        energy_tolerance,
        similarity_tolerance,
        similarity_pair_cor_max,
        use_mic=surface_aware,
        n_slab=n_slab,
        blocks=blocks,
        component_weights=component_weights,
        cross_weight=cross_weight,
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
        if not surface_aware:
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

    summary_path = os.path.join(final_dir, "final_unique_ts_summary.json")
    summary_data: dict[str, Any] = output_json_provenance(extra=run_context or {})
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
