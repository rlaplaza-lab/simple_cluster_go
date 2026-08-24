"""Core helper utilities for SCGO."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms as ASEFixAtoms
from ase.optimize.optimize import Optimizer
from ase.vibrations import Vibrations
from scipy.spatial import KDTree

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_FMAX_THRESHOLD,
    DEFAULT_PAIR_COR_MAX,
    MIN_ATOMIC_DISTANCE_WARNING,
    PENALTY_ENERGY,
)
from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.metadata.atoms import (
    get_tag,
    set_tags,
)
from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.composition import get_composition_counts
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _assign_penalty_energy(atoms: Atoms) -> float:
    """Assign penalty energy to atoms object when relaxation fails.

    Args:
        atoms: The ASE Atoms object to assign penalty energy to.

    Returns:
        The penalty energy value (PENALTY_ENERGY).
    """
    set_tags(
        atoms,
        potential_energy=PENALTY_ENERGY,
        raw_score=-PENALTY_ENERGY,
    )
    zero_forces = np.zeros((len(atoms), 3), dtype=np.float64)
    atoms.calc = SinglePointCalculator(atoms, energy=PENALTY_ENERGY, forces=zero_forces)
    return PENALTY_ENERGY


def adsorbate_primary_cell_shift(
    atoms: Atoms,
    *,
    n_slab: int,
) -> np.ndarray:
    """Cartesian shift that moves the mobile-region COM into the primary cell on PBC axes."""
    shift = np.zeros(3, dtype=float)
    n = len(atoms)
    if n_slab >= n:
        return shift
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    com = np.average(pos[n_slab:], axis=0, weights=masses[n_slab:])
    com_s = atoms.cell.scaled_positions(com.reshape(1, 3))[0]
    for i in range(3):
        if bool(atoms.pbc[i]):
            shift += float(np.floor(com_s[i])) * np.asarray(atoms.cell[i], dtype=float)
    return shift


def apply_primary_cell_shift(atoms: Atoms, shift: np.ndarray) -> None:
    """Apply a uniform Cartesian lattice shift and wrap (in-place)."""
    if np.any(shift != 0):
        atoms.positions = atoms.get_positions() - shift
    if np.any(atoms.get_pbc()):
        atoms.wrap()


def canonicalize_storage_frame(
    atoms: Atoms,
    *,
    center: bool = True,
    pbc_aware: bool = False,
    n_slab: int = 0,
) -> None:
    """Normalize Cartesian frame before persistence (translation / wrap only, in-place).

    Default (gas or non-slab): wrap periodic axes, then move the center of mass to
    the geometric center of the simulation cell.

    Slab + adsorbate (``pbc_aware=True`` and ``n_slab > 0``): shift all atoms by
    integer lattice translations so the mobile-region (indices ``n_slab:``)
    center of mass sits in the primary cell on periodic axes, then wrap. When
    ``n_slab == 0``, falls back to the gas-like path (wrap + optional centering).
    """
    if pbc_aware and n_slab > 0:
        n = len(atoms)
        if n_slab < n:
            apply_primary_cell_shift(
                atoms, adsorbate_primary_cell_shift(atoms, n_slab=n_slab)
            )
        return

    # Wrap periodic axes if needed
    if np.any(atoms.get_pbc()):
        atoms.wrap()

    if not center:
        return

    cell_center = 0.5 * (
        np.asarray(atoms.cell[0], dtype=float)
        + np.asarray(atoms.cell[1], dtype=float)
        + np.asarray(atoms.cell[2], dtype=float)
    )
    com = atoms.get_center_of_mass()
    atoms.positions = atoms.get_positions() + (cell_center - com)


def canonicalize_relaxed_for_storage(
    atoms: Atoms,
    *,
    surface_mode: bool = False,
    n_slab: int = 0,
) -> None:
    """Normalize relaxed structures immediately before GA database persistence.

    Gas clusters use ``atoms.center()`` so the cluster bounding box sits at the
    cell midpoint, matching :func:`perform_local_relaxation`. Slab+adsorbate
    systems use :func:`canonicalize_storage_frame` with ``pbc_aware=True`` to
    place the mobile region in the primary cell without recentering the slab.
    """
    if surface_mode and n_slab > 0:
        canonicalize_storage_frame(atoms, pbc_aware=True, center=False, n_slab=n_slab)
        return
    if np.any(atoms.get_pbc()):
        atoms.wrap()
    atoms.center()


def perform_local_relaxation(
    atoms: Atoms,
    calculator: Calculator,
    optimizer: type[Optimizer],
    fmax: float,
    steps: int,
    logfile: str | None = None,
    trajectory: str | None = None,
    *,
    center_after_relax: bool = True,
    surface_mode: bool = False,
    n_slab: int = 0,
) -> float:
    """Performs a local structure relaxation on an ASE Atoms object.

    Args:
        atoms: The ASE Atoms object to be relaxed.
        calculator: The ASE calculator for energy and force evaluations.
        optimizer: The ASE optimizer class (e.g., `FIRE`, `LBFGS`).
        fmax: The maximum force convergence criterion (in eV/Å).
        steps: The maximum number of optimization steps to perform.
        logfile: Optional path to a file for logging optimizer output.
        trajectory: Optional path to a file for saving the optimization trajectory.
        center_after_relax: If True (default), call
            :func:`.canonicalize_relaxed_for_storage` after relaxation (gas frame).
            Use False with ``surface_mode=True`` for slab+adsorbate systems.
        surface_mode: When True and ``center_after_relax`` is False, apply the
            slab-aware storage canonicalize (no bare PBC wrap / COM recenter).
        n_slab: Slab atom count for surface-mode canonicalize.

    Returns:
        The potential energy of the relaxed structure. Returns penalty energy if relaxation fails.
    """
    atoms.calc = calculator
    dyn: Optimizer = optimizer(atoms, trajectory=trajectory, logfile=logfile)

    try:
        positions = atoms.get_positions()
        if len(positions) > 1:
            tree = KDTree(positions)
            distances, _ = tree.query(positions, k=2)
            min_distance = np.min(distances[:, 1])
            if min_distance < MIN_ATOMIC_DISTANCE_WARNING:
                logger.warning(
                    "Atoms dangerously close (min distance: %.3f Å)", min_distance
                )
                logger.warning(
                    "This may cause numerical issues with some calculators (especially EMT)"
                )
        dyn.run(fmax=fmax, steps=steps)
        energy = atoms.get_potential_energy()
        forces = ensure_float64_forces(atoms)
        if center_after_relax:
            canonicalize_relaxed_for_storage(atoms)
        elif surface_mode:
            canonicalize_relaxed_for_storage(atoms, surface_mode=True, n_slab=n_slab)
        elif np.any(atoms.get_pbc()):
            atoms.wrap()

        set_tags(atoms, potential_energy=energy, raw_score=-energy)

        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

        return energy
    except KeyboardInterrupt:
        raise
    except (RuntimeError, ValueError, FloatingPointError) as e:
        logger.warning("Local relaxation failed: %s", e)
        logger.warning("Assigning large penalty energy to this structure")
        return _assign_penalty_energy(atoms)


def ensure_float64_forces(atoms: Atoms) -> np.ndarray:
    """Ensure forces in atoms object are float64 for database compatibility.

    Args:
        atoms: ASE Atoms object, modified in-place.

    Returns:
        The float64 forces array.
    """
    forces: np.ndarray | None = None
    if atoms.calc is not None:
        with contextlib.suppress(RuntimeError):
            forces = atoms.get_forces()

    if forces is None:
        if "forces" in atoms.arrays:
            forces = atoms.arrays["forces"]  # type: ignore[assignment]
        else:
            raise SCGORuntimeError(
                "Atoms object has no calculator and no forces in arrays."
            )

    forces_f64: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.asarray(
        forces, dtype=np.float64
    )
    atoms.arrays["forces"] = forces_f64

    if (
        atoms.calc is not None
        and hasattr(atoms.calc, "results")
        and "forces" in atoms.calc.results
    ):
        atoms.calc.results["forces"] = forces_f64

    return forces_f64


def copy_atoms(atoms: Atoms) -> Atoms:
    """Copy ``Atoms`` with nested ``info`` dicts isolated from the source.

    ASE's ``Atoms.copy()`` shallow-copies ``info``, so nested ``key_value_pairs``
    remain shared. TorchSim single-points then write ``raw_score`` into those
    shared dicts and corrupt minima used by later NEB pairs. Always use this
    helper when an Atoms object may receive calculator tag writes.
    """
    out = atoms.copy()
    out.info = dict(out.info)
    tags = out.info.get("key_value_pairs")
    if isinstance(tags, dict):
        out.info["key_value_pairs"] = dict(tags)
    return out


def extract_energy_from_atoms(atoms: Atoms) -> float | None:
    """Extract energy from atoms object, handling various formats.

    Attempts to extract energy from atoms object in order of preference:
    1. structure tag ``raw_score`` (ASE GA database format, returns -raw_score)
    2. get_potential_energy() method (if calculator is attached)

    Args:
        atoms: The Atoms object to extract energy from.

    Returns:
        Energy value in eV, or None if energy cannot be extracted.
    """
    raw = get_tag(atoms, "raw_score")
    if raw is not None:
        return -float(raw)

    # Fallback to calculator energy
    try:
        return atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        get_logger(__name__).debug(
            "Could not extract energy from atoms",
            exc_info=True,
        )
        return None


def validate_pair_id(pair_id: str) -> tuple[int, int]:
    r"""Validate canonical pair identifier 'i_j' and return (i, j).

    Raises:
        SCGOValidationError: If `pair_id` is not a string matching "^\d+_\d+$".
    """
    if not isinstance(pair_id, str):
        raise SCGOValidationError(f"Invalid pair_id type: {pair_id!r}")
    if not re.fullmatch(r"\d+_\d+", pair_id):
        raise SCGOValidationError(
            f"Invalid pair_id format: {pair_id!r} (expected 'i_j')"
        )
    i_str, j_str = pair_id.split("_")
    return int(i_str), int(j_str)


def extract_minima_from_database(
    candidates: list[Atoms],
) -> list[tuple[float, Atoms]]:
    """Extract energy and atoms from database candidates.

    Args:
        candidates: List of candidate objects from ASE database (each candidate
            is an Atoms object with info['key_value_pairs']['raw_score']).

    Returns:
        A list of (energy, Atoms) tuples sorted by energy (lowest first).
        Returns an empty list if no valid candidates are found.
    """
    if not candidates:
        return []

    all_minima = []
    for row in candidates:
        energy: float | None = extract_energy_from_atoms(row)
        if energy is not None:
            all_minima.append((energy, row))
    return sorted(all_minima, key=lambda x: x[0])


def get_cluster_formula(composition: list[str]) -> str:
    """Generate a chemical formula string from a list of atomic symbols.

    Args:
        composition: A list of atomic symbols, e.g., ["Au", "Pt", "Au"].

    Returns:
        A string representing the chemical formula, sorted alphabetically
        by element, e.g., 'Au2Pt'.
    """
    counts: Counter[str] = get_composition_counts(composition)
    return "".join(
        f"{elem}{count if count > 1 else ''}" for elem, count in sorted(counts.items())
    )


def get_ordered_formula(symbols: list[str]) -> str:
    """Generate a formula string preserving first-seen element order.

    Unlike :func:`get_cluster_formula`, this does not sort alphabetically, so
    ``["O", "H"]`` yields ``OH`` rather than ``HO``.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for symbol in symbols:
        key = str(symbol)
        if key not in counts:
            order.append(key)
            counts[key] = 0
        counts[key] += 1
    return "".join(
        f"{elem}{counts[elem] if counts[elem] > 1 else ''}" for elem in order
    )


def get_system_path_key(
    composition: list[str],
    *,
    adsorbate_definition: Any = None,
    surface_name: str | None = None,
) -> str:
    """Build an underscore-separated path key for campaign / formula directories.

    Parts (when present): nanoparticle formula, each adsorbate fragment formula
    (order-preserving), then surface name. Examples:
    ``Pt5``, ``Pt5_OH_OH``, ``Pt5_OH_OH_graphite``, ``Pt5_slab``.

    Chemical composition matching still uses :func:`.get_cluster_formula`.
    ``adsorbate_definition`` may be a plain dict or an
    :class:`~scgo.system_types.AdsorbateDefinition`.
    """
    # Deferred: ``system_types`` loads while ``utils.__init__`` is still importing
    # this module (via geometry_helpers → database → utils), so a top-level
    # import of ``as_adsorbate_definition`` would see a partial ``system_types``.
    from scgo.system_types import as_adsorbate_definition

    parts: list[str] = []
    ads_def = as_adsorbate_definition(adsorbate_definition)
    if ads_def is not None:
        core = list(ads_def.core_symbols)
        ads = list(ads_def.adsorbate_symbols)
        if core:
            parts.append(get_cluster_formula(core))
        lengths = list(ads_def.adsorbate_fragment_lengths)
        if ads:
            if lengths:
                if sum(lengths) != len(ads):
                    raise SCGOValidationError(
                        "adsorbate_fragment_lengths must sum to "
                        f"len(adsorbate_symbols) ({len(ads)}), got {lengths}."
                    )
                offset = 0
                for length in lengths:
                    frag = ads[offset : offset + length]
                    if frag:
                        parts.append(get_ordered_formula(frag))
                    offset += length
            else:
                parts.append(get_ordered_formula(ads))
        elif not core:
            formula = get_cluster_formula([str(s) for s in composition])
            if formula:
                parts.append(formula)
    else:
        formula = get_cluster_formula([str(s) for s in composition])
        if formula:
            parts.append(formula)

    if surface_name is not None:
        name = str(surface_name).strip()
        if name:
            parts.append(name)
    return "_".join(parts)


def is_true_minimum(
    atoms: Atoms,
    calculator: Calculator,
    fmax_threshold: float = DEFAULT_FMAX_THRESHOLD,
    check_hessian: bool = True,
    imag_freq_threshold: float = 50.0,
) -> bool:
    """Return True if ``atoms`` is a local minimum (force + optional Hessian check).

    The Hessian check is a Γ-point (single-supercell) analysis of the mobile
    subspace, which is the correct minimum-verification criterion for the
    supercell objective SCGO optimizes — it is **not** a phonon dispersion
    calculation and must not be read as one. A constrained structure (ASE
    ``FixAtoms``) therefore yields a *constrained* Hessian: only the free
    degrees of freedom are displaced, and the resulting immobile-block zero modes
    are the expected translational/rotational modes of the frozen block, not
    spurious saddle points.

    ASE ≥ 3.22 ``Vibrations`` derives its displaced-atom indices from internal
    ``FixAtoms`` constraints automatically, but the index derivation differs
    across versions and is not always reliable. To be version-independent we
    compute the vibration indices explicitly from ``atoms``' ``FixAtoms``
    constraints and pass them as ``Vibrations(atoms_check, indices=..., ...)``.

    ``imag_freq_threshold=50`` cm-1 is a finite-difference noise tolerance for
    MLIP force evaluations; callers with tighter needs may lower it.

    Args:
        atoms: Candidate structure (its ``FixAtoms`` constraints are honored).
        calculator: ASE calculator providing forces (and displaced forces).
        fmax_threshold: Maximum per-atom force magnitude (eV/Å) to accept.
        check_hessian: When False, only the force check runs.
        imag_freq_threshold: Imaginary-frequency magnitude (cm-1) above which a
            mode counts as a saddle-point signature.

    Returns:
        True when the structure passes the force check (and, if requested, the
        constrained-Hessian check).
    """
    atoms_check: Atoms = atoms.copy()
    atoms_check.calc = calculator

    forces = atoms_check.get_forces()
    max_force = np.sqrt((forces**2).sum(axis=1).max())

    if max_force > fmax_threshold:
        logger.debug(
            "Check failed: Max force (%.4f eV/Å) > threshold (%.4f eV/Å)",
            max_force,
            fmax_threshold,
        )
        return False

    if not check_hessian:
        logger.debug(
            "Check passed: Max force is below threshold (Hessian check skipped)"
        )
        return True

    logger.debug("Max force is OK. Performing vibrational analysis to check Hessian")

    # Derive vibration indices explicitly from any FixAtoms constraints on the
    # input so the Hessian is the constrained Hessian of the mobile subspace,
    # independent of ASE Vibrations' internal constraint handling (which changed
    # across 3.22 → 3.28). All atoms free -> full range; all atoms fixed ->
    # nothing to displace, skip the analysis.
    fixed_indices: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, ASEFixAtoms):
            fixed_indices.update(int(i) % len(atoms) for i in constraint.index)
    if fixed_indices:
        vib_indices = [i for i in range(len(atoms)) if i not in fixed_indices]
    else:
        vib_indices = list(range(len(atoms)))

    if len(vib_indices) == 0:
        logger.debug(
            "All atoms are fixed by constraints; skipping Hessian check "
            "(structure accepted as a constrained minimum on the force check)"
        )
        return True

    if (
        any(atoms.get_pbc())
        and not fixed_indices
        and logger.isEnabledFor(logging.WARNING)
    ):
        n_image_modes = 3 * len(atoms)
        logger.warning(
            "Hessian check on a periodic structure with no fixed atoms: SCGO "
            "evaluates the Γ-point supercell Hessian (%d modes). This is the "
            "correct minimum criterion for the supercell objective but is "
            "O(N) expensive; consider attaching FixAtoms for a cheaper "
            "constrained-Hessian check on slabs.",
            n_image_modes,
        )

    # A unique cache name per call (and a private temp dir) keeps concurrent
    # ProcessPoolExecutor workers from sharing/clobbering the same vib cache.
    vib_dir = tempfile.mkdtemp(prefix="scgo_vib_")
    vib_name = os.path.join(vib_dir, f"vib_check_{os.getpid()}_{uuid4().hex[:8]}")
    vib: Vibrations | None = None
    try:
        vib = Vibrations(atoms_check, indices=vib_indices, name=vib_name)
        vib.run()
        frequencies: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.atleast_1d(
            np.asarray(vib.get_frequencies())
        )
    except (RuntimeError, OSError, ValueError) as e:
        # Treat vibrational analysis failures as a non-minimum condition
        logger.warning("Vibrational analysis failed with error: %s", e)
        return False
    finally:
        if vib is not None:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                vib.clean()
        shutil.rmtree(vib_dir, ignore_errors=True)

    # ASE returns *complex* frequencies for imaginary modes (e.g. 0+965j), so a
    # plain ``frequencies < -threshold`` comparison never fires. Check the
    # imaginary part explicitly (magnitude, to stay sign-convention agnostic)
    # and keep a real-negative fallback for callers (and tests) that supply
    # purely real frequency arrays.
    imag_parts = np.abs(np.imag(frequencies))
    real_parts = np.real(frequencies)
    problematic_mask = (imag_parts > imag_freq_threshold) | (
        real_parts < -imag_freq_threshold
    )
    imag_count = int(np.sum(problematic_mask))
    if imag_count > 0:
        problematic_freqs = frequencies[problematic_mask]
        logger.debug(
            "Check failed: Found %s imaginary frequencies beyond %.1f cm-1: %s",
            imag_count,
            imag_freq_threshold,
            np.round(problematic_freqs, 2),
        )
        logger.debug("Structure is likely a saddle point")
        return False

    total_imag_count = int(np.sum((imag_parts > 0.0) | (real_parts < 0.0)))
    # Moments of inertia (and the 5/6 expected zero-mode count) only make sense
    # for an isolated, unconstrained, non-periodic structure. For slab/periodic
    # systems or constrained structures the free-block zero modes are a fixed
    # (and geometry-dependent) set, so we log the count as 0 and do not assert
    # on a fixed-mode expectation.
    is_constrained = bool(fixed_indices)
    is_periodic = bool(np.any(atoms.get_pbc()))
    if is_constrained or is_periodic:
        expected_zero_modes = 0
        moi = None
        is_linear = False
    else:
        moi = atoms_check.get_moments_of_inertia(vectors=False)
        is_linear = bool(any(np.isclose(moi, 0, atol=1e-5)))
        expected_zero_modes = 5 if is_linear else 6

    logger.debug(
        "Check passed: Found 0 imaginary frequencies below -%.1f cm-1",
        imag_freq_threshold,
    )
    logger.debug(
        "Total of %s imaginary/zero frequencies found (within threshold); "
        "expected %s translational/rotational zero modes "
        "(constrained=%s, periodic=%s, linear=%s)",
        total_imag_count,
        expected_zero_modes,
        is_constrained,
        is_periodic,
        is_linear,
    )
    logger.debug("Structure is confirmed as a true local minimum")
    return True


def _check_duplicate_in_energy_bins(
    atoms: Atoms,
    energy: float,
    bins_to_check: set[int],
    energy_bins: dict[int, list[tuple[float, Atoms]]],
    comparer: Any,
    energy_tolerance: float,
) -> bool:
    """Check if atoms structure is a duplicate in the specified energy bins.

    Args:
        atoms: Structure to check for duplicates.
        energy: Energy of the structure.
        bins_to_check: Set of bin indices to check (typically current bin ± 1).
        energy_bins: Dictionary mapping bin indices to lists of (energy, Atoms) tuples.
        comparer: Structure comparator object.
        energy_tolerance: Maximum energy difference for potential duplicates.

    Returns:
        True if a duplicate is found, False otherwise.
    """
    for check_bin in bins_to_check:
        if check_bin not in energy_bins:
            continue

        for unique_energy, unique_atoms_object in energy_bins[check_bin]:
            if len(atoms) != len(unique_atoms_object):
                continue

            energy_diff: float = abs(energy - unique_energy)
            if energy_diff > energy_tolerance:
                continue

            if comparer.looks_like(atoms, unique_atoms_object):
                return True

    return False


def _create_energy_bins(
    energy_tolerance: float, first_minimum: tuple[float, Atoms]
) -> tuple[Callable[[float], int], dict[int, list[tuple[float, Atoms]]]]:
    """Set up energy binning for duplicate detection.

    Args:
        energy_tolerance: Energy tolerance for duplicate detection.
        first_minimum: First (energy, Atoms) tuple to initialize bins.

    Returns:
        Tuple of (get_bin_index function, initialized energy_bins dictionary).
    """
    # Optimize with energy binning: group structures by energy bins
    # Structures in different bins can't be duplicates, reducing comparisons
    # Use bin width slightly larger than tolerance to catch edge cases
    bin_width: float = energy_tolerance * 1.5

    def get_bin_index(energy: float) -> int:
        """Get bin index for a given energy."""
        return int(energy / bin_width)

    energy_bins: dict[int, list[tuple[float, Atoms]]] = {}
    first_energy, first_atoms = first_minimum
    first_bin: int = get_bin_index(first_energy)
    energy_bins[first_bin] = [(first_energy, first_atoms)]

    return get_bin_index, energy_bins


def _find_unique_minima_with_binning(
    sorted_minima: list[tuple[float, Atoms]],
    comparer: Any,
    energy_tolerance: float,
    get_bin_index: Callable[[float], int],
    energy_bins: dict[int, list[tuple[float, Atoms]]],
) -> list[tuple[float, Atoms]]:
    """Find unique minima using energy binning optimization.

    Args:
        sorted_minima: List of (energy, Atoms) tuples sorted by energy
            (lowest first).
        comparer: Structure comparator object.
        energy_tolerance: Maximum energy difference for potential duplicates.
        get_bin_index: Function to get bin index for an energy value.
        energy_bins: Dictionary mapping bin indices to lists of (energy, Atoms) tuples.

    Returns:
        List of unique (energy, Atoms) tuples, preserving the input ``sorted_minima``
        energy order (energy-ascending by construction, since minima are appended in
        the order they appear in the already energy-sorted input).
    """
    unique_minima = []
    first_energy, first_atoms = sorted_minima[0]
    unique_minima.append((first_energy, first_atoms))

    for energy, atoms in sorted_minima[1:]:
        bin_idx: int = get_bin_index(energy)
        # Check same bin and adjacent bins to catch structures near bin boundaries
        bins_to_check: set[int] = {bin_idx - 1, bin_idx, bin_idx + 1}

        if not _check_duplicate_in_energy_bins(
            atoms, energy, bins_to_check, energy_bins, comparer, energy_tolerance
        ):
            unique_minima.append((energy, atoms))
            if bin_idx not in energy_bins:
                energy_bins[bin_idx] = []
            energy_bins[bin_idx].append((energy, atoms))

    return unique_minima


def filter_unique_minima(
    minima_list: list[tuple[float, Atoms]],
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    *,
    n_top: int,
    mic: bool = False,
    comparator_tol: float = DEFAULT_COMPARATOR_TOL,
    comparator_pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
) -> list[tuple[float, Atoms]]:
    """Filters a list of (energy, Atoms) tuples to identify unique structures.

    Args:
        minima_list: A list of (energy, Atoms) tuples, typically from one or
                     more optimization runs.
        energy_tolerance: The energy difference (in eV) below which two
                          structures are considered potential duplicates (if their
                          geometries also match). Defaults to `DEFAULT_ENERGY_TOLERANCE`.
        n_top: Trailing atoms to compare (system-type search-mobile count).
        mic: If True, use minimum-image convention for pairwise distances (slab PBC),
             matching :func:`scgo.algorithms.ga_common.create_structure_comparator`.
        comparator_tol: Cumulative structural difference tolerance.
        comparator_pair_cor_max: Maximum single-distance difference tolerance.

    Returns:
        A new list of (energy, Atoms) tuples containing only the unique
        structures, sorted by energy from lowest to highest.

    Note:
        Returned ``Atoms`` objects that lack a ``raw_score`` structure tag are
        annotated in-place with ``raw_score=-energy`` (ASE GA convention).
        Discarded duplicates are left unchanged.
    """
    if not minima_list:
        return []

    valid_minima: list[tuple[float, Atoms]] = [
        (energy, atoms) for energy, atoms in minima_list if np.isfinite(energy)
    ]

    if not valid_minima:
        return []

    comparer = PureInteratomicDistanceComparator(
        n_top=n_top,
        tol=comparator_tol,
        pair_cor_max=comparator_pair_cor_max,
        mic=mic,
    )

    sorted_minima: list[tuple[float, Atoms]] = sorted(
        valid_minima,
        key=lambda item: item[0],
    )

    # Set up energy binning
    get_bin_index, energy_bins = _create_energy_bins(energy_tolerance, sorted_minima[0])

    # Find unique minima using binning optimization
    unique_minima: list[tuple[float, Atoms]] = _find_unique_minima_with_binning(
        sorted_minima, comparer, energy_tolerance, get_bin_index, energy_bins
    )

    for energy, atoms in unique_minima:
        if get_tag(atoms, "raw_score") is None:
            set_tags(atoms, raw_score=-float(energy))

    return unique_minima


def _auto_scale_parameter(
    composition: list[str],
    *,
    base: int = 3,
    scaling: float = 35.0,
    min_val: int = 3,
    max_val: int = 1000,
) -> int:
    """Common scaling logic for auto parameters.

    Scales parameter value with cluster size using log1p.

    Args:
        composition: Cluster definition.
        base: Offset applied before scaling.
        scaling: Multiplier applied to log1p(n_atoms).
        min_val: Lower bound after scaling.
        max_val: Upper bound after scaling.

    Returns:
        Scaled integer parameter value.
    """
    n_atoms: int = max(len(composition), 1)
    scaled = base + scaling * np.log1p(n_atoms)
    return int(np.clip(scaled, min_val, max_val))


def auto_niter(
    composition: list[str],
    *,
    base: int = 3,
    scaling: float = 35.0,
    min_niter: int = 3,
    max_niter: int = 1000,
) -> int:
    """Heuristic iteration budget scaled by cluster size."""
    return _auto_scale_parameter(
        composition, base=base, scaling=scaling, min_val=min_niter, max_val=max_niter
    )


def auto_population_size(
    composition: list[str],
    *,
    base: int = 3,
    scaling: float = 35.0,
    min_population: int = 3,
    max_population: int = 1000,
) -> int:
    """Heuristic GA population size scaled by cluster size."""
    return _auto_scale_parameter(
        composition,
        base=base,
        scaling=scaling,
        min_val=min_population,
        max_val=max_population,
    )


def auto_niter_local_relaxation(
    composition: list[str],
    *,
    base: int = 50,
    scaling: float = 50.0,
    min_steps: int = 50,
    max_steps: int = 2000,
) -> int:
    """Heuristic number of relaxation steps scaled by cluster size."""
    return _auto_scale_parameter(
        composition,
        base=base,
        scaling=scaling,
        min_val=min_steps,
        max_val=max_steps,
    )


def auto_niter_ts(
    composition: list[str],
    *,
    base: int = 50,
    scaling: float = 180.0,
    min_steps: int = 150,
    max_steps: int = 5000,
) -> int:
    """Heuristic NEB/TS relaxation steps scaled by cluster size.

    Increased defaults to provide a larger automatic iteration budget for NEB/TS
    optimizations (e.g. Pt6 → ≈400 steps). The function preserves the
    log1p-scaling shape used by other auto helpers but raises the multiplier
    and minimum so `neb_steps='auto'` is more conservative for difficult NEBs.
    """
    return _auto_scale_parameter(
        composition,
        base=base,
        scaling=scaling,
        min_val=min_steps,
        max_val=max_steps,
    )


def deep_merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    copy_base: bool = True,
) -> dict[str, Any]:
    """Deep merge override dict into base dict.

    Recursively merges nested dictionaries, allowing override values
    to override base values while preserving unmodified base structure.

    Args:
        base: Base dictionary (from get_default_params).
        override: Override dictionary (user-provided minimal params).
        copy_base: If True (default), deep-copy ``base`` before merging so the
            caller receives an independent dict. Pass False when ``base`` is
            already a fresh copy (e.g. from :func:`~scgo.param_presets.get_default_params`).

    Returns:
        Merged dictionary with override values taking precedence.
    """
    if not override:
        return deepcopy(base) if copy_base else base

    merged: dict[str, Any] = deepcopy(base) if copy_base else base

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value, copy_base=False)
        else:
            merged[key] = value

    return merged


def ensure_directory_exists(path: str | Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
