"""Core helper utilities for SCGO."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize.optimize import Optimizer
from ase.vibrations import Vibrations
from scipy.spatial import KDTree

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
    MIN_ATOMIC_DISTANCE_WARNING,
    PENALTY_ENERGY,
)
from scgo.database.metadata import get_metadata
from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_shared_mobile_atom_indices,
)
from scgo.utils.logging import get_logger


def compute_final_id(atoms: Atoms, energy: float | None) -> str:
    """Compute a deterministic identifier for a final structure.

    The identifier is SHA256 over a canonical representation of the
    atomic species, rounded positions and the (optional) energy. This
    makes the id reproducible across runs for identical structures.

    Args:
        atoms: ASE Atoms object (positions are normalized by centering).
        energy: Energy value (may be None).

    Returns:
        Hexadecimal string (SHA256 digest).
    """
    # Use a copy and canonicalize positions to avoid mutating caller
    a = atoms.copy()
    # Only suppress attribute/type errors from ASE objects; do not hide unexpected issues
    with contextlib.suppress(AttributeError, TypeError):
        a.center()

    symbols = a.get_chemical_symbols()
    pos = a.get_positions()

    # Round positions for stable stringification
    pos_rounded = [[f"{x:.8f}" for x in triple] for triple in pos]

    parts: list[str] = ["|".join(symbols)]
    parts.extend([";".join(p) for p in pos_rounded])
    if energy is not None:
        parts.append(f"E={energy:.12e}")

    payload = "::".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assign_penalty_energy(atoms: Atoms) -> float:
    """Assign penalty energy to atoms object when relaxation fails.

    Args:
        atoms: The ASE Atoms object to assign penalty energy to.

    Returns:
        The penalty energy value (PENALTY_ENERGY).
    """
    from scgo.database.metadata import add_metadata

    add_metadata(
        atoms,
        potential_energy=PENALTY_ENERGY,
        raw_score=-PENALTY_ENERGY,
    )
    return PENALTY_ENERGY


def _rigid_shift_adsorbate_com_to_unit_cell(atoms: Atoms, n_slab: int) -> None:
    """Translate all atoms by a lattice vector so adsorbate COM lies in [0,1)^d in scaled space.

    Uses mass-weighted mean scaled position of atoms ``n_slab:``. For each axis with
    ``pbc`` True, subtracts ``floor(s_com[i])`` in fractional coordinates (equivalent
    Cartesian shift ``-floor @ cell``). Skips per-atom ``wrap()`` so clusters stay intact.

    Args:
        atoms: Combined slab + adsorbate system; modified in-place.
        n_slab: Number of leading slab atoms; adsorbate slice is ``n_slab:``.
    """
    n_tot = len(atoms)
    if n_slab <= 0 or n_slab >= n_tot:
        return

    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not np.any(pbc):
        return

    scaled = atoms.get_scaled_positions(wrap=False)
    masses = atoms.get_masses()[n_slab:]
    ads_scaled = scaled[n_slab:]
    s_com = np.average(ads_scaled, axis=0, weights=masses)

    n_int = np.zeros(3, dtype=float)
    for i in range(3):
        if pbc[i]:
            n_int[i] = np.floor(s_com[i])

    if not np.any(n_int):
        return

    cell = np.asarray(atoms.get_cell(), dtype=float)
    translation = -n_int @ cell
    atoms.positions = atoms.get_positions() + translation


def canonicalize_storage_frame(
    atoms: Atoms,
    *,
    pbc_aware: bool = True,
    center: bool = True,
    n_slab: int = 0,
) -> None:
    """Normalize atomic positions to a deterministic translation frame.

    Args:
        atoms: Atoms object to mutate in-place.
        pbc_aware: If True, apply PBC-aware normalization when any PBC axis is active.
        center: If True, center atoms in the current cell after optional wrap (ignored
            when ``n_slab > 0``).
        n_slab: If positive, treat atoms as slab (indices ``0..n_slab-1``) plus adsorbate.
            Uses a rigid lattice translation from adsorbate COM fractional coordinates on
            periodic axes (no per-atom ``wrap()``, no full-cell ``center()``).
    """
    if n_slab > 0:
        if pbc_aware:
            _rigid_shift_adsorbate_com_to_unit_cell(atoms, n_slab)
        return

    if pbc_aware and np.any(atoms.get_pbc()):
        atoms.wrap()
    if center:
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
        center_after_relax: If True (default), call ``atoms.center()`` after relaxation.
            Use False for slab+adsorbate systems.
        n_slab: If positive, slab+adsorbate rigid lattice canonicalization on periodic axes
            (see :func:`canonicalize_storage_frame`); use ``len(slab)`` for surface GA.

    Returns:
        The potential energy of the relaxed structure. Returns penalty energy if relaxation fails.
    """
    logger: Logger = get_logger(__name__)

    atoms.calc = calculator
    dyn: Optimizer = optimizer(atoms, trajectory=trajectory, logfile=logfile)

    try:
        positions: Any | np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            atoms.get_positions()
        )
        if len(positions) > 1:
            tree: KDTree[Any | np.ndarray[tuple[Any, ...], np.dtype[Any]]] = KDTree(
                positions
            )
            distances, indices = tree.query(positions, k=2)
            distances = np.asarray(distances)
            min_distance = np.min(distances[:, 1])

            if min_distance < MIN_ATOMIC_DISTANCE_WARNING:
                logger.warning(
                    f"Atoms dangerously close (min distance: {min_distance:.3f} Å)"
                )
                logger.warning(
                    "This may cause numerical issues with some calculators (especially EMT)"
                )

        dyn.run(fmax=fmax, steps=steps)
        energy = atoms.get_potential_energy()

        forces: np.ndarray[tuple[Any, ...], np.dtype[Any]] = ensure_float64_forces(
            atoms
        )

        canonicalize_storage_frame(
            atoms,
            pbc_aware=True,
            center=center_after_relax,
            n_slab=n_slab,
        )

        from scgo.database.metadata import add_metadata

        add_metadata(atoms, potential_energy=energy, raw_score=-energy)

        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

        return energy
    except KeyboardInterrupt:
        raise
    except (RuntimeError, ValueError, FloatingPointError) as e:
        logger.warning(f"Local relaxation failed: {e}")
        logger.warning("Assigning large penalty energy to this structure.")
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
            raise RuntimeError(
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


def extract_energy_from_atoms(atoms: Atoms) -> float | None:
    """Extract energy from atoms object, handling various formats.

    Attempts to extract energy from atoms object in order of preference:
    1. info['key_value_pairs']['raw_score'] (ASE GA database format, returns -raw_score)
    2. get_potential_energy() method (if calculator is attached)

    Args:
        atoms: The Atoms object to extract energy from.

    Returns:
        Energy value in eV, or None if energy cannot be extracted.
    """
    # Try unified metadata first
    raw = get_metadata(atoms, "raw_score", default=None)
    if raw is not None:
        return -float(raw)

    # Fallback to calculator energy
    try:
        return atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        return None


def validate_pair_id(pair_id: str) -> tuple[int, int]:
    r"""Validate canonical pair identifier 'i_j' and return (i, j).

    Raises:
        ValueError: if `pair_id` is not a string matching "^\d+_\d+$".
    """
    if not isinstance(pair_id, str):
        raise ValueError(f"Invalid pair_id type: {pair_id!r}")
    if not re.fullmatch(r"\d+_\d+", pair_id):
        raise ValueError(f"Invalid pair_id format: {pair_id!r} (expected 'i_j')")
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


def get_composition_counts(composition: list[str]) -> Counter[str]:
    """Get element counts for a composition.

    Args:
        composition: A list of atomic symbols, e.g., ["Au", "Pt", "Au"].

    Returns:
        A Counter mapping element symbols to their counts.
    """
    return Counter(composition)


def get_optimizer_db_filename(optimizer: str) -> str:
    """Get database filename for a given optimizer.

    Args:
        optimizer: Optimizer name (e.g., "bh", "ga", "simple").

    Returns:
        Database filename (e.g., "bh_go.db", "ga_go.db").
    """
    optimizer_name_lower: str = optimizer.lower()
    optimizer_db_map: dict[str, str] = {
        "bh": "bh_go.db",
        "simple": "bh_go.db",
        "ga": "ga_go.db",
    }
    return optimizer_db_map.get(optimizer_name_lower, f"{optimizer_name_lower}_go.db")


def get_provenance(atoms: Atoms) -> dict[str, Any]:
    """Get provenance (run_id, trial, trial_id) from canonical metadata.

    Returns an empty dict if metadata is not present or atoms.info doesn't exist.

    Args:
        atoms: The Atoms object to extract provenance from.

    Returns:
        Dictionary containing provenance information (may be empty).
    """
    return dict(getattr(atoms, "info", {}).get("metadata", {}))


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


def is_true_minimum(
    atoms: Atoms,
    calculator: Calculator,
    fmax_threshold: float = 0.05,
    check_hessian: bool = True,
    imag_freq_threshold: float = 50.0,
) -> bool:
    """Return True if `atoms` is a local minimum (force + optional Hessian checks)."""
    logger: Logger = get_logger(__name__)

    atoms_check: Atoms = atoms.copy()
    atoms_check.calc = calculator

    forces = atoms_check.get_forces()
    max_force = np.sqrt((forces**2).sum(axis=1).max())

    if max_force > fmax_threshold:
        logger.debug(
            f"Check failed: Max force ({max_force:.4f} eV/Å) > threshold ({fmax_threshold:.4f} eV/Å).",
        )
        return False

    if not check_hessian:
        logger.debug(
            "Check passed: Max force is below threshold (Hessian check skipped)."
        )
        return True

    logger.debug("Max force is OK. Performing vibrational analysis to check Hessian...")

    try:
        vib: Vibrations = Vibrations(atoms_check, name="vib_check")
        vib.run()
        frequencies: np.ndarray[tuple[Any, ...], np.dtype[Any]] = vib.get_frequencies()
        vib.clean()
    except (RuntimeError, OSError, ValueError) as e:
        # Treat vibrational analysis failures as a non-minimum condition
        logger.warning(f"Vibrational analysis failed with error: {e}")
        return False

    problematic_freqs = frequencies[frequencies < -imag_freq_threshold]
    if problematic_freqs.size > 0:
        logger.debug(
            f"Check failed: Found {problematic_freqs.size} imaginary frequencies "
            f"below -{imag_freq_threshold:.1f} cm-1: {np.round(problematic_freqs, 2)}.",
        )
        logger.debug("Structure is likely a saddle point.")
        return False

    total_imag_count = int(np.sum(frequencies < 0.0))
    moi = atoms_check.get_moments_of_inertia(vectors=False)
    is_linear: bool = any(np.isclose(moi, 0, atol=1e-5))
    expected_zero_modes: int = 5 if is_linear else 6

    logger.debug(
        f"Check passed: Found 0 imaginary frequencies above threshold ({imag_freq_threshold:.1f} cm-1).",
    )
    logger.debug(
        f"Total of {total_imag_count} imaginary/zero frequencies found (within threshold), "
        f"which is consistent with the {expected_zero_modes} expected translational/rotational modes.",
    )
    logger.debug("Structure is confirmed as a true local minimum.")
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

            comparison_indices = get_shared_mobile_atom_indices(
                atoms, unique_atoms_object
            )
            if comparer.looks_like(
                atoms[comparison_indices], unique_atoms_object[comparison_indices]
            ):
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
        sorted_minima: List of (energy, Atoms) tuples sorted by trial and energy.
        comparer: Structure comparator object.
        energy_tolerance: Maximum energy difference for potential duplicates.
        get_bin_index: Function to get bin index for an energy value.
        energy_bins: Dictionary mapping bin indices to lists of (energy, Atoms) tuples.

    Returns:
        List of unique (energy, Atoms) tuples.
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
) -> list[tuple[float, Atoms]]:
    """Filters a list of (energy, Atoms) tuples to identify unique structures.

    Args:
        minima_list: A list of (energy, Atoms) tuples, typically from one or
                     more optimization runs.
        energy_tolerance: The energy difference (in eV) below which two
                          structures are considered potential duplicates (if their
                          geometries also match). Defaults to `DEFAULT_ENERGY_TOLERANCE`.

    Returns:
        A new list of (energy, Atoms) tuples containing only the unique
        structures, sorted by energy from lowest to highest.
    """
    if not minima_list:
        return []

    valid_minima: list[tuple[float, Atoms]] = [
        (energy, atoms) for energy, atoms in minima_list if np.isfinite(energy)
    ]

    if not valid_minima:
        return []

    comparer: PureInteratomicDistanceComparator = PureInteratomicDistanceComparator(
        n_top=len(valid_minima[0][1]),
        tol=DEFAULT_COMPARATOR_TOL,
        pair_cor_max=DEFAULT_PAIR_COR_MAX,
        dE=energy_tolerance,
        mic=False,
    )

    sorted_minima: list[tuple[float, Atoms]] = sorted(
        valid_minima,
        key=lambda item: (
            get_provenance(item[1]).get("trial", float("inf")),
            item[0],
        ),
    )

    # Set up energy binning
    get_bin_index, energy_bins = _create_energy_bins(energy_tolerance, sorted_minima[0])

    # Find unique minima using binning optimization
    unique_minima: list[tuple[float, Atoms]] = _find_unique_minima_with_binning(
        sorted_minima, comparer, energy_tolerance, get_bin_index, energy_bins
    )

    # Sort by energy (lowest first)
    unique_minima.sort(key=lambda x: x[0])

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


def filter_dict_keys(d: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    """Filter dictionary to exclude specified keys.

    Args:
        d: Dictionary to filter.
        exclude: Set of keys to exclude.

    Returns:
        New dictionary with excluded keys removed.
    """
    return {k: v for k, v in d.items() if k not in exclude}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict.

    Recursively merges nested dictionaries, allowing override values
    to override base values while preserving unmodified base structure.

    Args:
        base: Base dictionary (from get_default_params).
        override: Override dictionary (user-provided minimal params).

    Returns:
        Merged dictionary with override values taking precedence.
    """
    merged: dict[str, Any] = deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def ensure_directory_exists(path: str | Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    This helper consolidates the common pattern of creating directories
    with exist_ok=True to avoid code duplication.

    Args:
        path: Directory path to ensure exists. Can be a string or Path object.
    """
    os.makedirs(path, exist_ok=True)


def _prepare_calculator_calculations(
    unique_minima: list[tuple[float, Atoms]],
    base_dir: str,
    calculator_name: str,
    write_function: Callable[..., None],
    **write_kwargs: Any,
) -> None:
    """Generic helper for preparing calculator input files for multiple structures.

    This function handles the common pattern of iterating through unique minima,
    creating subdirectories, and calling a write function for each structure.

    Args:
        unique_minima: A list of (energy, Atoms) tuples representing the
            unique structures to be calculated.
        base_dir: The base directory where subdirectories for each calculation
            will be created.
        calculator_name: Name of the calculator (for logging purposes).
        write_function: Function to call for each structure. Must accept
            `atoms` and `output_dir` as the first two positional arguments,
            followed by any additional kwargs.
        **write_kwargs: Additional keyword arguments to pass to write_function.
    """
    ensure_directory_exists(base_dir)
    logger: Logger = get_logger(__name__)
    logger.info(
        f"Preparing {calculator_name} inputs for {len(unique_minima)} unique minima in '{base_dir}'",
    )

    for i, (_energy, atoms) in enumerate(unique_minima):
        formula: str = get_cluster_formula(atoms.get_chemical_symbols())
        dir_name: str = f"minimum_{i + 1:02d}_{formula}"
        calc_dir: str = os.path.join(base_dir, dir_name)
        write_function(atoms=atoms, output_dir=calc_dir, **write_kwargs)
