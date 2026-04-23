"""SQLite database setup and helpers for SCGO (ASE ``DataConnection``)."""

from __future__ import annotations

import contextlib
import glob
import heapq
import multiprocessing
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase_ga.data import DataConnection, PrepareDB

from scgo.constants import PENALTY_ENERGY
from scgo.database.connection import (
    apply_sqlite_pragmas,
    close_data_connection,
    get_connection,
)
from scgo.database.discovery import list_discovered_db_paths_with_run_trial
from scgo.database.exceptions import DatabaseSetupError
from scgo.database.metadata import add_metadata, get_metadata
from scgo.database.registry import get_registry
from scgo.database.schema import (
    CURRENT_SCHEMA_VERSION,
    is_scgo_database,
    set_schema_version,
)
from scgo.database.streaming import _iter_relaxed_minima_from_da
from scgo.database.sync import PRESET_AGGRESSIVE, retry_on_lock, retry_with_backoff
from scgo.utils.helpers import (
    ensure_directory_exists,
    extract_energy_from_atoms,
    extract_minima_from_database,
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger
from scgo.utils.run_tracking import load_run_metadata

logger = get_logger(__name__)


def _ensure_database_indices(db_path: str) -> None:
    """Create SQLite indices for performance.

    Creates indices on commonly queried columns to improve performance
    of database operations. Indices are created with IF NOT EXISTS to
    allow safe re-running.

    Args:
        db_path: Path to the database file
    """
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            # Index on energy column for sorting/filtering
            conn.execute("CREATE INDEX IF NOT EXISTS idx_energy ON systems(energy)")

            # Index on id for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON systems(id)")

            # Index on unique_id if it exists (ASE GA tracking)
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_unique_id ON systems(unique_id)"
                )

            conn.commit()
            logger.debug(f"Database indices created for {db_path}")
    except sqlite3.OperationalError as e:
        # Log but don't fail - indices are optional optimizations
        logger.debug(f"Could not create all indices on {db_path}: {e}")
    except OSError as e:
        logger.warning(f"Unexpected error creating indices on {db_path}: {e}")


def _write_scgo_metadata(da: DataConnection, db_file: str) -> None:
    """Write minimal SCGO metadata (best-effort).

    Primary path uses ASE ``managed_connection`` via :func:`set_schema_version`.
    If that path fails and the file is still not marked SCGO, open SQLite
    directly once to create ``scgo_metadata`` (same keys).
    """
    with contextlib.suppress(
        sqlite3.DatabaseError, sqlite3.OperationalError, ValueError
    ):
        set_schema_version(da, CURRENT_SCHEMA_VERSION)

    if is_scgo_database(db_file):
        return

    try:
        with sqlite3.connect(db_file, timeout=5.0) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scgo_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('created_by', ?)",
                ("scgo",),
            )
            conn.execute(
                "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('schema_version', ?)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.debug("Failed direct sqlite metadata write for %s: %s", db_file, e)


def _trial_id_from_output_dir(base_path: Path) -> int | None:
    """Parse trial index from a canonical ``trial_<n>`` output directory name."""
    name = base_path.name
    if not name.startswith("trial_"):
        return None
    with contextlib.suppress(ValueError, IndexError):
        return int(name.split("_", 1)[1])
    return None


def _register_database_best_effort(
    base_dir: str | Path, db_file: str, atoms_template: Atoms | None, run_id: str | None
) -> None:
    """Best-effort register DB in persistent registry (no exceptions).

    If ``base_dir`` lies under a parent whose name ends with ``_searches``,
    registers only at that parent (one index file per campaign). Otherwise
    registers at ``base_dir``. ``trial_id`` is taken from a ``trial_<n>``
    directory name when present.
    """
    comp_list = None
    if atoms_template is not None:
        try:
            comp_list = atoms_template.get_chemical_symbols()
        except (AttributeError, TypeError) as e:
            # Non-fatal: composition extraction failed — continue without composition
            logger.debug(
                "Could not extract composition from atoms_template for %s: %s",
                db_file,
                e,
            )
            comp_list = None

    base_path = Path(base_dir)
    trial_id = _trial_id_from_output_dir(base_path)
    search_root = next(
        (p for p in base_path.parents if p.name.endswith("_searches")), None
    )
    if search_root is not None:
        registry_roots: list[Path] = [search_root]
    else:
        registry_roots = [base_path]

    for root in registry_roots:
        try:
            get_registry(root).register_database(
                Path(db_file),
                composition=comp_list,
                run_id=run_id,
                trial_id=trial_id,
            )
            logger.debug("Registered database in registry root %s: %s", root, db_file)
        except (ValueError, OSError, sqlite3.DatabaseError) as _e:
            logger.debug(
                "Registry registration failed for %s in %s: %s", db_file, root, _e
            )


def setup_database(
    output_dir: str | Path,
    db_filename: str,
    atoms_template: Atoms,
    initial_candidate: Atoms | None = None,
    initial_population: list[Atoms] | None = None,
    remove_existing: bool = True,
    remove_aux_files: bool = False,
    enable_wal_mode: bool = False,
    run_id: str | None = None,
) -> DataConnection:
    """Create/open an ASE `DataConnection` for `db_filename` in `output_dir`.

    Preserves optional initial candidates; supports WAL mode. Returns a
    DataConnection-like adapter.
    """
    output_dir_str = str(output_dir)
    ensure_directory_exists(output_dir_str)
    db_file = os.path.join(output_dir_str, db_filename)

    if remove_aux_files:
        for suffix in ["-shm", "-wal", "-journal"]:
            aux_file = db_file + suffix
            if os.path.exists(aux_file):
                with contextlib.suppress(OSError):
                    os.remove(aux_file)

    # If run_id is provided, default to not removing existing database
    # to preserve run history, unless explicitly requested
    if remove_existing and os.path.exists(db_file):

        def _remove_db():
            os.remove(db_file)

        try:
            retry_with_backoff(
                _remove_db,
                max_retries=5,
                initial_delay=0.1,
                backoff_factor=2.0,
                exception_types=(OSError,),
            )
        except OSError as e:
            logger.warning(f"Failed to remove database {db_file}: {e}")

    all_atom_numbers = [int(num) for num in atoms_template.get_atomic_numbers()]
    db = None
    try:
        db = PrepareDB(
            db_file_name=db_file,
            simulation_cell=atoms_template,
            stoichiometry=all_atom_numbers,
        )

        if initial_population is not None:
            for candidate in initial_population:
                db.add_unrelaxed_candidate(candidate)
        elif initial_candidate is not None:
            db.add_unrelaxed_candidate(initial_candidate)

        # Close PrepareDB to avoid SQLite locking issues
        try:
            # Vacuum the database to flush any pending writes
            # Note: ASE's SQLite3Database manages commits internally
            if hasattr(db.c, "vacuum"):
                db.c.vacuum()
        except (AttributeError, sqlite3.OperationalError) as e:
            logger.debug(
                f"Failed to vacuum database before opening DataConnection: {e}"
            )
    finally:
        # Clean up PrepareDB (ASE manages connection cleanup)
        if db is not None:
            try:
                del db
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Error cleaning up PrepareDB connection: {e}")

    # Opening DataConnection uses retry_on_lock below; do not rename or probe-lock
    # the SQLite file (unsafe on shared HPC filesystems and other writers).

    if enable_wal_mode:
        try:
            with sqlite3.connect(db_file, timeout=30.0) as conn:
                apply_sqlite_pragmas(
                    conn, wal_mode=True, busy_timeout=30000, cache_size_mb=64
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(
                f"Failed to enable WAL mode: {e}. Database will use default mode."
            )
    else:
        # HPC / shared FS: keep rollback journal; still tune cache and temp store.
        try:
            with sqlite3.connect(db_file, timeout=30.0) as conn:
                apply_sqlite_pragmas(
                    conn, wal_mode=False, busy_timeout=30000, cache_size_mb=64
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.debug("Non-WAL PRAGMA setup skipped for %s: %s", db_file, e)

    # Open DataConnection with centralized retry-on-lock semantics
    try:

        def _open_connection_impl():
            return DataConnection(db_file)

        _open_connection = retry_on_lock(
            config=PRESET_AGGRESSIVE,
            operation_name="open DataConnection",
            log_retries=True,
        )(_open_connection_impl)

        da = _open_connection()

        # Create performance indices (P3.1)
        _ensure_database_indices(db_file)

        db_path_obj = Path(db_file)
        try:
            sz = db_path_obj.stat().st_size if db_path_obj.exists() else 0
        except OSError:
            sz = -1
        logger.debug(
            "Database setup: path=%s size=%s wal=%s",
            db_file,
            sz,
            enable_wal_mode,
        )

        # SCGO metadata: use helper to keep main flow linear
        _write_scgo_metadata(da, db_file)

        # Register database in persistent registry (best-effort)
        _register_database_best_effort(output_dir_str, db_file, atoms_template, run_id)

        # Adapter: validate composition; preserve metadata round-trip for unrelaxed pool
        class _DBAdapter:
            def __init__(self, da_obj, expected_atomic_numbers):
                self._da = da_obj
                self._expected_atomic_numbers = expected_atomic_numbers
                self._last_unrelaxed_metadata = None

            def __getattr__(self, name):
                return getattr(self._da, name)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                with contextlib.suppress(OSError, RuntimeError, AttributeError):
                    close_data_connection(self._da)

            def add_relaxed_step(self, a, *args, **kwargs):
                # Validate composition against expected atomic numbers (order-insensitive)
                if Counter(int(x) for x in a.get_atomic_numbers()) != Counter(
                    self._expected_atomic_numbers
                ):
                    raise AssertionError(
                        "Candidate composition does not match database stoichiometry"
                    )

                # ASE GA DataConnection expects raw_score in key_value_pairs.
                if "key_value_pairs" not in a.info:
                    a.info["key_value_pairs"] = {}

                if "raw_score" not in a.info.get("key_value_pairs", {}):
                    try:
                        energy = a.get_potential_energy()
                        a.info.setdefault("key_value_pairs", {})["raw_score"] = -float(
                            energy
                        )
                    except (AttributeError, RuntimeError, ValueError):
                        # GA continuation when relax leaves no usable energy (missing calc / worker failure).
                        logger.warning(
                            "raw_score missing and energy could not be computed for candidate; "
                            "assigning PENALTY_ENERGY and continuing."
                        )
                        a.info.setdefault("metadata", {})["potential_energy"] = (
                            PENALTY_ENERGY
                        )
                        a.info.setdefault("key_value_pairs", {})[
                            "raw_score"
                        ] = -PENALTY_ENERGY
                        zero_forces = np.zeros((len(a), 3), dtype=np.float64)
                        a.calc = SinglePointCalculator(
                            a, energy=PENALTY_ENERGY, forces=zero_forces
                        )

                return self._da.add_relaxed_step(a, *args, **kwargs)

            def add_unrelaxed_candidate(self, a, *args, **kwargs):
                # Store metadata locally so immediate get_an_unrelaxed_candidate can return it
                self._last_unrelaxed_metadata = (
                    a.info.get("metadata", {}).copy() if a.info.get("metadata") else {}
                )

                # ASE DB persists key_value_pairs; copy metadata keys for discovery
                prov_src = a.info.get("metadata") or {}
                kv = a.info.setdefault("key_value_pairs", {})
                for _k in ("run_id", "trial_id", "trial", "confid", "gaid", "id"):
                    if _k in prov_src and _k not in kv:
                        kv[_k] = prov_src[_k]

                a.info.setdefault("data", {})
                return self._da.add_unrelaxed_candidate(a, *args, **kwargs)

            def get_an_unrelaxed_candidate(self, *args, **kwargs):
                u = self._da.get_an_unrelaxed_candidate(*args, **kwargs)
                # If metadata was lost by underlying DB, restore it from the last write
                if (
                    u is not None
                    and "metadata" not in u.info
                    and self._last_unrelaxed_metadata
                ):
                    u.info["metadata"] = self._last_unrelaxed_metadata.copy()
                return u

        return _DBAdapter(da, all_atom_numbers)
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Failed to open database after all retries: %s", e)
        raise DatabaseSetupError(f"Failed to setup database {db_file}: {e}") from e


def extract_minima_from_database_file(
    db_path: str | Path,
    run_id: str,
    trial_id: int | None = None,
    require_final: bool = True,
    persist: bool = False,
) -> list[tuple[float, Atoms]]:
    """Return minima from ``db_path`` annotated with ``run_id`` / ``trial_id``.

    By default only rows tagged ``final_unique_minimum`` are returned (the
    canonical global-optimization result). Pass ``require_final=False`` to
    include all relaxed non-TS structures.

    If ``persist`` is True, attempt to write provenance back to the DB.
    """
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    if not is_scgo_database(db_path):
        logger.debug("Skipping extract_minima: not an SCGO database %s", db_path)
        return []

    with get_connection(db_path) as da:
        try:
            candidates = [
                atoms
                for _, atoms in _iter_relaxed_minima_from_da(
                    da, Path(db_path), chunk_size=100
                )
            ]
            minima = extract_minima_from_database(candidates)

            # Transition-state rows live in the same relaxed table; never treat them as minima.
            minima = [
                (e, at)
                for e, at in minima
                if not get_metadata(at, "is_transition_state", False)
            ]

            final_minima = [
                (e, at)
                for (e, at) in minima
                if get_metadata(at, "final_unique_minimum", False)
            ]

            # require_final=True: final_unique_minimum only; False: all relaxed (non-TS) rows
            use_minima = final_minima if require_final else minima

            if require_final and not use_minima:
                logger.debug(
                    "No final_unique_minimum-tagged rows in %s (require_final=True)",
                    db_path,
                )

            # Add run_id and trial_id to provenance (in-memory) and optionally
            # persist into the DB rows when requested via `persist=True`.
            for _, atoms in use_minima:
                add_metadata(
                    atoms,
                    run_id=run_id,
                    trial_id=trial_id,
                    source_db=os.path.basename(db_path),
                )

            if persist:
                try:
                    with da.c.managed_connection() as conn:
                        cols = [
                            r[1]
                            for r in conn.execute(
                                "PRAGMA table_info(systems)"
                            ).fetchall()
                        ]
                        has_metadata_col = "metadata" in cols
                        col = "metadata" if has_metadata_col else "key_value_pairs"

                        for _, atoms in use_minima:
                            row_id = get_metadata(atoms, "systems_row_id", None)
                            if row_id is None:
                                continue
                            row_id = int(row_id)

                            conn.execute(
                                f"UPDATE systems SET {col} = json_set(COALESCE({col}, '{{}}'), '$.run_id', ?) WHERE id = ?",
                                (run_id, row_id),
                            )
                            if trial_id is not None:
                                conn.execute(
                                    f"UPDATE systems SET {col} = json_set(COALESCE({col}, '{{}}'), '$.trial', ?) WHERE id = ?",
                                    (trial_id, row_id),
                                )
                                conn.execute(
                                    f"UPDATE systems SET {col} = json_set(COALESCE({col}, '{{}}'), '$.trial_id', ?) WHERE id = ?",
                                    (trial_id, row_id),
                                )
                        conn.commit()
                except (
                    sqlite3.DatabaseError,
                    sqlite3.OperationalError,
                    OSError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.debug(
                        "Failed to persist provenance to DB %s: %s", db_path, e
                    )

            return use_minima
        except (sqlite3.DatabaseError, OSError, ValueError) as e:
            logger.warning("Failed to extract minima from %s: %s", db_path, e)
            return []


def extract_transition_states_from_database_file(
    db_path: str | Path,
    run_id: str,
    trial_id: int | None = None,
    require_final_unique_ts: bool = True,
) -> list[tuple[float, Atoms]]:
    """Return transition-state rows from ``db_path`` with provenance.

    By default only rows tagged ``final_unique_ts`` are returned (canonical
    deduplicated, converged output from the TS search tagging path). Pass
    ``require_final_unique_ts=False`` to include any relaxed row marked
    ``is_transition_state`` (e.g. from ``integrate_ts_to_database`` without the
    canonical tag).
    """
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    if not is_scgo_database(db_path):
        logger.debug(
            "Skipping extract_transition_states: not an SCGO database %s", db_path
        )
        return []

    with get_connection(db_path) as da:
        try:
            out: list[tuple[float, Atoms]] = []
            for _, atoms in _iter_relaxed_minima_from_da(
                da, Path(db_path), chunk_size=100
            ):
                if not get_metadata(atoms, "is_transition_state", False):
                    continue
                if require_final_unique_ts and not get_metadata(
                    atoms, "final_unique_ts", False
                ):
                    continue
                energy = extract_energy_from_atoms(atoms)
                if energy is None:
                    continue
                out.append((float(energy), atoms))

            out.sort(key=lambda x: x[0])

            for _, atoms in out:
                add_metadata(
                    atoms,
                    run_id=run_id,
                    trial_id=trial_id,
                    source_db=os.path.basename(db_path),
                )

            return out
        except (sqlite3.DatabaseError, OSError, ValueError, AttributeError) as e:
            logger.warning(
                "Failed to extract transition states from %s: %s", db_path, e
            )
            return []


def load_previous_run_results(
    base_output_dir: str,
    db_filename: str | None = None,
    composition: list[str] | None = None,
    current_run_id: str | None = None,
    prefer_final_unique: bool = True,
) -> list[tuple[float, Atoms]]:
    """Load minima from previous runs for a composition.

    By default only ``final_unique_minimum`` rows are loaded. Pass
    ``prefer_final_unique=False`` to include all relaxed structures.
    """
    # Delegate to the parallel-capable implementation — it preserves the
    # original semantics and will fall back to sequential processing when
    # appropriate (fewer files, running in worker process, or parallel=False).
    return load_previous_results_parallel(
        base_output_dir=base_output_dir,
        db_filename=db_filename,
        composition=composition,
        current_run_id=current_run_id,
        prefer_final_unique=prefer_final_unique,
    )


def load_reference_structures(
    db_glob_pattern: str,
    composition: list[str] | None = None,
    max_structures: int = 100,
    base_dir: str | Path | None = None,
) -> list[Atoms]:
    """Load up to `max_structures` final minima from databases matching pattern.

    If ``db_glob_pattern`` is relative, it is resolved against ``base_dir`` when
    given, otherwise against the current working directory. Absolute patterns
    are left unchanged (HPC submit-dir safe when callers pass an explicit base).
    """
    pattern_path = Path(db_glob_pattern)
    if pattern_path.is_absolute():
        search_glob = str(pattern_path)
    else:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        search_glob = str(root / db_glob_pattern)
    db_files = glob.glob(search_glob, recursive=True)

    if not db_files:
        logger.warning(f"No database files found matching pattern: {db_glob_pattern}")
        return []

    # Prepare composition filter
    target_counts = None
    if composition is not None:
        target_counts = get_composition_counts(composition)

    # Use heap to efficiently select top-k structures by energy (P3.2)
    heap: list[tuple[float, int, Atoms]] = []
    counter = 0

    # We prefer using the per-file extractor which can identify final-tagged minima
    for db_file in db_files:
        try:
            minima = extract_minima_from_database_file(
                db_file, run_id=os.path.basename(db_file)
            )
        except (sqlite3.DatabaseError, OSError, ValueError) as e:
            logger.debug(f"Failed to extract minima from {db_file}: {e}")
            continue

        for energy, atoms in minima:
            # Only include final-tagged minima for diversity references
            if not get_metadata(atoms, "final_unique_minimum", False):
                continue

            # Composition filter
            if target_counts is not None:
                atoms_symbols = atoms.get_chemical_symbols()
                atoms_counts = get_composition_counts(atoms_symbols)
                if atoms_counts != target_counts:
                    continue

            # Add to heap if we haven't reached limit, or if this is a better structure
            if len(heap) < max_structures:
                heapq.heappush(heap, (-energy, counter, atoms))
                counter += 1
            elif energy < -heap[0][0]:
                counter += 1
                heapq.heapreplace(heap, (-energy, counter, atoms))

    if not heap:
        logger.warning("No final unique minima found in databases matching the pattern")
        return []

    # Convert heap to sorted list (lowest energy first)
    sorted_structures = sorted(heap, key=lambda x: -x[0])

    reference_atoms = [atoms for _, _, atoms in sorted_structures]

    logger.info(
        f"Loaded {len(reference_atoms)} final reference structures for diversity calculation "
        f"from {len(db_files)} databases"
    )

    return reference_atoms


# ============================================================================
# Helper Functions
# ============================================================================


def _filter_minima_by_composition(
    minima: list[tuple[float, Atoms]],
    composition: list[str] | None = None,
) -> list[tuple[float, Atoms]]:
    """Filter minima by stoichiometric composition.

    Args:
        minima: List of (energy, Atoms) tuples
        composition: Optional list of atomic symbols to filter by

    Returns:
        Filtered list of (energy, Atoms) tuples
    """
    if composition is None:
        return minima

    target_counts = get_composition_counts(composition)
    filtered = []
    for energy, atoms in minima:
        atoms_counts = get_composition_counts(atoms.get_chemical_symbols())
        if atoms_counts == target_counts:
            filtered.append((energy, atoms))

    return filtered


# ============================================================================
# Parallel Database Loading (P3.3)
# ============================================================================


def _load_single_database_worker(
    db_path: str,
    composition: list[str] | None = None,
    run_id: str | None = None,
    trial_id: int | None = None,
    require_final: bool = False,
) -> list[tuple[float, Atoms]]:
    """Load minima from a single database in subprocess.

    Args:
        db_path: Database file path.
        composition: Optional composition filter.
        run_id: Run ID for provenance.
        trial_id: Optional trial ID for provenance.
        require_final: If True, only return rows tagged as `final_unique_minimum`.

    Returns:
        List of (energy, Atoms) tuples with provenance.
    """
    # Convert to string in case it's a Path object
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    # Delegate to extract_minima_from_database_file (SCGO-stamped DBs only).
    try:
        minima = extract_minima_from_database_file(
            db_path, run_id or "", trial_id, require_final=require_final
        )
    except (sqlite3.DatabaseError, OSError, ValueError) as e:
        logger.error(f"Failed to extract minima from {db_path} in worker: {e}")
        return []

    # Prepare composition filter if needed
    target_counts = None
    if composition is not None:
        target_counts = get_composition_counts(composition)

    # Filter by composition. Provenance is already added by the extractor.
    filtered_minima = []
    for energy, atoms in minima:
        if target_counts is not None:
            atoms_counts = get_composition_counts(atoms.get_chemical_symbols())
            if atoms_counts != target_counts:
                continue

        filtered_minima.append((energy, atoms))

    return filtered_minima


def load_previous_results_parallel(
    base_output_dir: str,
    db_filename: str | None = None,
    composition: list[str] | None = None,
    current_run_id: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    prefer_final_unique: bool = True,
) -> list[tuple[float, Atoms]]:
    """Parallel-capable loader for minima from previous runs.

    By default only ``final_unique_minimum`` rows are loaded. Pass
    ``prefer_final_unique=False`` to include all relaxed structures.
    """
    # First collect all database files to load
    all_db_files: list[tuple[str, str | None, int | None]] = []

    if not os.path.exists(base_output_dir):
        return []

    discovered_entries = list_discovered_db_paths_with_run_trial(
        base_output_dir,
        composition=composition,
        use_cache=True,
    )

    if discovered_entries:
        by_run: dict[str, list[tuple[str, int | None]]] = {}
        for db_path_str, run_id, trial_id in discovered_entries:
            if not run_id or run_id == current_run_id:
                continue
            by_run.setdefault(run_id, []).append((db_path_str, trial_id))
        for run_id, db_list in by_run.items():
            run_dir = os.path.join(base_output_dir, run_id)
            metadata = load_run_metadata(run_dir)
            if composition is not None and metadata and metadata.formula:
                expected_formula = get_cluster_formula(composition)
                if metadata.formula != expected_formula:
                    continue
            all_db_files.extend((p, run_id, t) for p, t in db_list)

    if not all_db_files:
        run_dir_pattern = os.path.join(base_output_dir, "run_*")
        run_dirs = sorted(glob.glob(run_dir_pattern))

        for run_dir in run_dirs:
            run_dir_name = os.path.basename(run_dir)
            if not run_dir_name.startswith("run_"):
                continue

            run_id_from_dir = run_dir_name
            if run_id_from_dir == current_run_id:
                continue

            # Quick check: load metadata to verify composition matches
            metadata = load_run_metadata(run_dir)
            if composition is not None and metadata and metadata.formula:
                expected_formula = get_cluster_formula(composition)
                if metadata.formula != expected_formula:
                    logger.debug(
                        f"Skipping run {run_id_from_dir}: formula mismatch "
                        f"({metadata.formula} != {expected_formula})"
                    )
                    continue

            # Scan for trial_*/ directories within this run
            trial_pattern = os.path.join(run_dir, "trial_*")
            trial_dirs = sorted(glob.glob(trial_pattern))

            for trial_dir in trial_dirs:
                trial_dir_name = os.path.basename(trial_dir)
                try:
                    trial_id = int(trial_dir_name.split("_")[1])
                except (IndexError, ValueError):
                    trial_id = None

                # Look for database files in this trial directory
                pattern = db_filename if db_filename else "*.db"
                db_files = glob.glob(os.path.join(trial_dir, pattern))

                all_db_files.extend(
                    (db_path, run_id_from_dir, trial_id) for db_path in db_files
                )

    if not all_db_files:
        logger.info(f"No databases found in {base_output_dir}")
        return []

    use_parallel = (
        parallel
        and len(all_db_files) >= 4
        and multiprocessing.current_process().name == "MainProcess"
    )

    all_minima: list[tuple[float, Atoms]] = []

    if use_parallel:
        # Use parallel loading with ProcessPoolExecutor
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() // 2)

        logger.info(
            f"Loading {len(all_db_files)} databases in parallel "
            f"with {max_workers} workers"
        )

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _load_single_database_worker,
                        db_path,
                        composition,
                        run_id,
                        trial_id,
                        prefer_final_unique,
                    ): (db_path, run_id, trial_id)
                    for db_path, run_id, trial_id in all_db_files
                }

                for future in as_completed(futures):
                    db_path, run_id, trial_id = futures[future]
                    try:
                        minima = future.result(timeout=30)
                        # minima are already filtered by composition and have provenance from worker
                        all_minima.extend(minima)
                        if minima:
                            logger.debug(
                                f"Loaded {len(minima)} minima from {os.path.basename(db_path)}"
                            )
                    except (
                        OSError,
                        sqlite3.DatabaseError,
                        RuntimeError,
                        TimeoutError,
                        ValueError,
                    ) as e:
                        logger.error(
                            f"Failed to load {db_path} in parallel worker: {e}"
                        )
        except (
            OSError,
            sqlite3.DatabaseError,
            RuntimeError,
            ValueError,
        ) as e:
            raise RuntimeError(
                f"Parallel minima loading failed for {base_output_dir}: {type(e).__name__}: {e}"
            ) from e

    else:
        logger.info(f"Loading {len(all_db_files)} databases sequentially")

        for db_path, run_id, trial_id in all_db_files:
            minima = extract_minima_from_database_file(
                db_path, run_id or "", trial_id, require_final=prefer_final_unique
            )
            # Apply composition filter using helper function
            filtered_minima = _filter_minima_by_composition(minima, composition)
            all_minima.extend(filtered_minima)
            if filtered_minima:
                logger.debug(
                    f"Loaded {len(filtered_minima)} minima from {os.path.basename(db_path)}"
                )

    logger.info(
        f"Loaded {len(all_minima)} total minima from previous runs "
        f"(excluding {current_run_id})"
    )
    return all_minima
