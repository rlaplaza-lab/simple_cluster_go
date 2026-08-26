"""SQLite database setup and helpers for SCGO (ASE ``DataConnection``)."""

from __future__ import annotations

import contextlib
import glob
import heapq
import os
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ase import Atoms
from ase.db import connect as ase_db_connect
from ase_ga.data import DataConnection

from scgo.database.connection import (
    _run_sqlite,
    apply_sqlite_pragmas,
    close_data_connection,
    get_connection,
    open_data_connection_for_setup,
)
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.discovery import (
    clear_discovery_cache,
    list_discovered_db_paths_with_run,
)
from scgo.database.exceptions import DatabaseSetupError
from scgo.database.registry import get_registry
from scgo.database.streaming import iter_database_minima, iter_relaxed_structures
from scgo.database.sync import PRESET_AGGRESSIVE, database_retry
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import ensure_final_id, get_tag, set_tags
from scgo.metadata.db_stamp import is_scgo_db, stamp_db
from scgo.metadata.run_dir import load_run_dir_record, resolve_run_id_from_db_path
from scgo.utils.helpers import (
    _assign_penalty_energy,
    ensure_directory_exists,
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


class SCGODataConnection:
    """Thin DataConnection wrapper that stamps tags via :mod:`scgo.metadata.atoms`.

    ``add_relaxed_step`` additionally enforces the database stoichiometry and
    guarantees that ``raw_score`` and ``final_id`` tags exist before the write.
    """

    def __init__(self, da_obj: DataConnection, expected_atomic_numbers: list[int]):
        self._da = da_obj
        self._expected_atomic_numbers = expected_atomic_numbers
        self._expected_counter = Counter(int(x) for x in expected_atomic_numbers)

    def __getattr__(self, name):
        return getattr(self._da, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            close_data_connection(self._da)
        except (OSError, RuntimeError, AttributeError) as e:
            # Close is best-effort: never let a close failure (whether the body
            # raised or not) mask a successful run or propagate out of the
            # context manager. Log and swallow.
            logger.warning("Best-effort data connection close failed: %s", e)

    def add_relaxed_step(self, a, *args, **kwargs):
        actual = Counter(int(x) for x in a.get_atomic_numbers())
        if actual != self._expected_counter:
            raise SCGOValidationError(
                f"Candidate composition {dict(actual)} does not match "
                f"database stoichiometry {dict(self._expected_counter)}"
            )

        if get_tag(a, "raw_score") is None:
            try:
                energy = a.get_potential_energy()
                set_tags(a, raw_score=-float(energy))
            except (AttributeError, RuntimeError, ValueError):
                logger.warning(
                    "Candidate has no raw_score and its energy could not be "
                    "computed; assigning PENALTY_ENERGY and continuing"
                )
                _assign_penalty_energy(a)

        ensure_final_id(a)
        return self._da.add_relaxed_step(a, *args, **kwargs)

    def add_unrelaxed_candidate(self, a, *args, **kwargs):
        # ASE GA requires key_value_pairs to exist before insert.
        a.info.setdefault("key_value_pairs", {})
        a.info.setdefault("data", {})
        return self._da.add_unrelaxed_candidate(a, *args, **kwargs)


def _ensure_database_indices(
    db_path: str,
    *,
    enable_expression_indexes: bool = True,
    enable_wal_mode: bool = False,
) -> None:
    """Apply performance pragmas and create SQLite indices on ``systems``."""
    try:

        def _create_indices(conn: sqlite3.Connection) -> None:
            apply_sqlite_pragmas(
                conn,
                wal_mode=enable_wal_mode,
                busy_timeout=30000,
                cache_size_mb=64,
            )
            # ``id`` is the INTEGER PRIMARY KEY (rowid alias → implicit index) and
            # ``unique_id`` already carries ASE's implicit unique index, so a
            # second user index on either column is redundant. Drop them on
            # reused DBs too so already-created files also shed the dead indices.
            conn.execute("DROP INDEX IF EXISTS idx_id")
            conn.execute("DROP INDEX IF EXISTS idx_unique_id")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_energy ON systems(energy)")

            if enable_expression_indexes:
                json_col = SYSTEMS_JSON_COLUMN
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_relaxed_json "
                        f"ON systems(json_extract({json_col}, '$.relaxed'))"
                    )
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_raw_score_json "
                        f"ON systems(CAST(json_extract({json_col}, '$.raw_score') AS REAL))"
                    )
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_final_unique_json "
                        f"ON systems(json_extract({json_col}, '$.final_unique_minimum'))"
                    )

        database_retry(
            lambda: _run_sqlite(db_path, _create_indices),
            config=PRESET_AGGRESSIVE,
            operation_name=f"create indices on {db_path}",
        )
        logger.debug("Database indices created for %s", db_path)
    except sqlite3.OperationalError as e:
        if enable_wal_mode:
            logger.warning(
                "Could not enable WAL mode or create indices for %s: %s "
                "(continuing with the default journal mode)",
                db_path,
                e,
            )
        else:
            logger.debug("Could not create all indices on %s: %s", db_path, e)
    except OSError:
        logger.exception("Unexpected error creating indices on %s", db_path)


def _register_database_best_effort(
    base_dir: str | Path, db_file: str, atoms_template: Atoms | None, run_id: str | None
) -> None:
    """Best-effort registry registration for ``db_file``.

    Expected registry and filesystem errors are logged, not raised. The entry is
    added to the enclosing ``*_searches`` root when there is one, otherwise to
    ``base_dir`` itself.
    """
    comp_list = None
    if atoms_template is not None:
        try:
            comp_list = atoms_template.get_chemical_symbols()
        except (AttributeError, TypeError) as e:
            logger.debug(
                "Could not extract composition from atoms_template for %s: %s",
                db_file,
                e,
            )
            comp_list = None

    base_path = Path(base_dir)

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
            )
            logger.debug("Registered database in registry root %s: %s", root, db_file)
            clear_discovery_cache(root)
        except (ValueError, OSError) as _e:
            logger.debug(
                "Registry registration failed for %s in %s: %s", db_file, root, _e
            )


def setup_database(
    output_dir: str | Path,
    db_filename: str,
    atoms_template: Atoms,
    initial_candidate: Atoms | None = None,
    remove_existing: bool = True,
    remove_aux_files: bool = False,
    enable_wal_mode: bool = False,
    enable_expression_indexes: bool = True,
    run_id: str | None = None,
) -> DataConnection:
    """Create or open the ASE database ``db_filename`` inside ``output_dir``.

    The template structure is written first (at most one ``simulation_cell=True``
    row), then indices are created, the file is stamped as an SCGO database and
    registered in the registry (both best effort), and the connection is wrapped
    so writes are validated and tagged.

    When ``remove_existing=False`` and the file already exists, a second template
    row is not written. The stored stoichiometry must match ``atoms_template``
    (else :exc:`~scgo.exceptions.SCGOValidationError`). More than one existing
    template row raises ``DatabaseSetupError``.

    Args:
        output_dir: Directory holding the database file (created if missing)
        db_filename: Database file name inside ``output_dir``
        atoms_template: Template structure defining the expected stoichiometry
        initial_candidate: Optional single unrelaxed starting candidate
        remove_existing: Delete an existing database file before writing
        remove_aux_files: Delete leftover ``-shm`` / ``-wal`` / ``-journal`` files
        enable_wal_mode: Enable SQLite WAL journaling (off by default, since
            SCGO targets shared HPC filesystems)
        enable_expression_indexes: Also create JSON expression indices on
            ``key_value_pairs``
        run_id: Run identifier stored in the registry entry

    Returns:
        A :class:`~scgo.database.helpers.SCGODataConnection` wrapping the ASE ``DataConnection``.

    If the database cannot be opened after all retries, a
    ``DatabaseSetupError`` is raised.
    """
    output_dir_str = str(output_dir)
    ensure_directory_exists(output_dir_str)
    db_file = os.path.join(output_dir_str, db_filename)

    # Track whether the file is being created fresh (or is being replaced by
    # remove_existing). VACUUM is only meaningful for a freshly created file;
    # re-vacuuming an existing reused DB (resume paths) just wastes time.
    db_file_existed_before = os.path.exists(db_file)

    if remove_aux_files:
        for suffix in ["-shm", "-wal", "-journal"]:
            aux_file = db_file + suffix
            if os.path.exists(aux_file):
                with contextlib.suppress(OSError):
                    os.remove(aux_file)

    if remove_existing and os.path.exists(db_file):

        def _remove_db():
            os.remove(db_file)

        try:
            database_retry(
                _remove_db,
                config=PRESET_AGGRESSIVE,
                exception_types=(OSError,),
            )
        except OSError as e:
            logger.warning("Failed to remove database %s: %s", db_file, e)

    all_atom_numbers = [int(num) for num in atoms_template.get_atomic_numbers()]

    with ase_db_connect(db_file) as prep_db:
        template_rows = list(prep_db.select(simulation_cell=True))
        n_templates = len(template_rows)

        if n_templates > 1:
            raise DatabaseSetupError(
                f"Database {db_file} has {n_templates} simulation_cell template "
                "rows; expected at most one. Remove the file or pass "
                "remove_existing=True to recreate it."
            )

        if n_templates == 1:
            stored_row = template_rows[0]
            stored_data = getattr(stored_row, "data", None) or {}
            stored_stoich = stored_data.get("stoichiometry")
            if stored_stoich is None:
                stored_stoich = [
                    int(n) for n in stored_row.toatoms().get_atomic_numbers()
                ]
            stored_stoich = [int(n) for n in stored_stoich]
            if Counter(stored_stoich) != Counter(all_atom_numbers):
                raise SCGOValidationError(
                    f"Reusing database {db_file}: stored stoichiometry "
                    f"{stored_stoich} does not match atoms_template "
                    f"{all_atom_numbers}."
                )
            all_atom_numbers = stored_stoich
        else:
            prep_db.write(
                atoms_template,
                data={"stoichiometry": all_atom_numbers},
                simulation_cell=True,
            )

            if initial_candidate is not None:
                gaid = prep_db.write(
                    initial_candidate,
                    origin="StartingCandidateUnrelaxed",
                    relaxed=0,
                    generation=0,
                    extinct=0,
                )
                prep_db.update(gaid, gaid=gaid)
                initial_candidate.info["confid"] = gaid

        if not db_file_existed_before or remove_existing:
            with contextlib.suppress(AttributeError, sqlite3.OperationalError):
                prep_db.vacuum()

    try:
        da = database_retry(
            lambda: open_data_connection_for_setup(
                db_file,
                wal_mode=enable_wal_mode,
            ),
            config=PRESET_AGGRESSIVE,
            operation_name=f"setup database connection for {db_file}",
        )

        _ensure_database_indices(
            db_file,
            enable_expression_indexes=enable_expression_indexes,
            enable_wal_mode=enable_wal_mode,
        )

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

        try:
            stamp_db(db_file)
        except (sqlite3.DatabaseError, OSError, ValueError) as e:
            logger.warning("Failed to stamp SCGO database %s: %s", db_file, e)

        _register_database_best_effort(output_dir_str, db_file, atoms_template, run_id)

        return SCGODataConnection(da, all_atom_numbers)
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        raise DatabaseSetupError(f"Failed to setup database {db_file}: {e}") from e


def _extract_structures_from_db(
    db_path: str | Path,
    run_id: str,
    *,
    iter_relaxed_kwargs: dict,
    sort: bool = False,
    persist: bool = False,
    source_db_relpath: str | None = None,
    empty_log: Callable[[], None] | None = None,
) -> list[tuple[float, Atoms]]:
    """Load relaxed structures from a stamped SCGO database file."""
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    if not is_scgo_db(db_path):
        logger.debug("Skipping extract: not an SCGO database %s", db_path)
        return []

    def _extract() -> list[tuple[float, Atoms]]:
        with get_connection(db_path) as da:
            out: list[tuple[float, Atoms]] = []
            for energy, atoms in iter_relaxed_structures(
                da,
                Path(db_path),
                chunk_size=100,
                **iter_relaxed_kwargs,
            ):
                out.append((float(energy), atoms))

            if sort:
                out.sort(key=lambda x: x[0])

            if empty_log is not None and not out:
                empty_log()

            tag_kwargs: dict[str, str] = {
                "run_id": run_id,
                "source_db": os.path.basename(db_path),
            }
            if source_db_relpath is not None:
                tag_kwargs["source_db_relpath"] = source_db_relpath
            for _, atoms in out:
                set_tags(atoms, **tag_kwargs)

            if persist:
                try:
                    for _, atoms in out:
                        row_id = get_tag(atoms, "systems_row_id", None)
                        if row_id is None:
                            continue
                        # Go through ASE so ``run_id`` also lands in the
                        # key-index tables (``keys`` / ``text_key_values``);
                        # a raw JSON UPDATE leaves ``db.select(run_id=...)``
                        # unable to find the row.
                        da.c.update(int(row_id), run_id=run_id)
                    conn = getattr(da.c, "connection", None)
                    if conn is not None:
                        conn.commit()
                except (
                    sqlite3.DatabaseError,
                    OSError,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.debug("Failed to persist run_id to DB %s: %s", db_path, e)

            return out

    try:
        return database_retry(
            _extract,
            operation_name=f"extract structures from {db_path}",
        )
    except (sqlite3.DatabaseError, OSError, ValueError, AttributeError) as e:
        logger.warning("Failed to extract structures from %s: %s", db_path, e)
        return []


def extract_minima_from_database_file(
    db_path: str | Path,
    run_id: str,
    *,
    require_final: bool = True,
    persist: bool = False,
    source_db_relpath: str | None = None,
) -> list[tuple[float, Atoms]]:
    """Return minima from ``db_path`` annotated with ``run_id``."""
    return _extract_structures_from_db(
        db_path,
        run_id,
        iter_relaxed_kwargs={
            "require_final_minimum": require_final,
            "exclude_transition_states": True,
        },
        persist=persist,
        source_db_relpath=source_db_relpath,
        empty_log=(
            lambda: logger.debug(
                "No final_unique_minimum-tagged rows in %s (require_final=True)",
                db_path,
            )
        )
        if require_final
        else None,
    )


def load_previous_run_results(
    base_output_dir: str,
    db_filename: str | None = None,
    composition: list[str] | None = None,
    current_run_id: str | None = None,
    prefer_final_unique: bool = True,
) -> list[tuple[float, Atoms]]:
    """Load minima from previous runs for a composition."""
    all_db_files: list[tuple[str, str | None]] = []

    if not os.path.exists(base_output_dir):
        return []

    discovered_entries = list_discovered_db_paths_with_run(
        base_output_dir,
        composition=composition,
        use_cache=True,
        db_filename=db_filename,
    )

    if discovered_entries:
        by_run: dict[str, list[str]] = {}
        for db_path_str, run_id in discovered_entries:
            if not run_id:
                logger.warning(
                    "Skipping database %s: could not resolve run_id from path layout",
                    db_path_str,
                )
                continue
            if run_id == current_run_id:
                continue
            by_run.setdefault(run_id, []).append(db_path_str)
        for run_id, db_list in by_run.items():
            run_dir = os.path.join(base_output_dir, run_id)
            metadata = load_run_dir_record(run_dir)
            if composition is not None and metadata and metadata.formula:
                expected_formula = get_cluster_formula(composition)
                if metadata.formula != expected_formula:
                    continue
            all_db_files.extend((p, run_id) for p in db_list)

    if not all_db_files:
        logger.info("No previous-run databases found in %s", base_output_dir)
        return []

    all_minima: list[tuple[float, Atoms]] = []

    logger.info("Loading %s databases sequentially", len(all_db_files))

    for db_path, run_id in all_db_files:
        minima = extract_minima_from_database_file(
            db_path, run_id or "", require_final=prefer_final_unique
        )
        filtered_minima = _filter_minima_by_composition(minima, composition)
        all_minima.extend(filtered_minima)
        if filtered_minima:
            logger.debug(
                "Loaded %s minima from %s",
                len(filtered_minima),
                os.path.basename(db_path),
            )

    logger.info(
        "Loaded %s total minima from previous runs (excluding %s)",
        len(all_minima),
        current_run_id,
    )
    return all_minima


def load_reference_structures(
    db_glob_pattern: str,
    composition: list[str] | None = None,
    max_structures: int = 100,
    base_dir: str | Path | None = None,
) -> list[Atoms]:
    """Load up to `max_structures` final minima from databases matching pattern."""
    pattern_path = Path(db_glob_pattern)
    if pattern_path.is_absolute():
        search_glob = str(pattern_path)
    else:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        search_glob = str(root / db_glob_pattern)
    db_files = [p for p in glob.glob(search_glob, recursive=True) if is_scgo_db(p)]

    if not db_files:
        logger.warning(
            "No SCGO database files found matching pattern: %s", db_glob_pattern
        )
        return []

    target_counts = None
    if composition is not None:
        target_counts = get_composition_counts(composition)

    heap: list[tuple[float, int, Atoms]] = []
    counter = 0

    for db_file in db_files:
        try:
            resolved_run_id = resolve_run_id_from_db_path(db_file, base_dir=base_dir)
            if not resolved_run_id:
                logger.warning(
                    "Skipping reference database %s: could not resolve run_id "
                    "from path layout",
                    db_file,
                )
                continue
            try:
                db_relpath = os.path.relpath(
                    db_file,
                    str(base_dir) if base_dir is not None else os.getcwd(),
                )
            except (OSError, ValueError):
                db_relpath = os.path.basename(db_file)

            for energy, atoms in iter_database_minima(
                db_file,
                chunk_size=200,
                require_final_minimum=True,
                exclude_transition_states=True,
            ):
                if target_counts is not None and not _composition_matches(
                    atoms, target_counts
                ):
                    continue

                if len(heap) < max_structures or energy < -heap[0][0]:
                    set_tags(
                        atoms,
                        run_id=resolved_run_id,
                        source_db_relpath=db_relpath,
                    )
                    entry = (-energy, counter, atoms)
                    counter += 1
                    if len(heap) < max_structures:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heapreplace(heap, entry)
        except (sqlite3.DatabaseError, OSError, ValueError) as e:
            logger.debug("Failed to extract minima from %s: %s", db_file, e)
            continue

    if not heap:
        logger.warning("No final unique minima found in databases matching the pattern")
        return []

    sorted_structures = sorted(heap, key=lambda x: -x[0])
    reference_atoms = [atoms for _, _, atoms in sorted_structures]

    logger.info(
        "Loaded %s final reference structures for diversity calculation from %s databases",
        len(reference_atoms),
        len(db_files),
    )

    return reference_atoms


def _composition_matches(atoms: Atoms, target_counts: Counter[str]) -> bool:
    """Return True if *atoms* has the exact stoichiometry in *target_counts*."""
    return get_composition_counts(atoms.get_chemical_symbols()) == target_counts


def _filter_minima_by_composition(
    minima: list[tuple[float, Atoms]],
    composition: list[str] | None = None,
) -> list[tuple[float, Atoms]]:
    """Filter minima by stoichiometric composition."""
    if composition is None:
        return minima

    target_counts = get_composition_counts(composition)
    filtered = []
    for energy, atoms in minima:
        if _composition_matches(atoms, target_counts):
            filtered.append((energy, atoms))

    return filtered
