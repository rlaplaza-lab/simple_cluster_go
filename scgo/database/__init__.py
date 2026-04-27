"""SCGO Database Module

Designed for **HPC** use: SQLite on shared filesystems (Lustre, GPFS, NFS-class),
batch jobs, and optional multi-process access. WAL mode is off by default; the
registry uses a cross-process lock file (``.scgo_db_registry.lock``) on Linux.
Prefer job-local scratch for heavy I/O when your site supports it.

Unified database management system for SCGO with:
- Simple context-manager pattern for automatic cleanup
- Health checks
- Stream processing for large datasets
- Schema versioning
- Metadata tracking and querying

Quick Start
-----------

Basic Database Operations:

    from scgo.database import open_db

    with open_db('output/run_xyz/db.db') as da:
        atoms = da.get_an_unrelaxed_candidate()
        da.add_relaxed_step(atoms)

High-Level Database Management:

    from scgo.database import SCGODatabaseManager

    with SCGODatabaseManager(base_dir='output', enable_caching=True) as manager:
        refs = manager.load_diversity_references(
            glob_pattern='**/*.db',
            composition=['Pt', 'Pt'],
            max_structures=100
        )

Setup New Database:

    from scgo.database import setup_database, close_data_connection

    da = setup_database(
        output_dir='output/run_xyz',
        db_filename='ga_go.db',
        atoms_template=initial_atoms,
        run_id='run_20260204_120000'
    )
    try:
        ...
    finally:
        close_data_connection(da)

Metadata Management:

    from scgo.database.metadata import add_metadata, get_metadata

    add_metadata(atoms, run_id='run_xyz', trial_id=1, fitness_score=0.95)
    run_id = get_metadata(atoms, 'run_id', default='unknown')

"""

from __future__ import annotations

from scgo.database.cache import (
    UnifiedCache,
    get_global_cache,
    reset_global_cache,
)
from scgo.database.connection import (
    close_data_connection,
    get_connection,
    open_db,
)
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.discovery import (
    DatabaseDiscovery,
    find_databases_simple,
)
from scgo.database.exceptions import (
    DatabaseError,
    DatabaseLockError,
    DatabaseMigrationError,
    DatabaseSetupError,
)
from scgo.database.health import (
    check_database_health,
    get_database_statistics,
)
from scgo.database.helpers import (
    extract_minima_from_database_file,
    extract_transition_states_from_database_file,
    load_previous_run_results,
    load_reference_structures,
    setup_database,
)
from scgo.database.manager import SCGODatabaseManager
from scgo.database.metadata import (
    add_metadata,
    filter_by_metadata,
    get_all_metadata,
    get_metadata,
    mark_final_minima_in_db,
    update_metadata,
)
from scgo.database.registry import (
    DatabaseRegistry,
    clear_registry_cache,
    get_registry,
)
from scgo.database.schema import (
    ensure_schema_version,
    get_schema_version,
    migrate_database,
    set_schema_version,
    stamp_scgo_database,
)
from scgo.database.streaming import (
    aggregate_relaxed_energy_stats,
    count_database_structures,
    iter_database_minima,
    iter_databases_minima,
)
from scgo.database.sync import (
    HPC_DATABASE_EXCEPTIONS,
    PRESET_AGGRESSIVE,
    PRESET_CONSERVATIVE,
    RetryConfig,
    database_retry,
    is_retryable_error,
    retry_on_lock,
    retry_transaction,
    retry_with_backoff,
)
from scgo.database.transactions import (
    database_transaction,
)

__all__ = [
    "UnifiedCache",
    "get_global_cache",
    "reset_global_cache",
    "SYSTEMS_JSON_COLUMN",
    "close_data_connection",
    "get_connection",
    "open_db",
    "DatabaseError",
    "DatabaseLockError",
    "DatabaseSetupError",
    "DatabaseMigrationError",
    "setup_database",
    "extract_minima_from_database_file",
    "extract_transition_states_from_database_file",
    "load_previous_run_results",
    "load_reference_structures",
    "DatabaseDiscovery",
    "find_databases_simple",
    "check_database_health",
    "get_database_statistics",
    "iter_database_minima",
    "iter_databases_minima",
    "aggregate_relaxed_energy_stats",
    "count_database_structures",
    "mark_final_minima_in_db",
    "SCGODatabaseManager",
    "database_transaction",
    "get_schema_version",
    "set_schema_version",
    "migrate_database",
    "ensure_schema_version",
    "stamp_scgo_database",
    "add_metadata",
    "get_metadata",
    "get_all_metadata",
    "update_metadata",
    "filter_by_metadata",
    "HPC_DATABASE_EXCEPTIONS",
    "RetryConfig",
    "PRESET_AGGRESSIVE",
    "PRESET_CONSERVATIVE",
    "database_retry",
    "is_retryable_error",
    "retry_on_lock",
    "retry_transaction",
    "retry_with_backoff",
    "DatabaseRegistry",
    "get_registry",
    "clear_registry_cache",
]
