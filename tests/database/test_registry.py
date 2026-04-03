import os
from pathlib import Path

from ase.db import connect

from scgo.database.registry import clear_registry_cache, get_registry


def test_database_registry_register_find_and_clear(tmp_path, single_atom):
    base = tmp_path
    dbpath = base / "some_dir" / "my.db"
    dbpath.parent.mkdir(parents=True)

    # Create simple ASE DB
    db = connect(str(dbpath))
    db.write(single_atom.copy(), relaxed=True)

    # Ensure fresh registry for this base_dir
    clear_registry_cache()
    reg = get_registry(base)
    reg.clear()

    # Register DB with explicit metadata
    reg.register_database(dbpath, composition=["Pt"], run_id="run_xyz", trial_id=1)

    # find_databases should locate the registered DB
    found = reg.find_databases(run_id="run_xyz", trial_id=1)
    assert len(found) == 1
    assert Path(found[0]).resolve() == dbpath.resolve()

    # get_all_databases should include it
    all_db = reg.get_all_databases()
    assert any(Path(p).resolve() == dbpath.resolve() for p in all_db)

    # get_database_entry should return metadata including run_id/trial_id
    entry = reg.get_database_entry(dbpath)
    assert entry is not None
    assert entry.get("run_id") == "run_xyz"
    assert entry.get("trial_id") == 1

    # unregister should remove it
    removed = reg.unregister_database(dbpath)
    assert removed is True
    assert reg.find_databases(run_id="run_xyz", trial_id=1) == []


def test_rebuild_and_invalidate(tmp_path):
    base = tmp_path
    dbpath = base / "other.db"

    # Create DB file
    db = connect(str(dbpath))
    from tests.test_utils import create_test_atoms

    db.write(create_test_atoms(["Pt", "Pt"], positions=[[0, 0, 0], [1.5, 0, 0]]))

    clear_registry_cache()
    reg = get_registry(base)
    reg.clear()

    # Rebuild from filesystem should register the DB
    registered = reg.rebuild_from_filesystem()
    assert registered >= 1
    entries = reg.get_all_databases()
    assert any(Path(p).resolve() == dbpath.resolve() for p in entries)

    # Remove the DB file and invalidate stale entries
    os.remove(dbpath)
    removed_count = reg.invalidate_stale_entries()
    assert removed_count >= 1
    # Registry should now be empty
    assert reg.get_all_databases() == []


def test_setup_database_registers_registry(tmp_path, pt2_atoms):
    from scgo.database.helpers import setup_database
    from scgo.database.registry import clear_registry_cache, get_registry

    # Ensure fresh registry for this base_dir
    clear_registry_cache()
    reg = get_registry(tmp_path)
    reg.clear()

    # Create DB via setup_database (should auto-register)
    setup_database(tmp_path, "auto_register.db", pt2_atoms, initial_candidate=pt2_atoms)
    db_path = tmp_path / "auto_register.db"

    entries = reg.get_all_databases()
    assert any(Path(p).resolve() == db_path.resolve() for p in entries)

    entry = reg.get_database_entry(db_path)
    assert entry is not None
    # composition_str should be canonical 'Pt2'
    assert entry.get("composition_str") == "Pt2"


def test_setup_database_registers_search_level_registry(tmp_path):
    """DB under trial_*/ inside *_searches is registered only at the search root."""

    from scgo.database.helpers import setup_database
    from scgo.database.registry import clear_registry_cache, get_registry

    # Build canonical run/trial layout under a search directory
    search_dir = tmp_path / "Pt6_searches"
    run_dir = search_dir / "run_000" / "trial_1"
    run_dir.mkdir(parents=True)

    # Ensure fresh registries
    clear_registry_cache()
    get_registry(run_dir).clear()
    get_registry(search_dir).clear()

    # Create DB in the trial directory (this is what the optimizer does)
    from tests.test_utils import create_test_atoms

    pt6 = create_test_atoms(["Pt"] * 6)
    setup_database(str(run_dir), "ga_go.db", pt6, initial_candidate=pt6)
    db_path = run_dir / "ga_go.db"

    # No per-trial registry file when under *_searches
    trial_entries = get_registry(run_dir).get_all_databases()
    assert trial_entries == []

    search_reg = get_registry(search_dir)
    search_entries = search_reg.get_all_databases()
    assert any(Path(p).resolve() == db_path.resolve() for p in search_entries)

    entry = search_reg.get_database_entry(db_path)
    assert entry is not None
    assert entry.get("trial_id") == 1


def test_create_preparedb_registers_registry(tmp_path, pt2_atoms):
    # Ensure fresh registry for this base_dir
    from scgo.database.registry import clear_registry_cache, get_registry
    from tests.test_utils import create_preparedb

    clear_registry_cache()
    reg = get_registry(tmp_path)
    reg.clear()

    # Create DB via PrepareDB helper (test utility)
    create_preparedb(pt2_atoms, tmp_path / "prepared.db", population_size=5)
    db_path = tmp_path / "prepared.db"

    entries = reg.get_all_databases()
    assert any(Path(p).resolve() == db_path.resolve() for p in entries)

    entry = reg.get_database_entry(db_path)
    assert entry is not None
    assert entry.get("composition_str") == "Pt2"


def test_register_database_best_effort_handles_bad_atoms_template(tmp_path):
    """_register_database_best_effort tolerates a bad atoms_template (composition may be None)."""
    from scgo.database.helpers import _register_database_best_effort
    from scgo.database.registry import clear_registry_cache, get_registry

    clear_registry_cache()
    reg = get_registry(tmp_path)
    reg.clear()

    # Create trial directory and db file path under base
    trial_dir = tmp_path / "run_1" / "trial_1"
    trial_dir.mkdir(parents=True)
    db_path = trial_dir / "ga_go.db"

    # atoms_template stub that raises when asked for chemical symbols
    class BadAtomsTemplate:
        def get_chemical_symbols(self):
            raise AttributeError("simulated - missing internals")

        def get_atomic_numbers(self):
            return [78, 78]

    # Call the best-effort registration helper — should not raise
    _register_database_best_effort(
        str(trial_dir), str(db_path), BadAtomsTemplate(), "run_1"
    )

    # Trial-level registry should now contain the entry (composition may be None)
    # The registry entry should exist even if the DB file is not yet present
    trial_registry = get_registry(trial_dir)
    entry = trial_registry.get_database_entry(db_path)
    assert entry is not None
    assert entry.get("path") == str(db_path.resolve().relative_to(trial_dir.resolve()))
    assert entry.get("trial_id") == 1

    # Also ensure the registry under the base tmp_path did not accidentally gain it
    entries = reg.get_all_databases()
    assert all(p.resolve().name != db_path.name for p in entries)


def test_get_database_entry_outside_base(tmp_path):
    """Paths outside the registry base must not raise and should return None."""
    from scgo.database.registry import clear_registry_cache, get_registry

    clear_registry_cache()
    reg = get_registry(tmp_path)
    reg.clear()

    # Use a path that is not under the registry base_dir
    outside = tmp_path.parent / "outside.db"
    assert reg.get_database_entry(outside) is None


def test_setup_database_context_manager(tmp_path, pt2_atoms):
    from scgo.database.connection import open_db
    from scgo.database.helpers import setup_database

    # Use context-manager returned by setup_database
    with setup_database(
        tmp_path, "cm.db", pt2_atoms, initial_candidate=pt2_atoms
    ) as da:
        from tests.test_utils import create_test_atoms

        a = create_test_atoms(["Pt", "Pt"], positions=[[0, 0, 0], [1.5, 0, 0]])
        # Insert as unrelaxed first so ASE assigns confid/identifiers as expected
        a.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_unrelaxed_candidate(a, description="cm:test")

        # Retrieve the unrelaxed candidate and promote to relaxed state
        u = da.get_an_unrelaxed_candidate()
        assert u is not None
        u.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_relaxed_step(u)
        assert len(da.get_all_relaxed_candidates()) >= 1

    # After exiting the context, DB should be readable and contain the relaxed row
    with open_db(tmp_path / "cm.db") as da2:
        assert len(da2.get_all_relaxed_candidates()) >= 1
