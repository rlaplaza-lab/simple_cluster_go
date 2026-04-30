import json
import sqlite3

from scgo.database.schema import stamp_scgo_database
from scgo.param_presets import get_testing_params
from scgo.run_minima import run_scgo_trials
from tests.test_utils import assert_db_final_row


def test_run_scgo_tags_final_minima(tmp_path):
    params = get_testing_params()
    # Ensure tagging is enabled
    params["tag_final_minima"] = True
    params["n_trials"] = 1
    # Run a trivial Pt2 search that completes quickly
    results = run_scgo_trials(
        ["Pt", "Pt"],
        "gas_cluster",
        params=params,
        seed=42,
        output_dir=str(tmp_path),
        verbosity=0,
    )
    assert len(results) > 0

    # Search for a database file in the output
    db_files = list(tmp_path.glob("**/*.db"))
    assert len(db_files) > 0, "No database files found"

    db_path = db_files[0]
    expected_run_id = db_path.parents[1].name if len(db_path.parents) >= 2 else None

    # Generic final-tag assertion (also asserts final_id presence)
    assert_db_final_row(str(db_path), expected_run_id, expect_final_id=True)


def test_mark_final_minima_fallback_scans_all_db(tmp_path):
    """Ensure mark_final_minima_in_db will find DB rows even when DB file is not
    under the usual run_xxx/trial_xxx path by scanning all DBs under base_dir.
    """

    from ase import Atoms
    from ase.db import connect

    run_id = "run_test_fallback"
    trial = 1

    # Create a DB somewhere under tmp_path but not under the canonical
    # run_xxx/trial_xxx layout so a simple glob won't find it by run/trial
    dbdir = tmp_path / "dbs"
    dbdir.mkdir()
    dbpath = dbdir / "other.db"

    with connect(str(dbpath)) as db:
        pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            pt2,
            relaxed=True,
            key_value_pairs={"run_id": run_id, "trial_id": trial, "raw_score": -3.4},
        )

    stamp_scgo_database(dbpath)

    # Register the DB explicitly so discovery can find it (no recursive scan)
    from scgo.database.registry import get_registry

    reg = get_registry(tmp_path)
    reg.register_database(dbpath, run_id=run_id, trial_id=trial)

    # Construct final_minima_info with provenance matching above run/trial
    atoms = pt2.copy()
    atoms.info["provenance"] = {"run_id": run_id, "trial_id": trial}

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(
        [{"atoms": atoms, "energy": None, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
    )

    # Verify DB row updated — DB row for the given run_id was marked final
    assert_db_final_row(str(dbpath), run_id, expect_final_id=False)


def test_mark_final_minima_prefers_relaxed_row(tmp_path):
    """If multiple DB rows match the same provenance, ensure the relaxed row is tagged."""

    from ase import Atoms
    from ase.db import connect

    run_id = "run_prefers_relaxed"
    trial = 1

    dbpath = tmp_path / "pref.db"
    with connect(str(dbpath)) as db:
        # Create unrelaxed row (relaxed=False)
        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            a1,
            relaxed=False,
            key_value_pairs={
                "run_id": run_id,
                "trial_id": trial,
                "raw_score": -1.0,
                "relaxed": False,
            },
        )

        # Create relaxed row (relaxed=True) - this should be preferred
        a2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        db.write(
            a2,
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "trial_id": trial,
                "raw_score": -2.0,
                "relaxed": True,
            },
        )

    stamp_scgo_database(dbpath)

    atoms = a2.copy()
    atoms.info["provenance"] = {"run_id": run_id, "trial_id": trial}

    # Register DB explicitly — strict discovery requires registration or
    # canonical run_xxx/trial_xxx layout.
    from scgo.database.registry import get_registry

    reg = get_registry(tmp_path)
    reg.register_database(dbpath, run_id=run_id, trial_id=trial)

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(
        [{"atoms": atoms, "energy": None, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
    )

    assert_db_final_row(str(dbpath), run_id, expect_final_id=False)

    # specific check: ensure the tagged row is the relaxed one
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        rows = cur.fetchall()
        assert any(
            (json.loads(r[0]) or {}).get("relaxed")
            and (json.loads(r[0]) or {}).get("final_unique_minimum")
            for r in rows
        ), "No relaxed row was tagged"


def test_mark_final_minima_prefers_confid_relaxed(tmp_path):
    """When matching by confid/gaid/id, prefer the relaxed row if multiples match."""

    from ase import Atoms
    from ase.db import connect

    dbpath = tmp_path / "conf.db"
    with connect(str(dbpath)) as db:
        # Two rows share same gaid/confid; one unrelaxed and one relaxed. Relaxed should be chosen.
        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            a1,
            relaxed=False,
            key_value_pairs={"gaid": 42, "raw_score": -1.0, "relaxed": False},
        )

        a2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        db.write(
            a2,
            relaxed=True,
            key_value_pairs={"gaid": 42, "raw_score": -2.0, "relaxed": True},
        )

    stamp_scgo_database(dbpath)

    atoms = a2.copy()
    # Provide confid via metadata to exercise exact-match logic
    from scgo.database.metadata import add_metadata

    add_metadata(atoms, confid=42)

    # Register DB explicitly — strict discovery requires registration or
    # canonical run_xxx/trial_xxx layout.
    from scgo.database.registry import get_registry

    reg = get_registry(tmp_path)
    reg.register_database(dbpath)

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(
        [{"atoms": atoms, "energy": None, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
    )

    # ensure at least one final-tagged row exists (generic)
    assert_db_final_row(str(dbpath), None, expect_final_id=False)

    # keep the relaxed-specific assertion (concise):
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        rows = cur.fetchall()
        assert any(
            json.loads(r[0]).get("relaxed")
            and json.loads(r[0]).get("final_unique_minimum")
            for r in rows
        ), "No relaxed row was tagged"
