import json
import sqlite3

from ase import Atoms

from scgo.database.metadata import add_metadata, mark_final_minima_in_db
from tests.test_utils import assert_db_final_row


def _iter_system_kvps(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        for (kvp_json,) in cur.fetchall():
            yield json.loads(kvp_json) if kvp_json else {}


def _assert_any_final_row_has(*, db_path, rank: int, final_written: str) -> None:
    for kvp in _iter_system_kvps(db_path):
        if kvp.get("final_unique_minimum"):
            assert kvp.get("final_rank") == rank
            assert kvp.get("final_written") == final_written
            return
    raise AssertionError("No final_unique_minimum rows found in DB")


def test_mark_final_minima_updates_db(tmp_path):
    base = tmp_path
    run_dir = base / "run_001" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    # Create a DB and add a candidate (use setup_database to initialize schema)
    from scgo.database.helpers import setup_database

    # Prepare template with metadata to be used as initial candidate
    template = Atoms("Pt", positions=[[0, 0, 0]])
    add_metadata(template, run_id="run_001", trial_id=1)
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"]["confid"] = "conf123"
    template.info["key_value_pairs"]["raw_score"] = -0.5

    # Create DB with initial candidate
    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    # Close prepare DB handle
    from scgo.database.connection import close_data_connection

    close_data_connection(da)
    # Prepare final minima info (simulate after writing .xyz files)
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_001"
    atoms2.info["provenance"]["trial"] = 1
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["confid"] = "conf123"

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_001_trial_1.xyz",
        }
    ]

    # Before calling helper, ensure one DB row has energy close to the final energy
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None, "No rows in systems table"

        # Pick a candidate row and set its energy to the final energy to allow best-match fallback
        target_id = row[0]
        cur.execute("UPDATE systems SET energy = ? WHERE id = ?", (-0.25, target_id))
        # Also update key_value_pairs to include a confid for traceability
        kv = json.loads(row[2]) if row[2] else {}
        kv.update({"confid": "conf123"})
        cur.execute(
            "UPDATE systems SET key_value_pairs = ? WHERE id = ?",
            (json.dumps(kv), target_id),
        )
        conn.commit()

    # Call helper
    mark_final_minima_in_db(final_info, base_dir=str(base))

    # assert DB contains a final-tagged row (do not require run_id present in DB row)
    assert_db_final_row(str(db_path), None, expect_final_id=False)

    _assert_any_final_row_has(
        db_path=db_path,
        rank=1,
        final_written="Pt_minimum_01_run_001_trial_1.xyz",
    )


def test_mark_final_minima_matches_kvp_run_trial(tmp_path):
    base = tmp_path
    run_dir = base / "run_002" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    from scgo.database.helpers import setup_database

    # Template uses legacy key_value_pairs for identifiers
    template = Atoms("Pt", positions=[[0, 0, 0]])
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"].update(
        {
            "confid": "conf_kvp",
            "run_id": "run_002",
            "trial": 1,
            "raw_score": -0.1,
        }
    )

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    # Ensure the stored row has run_id/trial in key_value_pairs and not in metadata
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()

        # Adapt to DB schema: metadata column may not exist on some ASE versions
        cols = [r[1] for r in cur.execute("PRAGMA table_info(systems)").fetchall()]
        if "metadata" in cols:
            cur.execute(
                "SELECT id, energy, metadata, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, metadata_col, kv_col = row
        else:
            cur.execute(
                "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, kv_col = row
            metadata_col = None

        import json

        metadata_json = json.loads(metadata_col) if metadata_col else {}
        metadata_json.pop("run_id", None)
        metadata_json.pop("trial_id", None)
        metadata_json.pop("confid", None)

        kv_json = json.loads(kv_col) if kv_col else {}
        kv_json.update({"confid": "conf_kvp", "run_id": "run_002", "trial": 1})

        if metadata_col is not None:
            cur.execute(
                "UPDATE systems SET energy = ?, metadata = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(metadata_json), json.dumps(kv_json), row_id),
            )
        else:
            cur.execute(
                "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(kv_json), row_id),
            )
        conn.commit()

    # Final minima info (provenance stores trial, metadata may store confid)
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_002"
    atoms2.info["provenance"]["trial"] = 1
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["confid"] = "conf_kvp"

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_002_trial_1.xyz",
        }
    ]

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(final_info, base_dir=str(base))

    assert_db_final_row(str(db_path), "run_002", expect_final_id=False)

    _assert_any_final_row_has(
        db_path=db_path,
        rank=1,
        final_written="Pt_minimum_01_run_002_trial_1.xyz",
    )


def test_mark_final_minima_matches_confid_in_kvp(tmp_path):
    base = tmp_path
    run_dir = base / "run_003" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    from scgo.database.helpers import setup_database

    # Create DB with metadata column present but confid only in key_value_pairs
    template = Atoms("Pt", positions=[[0, 0, 0]])
    template.info.setdefault("metadata", {})
    template.info.setdefault("key_value_pairs", {})
    # Do not store confid in metadata, only in key_value_pairs
    template.info["key_value_pairs"]["confid"] = "conf_only"
    template.info["key_value_pairs"]["raw_score"] = -0.2

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()

        # Adapt to DB schema: metadata column may not exist on some ASE versions
        cols = [r[1] for r in cur.execute("PRAGMA table_info(systems)").fetchall()]
        if "metadata" in cols:
            cur.execute(
                "SELECT id, energy, metadata, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, metadata_col, kv_col = row
        else:
            cur.execute(
                "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, kv_col = row
            metadata_col = None

        import json

        metadata_json = json.loads(metadata_col) if metadata_col else {}
        # ensure metadata does not contain confid
        metadata_json.pop("confid", None)

        kv_json = json.loads(kv_col) if kv_col else {}
        kv_json.update({"confid": "conf_only"})

        if metadata_col is not None:
            cur.execute(
                "UPDATE systems SET energy = ?, metadata = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(metadata_json), json.dumps(kv_json), row_id),
            )
        else:
            cur.execute(
                "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(kv_json), row_id),
            )
        conn.commit()

    # Final minima info with confid in metadata (matching kvp in DB)
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_003"
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["confid"] = "conf_only"

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_003_trial_1.xyz",
        }
    ]

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(final_info, base_dir=str(base))

    assert_db_final_row(str(db_path), "run_003", expect_final_id=False)

    # then verify final_rank/final_written as before (compact)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        for (kvp_json,) in cur.fetchall():
            kvp = json.loads(kvp_json) if kvp_json else {}
            if kvp.get("final_unique_minimum"):
                assert kvp.get("final_rank") == 1
                assert kvp.get("final_written") == "Pt_minimum_01_run_003_trial_1.xyz"


def test_mark_final_minima_matches_metadata_trial_key(tmp_path):
    # Ensure that a 'trial' key stored in metadata is accepted as a trial identifier
    base = tmp_path
    run_dir = base / "run_004" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    from scgo.database.helpers import setup_database

    # Template stores run_id and trial in key_value_pairs (legacy storage)
    template = Atoms("Pt", positions=[[0, 0, 0]])
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"].update(
        {
            "confid": "conf_trial",
            "run_id": "run_004",
            "trial": 1,
            "raw_score": -0.1,
        }
    )

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    # Update stored row to have matching energy and identifiers
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(systems)").fetchall()]
        if "metadata" in cols:
            cur.execute(
                "SELECT id, energy, metadata, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, metadata_col, kv_col = row
        else:
            cur.execute(
                "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            assert row is not None, "No DB row for final minima test"
            row_id, _, kv_col = row
            metadata_col = None

        import json

        # Ensure metadata contains a 'trial' key (this is the new compatibility case)
        meta_json = json.loads(metadata_col) if metadata_col else {}
        meta_json.update({"trial": 1})

        kv_json = json.loads(kv_col) if kv_col else {}
        kv_json.update({"confid": "conf_trial", "run_id": "run_004", "trial": 1})

        if metadata_col is not None:
            cur.execute(
                "UPDATE systems SET energy = ?, metadata = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(meta_json), json.dumps(kv_json), row_id),
            )
        else:
            cur.execute(
                "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
                (-0.25, json.dumps(kv_json), row_id),
            )
        conn.commit()

    # Final minima info has 'trial' stored in metadata (rather than provenance)
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["trial"] = 1
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_004"

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_004_trial_1.xyz",
        }
    ]

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(final_info, base_dir=str(base))

    assert_db_final_row(str(db_path), "run_004", expect_final_id=False)

    # keep the specific checks for final_rank/final_written
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        for (kvp_json,) in cur.fetchall():
            kvp = json.loads(kvp_json) if kvp_json else {}
            if kvp.get("final_unique_minimum"):
                assert kvp.get("final_rank") == 1
                assert kvp.get("final_written") == "Pt_minimum_01_run_004_trial_1.xyz"


def test_mark_final_minima_matches_id_in_metadata(tmp_path):
    # Ensure that an 'id' key present in metadata is used as an identifier for exact matching
    base = tmp_path
    run_dir = base / "run_005" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    from scgo.database.helpers import setup_database

    # Template stores id in key_value_pairs (legacy) and not confid
    template = Atoms("Pt", positions=[[0, 0, 0]])
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"].update(
        {
            "id": "id_777",
            "run_id": "run_005",
            "trial": 1,
            "raw_score": -0.1,
        }
    )

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None, "No rows in systems table"
        r_id = row[0]

        kv = json.loads(row[2]) if row[2] else {}
        kv.update({"id": "id_777"})
        cur.execute(
            "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
            (-0.25, json.dumps(kv), r_id),
        )
        conn.commit()

    # Final minima atoms include 'id' in metadata
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["id"] = "id_777"
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_005"

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_005_trial_1.xyz",
        }
    ]

    mark_final_minima_in_db(final_info, base_dir=str(base))

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, key_value_pairs FROM systems")
        rows = cur.fetchall()

        found = False
        for r in rows:
            _id2, kvp_json = r
            kvp = json.loads(kvp_json) if kvp_json else {}
            if kvp.get("final_unique_minimum"):
                found = True
                assert kvp.get("final_rank") == 1
                assert kvp.get("final_written") == "Pt_minimum_01_run_005_trial_1.xyz"

    assert found is True, "No row was tagged as final_unique_minimum (id in metadata)"


def test_mark_final_minima_prefers_closest_energy_within_tolerance(tmp_path):
    """When multiple DB rows share the same provenance, pick the row whose
    energy is closest to the provided final energy (within tolerance).
    """
    import json
    import sqlite3

    from ase import Atoms
    from ase.db import connect

    run_id = "run_energy_prefers"
    trial = 1

    dbpath = tmp_path / "energy.db"
    with connect(str(dbpath)) as db:
        # Two candidate rows with the same run_id/trial but slightly different energies
        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            a1,
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "trial": trial,
                "raw_score": -0.5,
                "relaxed": True,
            },
        )

        a2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        db.write(
            a2,
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "trial": trial,
                "raw_score": -0.5,
                "relaxed": True,
            },
        )

    # Update energies explicitly so we have two different energies in the DB
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, key_value_pairs FROM systems ORDER BY id ASC")
        rows = cur.fetchall()
        assert len(rows) >= 2
        id1 = rows[0][0]
        id2 = rows[1][0]

        # Set energies that are both within tolerance=0.0002 of final_energy
        cur.execute(
            "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
            (-0.50000, json.dumps({"run_id": run_id, "trial": trial}), id1),
        )
        cur.execute(
            "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
            (-0.50020, json.dumps({"run_id": run_id, "trial": trial}), id2),
        )
        conn.commit()

    from scgo.database.schema import stamp_scgo_database

    stamp_scgo_database(dbpath)

    # Final minima info: energy lies between the two DB energies and is closer to id1
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.info.setdefault("provenance", {})
    atoms.info["provenance"]["run_id"] = run_id
    atoms.info["provenance"]["trial"] = trial

    final_energy = -0.50005
    final_info = [
        {"atoms": atoms, "energy": final_energy, "rank": 1, "final_written": "foo.xyz"}
    ]

    # Register DB explicitly — strict discovery requires registration or canonical layout
    from scgo.database.registry import get_registry

    reg = get_registry(tmp_path)
    reg.register_database(dbpath, run_id=run_id, trial_id=trial)

    # Use a relaxed tolerance so the energy-based selection is used
    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(final_info, base_dir=str(tmp_path), tolerance=2e-4)

    # Verify the closest-energy row (id1) was tagged
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, key_value_pairs FROM systems")
        rows = cur.fetchall()
        found_id = None
        for r in rows:
            rid, kvp_json = r
            kv = json.loads(kvp_json) if kvp_json else {}
            if kv.get("final_unique_minimum"):
                found_id = rid
                break

    assert found_id == id1, f"Expected row {id1} to be tagged, got {found_id}"


def test_mark_final_minima_prefers_final_id(tmp_path):
    """If final_id is provided in final_minima_info, prefer exact match on it
    over ambiguous run/trial/energy matches.
    """
    import json
    import sqlite3

    from ase import Atoms
    from ase.db import connect

    run_id = "run_final_id"
    trial = 1
    dbpath = tmp_path / "fid.db"
    with connect(str(dbpath)) as db:
        # Two rows with identical energies/run/trial; one has a final_id stored
        a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            a1,
            relaxed=True,
            key_value_pairs={"run_id": run_id, "trial": trial, "raw_score": -0.5},
        )

        a2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        db.write(
            a2,
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "trial": trial,
                "raw_score": -0.5,
                "final_id": "persisted-fid",
            },
        )

    # Ensure both rows have the same energy in the energy column
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE systems SET energy = ?", (-0.5,))
        conn.commit()

    from scgo.database.schema import stamp_scgo_database

    stamp_scgo_database(dbpath)

    # Prepare final minima info that includes final_id matching second row
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
    atoms.info.setdefault("provenance", {})
    atoms.info["provenance"]["run_id"] = run_id
    atoms.info["provenance"]["trial"] = trial

    final_info = [
        {
            "atoms": atoms,
            "energy": -0.5,
            "rank": 1,
            "final_written": "foo.xyz",
            "final_id": "persisted-fid",
        }
    ]

    # Register DB explicitly — strict discovery requires registration or canonical layout
    from scgo.database.registry import get_registry

    reg = get_registry(tmp_path)
    reg.register_database(dbpath, run_id=run_id, trial_id=trial)

    from scgo.database.metadata import mark_final_minima_in_db

    mark_final_minima_in_db(final_info, base_dir=str(tmp_path))

    # generic assertion that final row exists and carries a final_id
    assert_db_final_row(str(dbpath), run_id, expect_final_id=True)

    # additional targeted check for the exact final_id value (keeps semantics):
    with sqlite3.connect(str(dbpath)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        rows = cur.fetchall()
        assert any(
            (json.loads(r[0]) or {}).get("final_unique_minimum")
            and (json.loads(r[0]) or {}).get("final_id") == "persisted-fid"
            for r in rows
        ), "Row with matching final_id was not tagged"
