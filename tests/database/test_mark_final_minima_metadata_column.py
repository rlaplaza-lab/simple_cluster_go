import contextlib
import json
import sqlite3

from ase import Atoms

from scgo.database.metadata import add_metadata, mark_final_minima_in_db
from tests.test_utils import assert_db_final_row


def test_mark_final_minima_updates_metadata_column(tmp_path):
    base = tmp_path
    run_dir = base / "run_002" / "trial_3"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    # Create a DB and add a candidate (use setup_database to initialize schema)
    from scgo.database.helpers import setup_database

    template = Atoms("Pt", positions=[[0, 0, 0]])
    # Store metadata values instead of key_value_pairs
    add_metadata(template, run_id="run_002", trial_id=3)
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"]["raw_score"] = -0.5

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    # Add a metadata column to simulate newer DB schema and populate it
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        with contextlib.suppress(sqlite3.OperationalError):
            # Some ASE/DB setups may already have `metadata` column; ignore if it exists
            cur.execute("ALTER TABLE systems ADD COLUMN metadata TEXT")

        cur.execute("SELECT id FROM systems ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        assert row is not None, "No systems rows found in systems table"

        target_id = row[0]
        # Set energy close to target
        cur.execute("UPDATE systems SET energy = ? WHERE id = ?", (-0.25, target_id))

        # Put confid into metadata JSON
        meta = {"confid": "meta_conf_1", "run_id": "run_002", "trial": 3}
        cur.execute(
            "UPDATE systems SET metadata = ? WHERE id = ?",
            (json.dumps(meta), target_id),
        )

        conn.commit()

    # Prepare final minima info using atoms with metadata.confid set
    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["confid"] = "meta_conf_1"
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_002"
    atoms2.info["provenance"]["trial"] = 3

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_002_trial_3.xyz",
        }
    ]

    # Call helper
    mark_final_minima_in_db(final_info, base_dir=str(base))

    assert_db_final_row(str(db_path), "run_002", expect_final_id=False)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, metadata, key_value_pairs FROM systems WHERE json_extract(metadata, '$.confid') = ?",
            ("meta_conf_1",),
        )
        row = cur.fetchone()
    assert row is not None, "No systems row for confid meta_conf_1"

    _id, meta_json, _kvp_json = row
    meta_dict = json.loads(meta_json) if meta_json else {}

    assert meta_dict.get("final_unique_minimum") is True, (
        "final_unique_minimum not set in metadata"
    )
    assert meta_dict.get("final_rank") == 1, "final_rank mismatch"
    assert meta_dict.get("final_written") == "Pt_minimum_01_run_002_trial_3.xyz", (
        "final_written mismatch"
    )
