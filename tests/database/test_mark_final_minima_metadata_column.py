import json
import sqlite3

from ase import Atoms

from scgo.database.metadata import add_metadata, mark_final_minima_in_db
from tests.test_utils import assert_db_final_row


def test_mark_final_minima_updates_key_value_pairs_json(tmp_path):
    """Final-minima flags are written to ``systems.key_value_pairs`` (SCGO layout)."""
    base = tmp_path
    run_dir = base / "run_002" / "trial_3"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    from scgo.database.helpers import setup_database

    template = Atoms("Pt", positions=[[0, 0, 0]])
    add_metadata(template, run_id="run_002", trial_id=3)
    template.info.setdefault("key_value_pairs", {})
    template.info["key_value_pairs"]["raw_score"] = -0.5
    template.info["key_value_pairs"]["confid"] = "kvp_conf_1"

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    # Match other mark_final tests: ensure identifiers are present in the row JSON
    # (initial write paths do not always round-trip every key_value_pairs key).
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, energy, key_value_pairs FROM systems ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None, "No systems rows found in systems table"
        target_id, _, kv_col = row
        kv = json.loads(kv_col) if kv_col else {}
        kv.update(
            {
                "confid": "kvp_conf_1",
                "run_id": "run_002",
                "trial_id": 3,
            }
        )
        cur.execute(
            "UPDATE systems SET energy = ?, key_value_pairs = ? WHERE id = ?",
            (-0.25, json.dumps(kv), target_id),
        )
        conn.commit()

    atoms2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms2.info.setdefault("metadata", {})
    atoms2.info["metadata"]["confid"] = "kvp_conf_1"
    atoms2.info.setdefault("provenance", {})
    atoms2.info["provenance"]["run_id"] = "run_002"
    atoms2.info["provenance"]["trial_id"] = 3

    final_info = [
        {
            "atoms": atoms2,
            "energy": -0.25,
            "rank": 1,
            "final_written": "Pt_minimum_01_run_002_trial_3.xyz",
        }
    ]

    mark_final_minima_in_db(final_info, base_dir=str(base))

    assert_db_final_row(str(db_path), "run_002", expect_final_id=False)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT key_value_pairs FROM systems "
            "WHERE json_extract(key_value_pairs, '$.confid') = ?",
            ("kvp_conf_1",),
        )
        row = cur.fetchone()
    assert row is not None, "No systems row for confid kvp_conf_1"

    (kvp_json,) = row
    kvp = json.loads(kvp_json) if kvp_json else {}
    assert kvp.get("final_unique_minimum") is True
    assert kvp.get("final_rank") == 1
    assert kvp.get("final_written") == "Pt_minimum_01_run_002_trial_3.xyz"
