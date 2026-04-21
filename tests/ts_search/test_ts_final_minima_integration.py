import json
import sqlite3

from ase import Atoms

from scgo.database.helpers import setup_database
from scgo.database.metadata import mark_final_minima_in_db
from scgo.ts_search.transition_state_run import run_transition_state_search
from tests.test_utils import assert_db_final_row


def test_ts_search_uses_only_tagged_final_minima(tmp_path):
    base = tmp_path

    # Build run dir structure: Pt_searches/run_test/trial_1/
    formula_dir = base / "Pt_searches"
    run_dir = formula_dir / "run_test"
    trial_dir = run_dir / "trial_1"
    trial_dir.mkdir(parents=True)

    db_path = trial_dir / "ga_go.db"

    # Create DB and add three minima with different energies
    template = Atoms("Pt", positions=[[0, 0, 0]])
    da = setup_database(trial_dir, "ga_go.db", template, initial_candidate=template)

    # Create three relaxed minima
    energies = [0.0, 0.2, 0.5]
    confid_list = ["c1", "c2", "c3"]

    positions = [[0, 0, 0], [0, 0.05, 0], [0, 0.10, 0]]
    for idx, (e, cid) in enumerate(zip(energies, confid_list, strict=False)):
        a = Atoms("Pt", positions=[positions[idx]])
        # Store raw_score as ASE GA expects (-energy)
        a.info.setdefault("key_value_pairs", {})
        a.info["key_value_pairs"]["raw_score"] = -e
        # Ensure required keys for ASE GA
        a.info.setdefault("data", {})
        # Add as unrelaxed candidate first to ensure GA confid is created
        da.add_unrelaxed_candidate(a, description=f"test:{cid}")
        # Mark as relaxed (adds to relaxed table)
        da.add_relaxed_step(a)

    # Ensure confid values are persisted in the DB key_value_pairs so marking can find exact matches
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, key_value_pairs FROM systems ORDER BY id ASC")
    rows = cur.fetchall()
    for row, cid in zip(rows, confid_list, strict=False):
        row_id = row[0]
        kvp = json.loads(row[1]) if row[1] else {}
        kvp.update({"confid": cid})
        cur.execute(
            "UPDATE systems SET key_value_pairs = ? WHERE id = ?",
            (json.dumps(kvp), row_id),
        )
    conn.commit()
    conn.close()

    from scgo.database.connection import close_data_connection

    close_data_connection(da)

    # Tag only two minima (c1 and c2) as final
    atoms_c1 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms_c1.info.setdefault("provenance", {})
    atoms_c1.info["provenance"]["run_id"] = "run_test"
    atoms_c1.info["provenance"]["trial"] = 1
    atoms_c1.info.setdefault("metadata", {})
    atoms_c1.info["metadata"]["confid"] = "c1"

    atoms_c2 = Atoms("Pt", positions=[[0, 0, 0]])
    atoms_c2.info.setdefault("provenance", {})
    atoms_c2.info["provenance"]["run_id"] = "run_test"
    atoms_c2.info["provenance"]["trial"] = 1
    atoms_c2.info.setdefault("metadata", {})
    atoms_c2.info["metadata"]["confid"] = "c2"

    final_info = [
        {"atoms": atoms_c1, "energy": 0.0, "rank": 1, "final_written": "f1"},
        {"atoms": atoms_c2, "energy": 0.2, "rank": 2, "final_written": "f2"},
    ]

    # Invoke helper to mark DB rows
    mark_final_minima_in_db(final_info, base_dir=str(formula_dir))

    # Now run TS search on the formula directory
    params = {"calculator": "EMT", "calculator_kwargs": {}}
    results = run_transition_state_search(
        ["Pt"],
        output_dir=str(formula_dir),
        params=params,
        verbosity=0,
        max_pairs=None,
    )

    # The load step in TS should find exactly two minima (the tagged ones) and attempt to pair them
    # Expect either 1 pair evaluated (pair of two minima) or zero if similarity removed it
    assert isinstance(results, list)

    # Additionally verify DB rows now contain final flag for tagged confid
    assert_db_final_row(str(db_path), None, expect_final_id=False)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT key_value_pairs FROM systems")
    rows = cur.fetchall()
    assert len(rows) > 0, "No rows written to systems"

    tagged = 0
    for (kvp_json,) in rows:
        kvp = json.loads(kvp_json) if kvp_json else {}
        if kvp.get("final_unique_minimum"):
            tagged += 1

    assert tagged == 2, f"Expected 2 tagged rows, found {tagged}"
    conn.close()
