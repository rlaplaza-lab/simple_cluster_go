from ase import Atoms
from ase.db import connect

from scgo.database.metadata import mark_final_minima_in_db
from tests.database.test_mark_final_minima import _iter_system_kvps
from tests.test_utils import assert_db_final_row


def test_mark_final_minima_accepts_db_paths_and_returns_summary(tmp_path):
    # Create a DB outside the canonical run_xxx/trial_xxx layout
    dbpath = tmp_path / "external.db"
    db = connect(str(dbpath))

    # Write a candidate row with identifiable confid/run_id
    db.write(
        Atoms("Pt", positions=[[0, 0, 0]]),
        relaxed=True,
        key_value_pairs={"run_id": "run_ext", "trial": 1, "raw_score": -0.1},
    )

    # Prepare final_minima_info matching the above provenance
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info.setdefault("provenance", {})
    atoms.info["provenance"]["run_id"] = "run_ext"
    atoms.info["provenance"]["trial"] = 1

    final_info = [
        {"atoms": atoms, "energy": -0.1, "rank": 1, "final_written": "foo.xyz"}
    ]

    # Call helper with explicit db_paths list (skip registry discovery)
    summary = mark_final_minima_in_db(
        final_info, base_dir=str(tmp_path), db_paths=[str(dbpath)]
    )

    assert isinstance(summary, dict)
    assert summary.get("rows_updated", 0) >= 1
    assert summary.get("dbs_touched", 0) >= 1
    assert str(dbpath) in summary.get("details", {})

    # Verify DB row was updated with final_unique_minimum and that summary matches actual DB
    assert_db_final_row(str(dbpath), "run_ext", expect_final_id=False)

    kv_list = list(_iter_system_kvps(dbpath))
    assert kv_list
    count_tagged = sum(1 for kv in kv_list if kv.get("final_unique_minimum"))
    assert count_tagged >= 1
    # summary should reflect number of rows actually updated
    assert summary.get("rows_updated", 0) == count_tagged
