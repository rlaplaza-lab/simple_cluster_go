import json
import sqlite3

import pytest

from tests.test_utils import assert_db_final_row


def _create_db(path) -> None:
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE systems (id INTEGER PRIMARY KEY AUTOINCREMENT, key_value_pairs TEXT, energy REAL)"
        )
        conn.commit()


def test_assert_db_final_row_includes_final_id_in_key_value_pairs(tmp_path):
    dbp = tmp_path / "finals.db"
    _create_db(dbp)

    with sqlite3.connect(str(dbp)) as conn:
        cur = conn.cursor()
        kv = json.dumps(
            {
                "run_id": "r1",
                "final_unique_minimum": True,
                "final_id": "fid",
            }
        )
        cur.execute(
            "INSERT INTO systems (key_value_pairs, energy) VALUES (?, ?)",
            (kv, -1.0),
        )
        conn.commit()

    assert_db_final_row(str(dbp), "r1", expect_final_id=True)


def test_assert_db_final_row_kvp_only(tmp_path):
    dbp = tmp_path / "kvp.db"
    _create_db(dbp)

    with sqlite3.connect(str(dbp)) as conn:
        cur = conn.cursor()
        kv = json.dumps({"final_unique_minimum": True, "run_id": "r2"})
        cur.execute(
            "INSERT INTO systems (key_value_pairs, energy) VALUES (?, ?)", (kv, -2.0)
        )
        conn.commit()

    # should not raise
    assert_db_final_row(str(dbp), "r2", expect_final_id=False)


def test_assert_db_final_row_requires_final_id_when_requested(tmp_path):
    dbp = tmp_path / "nofid.db"
    _create_db(dbp)

    with sqlite3.connect(str(dbp)) as conn:
        cur = conn.cursor()
        kv = json.dumps({"final_unique_minimum": True, "run_id": "r3"})
        cur.execute(
            "INSERT INTO systems (key_value_pairs, energy) VALUES (?, ?)", (kv, -3.0)
        )
        conn.commit()

    # expect an AssertionError because expect_final_id=True but no final_id stored
    with pytest.raises(AssertionError):
        assert_db_final_row(str(dbp), "r3", expect_final_id=True)


def test_assert_db_final_row_allows_none_run_id(tmp_path):
    dbp = tmp_path / "anyrun.db"
    _create_db(dbp)

    with sqlite3.connect(str(dbp)) as conn:
        cur = conn.cursor()
        kv = json.dumps({"final_unique_minimum": True, "run_id": "not_matching"})
        cur.execute(
            "INSERT INTO systems (key_value_pairs, energy) VALUES (?, ?)", (kv, -4.0)
        )
        conn.commit()

    # expected_run_id=None should skip run_id check
    assert_db_final_row(str(dbp), None, expect_final_id=False)
