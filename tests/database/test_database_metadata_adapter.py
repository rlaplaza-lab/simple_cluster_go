from ase import Atoms

from scgo.database import close_data_connection
from scgo.database.helpers import setup_database
from tests.test_utils import assert_run_id_persisted


def test_db_adapter_merges_metadata_into_key_value_pairs(tmp_path):
    """Ensure DB adapters merge metadata (run_id) into key_value_pairs.

    Regression test: previously metadata such as run_id could be lost when
    `key_value_pairs` already existed on `Atoms`. Both helpers and connection
    adapters should ensure `run_id` and other metadata are present in
    `atoms.info['key_value_pairs']` after writing to the DB.
    """

    outdir = tmp_path / "test_db_adapter"
    db = setup_database(
        output_dir=outdir,
        db_filename="test.db",
        atoms_template=Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]]),
        remove_existing=True,
    )

    # Create atoms with explicit metadata and existing key_value_pairs
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]])
    a.info["metadata"] = {"run_id": "run_test_123", "generation": 1}
    # Simulate existing ASE key_value_pairs lacking run_id
    # Use a non-reserved key so it won't be mutated by ASE write logic
    a.info["key_value_pairs"] = {"raw_score": -10.0, "user_note": "hello"}
    # Minimal 'data' mapping expected by ASE GA DataConnection.write
    a.info["data"] = {}

    # Add as unrelaxed candidate (adapter will call _ensure_kv)
    # description must be colon-separated per ASE GA expectations
    # Use a non-numeric description suffix to satisfy ASE DB checks
    db.add_unrelaxed_candidate(a, description="test:alpha")

    # Retrieve unrelaxed candidate to inspect stored key_value_pairs
    u = db.get_an_unrelaxed_candidate()
    assert u is not None

    # run_id should be present in either metadata or legacy key_value_pairs
    assert_run_id_persisted(u, "run_test_123")
    # existing non-reserved keys should be preserved in key_value_pairs
    kv = u.info.get("key_value_pairs", {})
    assert kv.get("user_note") == "hello"
    assert "raw_score" in kv

    close_data_connection(db)


def test_unrelaxed_metadata_persisted_for_cross_process_reads(tmp_path):
    """Ensure provenance for unrelaxed candidates is visible to other processes/readers.

    Regression: `_DBAdapter` previously cached metadata only in-memory which
    meant other connections could not discover provenance for unrelaxed rows.
    We persist critical keys into `key_value_pairs` to maintain backward
    compatibility and cross-process discovery.
    """
    outdir = tmp_path / "test_db_adapter_proc"
    db_file = outdir / "test.db"
    db = setup_database(
        output_dir=outdir,
        db_filename="test.db",
        atoms_template=Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]]),
        remove_existing=True,
    )

    # Writer: add unrelaxed candidate with metadata + existing key_value_pairs
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]])
    a.info["metadata"] = {"run_id": "run_proc_42", "trial_id": 99, "confid": "c-1"}
    a.info["key_value_pairs"] = {"raw_score": -5.0, "user_note": "keep-me"}
    a.info["data"] = {}
    db.add_unrelaxed_candidate(a, description="test:proc")

    # Reader: open a fresh DataConnection and retrieve the unrelaxed candidate
    from ase_ga.data import DataConnection

    da2 = DataConnection(str(db_file))
    u = da2.get_an_unrelaxed_candidate()
    assert u is not None
    # Critical provenance keys should be present on the retrieved Atoms
    assert_run_id_persisted(u, "run_proc_42")
    kv = u.info.get("key_value_pairs", {})
    assert kv.get("trial_id") == 99 or kv.get("trial") == 99
    assert kv.get("confid") == "c-1"

    # Existing user keys must be preserved
    assert kv.get("user_note") == "keep-me"

    close_data_connection(db)
    close_data_connection(da2)
