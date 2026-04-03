from scgo.param_presets import get_testing_params
from scgo.run_minima import run_scgo_trials
from tests.test_utils import assert_db_final_row


def test_main_writes_final_id_and_db_contains_it(tmp_path):
    """End-to-end: running scgo should write a deterministic final_id and
    mark it in the database when tagging is enabled.
    """
    params = get_testing_params()
    params["tag_final_minima"] = True
    params["n_trials"] = 1

    results = run_scgo_trials(
        ["Pt", "Pt"], params=params, seed=42, output_dir=str(tmp_path), verbosity=0
    )
    assert len(results) > 0

    db_files = list(tmp_path.glob("**/*.db"))
    assert db_files, "No DB files found"
    db_path = db_files[0]

    expected_run_id = db_path.parents[1].name if len(db_path.parents) >= 2 else None

    # assert at least one final-tag row exists and that it contains a final_id
    assert_db_final_row(str(db_path), expected_run_id, expect_final_id=True)
