from ase import Atoms

from scgo.database.metadata import persist_provenance


def test_persist_provenance_writes_trial_and_trial_id_and_run():
    a = Atoms("Pt", positions=[[0, 0, 0]])
    persist_provenance(a, run_id="run_007", trial_id=13)

    prov = a.info.get("provenance", {})
    assert prov.get("run_id") == "run_007"
    assert prov.get("trial") == 13
    assert prov.get("trial_id") == 13

    kv = a.info.get("key_value_pairs", {})
    assert kv.get("run_id") == "run_007"
    assert kv.get("trial") == 13
    assert kv.get("trial_id") == 13

    meta = a.info.get("metadata", {})
    assert meta.get("run_id") == "run_007"
    assert meta.get("trial_id") == 13


def test_persist_provenance_writes_trial_id_without_run():
    a = Atoms("Pt", positions=[[0, 0, 0]])
    persist_provenance(a, run_id=None, trial_id=42)

    prov = a.info.get("provenance", {})
    assert prov.get("trial") == 42
    assert prov.get("trial_id") == 42

    kv = a.info.get("key_value_pairs", {})
    assert kv.get("trial") == 42
    assert kv.get("trial_id") == 42

    meta = a.info.get("metadata", {})
    assert meta.get("trial_id") == 42
