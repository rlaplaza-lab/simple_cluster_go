"""Tests for parallel database worker loading SCGO-stamped SQLite databases."""

from __future__ import annotations

import pytest
from ase_ga.data import DataConnection

from scgo.database.helpers import _load_single_database_worker
from scgo.database.schema import stamp_scgo_database
from tests.test_utils import create_preparedb


def test_load_single_database_worker_extracts_scgo_db(tmp_path, pt2_atoms):
    """Worker uses extract_minima_from_database_file; DB must be SCGO-stamped."""
    run_dir = tmp_path / "run_000" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    create_preparedb(pt2_atoms, db_path)

    # Open connection and add a relaxed candidate
    da = DataConnection(str(db_path))
    a = pt2_atoms.copy()
    a.info.setdefault("key_value_pairs", {})
    a.info["key_value_pairs"]["raw_score"] = -10.0
    # Prepare minimal fields expected by ASE DataConnection
    a.info.setdefault("data", {})
    a.info.setdefault("confid", 1)
    # ASE's DataConnection expects an unrelaxed candidate to exist first
    da.add_unrelaxed_candidate(a, description="test:insert")
    da.add_relaxed_step(a)

    stamp_scgo_database(db_path)

    # Call the worker directly (as would be done in ProcessPool) and assert it returns the candidate
    minima = _load_single_database_worker(
        str(db_path), composition=None, run_id="run_000", trial_id=1
    )
    assert isinstance(minima, list)
    assert len(minima) == 1
    energy, atoms = minima[0]
    # raw_score = -10.0 -> energy = 10.0 by convention
    assert energy == pytest.approx(10.0)
    assert atoms is not None
    assert atoms.get_chemical_symbols() == pt2_atoms.get_chemical_symbols()
