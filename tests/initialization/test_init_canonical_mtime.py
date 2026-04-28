"""Tests for canonical mtime cache invalidation and deterministic refresh behavior."""

from __future__ import annotations

import os
import time

import pytest
from ase_ga.data import DataConnection

from scgo.database import close_data_connection
from scgo.database.helpers import setup_database
from scgo.initialization.candidate_discovery import invalidate_db_canonical_mtime
from scgo.initialization.initializers import _find_smaller_candidates


def test_canonical_mtime_invalidation_and_refresh(tmp_path, pt2_atoms):
    run_dir = tmp_path / "run_000" / "trial_1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "ga_go.db"

    # Create a SCGO DB and add a single non-final candidate
    da = setup_database(run_dir, "ga_go.db", pt2_atoms, initial_candidate=None)
    try:
        a = pt2_atoms.copy()
        a.info.setdefault("key_value_pairs", {})
        a.info["key_value_pairs"]["raw_score"] = -10.0
        a.info.setdefault("data", {})
        a.info.setdefault("confid", 1)
        da.add_unrelaxed_candidate(a, description="test:insert1")
        da.add_relaxed_step(a)
    finally:
        close_data_connection(da)

    # First call: should not find any seeds because we only accept final-tagged minima
    seeds = _find_smaller_candidates(["Pt", "Pt", "Pt"], db_glob_pattern=str(db_path))
    assert seeds == {}

    # Add a final-tagged candidate to the DB
    da2 = DataConnection(str(db_path))
    try:
        b = pt2_atoms.copy()
        b.info.setdefault("key_value_pairs", {})
        b.info["key_value_pairs"]["raw_score"] = -9.0
        b.info.setdefault("key_value_pairs", {})
        b.info["key_value_pairs"]["final_unique_minimum"] = True
        b.info.setdefault("data", {})
        b.info.setdefault("confid", 2)
        da2.add_unrelaxed_candidate(b, description="test:insert2")
        da2.add_relaxed_step(b)
    finally:
        close_data_connection(da2)

    # Ensure file mtime is updated
    now = time.time()
    os.utime(db_path, (now, now))

    # Explicitly invalidate canonical mtime cache and re-run finder
    invalidate_db_canonical_mtime(str(db_path))

    seeds_after = _find_smaller_candidates(
        ["Pt", "Pt", "Pt"], db_glob_pattern=str(db_path)
    )
    assert "Pt2" in seeds_after
    assert len(seeds_after["Pt2"]) >= 1
    energy, atoms = seeds_after["Pt2"][0]
    assert atoms is not None
    assert energy == pytest.approx(-9.0)
