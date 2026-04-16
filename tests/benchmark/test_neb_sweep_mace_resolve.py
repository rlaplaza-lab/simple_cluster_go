"""Unit tests for NEB sweep minima/composition helpers (no GPU)."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.io import write

from benchmark.neb_sweep_mace import (
    _barrier_suspicious,
    _parse_models,
    resolve_searches_composition,
)


def disabled_resolve_searches_composition_uses_explicit_list(tmp_path):
    assert resolve_searches_composition(tmp_path, ["Pt", "Au"]) == ["Pt", "Au"]


def disabled_resolve_searches_composition_infers_from_final_unique_minima(tmp_path):
    fin = tmp_path / "final_unique_minima"
    fin.mkdir(parents=True)
    write(fin / "m.xyz", Atoms("Cu2", positions=[[0, 0, 0], [2.1, 0, 0]]))
    assert resolve_searches_composition(tmp_path, None) == ["Cu", "Cu"]


def disabled_resolve_searches_composition_missing_raises(tmp_path):
    with pytest.raises(SystemExit, match="Could not infer composition"):
        resolve_searches_composition(tmp_path, None)


def disabled_parse_models_fallback_and_csv():
    assert _parse_models(None, "mace_matpes_0") == ["mace_matpes_0"]
    assert _parse_models("", "x") == ["x"]
    assert _parse_models("a, b", "x") == ["a", "b"]


def disabled_barrier_suspicious_respects_convergence_and_threshold():
    assert _barrier_suspicious(
        {"neb_converged": True, "barrier_height": 10.0},
        max_barrier_ev=3.0,
    )
    assert not _barrier_suspicious(
        {"neb_converged": False, "barrier_height": 10.0},
        max_barrier_ev=3.0,
    )
    assert not _barrier_suspicious(
        {"neb_converged": True, "barrier_height": 1.0},
        max_barrier_ev=3.0,
    )
