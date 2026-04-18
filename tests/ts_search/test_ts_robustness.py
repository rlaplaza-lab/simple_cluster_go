"""Robustness tests for TS search: OOM handling and malformed metadata."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from scgo.ts_search.transition_state_run import run_transition_state_search
from scgo.ts_search.ts_network import save_ts_network_metadata
from tests.cuda_skip import require_cuda
from tests.test_utils import create_preparedb, mark_test_minima_as_final


def test_save_ts_network_metadata_skips_malformed_success():
    """A `status=='success'` entry missing numeric fields should be skipped."""
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "reactant_energy": -5.0,
            "product_energy": -4.8,
            "ts_energy": None,  # malformed
            "barrier_height": None,
            "neb_converged": True,
            "n_images": 5,
        },
        {
            "pair_id": "1_2",
            "status": "success",
            "reactant_energy": -4.8,
            "product_energy": -4.6,
            "ts_energy": -4.3,
            "barrier_height": 0.5,
            "neb_converged": True,
            "n_images": 5,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_ts_network_metadata(
            ts_results, tmpdir, composition=["Cu", "Cu", "Cu"], minima_count=3
        )

        with open(path) as f:
            meta = json.load(f)

        # The malformed successful entry should be skipped
        assert meta["statistics"]["successful_ts"] == 1
        assert len(meta["ts_connections"]) == 1


def test_run_transition_state_search_handles_cuda_oom(monkeypatch):
    """Simulate a per-pair CUDA OOM and ensure the campaign continues and
    GPU cleanup is attempted. This test sets up a minimal mock DB locally.
    """
    require_cuda()

    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase_ga.data import DataConnection

    from scgo.database.metadata import add_metadata, update_metadata

    # Create a minimal mock database directory with a few relaxed minima
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_20260101_120000"
        run_dir.mkdir()
        db_path = run_dir / "candidates.db"

        db = create_preparedb(Atoms("Cu2"), db_path, population_size=20)

        # Minimum 1
        atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms1.center(vacuum=5.0)
        atoms1.calc = EMT()
        add_metadata(atoms1, raw_score=-10.0)
        atoms1.info["confid"] = 1
        db.add_unrelaxed_candidate(atoms1, description="Cu2_linear")

        # Minimum 2
        atoms2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])
        atoms2.center(vacuum=5.0)
        atoms2.calc = EMT()
        add_metadata(atoms2, raw_score=-10.0)
        atoms2.info["confid"] = 2
        db.add_unrelaxed_candidate(atoms2, description="Cu2_rotated")

        # Finalize: move unrelaxed -> relaxed (use DataConnection so add_relaxed_step persists)
        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            a.calc = EMT()
            update_metadata(a, raw_score=-a.get_potential_energy())
            da.add_relaxed_step(a)

        mark_test_minima_as_final(db_path)

        # Now patch the TS-finding call to raise a CUDA OOM and patch cleanup
        def fake_find_transition_state(*args, **kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.find_transition_state",
            fake_find_transition_state,
        )

        cleaned = {"called": False}

        def fake_cleanup(logger=None):
            cleaned["called"] = True

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.cleanup_torch_cuda", fake_cleanup
        )

        params = {"calculator": "EMT", "calculator_kwargs": {}}

        results = run_transition_state_search(
            composition=["Cu", "Cu"],
            base_dir=tmpdir,
            params=params,
            verbosity=0,
            max_pairs=1,
            neb_n_images=3,
            neb_fmax=0.5,
            neb_steps=10,
        )

        # Should return a list and include at least one failed result (not crash)
        assert isinstance(results, list)
        assert any(r.get("status") == "failed" for r in results)
        # cleanup should have been attempted
        assert cleaned["called"] is True
        # ensure no TS structure still holds a calculator reference
        for r in results:
            ts = r.get("transition_state")
            if ts is not None:
                assert ts.calc is None


def test_pairwise_cleanup_even_without_errors(monkeypatch):
    """GPU cleanup should be attempted after every pair, not only on OOMs.

    We simulate a minimal two-pair search and verify our patched
    `cleanup_torch_cuda` hook is invoked once per pair.  This regression test
    guards against future edits that accidentally remove the unconditional
    cleanup added by https://github.com/.../issue/xxx.
    """
    require_cuda()

    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase_ga.data import DataConnection

    from scgo.database.metadata import add_metadata, update_metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_20260101_120000"
        run_dir.mkdir()
        db_path = run_dir / "candidates.db"

        db = create_preparedb(Atoms("Cu2"), db_path, population_size=20)

        # add two minima (same as previous test setup)
        pairs = [
            ([[0, 0, 0], [2.5, 0, 0]], 1),
            ([[0, 0, 0], [1.8, 1.8, 0]], 2),
        ]
        for pos, confid in pairs:
            atoms = Atoms("Cu2", positions=pos)
            atoms.center(vacuum=5.0)
            atoms.calc = EMT()
            add_metadata(atoms, raw_score=-10.0)
            atoms.info["confid"] = confid
            db.add_unrelaxed_candidate(atoms, description=f"Cu2_{confid}")

        da = DataConnection(str(db_path))
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            a.calc = EMT()
            update_metadata(a, raw_score=-a.get_potential_energy())
            da.add_relaxed_step(a)

        mark_test_minima_as_final(db_path)

        calls = {"count": 0}

        def fake_cleanup(logger=None):
            calls["count"] += 1

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.cleanup_torch_cuda", fake_cleanup
        )

        params = {"calculator": "EMT", "calculator_kwargs": {}}

        results = run_transition_state_search(
            composition=["Cu", "Cu"],
            base_dir=tmpdir,
            params=params,
            verbosity=0,
            max_pairs=2,
            neb_n_images=3,
            neb_fmax=0.5,
            neb_steps=10,
        )

        assert isinstance(results, list)
        # at least two cleanup calls (one per pair)
        assert calls["count"] >= 2


def test_transition_state_results_do_not_retain_calculators(tmp_path):
    """Returned ts_results must not carry an attached calculator object."""
    params = {"calculator": "EMT", "calculator_kwargs": {}}
    results = run_transition_state_search(
        composition=["H", "H"],
        base_dir=tmp_path,
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    assert isinstance(results, list)
    for r in results:
        ts = r.get("transition_state")
        if ts is not None:
            assert ts.calc is None


def test_gpu_memory_does_not_grow(tmp_path, monkeypatch):
    """Repeated campaigns with a GPU-backed dummy calculator should not leak.

    This test monkeypatches ``get_calculator_class`` to return a simple
    ASE calculator that allocates a small CUDA tensor in its constructor.  We
    verify that memory usage immediately after two successive runs remains
    bounded (within a few MB) to catch regressions where calculators are
    retained by result structures.
    """
    require_cuda()
    from ase.calculators.emt import EMT

    from scgo.utils.run_helpers import get_calculator_class as _orig_get

    class GpuDummy(EMT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if torch.cuda.is_available():
                # allocate a tiny tensor to tag this instance
                self._buf = torch.zeros(1, device="cuda")

    def fake_get(name):
        if name == "GPUDUMMY":
            return GpuDummy
        return _orig_get(name)

    monkeypatch.setattr("scgo.utils.run_helpers.get_calculator_class", fake_get)
    # transition_state_run imports the function directly, so patch its copy too
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.get_calculator_class", fake_get
    )

    params = {"calculator": "GPUDUMMY", "calculator_kwargs": {}}

    # baseline memory
    before = torch.cuda.memory_allocated()
    run_transition_state_search(
        composition=["Cu", "Cu"],
        base_dir=str(tmp_path),
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    import gc

    gc.collect()
    mid = torch.cuda.memory_allocated()
    run_transition_state_search(
        composition=["Cu", "Cu"],
        base_dir=str(tmp_path),
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    gc.collect()
    after = torch.cuda.memory_allocated()

    # allow a small tolerance for driver bookkeeping
    assert mid <= before + 10_000_000
    assert after <= before + 10_000_000
