"""Pure-unit tests for parallel NEB module-level helpers (no MLIP / GPU).

The companion ``test_parallel_neb.py`` is module-marked ``requires_cuda`` /
``requires_mace`` because most of its cases build a real ``TorchSimBatchRelaxer``.
The helpers exercised here (`_stage1_band_climb_eligible`, `_evaluate_bands_in_chunks`)
are pure Python and are tested with fake relaxers so they run in the fast suite.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ase import Atoms

from scgo.ts_search.parallel_neb import (
    ParallelNEBBatch,
    _evaluate_bands_in_chunks,
    _stage1_band_climb_eligible,
)
from scgo.ts_search.transition_state import (
    TorchSimNEB,
    _image_has_cached_forces,
    evaluate_neb_image_energies,
    interpolate_path,
)

# Tagged so a GPU log scan can tell simulated OOM apart from a real one.
SIMULATED_OOM = (
    "CUDA out of memory [scgo-simulated-failure]. Tried to allocate 6.43 GiB."
)


# ---------------------------------------------------------------------------
# T2: two-stage CI-NEB climb eligibility predicate
# ---------------------------------------------------------------------------


def test_stage1_climb_eligible_converged_band():
    """A converged stage-1 band (steps > 0, not failed) is climb-eligible."""
    summary = {
        "converged": True,
        "steps_taken": 5,
        "final_fmax": 0.01,
        "error": None,
        "failed": False,
    }
    assert _stage1_band_climb_eligible(summary) is True


def test_stage1_climb_eligible_soft_nonconvergence():
    """The soft "did not converge" sentinel is the normal case and IS eligible.

    This is the regression: bands that merely exhausted stage-1's half budget
    were previously filtered out by ``not summary.get("error")`` so nothing
    climbed.
    """
    summary = {
        "converged": False,
        "steps_taken": 10,
        "final_fmax": 0.3,
        "error": "NEB did not converge after 10 steps",
        "failed": False,
    }
    assert _stage1_band_climb_eligible(summary) is True


def test_stage1_climb_not_eligible_nonfinite_forces():
    """A hard failure (non-finite forces) is never climb-eligible."""
    summary = {
        "converged": False,
        "steps_taken": 1,
        "final_fmax": float("nan"),
        "error": "NEB forces are non-finite (fmax=nan); refusing optimizer step",
        "failed": True,
    }
    assert _stage1_band_climb_eligible(summary) is False


def test_stage1_climb_not_eligible_never_ran():
    """A band that never took a step is not climb-eligible."""
    summary = {
        "converged": False,
        "steps_taken": 0,
        "final_fmax": None,
        "error": "NEB not processed",
        "failed": False,
    }
    assert _stage1_band_climb_eligible(summary) is False


def test_stage1_climb_string_sniff_fallback_when_no_failed_key():
    """Without the explicit ``failed`` boolean the predicate sniffs the error text.

    Summaries produced outside ``run_optimization`` (e.g. OOM-retry stubs) have
    no ``failed`` key: only an empty error or the soft "did not converge"
    sentinel stays eligible; non-finite / OOM / any other exception text is a
    hard failure.
    """
    # No error, steps taken -> eligible.
    assert _stage1_band_climb_eligible({"steps_taken": 3, "error": None}) is True
    # Soft sentinel -> eligible.
    assert _stage1_band_climb_eligible(
        {"steps_taken": 7, "error": "NEB did not converge after 7 steps"}
    )
    # Non-finite forces -> hard failure.
    assert not _stage1_band_climb_eligible(
        {"steps_taken": 1, "error": "NEB forces are non-finite (fmax=nan)"}
    )
    # CUDA OOM text -> hard failure.
    assert not _stage1_band_climb_eligible({"steps_taken": 2, "error": SIMULATED_OOM})
    # Any other exception message -> hard failure.
    assert not _stage1_band_climb_eligible(
        {"steps_taken": 4, "error": "boom: bad tensor"}
    )


def test_stage1_climb_failed_boolean_wins_over_error_text():
    """The explicit ``failed`` flag takes precedence over the error string."""
    # failed=False but a soft sentinel error present -> eligible.
    assert (
        _stage1_band_climb_eligible(
            {"steps_taken": 5, "failed": False, "error": "NEB did not converge"}
        )
        is True
    )
    # failed=True even though the error text alone looks soft -> not eligible.
    assert (
        _stage1_band_climb_eligible(
            {"steps_taken": 5, "failed": True, "error": "NEB did not converge"}
        )
        is False
    )


# ---------------------------------------------------------------------------
# T7: chunked pre-screen energy evaluation
# ---------------------------------------------------------------------------


def _make_bands(n_bands: int, n_images: int, n_atoms: int) -> list[list[Atoms]]:
    """Bands whose images encode a unique running index in ``positions[0, 0]``."""
    counter = itertools.count()
    bands: list[list[Atoms]] = []
    for _ in range(n_bands):
        images: list[Atoms] = []
        for _ in range(n_images):
            pos = np.zeros((n_atoms, 3))
            pos[0, 0] = float(next(counter))
            images.append(Atoms(numbers=[29] * n_atoms, positions=pos))
        bands.append(images)
    return bands


class _RecordingEnergyRelaxer:
    """Fake relaxer returning ``positions[0, 0]`` as energy; records batch sizes."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_atom_counts: list[int] = []

    def relax_batch(self, atoms_list, steps=0):
        self.calls += 1
        self.batch_atom_counts.append(sum(len(a) for a in atoms_list))
        results = []
        for a in atoms_list:
            ra = a.copy()
            ra.arrays["forces"] = np.zeros((len(a), 3))
            results.append((float(a.get_positions()[0, 0]), ra))
        return results


class _OomFirstCallRelaxer(_RecordingEnergyRelaxer):
    """Recording relaxer that raises a CUDA OOM on its first ``relax_batch``."""

    def relax_batch(self, atoms_list, steps=0):
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError(SIMULATED_OOM)
        return super().relax_batch(atoms_list, steps)


def _flatten(band_lists: list[list[float]]) -> list[float]:
    return [e for band in band_lists for e in band]


def test_evaluate_bands_in_chunks_respects_atom_budget_and_order():
    """(a) no batch exceeds the atom budget; (b) energies keep input order."""
    n_bands, n_images, n_atoms = 5, 5, 3
    band_cost = n_images * n_atoms  # 15
    atom_budget = 2 * band_cost  # 30 -> two bands per batch
    bands = _make_bands(n_bands, n_images, n_atoms)
    relaxer = _RecordingEnergyRelaxer()

    result = _evaluate_bands_in_chunks(
        bands, relaxer, atom_budget=atom_budget, band_cap=None
    )

    # (b) per-band shape and concatenated order preserved.
    assert [len(b) for b in result] == [n_images] * n_bands
    assert _flatten(result) == [float(i) for i in range(n_bands * n_images)]
    # (a) no batch exceeded the atom budget.
    assert relaxer.batch_atom_counts, "relaxer was never called"
    assert max(relaxer.batch_atom_counts) <= atom_budget
    # Greedy binning: [15,15]->30, [15,15]->30, [15]->15.
    assert relaxer.batch_atom_counts == [30, 30, 15]


def test_evaluate_bands_in_chunks_no_budget_single_batch():
    """No atom budget and no band cap -> a single fused batch."""
    bands = _make_bands(4, 3, 2)
    relaxer = _RecordingEnergyRelaxer()

    result = _evaluate_bands_in_chunks(bands, relaxer, atom_budget=None, band_cap=None)

    assert relaxer.calls == 1
    assert relaxer.batch_atom_counts == [4 * 3 * 2]
    assert _flatten(result) == [float(i) for i in range(4 * 3)]
    assert all(_image_has_cached_forces(img) for band in bands for img in band)


def test_evaluate_bands_in_chunks_band_cap_limits_bands_per_batch():
    """``band_cap`` caps the number of bands per batch even under a large budget."""
    bands = _make_bands(3, 4, 2)
    relaxer = _RecordingEnergyRelaxer()

    result = _evaluate_bands_in_chunks(
        bands, relaxer, atom_budget=1_000_000, band_cap=1
    )

    # One band per batch -> three calls.
    assert relaxer.calls == 3
    assert relaxer.batch_atom_counts == [4 * 2, 4 * 2, 4 * 2]
    assert _flatten(result) == [float(i) for i in range(3 * 4)]


def test_evaluate_bands_in_chunks_retries_once_on_cuda_oom(monkeypatch):
    """(c) an OOM on the first oversized batch still yields all energies."""
    cleanup_calls = {"n": 0}

    def _fake_cleanup(*args, **kwargs):
        cleanup_calls["n"] += 1

    monkeypatch.setattr("scgo.ts_search.parallel_neb.cleanup_torch_cuda", _fake_cleanup)

    n_bands, n_images, n_atoms = 4, 5, 3
    band_cost = n_images * n_atoms  # 15
    atom_budget = 2 * band_cost  # 30
    bands = _make_bands(n_bands, n_images, n_atoms)
    relaxer = _OomFirstCallRelaxer()

    result = _evaluate_bands_in_chunks(
        bands, relaxer, atom_budget=atom_budget, band_cap=None
    )

    # All energies returned, in input order, despite the first-batch OOM.
    assert [len(b) for b in result] == [n_images] * n_bands
    assert _flatten(result) == [float(i) for i in range(n_bands * n_images)]
    # cleanup_torch_cuda ran between the failed attempt and the retry.
    assert cleanup_calls["n"] >= 1
    # First call OOM'd (unrecorded); the retry re-binned that chunk at half cost
    # (15 atoms each), then the second chunk ran normally.
    assert relaxer.batch_atom_counts == [15, 15, 30]


def test_parallel_step0_reuses_energy_screen_forces(cu3_triangle, cu3_linear):
    """Energy-screen SP attaches forces; parallel step 0 must not re-evaluate."""
    relaxer = _RecordingEnergyRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    evaluate_neb_image_energies(images, relaxer)
    assert relaxer.calls == 1
    assert all(_image_has_cached_forces(img) for img in images)

    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    ParallelNEBBatch([neb], relaxer, max_total_steps=5).run_optimization(
        fmax=1.0, max_steps=1
    )
    # screen + post-loop PES refresh; step 0 reused the screen cache
    assert relaxer.calls == 2
    assert neb.get_force_calls() == 1


def test_run_parallel_neb_forwards_bond_tolerance_to_interpolate(
    monkeypatch, cu3_triangle, cu3_linear, tmp_path
):
    """The parallel runner forwards ``neb_interpolation_bond_tolerance_a`` (F13)."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search
    from scgo.utils.ts_runner_kwargs import NebRunConfig

    cfg_kwargs: dict = {
        "neb_n_images": 3,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": 2,
        "neb_climb": False,
        "neb_interpolation_method": "linear",
        "neb_align_endpoints": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_mic": False,
        "neb_tangent_method": "aseneb",
        "neb_surface_cell_remap": True,
        "neb_surface_lattice_rotation": True,
        "neb_surface_max_lattice_shift": 1,
        "n_slab": 0,
        "n_core_mobile": None,
        "n_adsorbate_mobile": None,
        "adsorbate_fragment_lengths": None,
        "max_endpoint_mismatch": None,
        "adsorbate_definition": None,
        "connectivity_factor": None,
        "allow_cluster_fragmentation": False,
        "allow_adsorbate_surface_detachment": False,
        "enforce_adsorbate_subgraph_integrity": True,
        "system_type": "gas_cluster",
        "surface_config": None,
        "torchsim_params": {},
        "neb_prescreen_clash_distance": 1.0,
        "min_saddle_prominence": 0.10,
        "neb_max_spurious_barrier": 8.0,
        "layer_cluster_threshold_ang": 0.4,
        "neb_interpolation_bond_tolerance_a": 0.37,
    }
    neb_cfg = NebRunConfig(**cfg_kwargs)

    captured: dict = {}

    def _fake_interpolate(a1, a2, **kwargs):
        captured.update(kwargs)
        return [a1.copy(), a2.copy()]

    class _FakeRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            out = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                out.append((0.0, ra))
            return out

    class _StubBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = neb_instances

        def run_optimization(self, fmax=0.05, max_steps=100):
            return [
                {"converged": False, "final_fmax": None, "steps_taken": 1}
                for _ in self.neb_instances
            ]

    with (
        patch(
            "scgo.ts_search.parallel_neb.prepare_neb_endpoints",
            side_effect=lambda a, b, _cfg: (a.copy(), b.copy()),
        ),
        patch(
            "scgo.ts_search.parallel_neb.interpolate_path",
            side_effect=_fake_interpolate,
        ),
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path"),
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_FakeRelaxer(),
        ),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _StubBatch),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result"),
    ):
        results, _meta = run_parallel_neb_search(
            [(0, 1)],
            [(0.0, cu3_triangle), (0.5, cu3_linear)],
            neb_cfg=neb_cfg,
            run_dir=tmp_path,
            rng=None,
            verbosity=0,
        )

    assert results
    assert captured["neb_interpolation_bond_tolerance_a"] == pytest.approx(0.37)
