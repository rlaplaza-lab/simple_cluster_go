"""Tests for parallel NEB optimization with batched GPU force evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from ase.constraints import FixAtoms

from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.ts_search.neb_endpoints import prepare_neb_endpoints
from scgo.ts_search.parallel_neb import ParallelNEBBatch, _neb_image_dedup_key
from scgo.ts_search.transition_state import TorchSimNEB, interpolate_path
from scgo.utils.ts_runner_kwargs import NebRunConfig

pytestmark = [pytest.mark.requires_cuda, pytest.mark.requires_mace]


def _gas_neb_cfg(**overrides) -> NebRunConfig:
    """Minimal gas-cluster NebRunConfig for parallel NEB unit tests."""
    kwargs: dict = {
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
    }
    # Per-system-type required fields (mirrors TS_DEFAULTS_BY_SYSTEM_TYPE). Set
    # here, before kwargs.update(overrides), so individual tests can still
    # override them. Must stay in sync with NebRunConfig's required fields.
    _surface_defaults = ("surface_cluster", "surface", "surface_adsorbate")
    _is_surface = overrides.get("system_type", "gas_cluster") in _surface_defaults
    if _is_surface:
        _clash, _prom, _spurious = 0.7, 0.40, 8.0
    else:
        _clash, _prom, _spurious = 1.0, 0.10, 8.0
    kwargs.update(
        neb_prescreen_clash_distance=_clash,
        min_saddle_prominence=_prom,
        neb_max_spurious_barrier=_spurious,
        layer_cluster_threshold_ang=0.4,
        neb_interpolation_bond_tolerance_a=0.5,
    )
    kwargs.update(overrides)
    return NebRunConfig(**kwargs)


def _unique_neb_image_count(*image_lists: list) -> int:
    keys = {_neb_image_dedup_key(atoms) for images in image_lists for atoms in images}
    return len(keys)


class _CountingFakeRelaxer:
    """Relaxer stub that records batch sizes and returns zero forces.

    Energies use ``sum(positions**2)`` so ASE ``improvedtangent`` sees a
    non-flat band (``sum(positions)`` is accidentally constant along the Cu3
    IDPP path). Identical images still share the same energy for dedup.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def relax_batch(self, atoms_list, steps=0):
        self.calls += 1
        self.batch_sizes.append(len(atoms_list))
        results = []
        for a in atoms_list:
            ra = a.copy()
            ra.arrays["forces"] = np.zeros((len(a), 3))
            energy = float(np.sum(a.get_positions() ** 2))
            results.append((energy, ra))
        return results


def _assert_one_global_relax_batch(
    neb1, neb2, relaxer, *, expected_unique: int
) -> None:
    batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=5)
    batch.run_optimization(fmax=1.0, max_steps=1)
    # One optimization eval + one post-loop PES refresh
    assert relaxer.calls == 2
    assert relaxer.batch_sizes == [expected_unique, expected_unique]
    assert neb1.get_force_calls() >= 1
    assert neb2.get_force_calls() >= 1


class TestParallelNEBBatch:
    """Tests for ParallelNEBBatch parallel NEB optimization."""

    def test_parallel_neb_initialization(self, cu3_triangle, cu3_linear):
        """Test ParallelNEBBatch initialization with multiple NEBs."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.05,
            max_steps=100,
        )

        # Create two NEB paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(
            cu3_triangle, cu3_linear, n_images=3, method="linear"
        )

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=50)

        assert len(batch.neb_instances) == 2
        assert len(batch.active_nebs) == 2
        assert batch.step_count == 0

    @pytest.mark.slow
    def test_parallel_neb_basic_run(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test basic parallel NEB optimization with multiple paths."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
            autobatcher=True,
        )

        # Create three NEB paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_bent, n_images=3, method="linear")
        images3 = interpolate_path(cu3_linear, cu3_bent, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
        neb3 = TorchSimNEB(images3, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2, neb3], relaxer, max_total_steps=100)

        # Run optimization
        results = batch.run_optimization(fmax=0.5, max_steps=50)

        # Check results structure
        assert len(results) == 3
        for result in results:
            assert "converged" in result
            assert "steps_taken" in result
            assert "final_fmax" in result
            assert "force_calls" in result
            assert "error" in result

        # All should have run at least one step
        for result in results:
            assert result["steps_taken"] > 0 or result["error"] is not None

    @pytest.mark.slow
    def test_parallel_neb_summary(self, cu3_triangle, cu3_linear):
        """Test ParallelNEBBatch summary statistics."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=50,
        )

        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=30)
        batch.run_optimization(fmax=1.0, max_steps=30)

        summary = batch.get_summary()

        assert summary["total_nebs"] == 2
        assert summary["total_steps"] > 0
        assert summary["converged"] + summary["failed"] <= 2

    def test_parallel_neb_batching_efficiency(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test that parallel NEB batches images from multiple NEBs together."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
        )

        # Create two NEBs with 5 images each
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=5, method="idpp")
        images2 = interpolate_path(cu3_linear, cu3_bent, n_images=5, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=50)

        # Run one step and verify both NEBs got evaluated
        results = batch.run_optimization(fmax=1.0, max_steps=1)

        # After 1 step, both should have attempted force evaluation
        assert results[0]["steps_taken"] >= 1 or results[0]["error"] is not None
        assert results[1]["steps_taken"] >= 1 or results[1]["error"] is not None

    def test_parallel_neb_partial_convergence(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test parallel NEB with different convergence rates."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
        )

        # Create two different paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_bent, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=200)

        # Run with loose convergence to allow some to finish
        results = batch.run_optimization(fmax=2.0, max_steps=200)

        # Check that at least one NEB attempted optimization or both have errors
        assert any(r["steps_taken"] > 0 for r in results) or any(
            r["error"] is not None for r in results
        )


def test_parallel_neb_relax_batch_dedups_identical_cu3_paths(cu3_triangle, cu3_linear):
    """Cu3 triangle→linear: IDPP matches linear, so 10 slots collapse to 5.

    ``ParallelNEBBatch`` still evaluates once globally and fans results out to
    both bands (no per-band second ``relax_batch``).
    """
    relaxer = _CountingFakeRelaxer()
    images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    images2 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="linear")

    assert len(images1) == len(images2) == 5
    assert np.allclose(images1[2].positions, images2[2].positions)
    expected_unique = _unique_neb_image_count(images1, images2)
    assert expected_unique == 5
    assert expected_unique < len(images1) + len(images2)

    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    _assert_one_global_relax_batch(neb1, neb2, relaxer, expected_unique=expected_unique)


def test_parallel_neb_relax_batch_keeps_distinct_ir4_interiors(
    ir4_tetrahedron, ir4_tetrahedron_atom_swapped
):
    """Ir4 tet→atom-swapped tet: IDPP interiors differ from linear.

    Shared endpoints still dedupe (2), but distinct interiors keep 3+3, so the
    first ``relax_batch`` sees 8 unique images rather than 5 or 10.
    """
    relaxer = _CountingFakeRelaxer()
    images1 = interpolate_path(
        ir4_tetrahedron, ir4_tetrahedron_atom_swapped, n_images=3, method="idpp"
    )
    images2 = interpolate_path(
        ir4_tetrahedron, ir4_tetrahedron_atom_swapped, n_images=3, method="linear"
    )

    assert len(images1) == len(images2) == 5
    assert not np.allclose(images1[2].positions, images2[2].positions)
    expected_unique = _unique_neb_image_count(images1, images2)
    assert expected_unique == 8
    assert expected_unique > len(images1)
    assert expected_unique < len(images1) + len(images2)

    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    _assert_one_global_relax_batch(neb1, neb2, relaxer, expected_unique=expected_unique)


def test_parallel_neb_uses_neb_forces_for_stepping(cu3_triangle, cu3_linear):
    """Verify position updates use NEB forces (not raw PES forces).

    We provide PES forces = 0 via the relaxer but mock `neb.get_forces()` to
    return a known NEB force on the intermediate image and then assert that
    the intermediate positions change according to that NEB force and the
    batch.step_size.
    """

    class FakeRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                results.append((0.0, ra))
            return results

    relaxer = FakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Replace neb.get_forces() with an NEB-force array that exerts a known
    # force on the first interior image (band index 1). ASE NEB.get_forces()
    # returns interior images only, shape (n_interior_atoms, 3).
    def fake_neb_get_forces():
        n_images = len(neb.images)
        natoms = len(neb.images[0])
        n_int_atoms = (n_images - 2) * natoms
        forces = np.zeros((n_int_atoms, 3), dtype=float)
        # Apply +0.5 eV/Ang in x for all atoms of the first interior image
        forces[0:natoms, 0] = 0.5
        return forces

    neb.get_forces = fake_neb_get_forces

    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=1)

    # Record initial position of the first atom in the intermediate image
    init_pos = neb.images[1].positions[0].copy()

    # Use a smaller fmax to force an optimizer step (NEB force = 0.5)
    batch.run_optimization(fmax=0.1, max_steps=1)

    # Position should have moved in the same direction as the optimizer
    # gradient (FIRE uses the supplied NEB forces directly). Force was +0.5
    new_pos = neb.images[1].positions[0]
    assert new_pos[0] > init_pos[0]
    assert not np.allclose(new_pos, init_pos)

    # An ASE optimizer should have been created for this NEB (default: FIRE)
    assert 0 in batch._optimizers
    from ase.optimize import FIRE as ASE_FIRE

    assert isinstance(batch._optimizers[0], ASE_FIRE)


def test_torchsimneb_get_forces_skips_relax_if_forces_present(cu3_triangle, cu3_linear):
    """TorchSimNEB.get_forces() should not call relax_batch when images
    already contain PES forces/calculators (cached results).
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    class DummyRelaxer:
        def __init__(self):
            self.calls = 0

        def relax_batch(self, images, steps=0):
            self.calls += 1
            results = []
            for a in images:
                ra = a.copy()
                ra.arrays["forces"] = np.ones((len(a), 3))
                results.append((0.0, ra))
            return results

    relaxer = DummyRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

    # Pre-attach SinglePointCalculator (forces present) to simulate cached PES
    for img in images:
        img.calc = SinglePointCalculator(
            img, energy=0.0, forces=np.zeros((len(img), 3))
        )

    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Should skip calling relax_batch because forces are already present
    neb.get_forces()
    assert relaxer.calls == 0


def test_parallel_neb_skips_endpoints_after_first_step(cu3_triangle, cu3_linear):
    """After step 0, only interior images are batch-evaluated."""
    relaxer = _CountingFakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    assert len(images) == 5
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=5)
    # Force at least two steps by keeping fmax tiny and get_forces returning large force.
    original_get_forces = neb.get_forces

    def always_high_forces():
        forces = original_get_forces()
        return np.ones_like(forces) * 10.0

    neb.get_forces = always_high_forces  # type: ignore[method-assign]
    batch.run_optimization(fmax=1e-6, max_steps=2)
    # step 0: all images; step 1: interiors; final PES refresh: all images again
    assert relaxer.calls == 3
    assert relaxer.batch_sizes[0] == 5  # all images on step 0
    assert relaxer.batch_sizes[1] == 3  # interiors only
    assert relaxer.batch_sizes[2] == 5  # post-loop refresh


def test_parallel_neb_refuses_step_on_nonfinite_fmax(cu3_triangle, cu3_linear):
    """Non-finite NEB fmax must fail the band without calling the optimizer."""

    class _NanForceRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            out = []
            for a in atoms_list:
                ra = a.copy()
                # Direct non-finite forces (ASE improvedtangent flat-band NaNs
                # are version-dependent; assert the refuse-step path itself).
                ra.arrays["forces"] = np.full((len(a), 3), np.nan)
                out.append((0.0, ra))
            return out

    relaxer = _NanForceRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False, method="improvedtangent")
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=5)
    results = batch.run_optimization(fmax=0.1, max_steps=2)

    assert results[0]["converged"] is False
    assert not np.isfinite(results[0]["final_fmax"])
    assert "non-finite" in (results[0]["error"] or "")
    assert 0 not in batch._optimizers
    assert 0 in batch.failed_nebs


def test_parallel_neb_refresh_keeps_energies_after_steps(cu3_triangle, cu3_linear):
    """Post-loop PES refresh must leave readable energies after FIRE steps."""
    from scgo.ts_search.transition_state import (
        _finalize_neb_result,
        _image_potential_energy,
        make_ts_result,
    )

    class _SteppingRelaxer:
        """Nonzero forces so FIRE moves interiors; band energies are not flat.

        Flat equal energies make ASE ``improvedtangent`` emit NaN tangents
        (zero-length tangent when all neighbor ΔE are 0). ``sum(positions)``
        is constant on the Cu3 IDPP path; use ``sum(positions**2)`` instead.
        """

        def __init__(self) -> None:
            self.calls = 0

        def relax_batch(self, atoms_list, steps=0):
            self.calls += 1
            out = []
            for a in atoms_list:
                ra = a.copy()
                forces = np.zeros((len(a), 3))
                forces[0, 0] = 0.5  # enough to move when not converged
                ra.arrays["forces"] = forces
                energy = float(np.sum(a.get_positions() ** 2)) - 0.01 * self.calls
                out.append((energy, ra))
            return out

    relaxer = _SteppingRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=5)
    results = batch.run_optimization(fmax=1e-8, max_steps=2)
    assert results[0]["steps_taken"] == 2
    assert relaxer.calls >= 3  # includes refresh
    for img in neb.images:
        energy = _image_potential_energy(img)
        assert np.isfinite(energy)
        assert img.calc is not None
        assert np.isfinite(float(img.get_potential_energy()))

    result = make_ts_result(
        pair_id="0_1",
        n_images=3,
        spring_constant=0.1,
        use_torchsim=True,
        fmax=1e-8,
        neb_steps=2,
        interpolation_method="idpp",
        climb=False,
        align_endpoints=True,
        perturb_sigma=0.0,
        neb_interpolation_mic=False,
        neb_tangent_method="improvedtangent",
        use_parallel_neb=True,
        reactant_energy=0.0,
        product_energy=0.0,
    )
    result["neb_converged"] = False
    _finalize_neb_result(result, neb.images)
    assert result["barrier_height"] is not None
    assert result.get("error") != 'The property "energy" is not available.'


def test_parallel_neb_require_forces_raises_when_missing(cu3_triangle, cu3_linear):
    """Missing forces from relax_batch must raise (require_forces=True)."""
    from scgo.exceptions import SCGORuntimeError

    class EnergyOnlyRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            return [(0.0, a.copy()) for a in atoms_list]

    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    relaxer = EnergyOnlyRelaxer()
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=1)
    with pytest.raises(SCGORuntimeError, match="did not return forces"):
        batch.run_optimization(fmax=1.0, max_steps=1)


def test_prepare_neb_endpoints_attaches_slab_fixatoms():
    """Shared prep copies endpoints and attaches FixAtoms from surface_config."""
    from ase import Atoms
    from ase.build import fcc111

    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Cu", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max() + 2.0)
    atoms_a = slab.copy() + Atoms("Cu2", positions=[[1.0, 1.0, z0], [3.0, 1.0, z0]])
    atoms_b = slab.copy() + Atoms("Cu2", positions=[[1.5, 1.5, z0], [3.5, 1.5, z0]])
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=True,
    )
    neb_cfg = _gas_neb_cfg(
        system_type="surface_cluster",
        surface_config=cfg,
        n_slab=n_slab,
        neb_align_endpoints=False,
    )
    react, prod = prepare_neb_endpoints(atoms_a, atoms_b, neb_cfg)
    assert any(isinstance(c, FixAtoms) for c in react.constraints)
    assert any(isinstance(c, FixAtoms) for c in prod.constraints)
    assert react is not atoms_a
    assert prod is not atoms_b


def test_run_parallel_neb_search_skips_invalid_pair(tmp_path, cu3_triangle, cu3_linear):
    """Validation failure on one pair skips it without aborting the batch."""
    from unittest.mock import patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    with (
        patch(
            "scgo.ts_search.parallel_neb.prepare_neb_endpoints",
            side_effect=ValueError("bad structure"),
        ),
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_CountingFakeRelaxer(),
        ),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(),
            run_dir=tmp_path,
            rng=None,
        )

    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert "bad structure" in str(results[0].get("error", ""))


def test_run_parallel_neb_preserves_batch_oom_error(tmp_path, cu3_triangle, cu3_linear):
    """Batch OOM (no steps) must keep the real error, not endpoint-as-TS."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]
    # Tagged for the Kaggle runner's real-vs-simulated OOM log scan.
    oom = "CUDA out of memory [scgo-simulated-failure]. Tried to allocate 6.43 GiB."

    class _OomBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = neb_instances

        def run_optimization(self, fmax=0.05, max_steps=100):
            return [
                {
                    "converged": False,
                    "final_fmax": None,
                    "steps_taken": 0,
                    "error": oom,
                }
                for _ in self.neb_instances
            ]

    with (
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_CountingFakeRelaxer(),
        ),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _OomBatch),
        patch(
            "scgo.ts_search.parallel_neb._finalize_neb_result",
            MagicMock(side_effect=AssertionError("finalize must be skipped")),
        ) as finalize_mock,
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(),
            run_dir=tmp_path,
            rng=None,
        )

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["neb_converged"] is False
    assert oom in str(results[0].get("error", ""))
    # Real OOM/error text must remain; do not overwrite with endpoint-saddle message.
    err = str(results[0].get("error", "")).lower()
    assert "no interior saddle" not in err
    assert "endpoint as ts" not in err
    finalize_mock.assert_not_called()


def test_parallel_endpoint_max_idpp_uses_single_stage_climb(
    tmp_path, cu3_triangle, cu3_linear
):
    """Endpoint-max IDPP must climb from step 0 (no no-climb pre-relax)."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]
    calls: list[dict] = []

    class _RecordingBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)
            self.climb_flags = [bool(n.climb) for n in self.neb_instances]

        def run_optimization(self, fmax=0.05, max_steps=100):
            calls.append(
                {
                    "n": len(self.neb_instances),
                    "climb": list(self.climb_flags),
                    "max_steps": int(max_steps),
                }
            )
            return [
                {
                    "converged": True,
                    "final_fmax": 0.01,
                    "steps_taken": 5,
                    "error": None,
                }
                for _ in self.neb_instances
            ]

    # Flat / endpoint-max IDPP energies → single-stage climb.
    # Endpoint SP energies must match minima tuple energies (drift gate).
    flat_band = [0.0, 0.1, 0.2, 0.3, 1.0]

    with (
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_CountingFakeRelaxer(),
        ),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _RecordingBatch),
        patch(
            "scgo.ts_search.parallel_neb.evaluate_neb_image_energies",
            return_value=flat_band,
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(
                system_type="gas_cluster_adsorbate",
                neb_steps=20,
                neb_climb=True,
                max_endpoint_mismatch=1.25,
            ),
            run_dir=tmp_path,
            rng=None,
        )

    assert len(results) == 1
    assert len(calls) == 1, f"expected single-stage only, got {calls!r}"
    assert calls[0]["climb"] == [True]
    assert calls[0]["max_steps"] == 20


def test_parallel_two_stage_climb_runs_after_stage1_converges(
    tmp_path, cu3_triangle, cu3_linear
):
    """Interior-max IDPP still uses two-stage; stage-1 fmax hit must climb after."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]
    calls: list[dict] = []

    class _RecordingBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)
            self.climb_flags = [bool(n.climb) for n in self.neb_instances]

        def run_optimization(self, fmax=0.05, max_steps=100):
            calls.append(
                {
                    "n": len(self.neb_instances),
                    "climb": list(self.climb_flags),
                    "max_steps": int(max_steps),
                }
            )
            return [
                {
                    "converged": True,
                    "final_fmax": 0.01,
                    "steps_taken": 5 if len(calls) == 1 else 3,
                    "error": None,
                }
                for _ in self.neb_instances
            ]

    # Endpoint SP energies must match minima tuple energies (drift gate).
    # Interior max must clear prominence (>=0.40) and two-stage barrier (>=1.0).
    interior_max_band = [0.0, 0.5, 1.6, 0.4, 1.0]

    with (
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_CountingFakeRelaxer(),
        ),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _RecordingBatch),
        patch(
            "scgo.ts_search.parallel_neb.evaluate_neb_image_energies",
            return_value=interior_max_band,
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(
                system_type="gas_cluster_adsorbate",
                neb_steps=20,
                neb_climb=True,
                max_endpoint_mismatch=1.25,
            ),
            run_dir=tmp_path,
            rng=None,
        )

    assert len(results) == 1
    assert len(calls) == 2, f"expected stage1+climb, got {calls!r}"
    assert calls[0]["climb"] == [False]
    assert calls[1]["climb"] == [True]
    assert calls[1]["max_steps"] == 15


def test_run_parallel_neb_batches_single_screen_relax_batch(
    tmp_path, cu3_triangle, cu3_linear
):
    """P3: all candidate bands are fused into ONE relax_batch for the screen.

    With multiple pairs and ``max_endpoint_mismatch`` gating on, the per-pair
    energy evals are concatenated and relaxed in a single ``relax_batch(steps=0)``
    call instead of O(n_pairs) tiny launches.
    """
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search
    from scgo.ts_search.transition_state import interpolate_path

    # Three pairs -> three NEB bands in the screen.
    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (1, 2), (2, 3)]
    # interpolate_path yields more images than neb_n_images (endpoints included).
    images_per_band = len(
        interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="linear")
    )
    total_screen_images = len(pairs) * images_per_band

    screen_relaxer = _CountingFakeRelaxer()

    with (
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_path",
            MagicMock(),
        ),
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_energy_profile",
            MagicMock(),
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(
                neb_n_images=3,
                max_endpoint_mismatch=1.25,
            ),
            run_dir=tmp_path,
            rng=None,
            relaxer=screen_relaxer,
        )

    # The screen is the FIRST relax_batch and must cover every image of every
    # band in one launch. Before P3 this was O(n_pairs) calls of images_per_band.
    # Later entries in batch_sizes are the NEB optimization/PES-refresh calls
    # (deduplicated across bands), which this test does not constrain.
    assert screen_relaxer.batch_sizes[0] == total_screen_images, (
        f"expected the fused screen relax_batch of {total_screen_images} images "
        f"to be the first call, got {screen_relaxer.batch_sizes}"
    )

    # Cross-check against the same run with the energy screen disabled: the
    # screen SP attaches forces that step 0 reuses, so enabling the screen does
    # not add a net extra relax_batch (screen replaces the step-0 eval).
    no_screen_relaxer = _CountingFakeRelaxer()
    with (
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_path",
            MagicMock(),
        ),
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_energy_profile",
            MagicMock(),
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3, max_endpoint_mismatch=None),
            run_dir=tmp_path,
            rng=None,
            relaxer=no_screen_relaxer,
        )

    assert screen_relaxer.calls == no_screen_relaxer.calls, (
        f"screen SP should be reused at step 0 (no net extra call): "
        f"{screen_relaxer.batch_sizes} vs {no_screen_relaxer.batch_sizes}"
    )


def test_run_parallel_neb_screen_skip_records_correct_pair_indices(
    tmp_path, cu3_triangle, cu3_linear
):
    """Energy-screen skips must record their own pair indices, not the last pair's.

    The batched screen slices energies in a second loop; provenance for a pair
    dropped there must still come from that pair's own (i, j).
    """
    from unittest.mock import MagicMock, patch

    from scgo.exceptions import SCGOValidationError
    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (2, 3)]

    # Fail the energy profile only for the FIRST pair (0, 1).
    calls = {"n": 0}

    def _profile(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SCGOValidationError("synthetic energy-profile rejection")

    with (
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_path",
            MagicMock(),
        ),
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_energy_profile",
            _profile,
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3, max_endpoint_mismatch=1.25),
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
        )

    skipped = results[0]
    assert skipped["status"] == "skipped"
    assert skipped["pair_id"] == "0_1"
    # Provenance must reference minima 0 and 1. With the stale-index bug the
    # second loop reused (i, j) leaked from the setup loop (the last pair, 2/3).
    assert skipped["minima_indices"] == [0, 1], skipped["minima_indices"]
    assert skipped["reactant_energy"] == 0.0
    assert skipped["product_energy"] == 1.0


def test_run_parallel_neb_reuses_injected_relaxer(tmp_path, cu3_triangle, cu3_linear):
    """P4: an injected relaxer is reused; no second TorchSimBatchRelaxer built."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    with (
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_path",
            MagicMock(),
        ),
        patch(
            "scgo.ts_search.parallel_neb.validate_initial_neb_energy_profile",
            MagicMock(),
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            MagicMock(),
        ) as relaxer_ctor,
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(
                neb_n_images=3,
                max_endpoint_mismatch=None,
            ),
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
        )

    assert len(results) == 1
    # The injected relaxer means the function must NOT construct its own.
    relaxer_ctor.assert_not_called()


# ---------------------------------------------------------------------------
# B1: the optimizer step must reuse the already-batched NEB forces
# ---------------------------------------------------------------------------


class _SteppingCountingRelaxer(_CountingFakeRelaxer):
    """Counting relaxer with nonzero PES forces so bands keep stepping.

    ``_CountingFakeRelaxer`` returns zero forces, which converges (or NaNs) a
    band immediately; these tests need several real optimizer steps.
    """

    def relax_batch(self, atoms_list, steps=0):
        self.calls += 1
        self.batch_sizes.append(len(atoms_list))
        results = []
        for a in atoms_list:
            ra = a.copy()
            forces = np.zeros((len(a), 3))
            forces[0, 0] = 0.5  # enough to keep FIRE moving
            ra.arrays["forces"] = forces
            energy = float(np.sum(a.get_positions() ** 2)) - 0.01 * self.calls
            results.append((energy, ra))
        return results


def test_parallel_neb_step_reuses_batched_forces(cu3_triangle, cu3_linear):
    """B1: stepping must not dispatch an extra unbatched ``relax_batch``.

    ``optimizer.step()`` is called with no forces argument (ASE 3.28 removes
    ``Optimizer.step(f)``), so FIRE recomputes the gradient via
    ``NEBOptimizable.get_gradient`` -> ``neb.get_forces()``. That is only safe
    because every image still carries the SinglePoint results the batch runner
    just attached, so ``TorchSimNEB.get_forces`` takes its cached-forces fast
    path instead of re-entering TorchSim per band.
    """
    relaxer = _SteppingCountingRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=3)
    results = batch.run_optimization(fmax=1e-8, max_steps=2)

    assert results[0]["steps_taken"] == 2
    # step 0 (all images) + step 1 (interiors) + post-loop refresh = 3 batches.
    # Any per-band re-entry into TorchSim would push this higher.
    assert relaxer.calls == 3, relaxer.batch_sizes


def test_parallel_neb_never_passes_forces_to_step(cu3_triangle, cu3_linear):
    """``optimizer.step`` is always called with no arguments (ASE 3.28 ready)."""

    class _RecordingOptimizer:
        instances: list[_RecordingOptimizer] = []

        def __init__(self, neb, logfile=None, trajectory=None):
            self.neb = neb
            self.step_args: list[tuple] = []
            _RecordingOptimizer.instances.append(self)

        def step(self, *args):
            self.step_args.append(args)

    _RecordingOptimizer.instances.clear()
    relaxer = _SteppingCountingRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch(
        [neb], relaxer, max_total_steps=1, optimizer=_RecordingOptimizer
    )
    batch.run_optimization(fmax=1e-8, max_steps=1)
    assert _RecordingOptimizer.instances
    assert _RecordingOptimizer.instances[0].step_args == [()]


# ---------------------------------------------------------------------------
# B2: force_calls == number of batched evaluations the band took part in
# ---------------------------------------------------------------------------


def test_parallel_neb_force_calls_match_batch_participations(cu3_triangle, cu3_linear):
    """B2: each band counts exactly one force call per batch it participates in."""
    relaxer = _SteppingCountingRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=5)
    results = batch.run_optimization(fmax=1e-8, max_steps=3)

    steps_taken = int(results[0]["steps_taken"])
    assert steps_taken == 3
    # One batched relax_batch per optimization step (the post-loop PES refresh is
    # not an optimization evaluation and is not counted).
    assert neb.get_force_calls() == steps_taken


def test_parallel_neb_does_not_double_count_force_calls(cu3_triangle, cu3_linear):
    """B2: the batch runner suppresses TorchSimNEB's own force_calls increment."""
    relaxer = _CountingFakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    assert neb._force_calls_counted_externally is False

    ParallelNEBBatch([neb], relaxer, max_total_steps=1)
    assert neb._force_calls_counted_externally is True

    # A cache miss inside get_forces must no longer bump the counter: the batch
    # runner owns it. (The serial fallback keeps the flag False and self-counts.)
    for img in neb.images:
        img.calc = None
        img.arrays.pop("forces", None)
    before = neb.get_force_calls()
    neb.get_forces()
    assert neb.get_force_calls() == before


def test_serial_torchsim_neb_still_counts_its_own_force_calls(cu3_triangle, cu3_linear):
    """The serial fallback keeps owning force_calls (no ParallelNEBBatch)."""
    relaxer = _CountingFakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    assert neb.get_force_calls() == 0
    neb.get_forces()
    assert neb.get_force_calls() == 1


# ---------------------------------------------------------------------------
# B6: dedup key must distinguish cells / pbc
# ---------------------------------------------------------------------------


def test_neb_dedup_key_distinguishes_cell_and_pbc(cu3_triangle):
    """B6: identical positions in different cells must not collide.

    Surface bands enable ``neb_surface_cell_remap`` / lattice rotation, which
    genuinely produce the same Cartesian positions in different cells. Without
    cell+pbc in the key, one image would receive the other's energy/forces.
    """
    a = cu3_triangle.copy()
    b = cu3_triangle.copy()
    assert _neb_image_dedup_key(a) == _neb_image_dedup_key(b)

    b.set_cell([20.0, 20.0, 20.0])
    assert _neb_image_dedup_key(a) != _neb_image_dedup_key(b)

    c = cu3_triangle.copy()
    c.set_pbc([True, True, True])
    assert _neb_image_dedup_key(a) != _neb_image_dedup_key(c)


def test_parallel_neb_does_not_dedup_across_different_cells(cu3_triangle, cu3_linear):
    """B6 end-to-end: same-position bands in different cells stay distinct."""
    relaxer = _CountingFakeRelaxer()
    images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    images2 = [img.copy() for img in images1]
    for img in images2:
        img.set_cell([30.0, 30.0, 30.0])
        img.set_pbc([True, True, True])

    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=1)
    batch.run_optimization(fmax=1.0, max_steps=1)

    # 5 + 5 distinct images: positions match but cells do not, so no collapse.
    assert relaxer.batch_sizes[0] == 10, relaxer.batch_sizes


# ---------------------------------------------------------------------------
# T5: atom-budget chunking
# ---------------------------------------------------------------------------


def test_chunk_band_indices_by_atom_budget_splits_as_expected():
    """Greedy bin-packing preserves order and respects the atom budget."""
    from scgo.ts_search.parallel_neb import chunk_band_indices_by_atom_budget

    costs = [1000, 1000, 1000, 1000, 1000]
    indices = list(range(5))

    assert chunk_band_indices_by_atom_budget(indices, costs, 2000) == [
        [0, 1],
        [2, 3],
        [4],
    ]
    assert chunk_band_indices_by_atom_budget(indices, costs, 3000) == [
        [0, 1, 2],
        [3, 4],
    ]
    # Budget >= total -> a single chunk.
    assert chunk_band_indices_by_atom_budget(indices, costs, 5000) == [indices]
    # No budget -> a single chunk.
    assert chunk_band_indices_by_atom_budget(indices, costs, None) == [indices]
    assert chunk_band_indices_by_atom_budget(indices, costs, 0) == [indices]
    assert chunk_band_indices_by_atom_budget([], costs, 1000) == []


def test_chunk_band_indices_by_atom_budget_keeps_oversized_band():
    """A band larger than the whole budget still gets its own chunk."""
    from scgo.ts_search.parallel_neb import chunk_band_indices_by_atom_budget

    costs = [500, 9000, 500]
    assert chunk_band_indices_by_atom_budget([0, 1, 2], costs, 1000) == [
        [0],
        [1],
        [2],
    ]


def _record_chunk_sizes(tmp_path, minima, pairs, neb_cfg, **run_kwargs):
    """Run ``run_parallel_neb_search`` with a stub batch; return chunk sizes."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    sizes: list[int] = []

    class _RecordingBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)
            sizes.append(len(self.neb_instances))

        def run_optimization(self, fmax=0.05, max_steps=100):
            return [
                {
                    "converged": True,
                    "final_fmax": 0.01,
                    "steps_taken": 1,
                    "error": None,
                }
                for _ in self.neb_instances
            ]

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _RecordingBatch),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=neb_cfg,
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
            **run_kwargs,
        )
    return sizes


def test_run_parallel_neb_chunks_by_atom_budget(tmp_path, cu3_triangle, cu3_linear):
    """T5: with no band cap, bands are binned by the atom budget."""
    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (1, 2), (2, 3)]
    # Cu3 with n_images=3 -> interpolate_path yields 5 images x 3 atoms = 15
    # atoms per band. A 30-atom budget therefore fits exactly two bands.
    sizes = _record_chunk_sizes(
        tmp_path,
        minima,
        pairs,
        _gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=30),
    )
    assert sizes == [2, 1], sizes


def test_run_parallel_neb_atom_budget_single_chunk_when_none(
    tmp_path, cu3_triangle, cu3_linear
):
    """No band cap and no atom budget -> all bands share one force batch."""
    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (1, 2), (2, 3)]
    sizes = _record_chunk_sizes(
        tmp_path,
        minima,
        pairs,
        _gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=None),
    )
    assert sizes == [3], sizes


def test_run_parallel_neb_max_bands_overrides_atom_budget(
    tmp_path, cu3_triangle, cu3_linear
):
    """T5: an explicit ``parallel_neb_max_bands`` wins over the atom budget."""
    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (1, 2), (2, 3)]
    # A generous atom budget would put all 3 bands in one chunk; max_bands=1
    # must still force one band per batch.
    sizes = _record_chunk_sizes(
        tmp_path,
        minima,
        pairs,
        _gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=100000),
        parallel_neb_max_bands=1,
    )
    assert sizes == [1, 1, 1], sizes


# ---------------------------------------------------------------------------
# B4: non-finite forces must not reach finalize
# ---------------------------------------------------------------------------


def test_run_parallel_neb_marks_nonfinite_band_failed(
    tmp_path, cu3_triangle, cu3_linear
):
    """B4: a band whose forces went non-finite must fail, skipping finalize."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    class _NanBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)
            # Non-finite bands still take steps and log force calls, so the old
            # ``batch_never_ran`` (force_calls == 0) guard did not catch them.
            for neb in self.neb_instances:
                neb._force_calls += 1

        def run_optimization(self, fmax=0.05, max_steps=100):
            return [
                {
                    "converged": False,
                    "final_fmax": float("nan"),
                    "steps_taken": 1,
                    "error": (
                        "NEB forces are non-finite (fmax=nan); refusing optimizer "
                        "step [scgo-simulated-failure]"
                    ),
                }
                for _ in self.neb_instances
            ]

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _NanBatch),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb._finalize_neb_result",
            MagicMock(side_effect=AssertionError("finalize must be skipped")),
        ) as finalize_mock,
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3),
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
        )

    assert results[0]["status"] == "failed"
    assert results[0]["neb_converged"] is False
    assert "non-finite" in str(results[0].get("error", ""))
    finalize_mock.assert_not_called()


# ---------------------------------------------------------------------------
# B5: one OOM retry per chunk (driving the real ParallelNEBBatch)
# ---------------------------------------------------------------------------
#
# These tests deliberately do *not* stub ``ParallelNEBBatch.run_optimization``.
# The retry wrapper only ever sees an exception if the real
# ``run_optimization`` re-raises CUDA OOM out of ``relaxer.relax_batch`` — a
# fake that raises from ``run_optimization`` would pass even while the
# production path swallowed the OOM and silently produced zero saddles (which
# is exactly the regression this suite exists to catch). So the fault is
# injected at the ``relax_batch`` boundary and the real class runs.

# Tagged so the Kaggle GPU runner's log scan can tell simulated OOM apart from a
# real one (see .github/scripts/kaggle_gpu_runner.template.py).
SIMULATED_OOM = (
    "CUDA out of memory [scgo-simulated-failure]. Tried to allocate 6.43 GiB."
)


class _OomRelaxer(_CountingFakeRelaxer):
    """Fake relaxer whose ``relax_batch`` raises CUDA OOM on the first N calls."""

    def __init__(self, *, fail_first: int | None = None) -> None:
        super().__init__()
        self.fail_first = fail_first  # None -> always fail
        self.failures = 0

    def relax_batch(self, atoms_list, steps=0):
        if self.fail_first is None or self.failures < self.fail_first:
            self.failures += 1
            raise RuntimeError(SIMULATED_OOM)
        return super().relax_batch(atoms_list, steps=steps)


def _recording_batch_cls(attempts: list[int]):
    """Subclass of the real ``ParallelNEBBatch`` that records chunk sizes."""

    class _RecordingBatch(ParallelNEBBatch):
        def __init__(self, neb_instances, *args, **kwargs):
            attempts.append(len(neb_instances))
            super().__init__(neb_instances, *args, **kwargs)

    return _RecordingBatch


def test_run_optimization_reraises_cuda_oom(cu3_triangle, cu3_linear):
    """T3: CUDA OOM must escape ``run_optimization`` instead of failing bands.

    Swallowing it here is what made ``_run_chunk_with_oom_retry`` dead code.
    """
    relaxer = _OomRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=2)

    with pytest.raises(RuntimeError, match="out of memory"):
        batch.run_optimization(fmax=0.05, max_steps=2)

    # The band must not have been quietly marked failed on the way out.
    assert batch.failed_nebs == {}


def test_run_optimization_still_fails_bands_on_non_oom_error(cu3_triangle, cu3_linear):
    """Non-OOM ``relax_batch`` failures stay contained (bad input, not GPU pressure)."""

    class _BrokenRelaxer(_CountingFakeRelaxer):
        def relax_batch(self, atoms_list, steps=0):
            raise RuntimeError("model weights are corrupt [scgo-simulated-failure]")

    relaxer = _BrokenRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=2)

    results = batch.run_optimization(fmax=0.05, max_steps=2)
    assert "model weights are corrupt" in str(results[0]["error"])
    assert batch.failed_nebs


def test_run_parallel_neb_retries_chunk_once_on_cuda_oom(
    tmp_path, cu3_triangle, cu3_linear
):
    """B5: a CUDA-OOM chunk is re-binned smaller and recovers, via the real class."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [
        (0.0, cu3_triangle),
        (1.0, cu3_linear),
        (2.0, cu3_triangle),
        (3.0, cu3_linear),
    ]
    pairs = [(0, 1), (1, 2), (2, 3)]
    attempts: list[int] = []
    relaxer = _OomRelaxer(fail_first=1)

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb.ParallelNEBBatch",
            _recording_batch_cls(attempts),
        ),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=None),
            run_dir=tmp_path,
            rng=None,
            relaxer=relaxer,
        )

    # First attempt: one 3-band chunk (OOM). Retry: re-binned to smaller chunks.
    assert attempts[0] == 3
    assert len(attempts) > 1, attempts
    assert all(n < 3 for n in attempts[1:]), attempts
    assert relaxer.failures == 1
    # Every band recovered, so none is marked failed.
    assert all(r.get("error") is None for r in results), [
        r.get("error") for r in results
    ]


def test_run_parallel_neb_fails_bands_when_oom_retry_also_fails(
    tmp_path, cu3_triangle, cu3_linear
):
    """B5: a persistently OOM chunk marks its bands failed instead of raising."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear), (2.0, cu3_triangle)]
    pairs = [(0, 1), (1, 2)]

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb._finalize_neb_result",
            MagicMock(side_effect=AssertionError("finalize must be skipped")),
        ) as finalize_mock,
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=None),
            run_dir=tmp_path,
            rng=None,
            relaxer=_OomRelaxer(),
        )

    for result in results:
        assert result["status"] == "failed"
        assert SIMULATED_OOM in str(result.get("error", ""))
        assert not result.get("steps_taken")
    finalize_mock.assert_not_called()


def test_run_parallel_neb_does_not_retry_non_oom_relax_batch_errors(
    tmp_path, cu3_triangle, cu3_linear
):
    """A non-OOM ``relax_batch`` failure fails the bands without re-binning."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear), (2.0, cu3_triangle)]
    pairs = [(0, 1), (1, 2)]
    attempts: list[int] = []

    class _BrokenRelaxer(_CountingFakeRelaxer):
        def relax_batch(self, atoms_list, steps=0):
            raise RuntimeError("model weights are corrupt [scgo-simulated-failure]")

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb.ParallelNEBBatch",
            _recording_batch_cls(attempts),
        ),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
        patch(
            "scgo.ts_search.parallel_neb._finalize_neb_result",
            MagicMock(side_effect=AssertionError("finalize must be skipped")),
        ),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3, parallel_neb_max_batch_atoms=None),
            run_dir=tmp_path,
            rng=None,
            relaxer=_BrokenRelaxer(),
        )

    # Exactly one chunk attempt: no half-budget re-binning for non-OOM errors.
    assert attempts == [2], attempts
    for result in results:
        assert result["status"] == "failed"
        assert "model weights are corrupt" in str(result.get("error", ""))


def test_run_parallel_neb_propagates_non_oom_error_from_run_optimization(
    tmp_path, cu3_triangle, cu3_linear
):
    """Wrapper contract: only CUDA OOM is retried; anything else propagates."""
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    class _BrokenBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)

        def run_optimization(self, fmax=0.05, max_steps=100):
            raise RuntimeError("optimizer exploded")

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _BrokenBatch),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
        pytest.raises(RuntimeError, match="optimizer exploded"),
    ):
        run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3),
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
        )


# ---------------------------------------------------------------------------
# B7: per-pair timing keys are chunk averages
# ---------------------------------------------------------------------------


def test_run_parallel_neb_reports_avg_timing_keys(tmp_path, cu3_triangle, cu3_linear):
    """B7: per-pair timings are labelled ``*_avg_s`` (chunk time / n pairs).

    The old ``neb_optimization_s`` alias is gone; the run-level rollup in
    :func:`scgo.utils.timing_report.sum_neb_seconds_from_ts_results` reads the
    ``*_avg_s`` key directly.
    """
    from unittest.mock import MagicMock, patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search
    from scgo.utils.timing_report import sum_neb_seconds_from_ts_results

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    class _FastBatch:
        def __init__(self, neb_instances, *args, **kwargs):
            self.neb_instances = list(neb_instances)

        def run_optimization(self, fmax=0.05, max_steps=100):
            return [
                {
                    "converged": True,
                    "final_fmax": 0.01,
                    "steps_taken": 1,
                    "error": None,
                }
                for _ in self.neb_instances
            ]

    with (
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path", MagicMock()),
        patch("scgo.ts_search.parallel_neb.ParallelNEBBatch", _FastBatch),
        patch("scgo.ts_search.parallel_neb._finalize_neb_result", MagicMock()),
        patch("scgo.ts_search.parallel_neb.save_neb_result", MagicMock()),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=_gas_neb_cfg(neb_n_images=3),
            run_dir=tmp_path,
            rng=None,
            relaxer=_CountingFakeRelaxer(),
        )

    timings = results[0]["timings_s"]
    for key in ("total_wall_avg_s", "neb_optimization_avg_s", "cpu_non_relax_avg_s"):
        assert key in timings, timings
        assert timings[key] >= 0.0
    # The back-compat alias is removed.
    assert "neb_optimization_s" not in timings
    assert sum_neb_seconds_from_ts_results(results) == pytest.approx(
        timings["neb_optimization_avg_s"]
    )
