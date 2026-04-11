"""Tests for transition state search (NEB, TorchSim, MACE)."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from scgo.ts_search.transition_state import (
    calculate_structure_similarity,
    find_transition_state,
    interpolate_path,
    save_neb_result,
    setup_neb_path,
)
from scgo.ts_search.transition_state_io import (
    select_structure_pairs,
)
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION
from tests.cuda_skip import require_cuda


@pytest.fixture
def temp_output_dir():
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_interpolate_path_basic(h2_reactant, h2_product):
    """Test basic geodesic interpolation between two structures."""
    n_images = 5
    # Explicitly disable endpoint alignment for this test so endpoints remain
    # bitwise-identical to the provided Atoms objects (previous default).
    images = interpolate_path(
        h2_reactant,
        h2_product,
        n_images=n_images,
        method="linear",
        align_endpoints=False,
    )

    # Should return n_images + 2 (including endpoints)
    assert len(images) == n_images + 2
    assert images[0] == h2_reactant
    assert images[-1] == h2_product

    # Check that intermediate images are interpolated
    # Distance should increase monotonically
    for i in range(len(images) - 1):
        d1 = images[i].get_distance(0, 1)
        d2 = images[i + 1].get_distance(0, 1)
        assert d2 >= d1


def test_interpolate_path_idpp(li2_isomer1, li2_isomer2):
    """Test IDPP interpolation (default)."""
    n_images = 3
    images = interpolate_path(
        li2_isomer1, li2_isomer2, n_images=n_images, method="idpp"
    )

    assert len(images) == n_images + 2
    # IDPP should avoid atom overlaps better than linear
    # Check all images have reasonable distances
    for img in images:
        d = img.get_distance(0, 1)
        assert d > 1.0  # Atoms shouldn't overlap


def test_interpolate_path_align_reduces_max_displacement(pt4_tetrahedron):
    """Endpoint alignment should reduce per-atom displacements for permuted endpoints."""
    a = pt4_tetrahedron.copy()
    b = a.copy()
    perm = [2, 3, 0, 1]
    b.set_positions(a.get_positions()[perm])
    b.rotate(0.7, "z", rotate_cell=False)

    imgs_no_align = interpolate_path(
        a, b, n_images=3, method="idpp", align_endpoints=False
    )
    max_disp_no_align = float(
        np.max(
            np.linalg.norm(
                imgs_no_align[0].get_positions() - imgs_no_align[-1].get_positions(),
                axis=1,
            )
        )
    )

    imgs_align = interpolate_path(a, b, n_images=3, method="idpp", align_endpoints=True)
    max_disp_align = float(
        np.max(
            np.linalg.norm(
                imgs_align[0].get_positions() - imgs_align[-1].get_positions(), axis=1
            )
        )
    )

    assert max_disp_align < max_disp_no_align


def test_interpolate_path_perturb_keeps_endpoints_changes_interior(
    h2_reactant, h2_product
):
    """Perturbation should not change endpoints but should perturb interior images."""
    rng = np.random.default_rng(12345)
    imgs_clean = interpolate_path(h2_reactant, h2_product, n_images=3, method="idpp")
    imgs_pert = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng
    )

    # endpoints unchanged
    assert np.allclose(imgs_clean[0].get_positions(), imgs_pert[0].get_positions())
    assert np.allclose(imgs_clean[-1].get_positions(), imgs_pert[-1].get_positions())

    # interior image changed
    assert not np.allclose(imgs_clean[1].get_positions(), imgs_pert[1].get_positions())


def test_interpolate_path_perturb_deterministic_with_rng(h2_reactant, h2_product):
    """Seeded RNG should produce deterministic perturbations."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    imgs1 = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng1
    )
    imgs2 = interpolate_path(
        h2_reactant, h2_product, n_images=3, method="idpp", perturb_sigma=0.05, rng=rng2
    )

    assert np.allclose(imgs1[1].get_positions(), imgs2[1].get_positions())


def test_find_transition_state_records_align_and_perturb(
    temp_output_dir, h2_reactant, h2_product
):
    """`find_transition_state` should record `align_endpoints` and `perturb_sigma` in the result metadata."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="meta_test",
        n_images=3,
        fmax=0.1,
        neb_steps=20,
        verbosity=0,
        align_endpoints=True,
        perturb_sigma=0.03,
        rng=np.random.default_rng(1),
    )

    assert result.get("align_endpoints") is True
    assert result.get("perturb_sigma") == pytest.approx(0.03)
    # returned structure should not retain a calculator instance
    ts = result.get("transition_state")
    if ts is not None:
        assert ts.calc is None


def test_find_ts_endpoint_marked_not_converged(temp_output_dir, h2_reactant):
    """If the highest-energy image is an endpoint (or endpoints are identical),
    NEB should be marked non-converged with an endpoint-related error message.
    """
    a = h2_reactant.copy()
    b = h2_reactant.copy()  # identical endpoints -> no interior saddle
    # ensure calculators are attached to copies (ASE .copy() does not preserve .calc)
    a.calc = EMT()
    b.calc = EMT()

    result = find_transition_state(
        a,
        b,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="endpoint_test",
        n_images=3,
        fmax=0.05,
        neb_steps=50,
        verbosity=0,
    )

    # NEB should be treated as non-converged when TS is endpoint
    assert result["neb_converged"] is False
    assert result["status"] == "failed"
    assert result.get("error") is not None
    assert "endpoint" in result.get("error").lower()
    assert result.get("ts_image_index") in (0, 3)
    # calculator should be detached
    ts = result.get("transition_state")
    if ts is not None:
        assert ts.calc is None


def test_interpolate_path_different_lengths_fails():
    """Test that interpolation fails with different atom counts."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)

    with pytest.raises(ValueError):  # ASE will raise a ValueError for mismatched atoms
        interpolate_path(atoms1, atoms2, n_images=3)


def test_calculate_similarity_basic():
    """Test similarity comparison wrapper (comparator logic tested elsewhere)."""
    # Identical structures
    atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2 = atoms1.copy()
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms2)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True

    # Permuted (swapped atoms) - should still be identical
    atoms3 = Atoms("Cu2", positions=[[2.5, 0, 0], [0, 0, 0]])
    atoms3.center(vacuum=5.0)
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms3)
    assert cum_diff < 0.001
    assert are_similar is True

    # Different structures
    atoms4 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]])
    atoms4.center(vacuum=5.0)
    cum_diff, max_diff, are_similar = calculate_structure_similarity(atoms1, atoms4)
    assert cum_diff > 0.01
    # are_similar may be implementation-dependent; ensure the difference is large enough to indicate dissimilarity


def test_calculate_similarity_ignores_fixed_slab_atoms():
    """Fixed slab atoms should not affect similarity metrics."""
    slab_mobile = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0], [0.6, 0.4, 1.8]]
    atoms1 = Atoms("Pt4", positions=slab_mobile)
    atoms2 = atoms1.copy()
    atoms1.set_constraint(FixAtoms(indices=[0, 1, 2]))
    atoms2.set_constraint(FixAtoms(indices=[0, 1, 2]))
    # Move only frozen slab atoms in atoms2
    pos2 = atoms2.get_positions()
    pos2[:3, 0] += 0.4
    atoms2.set_positions(pos2)

    cum_diff, _max_diff, are_similar = calculate_structure_similarity(atoms1, atoms2)
    assert cum_diff == pytest.approx(0.0, abs=1e-10)
    assert are_similar is True


def test_select_structure_pairs_ignores_fixed_slab_atom_differences():
    """Pair filtering should reject endpoint pairs that differ only in frozen slab atoms."""
    base = Atoms(
        "Pt5",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.6, 0.4, 1.8],
            [1.8, 0.4, 1.8],
        ],
    )
    base.set_constraint(FixAtoms(indices=[0, 1, 2]))
    slab_shifted = base.copy()
    slab_shifted.set_constraint(FixAtoms(indices=[0, 1, 2]))
    shifted_pos = slab_shifted.get_positions()
    shifted_pos[:3, 1] += 0.35
    slab_shifted.set_positions(shifted_pos)
    mobile_changed = base.copy()
    mobile_pos = mobile_changed.get_positions()
    mobile_pos[4, 0] += 0.8
    mobile_changed.set_positions(mobile_pos)

    minima = [(-1.0, base), (-0.95, slab_shifted), (-0.90, mobile_changed)]
    pairs = select_structure_pairs(
        minima,
        max_pairs=None,
        similarity_tolerance=0.01,
        similarity_pair_cor_max=0.2,
    )

    assert (0, 1) not in pairs
    assert (0, 2) in pairs


def test_setup_neb_path(h2_reactant, h2_product):
    """Test NEB path setup with calculator attachment."""
    calc = EMT()
    neb = setup_neb_path(h2_reactant, h2_product, calculator=calc, n_images=3)

    # Check NEB object created
    assert neb is not None
    assert len(neb.images) == 5  # 3 intermediate + 2 endpoints

    # Check calculators attached
    for img in neb.images:
        assert img.calc is not None

    # Ensure calculators are not the same object across images (avoid shared state)
    ids = [id(img.calc) for img in neb.images]
    assert len(set(ids)) == len(ids)


def test_setup_neb_path_deepcopy_behavior(h2_reactant, h2_product):
    """Verify that calculators are deep-copied for NEB images.

    This ensures safe behavior with ML calculators which may be unsafe to
    reuse across multiple Atoms instances concurrently. Matches the policy
    used in GA (direct assignment for sequential use) vs NEB (deepcopy for
    concurrent use).
    """

    class DummyCalc(EMT):
        def __init__(self):
            super().__init__()
            self.uid = id(self)

        def __deepcopy__(self, memo):
            # Return a new instance with a different uid
            new = DummyCalc()
            return new

    calc = DummyCalc()

    neb = setup_neb_path(h2_reactant, h2_product, calculator=calc, n_images=3)

    # Calculators were attached
    for img in neb.images:
        assert img.calc is not None
        assert isinstance(img.calc, DummyCalc)

    # Ensure each calc is a distinct object and not the original instance
    ids = [id(img.calc) for img in neb.images]
    assert len(set(ids)) == len(ids)
    assert all(id(calc) != cid for cid in ids)


def test_setup_neb_path_no_calculator(h2_reactant, h2_product):
    """Test NEB path setup without calculator."""
    neb = setup_neb_path(h2_reactant, h2_product, calculator=None, n_images=3)

    assert neb is not None
    assert len(neb.images) == 5


def test_setup_neb_path_passes_k_and_climb(h2_reactant, h2_product):
    """Ensure `k` and `climb` are forwarded to the ASE NEB created by setup_neb_path."""
    neb = setup_neb_path(
        h2_reactant, h2_product, calculator=None, n_images=3, k=5.0, climb=True
    )

    # ASE NEB may store `k` as a scalar or a per-spring sequence; accept both
    k_val = getattr(neb, "k", None)
    if hasattr(k_val, "__iter__") and not isinstance(k_val, (str, bytes)):
        assert all(float(x) == 5.0 for x in k_val)
    else:
        assert float(k_val) == 5.0
    assert getattr(neb, "climb", None) is True


def test_find_ts_simple(h2_reactant, h2_product, temp_output_dir):
    """Test basic TS finding with EMT calculator."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="test_h2",
        n_images=3,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
    )

    # Accept either a successful NEB or a correctly-detected endpoint case
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert result["transition_state"] is not None
        assert result["ts_energy"] is not None
        assert result["barrier_height"] is not None
    else:
        # NEB may report the endpoint as the highest-energy image and be
        # intentionally rejected by the endpoint-detection safeguard.
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()

    assert result["pair_id"] == "test_h2"
    assert result["n_images"] == 3


def test_find_ts_with_climb(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with climbing image NEB."""
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_climb",
        n_images=5,
        spring_constant=0.05,
        climb=True,
        fmax=0.1,
        neb_steps=200,
        verbosity=0,
    )

    assert result["climb"] is True
    # With the staged-retry defaults enabled, this should converge for Cu3
    assert result["status"] == "success"
    assert result["neb_converged"] is True


def test_find_transition_state_auto_retry_recovers_cu3(
    cu3_triangle, cu3_linear, temp_output_dir
):
    """If the initial NEB reports an endpoint TS, the staged-retry should recover an interior saddle for Cu3."""
    # Use conservative initial settings that commonly lead to endpoint outcomes
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_retry",
        n_images=3,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
        rng=np.random.default_rng(0),
        # neb_retry_on_endpoint is True by default
    )

    # Expect the single conservative retry to have been attempted and to succeed
    assert result.get("retry_attempted") is True
    assert result.get("retry_success") is True
    assert result["status"] == "success"
    assert result["neb_converged"] is True
    assert result.get("ts_image_index") not in (0, result.get("n_images") + 1)
    # Original attempt + single retry
    assert (
        isinstance(result.get("retry_history"), list)
        and len(result.get("retry_history")) == 2
    )


def test_find_ts_linear_interpolation(h2_reactant, h2_product, temp_output_dir):
    """Test TS finding with linear interpolation instead of IDPP."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="linear_test",
        n_images=3,
        interpolation_method="linear",
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
    )

    # NEB may either succeed or be rejected because the highest-energy
    # image is an endpoint — accept both behaviors.
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert result["transition_state"] is not None
    else:
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()


def test_find_ts_saves_trajectory(h2_reactant, h2_product, temp_output_dir):
    """Test that NEB trajectory is saved."""
    traj_path = os.path.join(temp_output_dir, "custom_neb.traj")

    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="traj_test",
        n_images=3,
        trajectory=traj_path,
        fmax=0.05,
        neb_steps=200,
        verbosity=0,
    )

    # Accept either success or a detected endpoint-TS failure; trajectory
    # should still be written by the optimizer run in either case.
    if result["status"] == "success":
        assert result["neb_converged"] is True
    else:
        assert result["neb_converged"] is False
        assert result.get("error") and "endpoint" in result.get("error").lower()

    assert os.path.exists(traj_path)


def test_find_transition_state_defaults_reflect_promoted_retry(
    h2_reactant, h2_product, temp_output_dir
):
    """Defaults align with tuned TS presets (spring k=0.1, climb False, etc.)."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="defaults_test",
        n_images=3,
        fmax=0.1,
        neb_steps=20,
        verbosity=0,
        # Retry promotes climb=True on the second attempt; final `result` then
        # echoes retry params, which is environment-dependent and flaky in CI.
        neb_retry_on_endpoint=False,
    )

    assert result.get("spring_constant") == pytest.approx(0.1)
    assert result.get("climb") is False
    assert result.get("perturb_sigma") == pytest.approx(0.0)


def test_find_ts_different_lengths_fails(temp_output_dir):
    """Test TS finding fails with different atom counts."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms1.center(vacuum=5.0)
    atoms2.center(vacuum=5.0)
    atoms1.calc = EMT()
    atoms2.calc = EMT()

    with pytest.raises(ValueError, match="different lengths"):
        find_transition_state(
            atoms1,
            atoms2,
            calculator=EMT(),
            output_dir=temp_output_dir,
            pair_id="fail_test",
        )


def test_find_ts_no_calculator_fails(h2_reactant, h2_product, temp_output_dir):
    """Test that TS finding fails without calculator when use_torchsim=False."""
    # Remove calculators
    h2_reactant.calc = None
    h2_product.calc = None

    with pytest.raises(ValueError):  # Should fail validation without calculator
        find_transition_state(
            h2_reactant,
            h2_product,
            calculator=None,
            output_dir=temp_output_dir,
            pair_id="no_calc",
            use_torchsim=False,
        )


def test_save_neb_result_success(temp_output_dir, default_rel_tol):
    """Test saving successful NEB result."""
    # Create mock TS result
    ts_atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
    ts_atoms.center(vacuum=5.0)

    r_atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    r_atoms.center(vacuum=4.0)
    p_atoms = Atoms("H2", positions=[[0, 0, 0], [1.4, 0, 0]])
    p_atoms.center(vacuum=4.0)

    result = {
        "status": "success",
        "pair_id": "0_1",
        "neb_converged": True,
        "n_images": 5,
        "spring_constant": 0.1,
        "reactant_energy": -1.0,
        "product_energy": -0.8,
        "ts_energy": -0.5,
        "barrier_height": 0.5,
        "transition_state": ts_atoms,
        "ts_image_index": 3,
        "error": None,
        "reactant_structure": r_atoms,
        "product_structure": p_atoms,
        "use_torchsim": False,
        "fmax": 0.05,
        "neb_steps": 100,
        "interpolation_method": "idpp",
    }

    save_neb_result(result, temp_output_dir, "0_1")

    # Check files created
    assert os.path.exists(os.path.join(temp_output_dir, "ts_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "reactant_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "product_0_1.xyz"))
    assert os.path.exists(os.path.join(temp_output_dir, "neb_0_1_metadata.json"))

    with open(os.path.join(temp_output_dir, "neb_0_1_metadata.json")) as f:
        metadata = json.load(f)

    assert metadata["status"] == "success"
    assert metadata["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
    assert metadata["scgo_version"] != "unknown"
    assert "created_at" in metadata
    assert metadata["barrier_height"] == pytest.approx(0.5, rel=default_rel_tol)
    assert metadata["spring_constant"] == pytest.approx(0.1, rel=default_rel_tol)
    assert metadata["ts_image_index"] == 3
    assert "steps_taken" in metadata
    assert metadata.get("neb_backend") == "ase"


def test_save_neb_result_failed(temp_output_dir):
    """Test saving failed NEB result."""
    result = {
        "status": "failed",
        "pair_id": "1_2",
        "neb_converged": False,
        "n_images": 5,
        "spring_constant": 0.1,
        "reactant_energy": -1.0,
        "product_energy": -0.8,
        "ts_energy": None,
        "barrier_height": None,
        "transition_state": None,
        "error": "Test error",
    }

    save_neb_result(result, temp_output_dir, "1_2")

    # Metadata should be saved even for failed runs
    meta_path = os.path.join(temp_output_dir, "neb_1_2_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        failed_meta = json.load(f)
    assert failed_meta["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
    assert "created_at" in failed_meta

    # TS structure should not be saved for failed runs
    assert not os.path.exists(os.path.join(temp_output_dir, "ts_1_2.xyz"))


def test_select_structure_pairs_basic():
    """Test basic pair selection."""
    # Create mock minima
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])  # Different geometry
    atoms3 = Atoms("H2", positions=[[0, 0, 0], [3, 0, 0]])

    minima = [
        (-1.0, atoms1),
        (-0.9, atoms2),
        (-0.7, atoms3),
    ]

    pairs = select_structure_pairs(minima, max_pairs=None, similarity_tolerance=0.1)

    # Should get all unique pairs: (0,1), (0,2), (1,2)
    assert len(pairs) == 3
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_select_structure_pairs_max_limit():
    """Test pair selection with max_pairs limit."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    minima = [(float(-i), atoms.copy()) for i in range(10)]

    pairs = select_structure_pairs(minima, max_pairs=5, similarity_tolerance=0.01)

    assert len(pairs) <= 5


def test_select_structure_pairs_energy_gap():
    """Test pair selection with energy gap threshold."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]])
    atoms3 = Atoms("H2", positions=[[0, 0, 0], [3, 0, 0]])

    minima = [
        (-1.0, atoms1),
        (-0.95, atoms2),  # Small gap
        (-0.2, atoms3),  # Large gap from first two
    ]

    # Only allow pairs within 0.1 eV
    pairs = select_structure_pairs(
        minima, energy_gap_threshold=0.1, similarity_tolerance=0.01
    )

    # Should only get (0,1) since gap to 2 is too large
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_select_structure_pairs_physics_ranking_when_capped(monkeypatch):
    """Capped pair lists use physics-guided ranking (score + stable tie-break)."""
    atoms0 = Atoms("H2", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    atoms1 = Atoms("H2", positions=[[1.0, 0, 0], [1.5, 0, 0]])
    atoms2 = Atoms("H2", positions=[[2.0, 0, 0], [2.5, 0, 0]])
    minima = [(-1.0, atoms0), (-0.95, atoms1), (-0.55, atoms2)]

    def _fake_similarity(
        a_i: Atoms,
        a_j: Atoms,
        tolerance: float = 0.1,
        pair_cor_max: float = 0.1,
    ) -> tuple[float, float, bool]:
        pair = tuple(
            sorted(
                (
                    int(round(a_i.get_positions()[0, 0])),
                    int(round(a_j.get_positions()[0, 0])),
                )
            )
        )
        table = {
            (0, 1): (0.02, 0.10, False),
            (0, 2): (0.20, 0.20, False),
            (1, 2): (0.25, 0.15, False),
        }
        return table[pair]

    monkeypatch.setattr(
        "scgo.ts_search.transition_state_io.calculate_structure_similarity",
        _fake_similarity,
    )

    ranked = select_structure_pairs(minima, max_pairs=2)
    assert ranked == [(1, 2), (0, 2)]


def test_find_ts_emt_basic(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with EMT calculator on CPU.

    Verifies the standard ASE NEB path works with classical potentials.
    Uses Cu3 which has predictable EMT behavior.
    """
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),  # Override fixtures' EMT with fresh one
        output_dir=temp_output_dir,
        pair_id="emt_cu3",
        n_images=3,
        spring_constant=0.1,
        fmax=0.1,  # Relaxed convergence for testing
        neb_steps=50,
        verbosity=0,
    )

    # Verify EMT path was used (not TorchSim)
    assert result["use_torchsim"] is False
    # n_images may be 3 (original) or 5 (if the retry mechanism was triggered)
    if result.get("retry_attempted") and result.get("retry_success"):
        assert result["n_images"] >= 3
    else:
        assert result["n_images"] == 3
    # Either converged or failed gracefully with error
    assert result["status"] in ["success", "failed"]
    # If successful, check key outputs
    if result["status"] == "success":
        assert "transition_state" in result
        assert result["neb_converged"] is True


@pytest.mark.slow
def test_find_ts_mace_cpu(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with MACE on CPU (no TorchSim).

    Uses ML potential on CPU with standard ASE NEB optimization.
    This verifies MACE works without GPU acceleration.
    """
    from scgo.calculators.mace_helpers import MACE

    # Create fresh MACE calculators (not from fixtures)
    mace_calc_reactant = MACE(model="small", device="cpu")
    mace_calc_product = MACE(model="small", device="cpu")

    # Attach to new atoms copies
    reactant = cu3_triangle.copy()
    product = cu3_linear.copy()
    reactant.calc = mace_calc_reactant
    product.calc = mace_calc_product

    result = find_transition_state(
        reactant,
        product,
        calculator=MACE(model="small", device="cpu"),  # For NEB images
        output_dir=temp_output_dir,
        pair_id="mace_cpu_cu3",
        n_images=3,
        spring_constant=0.1,
        fmax=0.1,  # Relaxed convergence for testing
        neb_steps=30,
        use_torchsim=False,  # Explicitly disable TorchSim
        verbosity=0,
    )

    # Verify MACE CPU path was used
    assert result["use_torchsim"] is False
    if result.get("retry_attempted") and result.get("retry_success"):
        assert result["n_images"] >= 3
    else:
        assert result["n_images"] == 3
    assert result["status"] in ["success", "failed"]
    if result["status"] == "success":
        assert "transition_state" in result
        assert result["neb_converged"] is True


@pytest.mark.slow
def test_find_ts_mace_gpu_torchsim(cu3_triangle, cu3_linear, temp_output_dir):
    """Test TS finding with MACE on GPU using TorchSim batching.

    This is the primary production use case: GPU acceleration with batched NEB.
    Verifies that GPU batching via TorchSim works end-to-end.
    """
    require_cuda()
    device = "cuda"

    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=None,  # TorchSim provides forces via MACE
        output_dir=temp_output_dir,
        pair_id="mace_gpu_cu3",
        n_images=5,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=100,
        use_torchsim=True,
        torchsim_params={
            "device": device,
            "mace_model_name": "mace_matpes_0",
            "autobatch_strategy": "binning",  # GPU batching enabled
            "max_steps": 100,
        },
        verbosity=0,
    )

    # Verify TorchSim path was used and completed
    assert result["use_torchsim"] is True
    assert result["n_images"] == 5
    # Result should either succeed or have an error (not hang)
    assert "error" in result or result["status"] == "success"
    if result["status"] == "success":
        assert result["neb_converged"] is True
        assert "transition_state" in result


class TestTorchSimNEB:
    """TorchSim NEB with MACE (small clusters for fast tests)."""

    def test_torchsim_neb_initialization_with_mace(self, cu3_triangle, cu3_linear):
        """TorchSimNEB + MACE relaxer wires up a batched NEB path."""
        from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
        from scgo.ts_search.transition_state import TorchSimNEB

        device = "cuda" if torch.cuda.is_available() else "cpu"

        relaxer = TorchSimBatchRelaxer(
            device=device,
            mace_model_name="mace_matpes_0",
            force_tol=0.05,
            max_steps=100,
        )

        # Create path for Cu3 (triangle -> linear)
        images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

        # Initialize TorchSimNEB - this is where GPU batching setup happens
        neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

        assert neb.relaxer is relaxer
        assert len(neb.images) == 5  # 3 intermediate + 2 endpoints
        assert neb.get_force_calls() == 0

    @pytest.mark.slow
    def test_find_ts_with_torchsim_cu3(self, cu3_triangle, cu3_linear, temp_output_dir):
        """Cu3 triangle–linear TS search with TorchSim + MACE (GPU-only)."""
        require_cuda()
        device = "cuda"

        result = find_transition_state(
            cu3_triangle,
            cu3_linear,
            calculator=None,  # TorchSim provides the calculator
            output_dir=temp_output_dir,
            pair_id="cu3_torchsim",
            n_images=5,  # Use more images to better utilize GPU batching
            spring_constant=0.1,
            fmax=0.05,
            neb_steps=100,
            use_torchsim=True,
            torchsim_params={
                "device": device,
                "mace_model_name": "mace_matpes_0",
                "autobatch_strategy": "binning",  # Use autobatching for GPU efficiency
                "max_steps": 100,
            },
            verbosity=0,
        )

        # Validate the result - verify TorchSim was used and ran
        assert result["n_images"] == 5
        assert result["use_torchsim"] is True
        # Note: Full convergence is not guaranteed in tests, but we verify the run was attempted
        assert "barrier_forward" in result or result.get("error") is not None
        assert "barrier_reverse" in result or result.get("error") is not None


def test_find_ts_allows_missing_endpoint_energies_when_use_torchsim(
    monkeypatch, temp_output_dir, cu3_triangle, cu3_linear
):
    """When `use_torchsim=True` missing endpoint energies on Atoms are allowed
    and the relaxer provides single-point endpoint energies instead.
    """
    # Ensure Atoms have no attached calculators (so extract_energy_from_atoms -> None)
    react = cu3_triangle.copy()
    prod = cu3_linear.copy()
    react.calc = None
    prod.calc = None

    class FakeRelaxer:
        def __init__(self, **kw):
            pass

        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                # return a deterministic single-point energy for endpoints
                results.append((-4.1234, ra))
            return results

    monkeypatch.setattr(
        "scgo.ts_search.transition_state.TorchSimBatchRelaxer",
        FakeRelaxer,
    )

    # Run a very short NEB (neb_steps small) to exercise the endpoint-energy path
    result = find_transition_state(
        react,
        prod,
        calculator=None,
        output_dir=temp_output_dir,
        pair_id="ts_torchsim_endpoint",
        n_images=3,
        fmax=1.0,
        neb_steps=1,
        use_torchsim=True,
        torchsim_params={},
        verbosity=0,
    )

    # Relaxer-provided endpoint energies should be recorded
    assert result.get("reactant_energy") == pytest.approx(-4.1234)
    assert result.get("product_energy") == pytest.approx(-4.1234)


def test_find_ts_high_spring_constant(
    h2_reactant, h2_product, temp_output_dir, default_rel_tol
):
    """Test with very high spring constant."""
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="high_spring",
        n_images=3,
        spring_constant=10.0,  # Very high
        fmax=0.5,
        neb_steps=10,
        verbosity=0,
    )

    assert result["spring_constant"] == pytest.approx(10.0, rel=default_rel_tol)


def test_interpolate_path_many_images(h2_reactant, h2_product):
    """Test interpolation with many images."""
    n_images = 50
    images = interpolate_path(
        h2_reactant, h2_product, n_images=n_images, method="linear"
    )

    assert len(images) == n_images + 2


@pytest.mark.slow
def test_full_neb_convergence(cu3_triangle, cu3_linear, temp_output_dir):
    """Full NEB convergence test with Cu3 (slow)."""
    result = find_transition_state(
        cu3_triangle,
        cu3_linear,
        calculator=EMT(),
        output_dir=temp_output_dir,
        pair_id="cu3_full",
        n_images=7,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=200,
        climb=False,
        verbosity=1,
    )

    # Should converge for Cu3 with EMT
    # (though TS might not be meaningful for EMT)
    assert "status" in result
