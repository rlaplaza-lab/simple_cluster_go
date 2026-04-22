"""Shared pytest fixtures for the SCGO test suite."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from tests.constants import INITIALIZATION_MODES
from tests.test_utils import setup_test_atoms


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="function")
def rng():
    """Provide a fixed RNG for reproducible tests.

    Function scope ensures each test gets a fresh RNG instance,
    preventing race conditions in parallel/multiprocess test runs.
    """
    return np.random.default_rng(42)


@pytest.fixture
def default_rel_tol():
    """Default relative tolerance for floating point comparisons in tests."""
    return 1e-6


@pytest.fixture
def seed_random():
    """Seed Python and NumPy RNGs for deterministic behavior where needed.

    This is the canonical fixture to opt into deterministic seeding. It
    sets both the Python `random` and `numpy.random` state to a fixed seed
    for the duration of the test and restores the previous RNG states
    after the test finishes to avoid global side-effects.
    """
    import random

    # Save previous states
    prev_py_state = random.getstate()
    prev_np_state = np.random.get_state()

    # Set deterministic seed for the test
    random.seed(42)
    np.random.seed(42)

    try:
        yield 42
    finally:
        # Restore previous RNG states
        random.setstate(prev_py_state)
        np.random.set_state(prev_np_state)


@pytest.fixture(params=INITIALIZATION_MODES)
def init_mode(request):
    """Parametrized fixture providing all initialization modes.

    Use this fixture in tests that should work across all modes:
    - random_spherical
    - seed+growth
    - template
    - smart

    Example:
        def test_all_modes_work(init_mode, rng):
            atoms = create_initial_cluster([...], mode=init_mode, rng=rng)
            assert_cluster_valid(atoms, [...])
    """
    return request.param


@pytest.fixture
def pt3_atoms():
    """Create a simple Pt3 cluster for testing."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
    return setup_test_atoms(atoms)


@pytest.fixture
def au2pt2_atoms():
    """Create a bimetallic Au2Pt2 cluster for testing."""
    atoms = Atoms(
        "Au2Pt2",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]],
    )
    return setup_test_atoms(atoms)


@pytest.fixture
def pt4_tetrahedron():
    """Create a Pt4 tetrahedron for testing."""
    atoms = Atoms(
        "Pt4",
        positions=[
            [0, 0, 0],
            [2.5, 0, 0],
            [1.25, 2.165, 0],  # sqrt(3)/2 * 2.5
            [1.25, 0.721, 2.357],  # sqrt(2/3) * 2.5
        ],
    )
    return setup_test_atoms(atoms)


@pytest.fixture
def pt2_atoms():
    """Create a simple Pt2 dimer for testing."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    return setup_test_atoms(atoms)


@pytest.fixture
def pt_slab_small():
    """Small Pt(111) slab for adsorbate-on-surface tests."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    return slab


@pytest.fixture
def h2_reactant():
    """H2 molecule at equilibrium distance (~0.74 Å)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def h2_product():
    """H2 molecule stretched to ~1.5 Å (higher energy)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def li2_isomer1():
    """Li2 cluster - linear configuration."""
    atoms = Atoms("Li2", positions=[[0, 0, 0], [3.0, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def li2_isomer2():
    """Li2 cluster - rotated configuration."""
    atoms = Atoms("Li2", positions=[[0, 0, 0], [2.1, 2.1, 0]])
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def cu3_triangle():
    """Cu3 equilateral triangle."""
    d = 2.5  # Cu-Cu distance
    positions = [
        [0, 0, 0],
        [d, 0, 0],
        [d / 2, d * np.sqrt(3) / 2, 0],
    ]
    atoms = Atoms("Cu3", positions=positions)
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def cu3_linear():
    """Cu3 linear chain."""
    d = 2.5
    positions = [[0, 0, 0], [d, 0, 0], [2 * d, 0, 0]]
    atoms = Atoms("Cu3", positions=positions)
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def cu3_bent():
    """Cu3 bent chain."""
    positions = [[0, 0, 0], [2.5, 0, 0], [2.5, 2.5, 0]]
    atoms = Atoms("Cu3", positions=positions)
    atoms.center(vacuum=5.0)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def empty_atoms():
    """Create an empty Atoms object for testing."""
    atoms = Atoms()
    return setup_test_atoms(atoms)


@pytest.fixture
def single_atom():
    """Create a single Pt atom for testing."""
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    return setup_test_atoms(atoms)


@pytest.fixture
def pt2_with_calc():
    """Create a Pt2 dimer with EMT calculator attached."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    setup_test_atoms(atoms)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def pt3_with_calc():
    """Create a Pt3 cluster with EMT calculator attached."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
    setup_test_atoms(atoms)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def test_database(tmp_path):
    """Create a temporary database with sample structures.

    Provides a database path with pre-populated test structures:
    - Pt2 dimer (energy=-10.0)
    - Pt3 triangle (energy=-15.0)
    - Au2 dimer (energy=-8.0)
    """
    from ase.db import connect

    db_path = tmp_path / "test.db"
    with connect(str(db_path)) as db:
        pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            pt2,
            relaxed=True,
            key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
            gaid=1,
        )

        pt3 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        db.write(
            pt3,
            relaxed=True,
            key_value_pairs={"raw_score": -15.0, "final_unique_minimum": True},
            gaid=2,
        )

        au2 = Atoms("Au2", positions=[[0, 0, 0], [2.8, 0, 0]])
        db.write(
            au2,
            relaxed=True,
            key_value_pairs={"raw_score": -8.0, "final_unique_minimum": True},
            gaid=3,
        )

    from scgo.database.schema import stamp_scgo_database

    stamp_scgo_database(db_path)
    return str(db_path)


@pytest.fixture
def calculator_factory():
    """Factory fixture for creating calculators.

    Returns a function that can create calculators by name.
    Useful for parametrized tests that need different calculator types.

    Example:
        def test_with_calc(calculator_factory):
            calc = calculator_factory("EMT")
            assert calc is not None
    """

    def _create_calculator(name: str = "EMT", **kwargs):
        """Create a calculator by name.

        Args:
            name: Calculator name ("EMT" supported by default)
            **kwargs: Additional arguments to pass to calculator

        Returns:
            Calculator instance
        """
        if name == "EMT":
            return EMT(**kwargs)
        else:
            raise ValueError(f"Unknown calculator: {name}")

    return _create_calculator


@pytest.fixture(autouse=True)
def clear_initialization_caches():
    """Clear template-rotation and geometry caches so reproducibility tests stay isolated."""
    from scgo.database.cache import get_global_cache
    from scgo.initialization import geometry_helpers, initializers

    get_global_cache().clear_namespace(initializers._TEMPLATE_ROTATIONS_CACHE_NS)
    geometry_helpers.clear_convex_hull_cache()

    yield

    get_global_cache().clear_namespace(initializers._TEMPLATE_ROTATIONS_CACHE_NS)


@pytest.fixture(autouse=True)
def maybe_force_mace_gpu(request, monkeypatch):
    """If the test is marked with `force_mace_gpu`, force MACE to use GPU.

    Only applies the patch when the test has the `force_mace_gpu` marker.
    If CUDA is requested but unavailable, the test is skipped.
    """
    marker = request.node.get_closest_marker("force_mace_gpu")
    if not marker:
        return

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA requested via 'force_mace_gpu' marker but CUDA is not available"
        )

    from scgo.calculators import mace_helpers

    original_init = mace_helpers.MACE.__init__

    def patched_init(self, *args, **kwargs):
        if "device" not in kwargs or kwargs.get("device") is None:
            kwargs["device"] = "cuda"
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(mace_helpers.MACE, "__init__", patched_init)


@pytest.fixture(autouse=True)
def maybe_force_njobs_parallel(request, monkeypatch):
    """If the test is marked with `force_njobs_parallel`, force n_jobs=-2.

    Only applies the patch when the test has the `force_njobs_parallel` marker.
    """
    marker = request.node.get_closest_marker("force_njobs_parallel")
    if not marker:
        return

    from scgo.initialization import initializers

    original_batch = initializers.create_initial_cluster_batch

    def patched_batch(*args, **kwargs):
        if "n_jobs" not in kwargs:
            kwargs["n_jobs"] = -2
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(initializers, "create_initial_cluster_batch", patched_batch)

    original_pop = initializers.generate_initial_population

    def patched_pop(*args, **kwargs):
        if "n_jobs" not in kwargs:
            kwargs["n_jobs"] = -2
        return original_pop(*args, **kwargs)

    monkeypatch.setattr(initializers, "generate_initial_population", patched_pop)
