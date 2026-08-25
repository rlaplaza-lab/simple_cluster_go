"""Shared pytest fixtures for the SCGO test suite."""

import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.surface.config import SurfaceSystemConfig
from tests.helpers import create_test_atoms


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    if item.get_closest_marker("requires_multicore") and (os.cpu_count() or 1) < 2:
        pytest.skip("Requires >=2 CPUs")


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
def pt3_atoms():
    """Create a simple Pt3 cluster for testing."""
    return create_test_atoms(
        "Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]]
    )


@pytest.fixture
def au2pt2_atoms():
    """Create a bimetallic Au2Pt2 cluster for testing."""
    return create_test_atoms(
        "Au2Pt2",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]],
    )


@pytest.fixture
def pt4_tetrahedron():
    """Create a Pt4 tetrahedron for testing."""
    return create_test_atoms(
        "Pt4",
        positions=[
            [0, 0, 0],
            [2.5, 0, 0],
            [1.25, 2.165, 0],  # sqrt(3)/2 * 2.5
            [1.25, 0.721, 2.357],  # sqrt(2/3) * 2.5
        ],
    )


@pytest.fixture
def pt2_atoms():
    """Create a simple Pt2 dimer for testing."""
    return create_test_atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])


@pytest.fixture
def pt_slab_small():
    """Small Pt(111) slab for adsorbate-on-surface tests."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    return slab


@pytest.fixture
def surface_config_pt111(pt_slab_small):
    """Standard Pt(111) surface config used across surface GA tests."""
    return SurfaceSystemConfig(
        slab=pt_slab_small,
        adsorption_height_min=1.6,
        adsorption_height_max=2.2,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )


@pytest.fixture
def minimal_ga_kwargs():
    """Minimal GA parameters for fast smoke tests."""
    return {
        "niter": 2,
        "population_size": 4,
        "offspring_fraction": 0.5,
        "niter_local_relaxation": 50,
        "n_jobs_population_init": 1,
        "early_stopping_niter": 0,
    }


@pytest.fixture
def ts_minima_db(tmp_path):
    """Temporary GA database with marked final minima for TS integration tests."""
    from ase import Atoms

    from tests.helpers import create_preparedb, mark_test_minima_as_final

    db_path = tmp_path / "ts_minima.db"
    db = create_preparedb(Atoms("Pt2"), db_path, population_size=10)
    pt3 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
    db.add_unrelaxed_candidate(pt3, description="pt3")
    mark_test_minima_as_final(db_path)
    return str(db_path)


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
def ir4_tetrahedron():
    """Ir4 regular tetrahedron."""
    d = 2.7
    positions = [
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [d / 2, d * np.sqrt(3) / 2, 0.0],
        [d / 2, d / (2 * np.sqrt(3)), d * np.sqrt(2.0 / 3.0)],
    ]
    atoms = Atoms("Ir4", positions=positions)
    atoms.center(vacuum=8.0)
    return atoms


@pytest.fixture
def ir4_tetrahedron_atom_swapped():
    """Ir4 tetrahedron with two atoms swapped.

    A label swap of a *regular* tetrahedron is the same minimum up to atom
    permutation: canonical endpoint alignment (fingerprint + Kabsch + spatial
    rematch) maps it onto the reactant geometry exactly, so every interpolation
    of this pair yields identical images and ``validate_initial_neb_path``
    rejects it as degenerate. Used to pin that rejection.
    """
    d = 2.7
    positions = [
        [d, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [d / 2, d / (2 * np.sqrt(3)), d * np.sqrt(2.0 / 3.0)],
        [d / 2, d * np.sqrt(3) / 2, 0.0],
    ]
    atoms = Atoms("Ir4", positions=positions)
    atoms.center(vacuum=8.0)
    return atoms


@pytest.fixture
def empty_atoms():
    """Create an empty Atoms object for testing."""
    return create_test_atoms("")


@pytest.fixture
def single_atom():
    """Create a single Pt atom for testing."""
    return create_test_atoms("Pt", positions=[[0, 0, 0]])


@pytest.fixture
def pt2_with_calc():
    """Create a Pt2 dimer with EMT calculator attached."""
    atoms = create_test_atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def pt3_with_calc():
    """Create a Pt3 cluster with EMT calculator attached."""
    atoms = create_test_atoms(
        "Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]]
    )
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

    from scgo.metadata.db_stamp import stamp_db

    stamp_db(db_path)
    return str(db_path)


def _needs_initialization_cache_isolation(request: pytest.FixtureRequest) -> bool:
    """Return True when a test is likely to depend on clean initialization caches."""
    node = request.node
    return (
        node.get_closest_marker("requires_cache_isolation") is not None
        or node.get_closest_marker("reproducibility") is not None
    )


@pytest.fixture(autouse=True)
def clear_initialization_caches(request: pytest.FixtureRequest):
    """Clear expensive initialization caches only for isolation-sensitive tests."""
    if not _needs_initialization_cache_isolation(request):
        yield
        return

    from scgo.database.cache import get_global_cache
    from scgo.initialization import geometry_helpers
    from scgo.initialization.initialization_config import _COMPOSITION_CACHE_NS
    from scgo.initialization.templates import _TEMPLATE_ROTATIONS_CACHE_NS

    get_global_cache().clear_namespace(_TEMPLATE_ROTATIONS_CACHE_NS)
    get_global_cache().clear_namespace(_COMPOSITION_CACHE_NS)
    geometry_helpers.clear_convex_hull_cache()

    yield

    get_global_cache().clear_namespace(_TEMPLATE_ROTATIONS_CACHE_NS)
    get_global_cache().clear_namespace(_COMPOSITION_CACHE_NS)
    geometry_helpers.clear_convex_hull_cache()
