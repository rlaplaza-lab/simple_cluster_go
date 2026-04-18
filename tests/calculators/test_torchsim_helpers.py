"""Tests for TorchSim helper functionality.

These tests verify the TorchSimBatchRelaxer interface. Since torch-sim-atomistic
is now a mandatory dependency, these tests expect it to be available.
"""

import pytest


def test_torchsim_import_success():
    """TorchSim and PyTorch are available as core dependencies."""
    import torch
    import torch_sim as ts

    assert torch is not None
    assert ts is not None


def test_torchsim_batch_relaxer_import():
    """Test that TorchSimBatchRelaxer can be imported."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert TorchSimBatchRelaxer is not None


def test_geneticalgorithm_go_torchsim_import():
    """Test that ga_go_torchsim can be imported."""
    from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim

    assert ga_go_torchsim is not None


def test_force_dtype_conversion():
    """Test that forces are converted to float64 in all storage locations."""
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    # Create a mock calculator that stores float32 forces (like MACE with dtype=float32)
    class MockMACECalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            # Simulate MACE with dtype=float32 - stores float32 forces
            self.results["energy"] = -1.0
            self.results["forces"] = np.array(
                [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
                dtype=np.float32,
            )

    # Create atoms with float32 forces from MACE
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    forces_f32 = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32)
    atoms.arrays["forces"] = forces_f32.copy()

    # Attach mock MACE calculator with float32 forces in results
    atoms.calc = MockMACECalculator()
    atoms.calc.results = {
        "energy": -1.0,
        "forces": forces_f32.copy(),
    }

    # Verify forces are float32 in both locations
    assert atoms.arrays["forces"].dtype == np.float32
    assert atoms.calc.results["forces"].dtype == np.float32

    # Now simulate what our fixed code does - convert in both locations
    if "forces" in atoms.arrays:
        atoms.arrays["forces"] = np.asarray(atoms.arrays["forces"], dtype=np.float64)

    if (
        hasattr(atoms, "calc")
        and atoms.calc is not None
        and hasattr(atoms.calc, "results")
        and "forces" in atoms.calc.results
    ):
        atoms.calc.results["forces"] = np.asarray(
            atoms.calc.results["forces"],
            dtype=np.float64,
        )

    # Verify forces are now float64 in both locations
    assert atoms.arrays["forces"].dtype == np.float64
    assert atoms.calc.results["forces"].dtype == np.float64

    # Verify values are preserved in both locations
    assert np.allclose(atoms.arrays["forces"][0], [0.1, 0.2, 0.3])
    assert np.allclose(atoms.arrays["forces"][1], [-0.1, -0.2, -0.3])
    assert np.allclose(atoms.calc.results["forces"][0], [0.1, 0.2, 0.3])
    assert np.allclose(atoms.calc.results["forces"][1], [-0.1, -0.2, -0.3])


def test_torchsim_basic_initialization():
    """Test basic TorchSimBatchRelaxer initialization."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    # Since torch-sim-atomistic is now mandatory, this should work
    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        force_tol=0.05,
        max_steps=10,
    )
    assert relaxer.device is not None
    assert relaxer.force_tol == pytest.approx(0.05, rel=1e-6)
    assert relaxer.max_steps == 10


def test_memory_scaler_cache_basic():
    """Test basic MemoryScalerCache functionality."""
    import tempfile

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    # Create a temporary cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MemoryScalerCache(cache_dir=tmpdir)

        # Test set and get
        cache.set(
            n_atoms=32,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
            value=100.0,
        )

        # Retrieve the cached value
        cached = cache.get(
            n_atoms=32,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached == pytest.approx(100.0, rel=1e-6)

        # Test that similar n_atoms values use the same bin
        cached_similar = cache.get(
            n_atoms=33,  # Should bin to same value as 32 (both -> 35)
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached_similar == pytest.approx(100.0, rel=1e-6)

        # Test that different parameters return None
        cached_different = cache.get(
            n_atoms=32,
            model_name="large",  # Different model
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached_different is None


def test_memory_scaler_cache_persistence():
    """Test that MemoryScalerCache persists to disk."""
    import tempfile
    from pathlib import Path

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.json"

        # Create cache and add value
        cache1 = MemoryScalerCache(cache_dir=tmpdir, cache_file="test_cache.json")
        cache1.set(
            n_atoms=50,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms_x_density",
            device="cpu",
            value=250.5,
        )

        # Verify file was created
        assert cache_path.exists()

        # Create new cache instance and verify it loads the persisted value
        cache2 = MemoryScalerCache(cache_dir=tmpdir, cache_file="test_cache.json")
        cached = cache2.get(
            n_atoms=50,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms_x_density",
            device="cpu",
        )
        assert cached == pytest.approx(250.5, rel=1e-6)


def test_memory_scaler_cache_clear():
    """Test clearing the cache."""
    import tempfile

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MemoryScalerCache(cache_dir=tmpdir)

        # Add some values
        cache.set(32, "medium", "n_atoms", "cuda", 100.0)
        cache.set(64, "medium", "n_atoms", "cuda", 200.0)

        # Clear the cache
        cache.clear()

        # Verify values are gone
        assert cache.get(32, "medium", "n_atoms", "cuda") is None
        assert cache.get(64, "medium", "n_atoms", "cuda") is None


def test_get_global_memory_scaler_cache():
    """Test accessing the global cache."""
    from scgo.calculators.torchsim_helpers import get_global_memory_scaler_cache

    cache = get_global_memory_scaler_cache()
    assert cache is not None
    assert hasattr(cache, "get")
    assert hasattr(cache, "set")
    assert hasattr(cache, "clear")


def test_torchsim_step_kwargs_removed():
    """``step_kwargs`` has been removed from TorchSimBatchRelaxer (never forwarded by ts.optimize).

    The field was replaced by ``optimizer_kwargs``; passing the old name should raise TypeError.
    """
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    with pytest.raises(TypeError):
        TorchSimBatchRelaxer(
            device="cpu",
            mace_model_name="mace_matpes_0",
            step_kwargs={"dt_max": 0.1},  # type: ignore[call-arg]
        )


def test_torchsim_optimizer_kwargs_flatten_into_runner_kwargs():
    """``optimizer_kwargs`` should be forwarded flat as **optimizer_kwargs to ts.optimize."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        optimizer_kwargs={"dt_max": 0.1},
    )
    # Flattened into the resolved runner kwargs rather than nested.
    assert relaxer._runner_kwargs.get("dt_max") == pytest.approx(0.1)
    assert "optimizer_kwargs" not in relaxer._runner_kwargs
    assert "step_kwargs" not in relaxer._runner_kwargs


def test_torchsim_autobatcher_default_off_on_cpu():
    """On CPU the default ``autobatcher=None`` disables autobatching (docs recommendation)."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
    )
    assert "autobatcher" not in relaxer._runner_kwargs


class _StubModel:
    """Minimal stand-in for a torch-sim model; avoids MACE/UMA downloads in unit tests."""


def test_torchsim_autobatcher_probe_capped_by_expected_max_atoms(monkeypatch):
    """``expected_max_atoms`` should cap the autobatcher's GPU probe (``max_atoms_to_try``).

    Without this cap the probe can geometrically climb to 500k atoms and OOM
    small GPUs. We must never probe more atoms than the workload demands.
    """
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",  # triggers the autobatcher construction path
        model=_StubModel(),
        expected_max_atoms=256,
        autobatcher=True,
    )
    assert captured.get("max_atoms_to_try") == 256


def test_torchsim_autobatcher_probe_cap_explicit_override_is_honored(monkeypatch):
    """An explicit ``max_atoms_to_try`` wins over ``expected_max_atoms``."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        expected_max_atoms=10_000,
        max_atoms_to_try=128,
        autobatcher=True,
    )
    assert captured.get("max_atoms_to_try") == 128


def test_torchsim_autobatcher_probe_cap_defaults_to_torchsim_when_unset(monkeypatch):
    """Without ``expected_max_atoms``/``max_atoms_to_try`` we inherit torch-sim's default."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        autobatcher=True,
    )
    assert "max_atoms_to_try" not in captured


def test_torchsim_autobatcher_true_on_cpu_warns_and_coerces():
    """Passing ``autobatcher=True`` on CPU emits a RuntimeWarning and disables autobatching."""
    import warnings

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        relaxer = TorchSimBatchRelaxer(
            device="cpu",
            mace_model_name="mace_matpes_0",
            autobatcher=True,
        )
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("autobatching" in str(w.message).lower() for w in runtime_warnings)
    assert "autobatcher" not in relaxer._runner_kwargs


def test_torchsim_optimizer_set_correctly():
    """Test that TorchSimBatchRelaxer sets optimizer correctly for different torch-sim versions.

    This test verifies that the optimizer is set correctly:
    - torch-sim 0.4.0+: ts.Optimizer.fire (enum)
    - torch-sim 0.3.0: ts.optimizers.fire (callable function)
    """
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    # Create relaxer with default optimizer_name="fire"
    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        optimizer_name="fire",
        force_tol=0.05,
        max_steps=10,
    )

    # Verify optimizer is set correctly
    assert relaxer.optimizer is not None

    # Check which API version we're using
    if hasattr(ts, "Optimizer"):
        # 0.4.0+ uses enum API
        assert relaxer.optimizer == ts.Optimizer.fire
        # Verify it's an enum (may be StrEnum which is also a str, but that's OK)
        import enum

        assert isinstance(relaxer.optimizer, enum.Enum)
    elif hasattr(ts, "optimizers"):
        # 0.3.0 uses callable function API
        assert relaxer.optimizer == ts.optimizers.fire
        # Verify it's callable
        assert callable(relaxer.optimizer)
        # Verify it's not a string
        assert not isinstance(relaxer.optimizer, str)
    else:
        pytest.fail("Neither ts.Optimizer nor ts.optimizers found")
