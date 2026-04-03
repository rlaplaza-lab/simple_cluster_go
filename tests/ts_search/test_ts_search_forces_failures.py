import pytest
from ase import Atoms

from scgo.ts_search.transition_state import TorchSimNEB


class _FakeCalcRaises:
    def __init__(self, exc):
        self._exc = exc

    def get_forces(self, *args, **kwargs):
        raise self._exc


class _FakeRelaxer:
    def __init__(self, relaxed_atoms_results):
        # relaxed_atoms_results should be a list of Atoms to return as relaxed atoms
        self._results = [(0.0, a) for a in relaxed_atoms_results]

    def relax_batch(self, atoms_list, steps=0):
        # Return a list matching the input length
        return self._results


def _make_images(n=3):
    images = [Atoms("Pt", positions=[[0, 0, 0]]) for _ in range(n)]
    return images


def test_get_forces_handles_attributeerror_from_calc():
    images = _make_images(3)
    # Create relaxed atoms with no forces array and a calc that raises AttributeError
    relaxed = Atoms("Pt", positions=[[0, 0, 0]])
    relaxed.arrays.clear()
    relaxed.calc = _FakeCalcRaises(AttributeError("no forces"))

    relaxer = _FakeRelaxer([relaxed] * len(images))
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Expect the outer call to raise RuntimeError when no forces are available
    with pytest.raises(RuntimeError, match="TorchSim did not return forces"):
        neb.get_forces()


def test_get_forces_handles_notimplementederror_from_calc():
    images = _make_images(2)
    relaxed = Atoms("Pt", positions=[[0, 0, 0]])
    relaxed.arrays.clear()
    relaxed.calc = _FakeCalcRaises(NotImplementedError("not implemented"))

    relaxer = _FakeRelaxer([relaxed] * len(images))
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    with pytest.raises(RuntimeError, match="TorchSim did not return forces"):
        neb.get_forces()


def test_get_forces_propagates_unexpected_exceptions():
    images = _make_images(2)
    relaxed = Atoms("Pt", positions=[[0, 0, 0]])
    relaxed.arrays.clear()
    relaxed.calc = _FakeCalcRaises(ValueError("boom"))

    relaxer = _FakeRelaxer([relaxed] * len(images))
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Unexpected exceptions (ValueError) should propagate rather than being silently suppressed
    with pytest.raises(ValueError, match="boom"):
        neb.get_forces()
