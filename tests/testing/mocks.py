import numpy as np
from ase import Atoms


class MockBaseRelaxer:
    """Base class for mock relaxers to avoid code duplication across tests.

    This provides a mock `relax` method that can be configured to succeed
    or fail, with predictable energy outputs.
    """

    def __init__(self, rng=None, fail_rate=0.0):
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.fail_rate = fail_rate
        self.relax_calls = 0

    def relax(self, atoms: Atoms) -> Atoms | None:
        self.relax_calls += 1

        # Simulate relaxation failure
        if self.rng.random() < self.fail_rate:
            return None

        a = atoms.copy()

        # Simple simulated relaxation: jitter positions and deterministic energy
        pos = a.get_positions()
        jitter = self.rng.normal(0, 0.05, pos.shape)
        a.set_positions(pos + jitter)

        # Consistent mock energy: depends on size and an arbitrary term
        energy = -1.0 * len(a) + self.rng.random()
        a.calc = MockCalculator(energy)

        # Preserve info Dict
        if not hasattr(a, "info") or a.info is None:
            a.info = {}
        for key, value in getattr(atoms, "info", {}).items():
            a.info[key] = value

        return a


class MockCalculator:
    """Simple mock calculator returning a fixed energy."""

    def __init__(self, energy: float):
        self.energy = energy
        self.results = {"energy": energy}

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        return np.zeros((len(atoms) if atoms else 1, 3))


class MockBatchRelaxer:
    """Minimal test relaxer for batch processing that returns varying energies."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def relax_batch(self, batch):
        # return (energy, atoms) spaced to avoid duplicate collapse
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]
