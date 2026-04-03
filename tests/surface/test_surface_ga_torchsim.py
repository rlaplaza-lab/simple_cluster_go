"""TorchSim GA path with `surface_config` (MockRelaxer, no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim
from scgo.calculators.mace_helpers import MACE
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import slab_surface_extreme


class MockRelaxer:
    """Minimal relaxer matching tests/algorithms/test_geneticalgorithm_generational.py."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def relax_batch(self, batch):
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


@pytest.fixture
def pt_slab_small():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    return slab


def test_ga_go_torchsim_surface_config_mock_relaxer(pt_slab_small, tmp_path, rng):
    """Exercise TorchSim batching + slab constraints without CUDA or MACE."""
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    out = tmp_path / "surface_ga_torchsim"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go_torchsim(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=3,
        offspring_fraction=0.5,
        niter_local_relaxation=20,
        batch_size=2,
        verbosity=0,
        rng=rng,
        surface_config=surface_config,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + 2
    z_top = slab_surface_extreme(slab, 2, upper=True)
    ads_z = best.get_positions()[n_slab:, 2]
    assert np.min(ads_z) > z_top - 0.2


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_ga_go_torchsim_surface_config_mace_cuda(pt_slab_small, tmp_path, rng):
    """Optional real GPU path: MACE + CUDA when available (conda scgo on a GPU box)."""
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    out = tmp_path / "surface_ga_torchsim_cuda"
    out.mkdir(parents=True, exist_ok=True)

    calc = MACE(model_name="small", device="cuda")
    minima = ga_go_torchsim(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=calc,
        niter=1,
        population_size=3,
        offspring_fraction=0.5,
        niter_local_relaxation=30,
        batch_size=2,
        verbosity=0,
        rng=rng,
        surface_config=surface_config,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    _e, best = minima[0]
    assert len(best) == len(slab) + 2
