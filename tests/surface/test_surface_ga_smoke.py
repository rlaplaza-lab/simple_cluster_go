"""End-to-end smoke tests for adsorbate-on-slab GA (`ga_go` + `SurfaceSystemConfig`)."""

from __future__ import annotations

import numpy as np
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.algorithms.geneticalgorithm_go import ga_go
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import slab_surface_extreme


def test_ga_go_surface_config_emt_smoke(pt_slab_small, tmp_path):
    """Minimal GA on a slab: same recipe as examples/surface_ga_pt_dimer_demo.py."""
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    rng = default_rng(42)
    out = tmp_path / "surface_ga_emt"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        niter=2,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=50,
        n_jobs_population_init=1,
        early_stopping_niter=0,
        verbosity=0,
        rng=rng,
        surface_config=surface_config,
    )

    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + 2
    z_top = slab_surface_extreme(slab, 2, upper=True)
    ads_z = best.get_positions()[n_slab:, 2]
    assert np.min(ads_z) > z_top - 0.2
