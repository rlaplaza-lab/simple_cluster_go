"""`run_scgo_trials` with adsorbate-on-slab GA via `optimizer_params['ga']`.

`run_scgo_trials` selects the algorithm from len(composition) only. For slab GA,
``surface_config`` must live under ``optimizer_params['ga']``, which is only read
when the chosen algorithm is ``ga`` — so use **at least four** adsorbate atoms.
"""

from __future__ import annotations

import numpy as np

from scgo.param_presets import get_testing_params
from scgo.run_minima import run_scgo_trials
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import slab_surface_extreme
from scgo.utils.helpers import deep_merge_dicts


def test_run_scgo_trials_passes_surface_config_when_ga_selected(
    pt_slab_small, tmp_path
):
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    base = get_testing_params()
    params = deep_merge_dicts(
        base,
        {
            "optimizer_params": {
                "ga": {
                    "niter": 2,
                    "population_size": 4,
                    "offspring_fraction": 0.5,
                    "niter_local_relaxation": 400,
                    "n_jobs_population_init": 1,
                    "early_stopping_niter": 0,
                    "surface_config": surface_config,
                }
            }
        },
    )

    # Four adsorbate Pt atoms => chosen_go == "ga" (see run_minima._select_algorithm).
    composition = ["Pt", "Pt", "Pt", "Pt"]

    minima = run_scgo_trials(
        composition=composition,
        params=params,
        seed=42,
        verbosity=0,
        output_dir=str(tmp_path / "surf_go"),
    )

    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + 4
    z_top = slab_surface_extreme(slab, 2, upper=True)
    ads_z = best.get_positions()[n_slab:, 2]
    assert np.min(ads_z) > z_top - 0.2
