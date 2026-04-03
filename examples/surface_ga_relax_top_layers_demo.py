#!/usr/bin/env python3
"""GA for a Pt2 adsorbate on Pt(111) with partial slab freezing (EMT).

Only the top ``n_relax_top_slab_layers`` distinct slab layers along the surface
normal are free to relax; deeper layers stay fixed. This matches the common
“freeze bulk, relax surface” setup without counting ``L`` manually.

For a **fully rigid** slab, use ``fix_all_slab_atoms=True`` (default); see
``surface_ga_pt_dimer_demo.py``. For a **fully mobile** slab during local
relaxation, use ``fix_all_slab_atoms=False`` with both layer fields unset.

Run from the repo root::

    python examples/surface_ga_relax_top_layers_demo.py

Requires ASE. Uses few generations for a quick smoke test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ase.build import fcc111
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo import SurfaceSystemConfig, ga_go


def main() -> None:
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True

    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=2,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )

    rng = default_rng(43)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "surface_ga_relax_top_demo"
        out.mkdir(parents=True, exist_ok=True)
        results = ga_go(
            composition=["Pt", "Pt"],
            output_dir=str(out),
            calculator=EMT(),
            niter=2,
            population_size=4,
            offspring_fraction=0.5,
            niter_local_relaxation=50,
            n_jobs_population_init=1,
            early_stopping_niter=0,
            verbosity=1,
            rng=rng,
            surface_config=surface_config,
        )
        assert len(results) >= 1
        _e, best = results[0]
        assert len(best) == len(slab) + 2
        print(
            f"Demo OK (relax top {surface_config.n_relax_top_slab_layers} slab layers): "
            f"{len(results)} minima, first energy={_e:.4f} eV"
        )


if __name__ == "__main__":
    main()
