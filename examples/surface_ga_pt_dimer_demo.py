#!/usr/bin/env python3
"""Minimal demo: GA for a Pt2 adsorbate on a small Pt(111) slab (EMT).

The slab is **fully frozen** during local relaxations (``fix_all_slab_atoms=True``,
the default). For partial freezing (relax only the top N slab layers) or a
fully mobile slab, see ``surface_ga_relax_top_layers_demo.py`` and the README
section “Adsorbate on a surface (supported-cluster GA)”.

Requires ASE. Run from the repo root::

    python examples/surface_ga_pt_dimer_demo.py

Uses few generations and a small population for a quick smoke test.
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
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )

    rng = default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "surface_ga_demo"
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
        print(f"Demo OK: {len(results)} minima, first energy={_e:.4f} eV")


if __name__ == "__main__":
    main()
