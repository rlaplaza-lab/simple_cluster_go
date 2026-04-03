#!/usr/bin/env python3
"""Demo: relax a small Pt cluster with adsorbed OH (EMT).

Shows the composable workflow: optional global search for the bare cluster,
then :func:`relax_metal_cluster_with_oh` to add OH. The combined system is
checked for connectivity (and clashes) before and after relaxation; internal
O–H length is not fixed unless you use :func:`relax_metal_cluster_with_adsorbate`
with ``bond_pairs``.

Run from the repo root::

    python examples/cluster_oh_relax_demo.py

Requires ASE (EMT). For production, swap in MACE or another ML calculator.
"""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo import (
    ClusterOHConfig,
    relax_metal_cluster_with_oh,
    run_scgo_trials,
)


def main() -> None:
    rng = default_rng(7)
    # Optional: global optimization for bare Pt3 (3 atoms → basin hopping by default)
    params = {
        "calculator": "EMT",
        "n_trials": 1,
        "optimizer_params": {
            "bh": {"niter": 30},
        },
    }
    minima = run_scgo_trials(
        ["Pt", "Pt", "Pt"],
        params=params,
        seed=7,
        verbosity=0,
    )
    core = minima[0][1].copy()

    cfg = ClusterOHConfig(max_placement_attempts=250)
    relaxed, info = relax_metal_cluster_with_oh(
        core,
        EMT(),
        rng=rng,
        config=cfg,
        fix_core=True,
        fmax=0.1,
        steps=150,
    )
    print("initial_energy_eV", info["initial_energy"])
    print("final_energy_eV", info["final_energy"])
    print("structure_ok_initial", info["structure_ok_initial"])
    print("structure_ok_final", info["structure_ok_final"])
    print("OH_distance_A", info.get("oh_distance"))
    print("symbols", relaxed.get_chemical_symbols())

    out = {k: v for k, v in info.items() if k != "bond_lengths"}
    out["bond_lengths"] = {f"{i}_{j}": v for (i, j), v in info["bond_lengths"].items()}
    Path("cluster_oh_relax_result.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
