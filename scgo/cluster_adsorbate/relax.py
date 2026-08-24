"""Local relaxation of a metal cluster with a small adsorbate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer
from numpy.random import Generator

from scgo.cluster_adsorbate.combine import (
    combine_core_adsorbate,
    expand_cubic_cell_to_fit,
)
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import attach_fix_bond_lengths
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.constants import DEFAULT_FMAX_THRESHOLD
from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.metadata.provenance import output_json_provenance
from scgo.utils.rng_helpers import ensure_rng_or_create


def _validate_combined(
    combined: Atoms, config: ClusterAdsorbateConfig
) -> tuple[bool, str]:
    return validate_combined_cluster_structure(
        combined,
        min_distance_factor=config.structure_min_distance_factor,
        connectivity_factor=config.structure_connectivity_factor,
        check_clashes=config.structure_check_clashes,
        check_connectivity=config.structure_check_connectivity,
    )


def relax_metal_cluster_with_adsorbate(
    core: Atoms,
    calculator: Calculator,
    fragment_template: Atoms,
    rng: Generator | None = None,
    config: ClusterAdsorbateConfig | None = None,
    *,
    anchor_index: int = 0,
    bond_axis: tuple[int, int] | None = None,
    bond_pairs: Sequence[tuple[int, int]] = (),
    fix_core: bool = True,
    optimizer: type[Optimizer] = LBFGS,
    fmax: float = DEFAULT_FMAX_THRESHOLD,
    steps: int = 200,
    preplaced: Atoms | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    """Place a rigid fragment (unless ``preplaced``), optionally relax with constraints.

    ``bond_pairs`` uses fragment-local indices. Validates connectivity before and
    after relaxation when ``config.validate_combined_structure`` is True.
    """
    if config is None:
        config = ClusterAdsorbateConfig()

    core_work = core.copy()
    n_core = len(core_work)
    n_frag = len(fragment_template)
    if n_frag == 0:
        raise SCGOValidationError("fragment_template must be non-empty")

    ref_syms = fragment_template.get_chemical_symbols()

    if preplaced is not None:
        if len(preplaced) != n_frag:
            raise SCGOValidationError(
                f"preplaced must have {n_frag} atoms like fragment_template, "
                f"got {len(preplaced)}"
            )
        if preplaced.get_chemical_symbols() != ref_syms:
            raise SCGOValidationError(
                "preplaced chemical symbols must match fragment_template: "
                f"{preplaced.get_chemical_symbols()!r} vs {ref_syms!r}"
            )
        frag = preplaced.copy()
    else:
        rng = ensure_rng_or_create(rng)
        frag = place_fragment_on_cluster(
            core_work,
            fragment_template,
            rng,
            config,
            anchor_index=anchor_index,
            bond_axis=bond_axis,
        )
        if frag is None:
            raise SCGORuntimeError(
                "Adsorbate placement failed; increase max_placement_attempts "
                "or adjust height range"
            )

    combined = combine_core_adsorbate(core_work, frag)
    # Only gas-phase (non-periodic) systems get a cubic bounding box; a periodic
    # core keeps its own cell/pbc.
    if not any(combined.pbc):
        expand_cubic_cell_to_fit(combined, config.cell_margin)

    ok0, err0 = True, ""
    if config.validate_combined_structure:
        ok0, err0 = _validate_combined(combined, config)
        if not ok0:
            raise SCGOValidationError(
                "Combined core+adsorbate failed structure validation before relax: "
                + err0
            )

    constraints: list = []
    if fix_core:
        constraints.append(FixAtoms(indices=list(range(n_core))))

    global_pairs = [(n_core + i, n_core + j) for i, j in bond_pairs]
    if global_pairs:
        if fix_core:
            combined.set_constraint(constraints)
        attach_fix_bond_lengths(combined, global_pairs)
    elif fix_core:
        combined.set_constraint(constraints)

    combined.calc = calculator
    e0 = float(combined.get_potential_energy())

    dyn = optimizer(combined, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    e1 = float(combined.get_potential_energy())

    ok1, err1 = True, ""
    if config.validate_combined_structure:
        ok1, err1 = _validate_combined(combined, config)
        if not ok1:
            raise SCGORuntimeError(
                "Combined structure failed validation after relax: " + err1
            )

    bond_lengths: dict[tuple[int, int], float] = {}
    for i, j in bond_pairs:
        gi, gj = n_core + i, n_core + j
        bond_lengths[(i, j)] = float(combined.get_distance(gi, gj))

    info: dict[str, Any] = {
        "initial_energy": e0,
        "final_energy": e1,
        "n_core": n_core,
        "n_frag": n_frag,
        "structure_ok_initial": ok0,
        "structure_error_initial": err0,
        "structure_ok_final": ok1,
        "structure_error_final": err1,
        "bond_lengths": bond_lengths,
    }
    calc_obj = combined.calc
    info.update(
        output_json_provenance(
            extra={
                "formula": combined.get_chemical_formula(),
                "n_core": n_core,
                "n_frag": n_frag,
                "calculator_class": (
                    calc_obj.__class__.__name__ if calc_obj is not None else None
                ),
                "placement": {
                    "anchor_index": anchor_index,
                    "bond_axis": list(bond_axis) if bond_axis is not None else None,
                    "preplaced": preplaced is not None,
                },
                "relax": {
                    "optimizer": optimizer.__name__,
                    "fmax": fmax,
                    "steps": steps,
                    "fix_core": fix_core,
                    "bond_pairs": [list(p) for p in bond_pairs],
                },
                "config": asdict(config),
            }
        )
    )
    sym = combined.get_chemical_symbols()
    if (
        n_frag == 2
        and n_core + 1 < len(sym)
        and sym[n_core] == "O"
        and sym[n_core + 1] == "H"
    ):
        info["oh_distance"] = (
            bond_lengths.get((0, 1))
            if bond_pairs and {tuple(sorted(p)) for p in bond_pairs} == {(0, 1)}
            else float(combined.get_distance(n_core, n_core + 1))
        )
    return combined, info
