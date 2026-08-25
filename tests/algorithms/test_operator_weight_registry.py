"""Weight-table keys must resolve to registered operators for every setup.

An unmatched key (typo, or an operator not registered for that system type)
silently carries no selector mass; these tests pin table/registry coherence so
such drift fails loudly here instead of skewing selection silently at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.algorithms.ga_common import (
    create_mutation_operators,
    unmatched_operator_weight_keys,
)
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.system_types import AdsorbateDefinition
from scgo.utils.mutation_weights import (
    calculate_system_type_weights,
    get_adaptive_mutation_config,
)

_OH = {
    "core_symbols": ["Pt", "Pt", "Pt"],
    "adsorbate_symbols": ["O", "H"],
    "adsorbate_fragment_lengths": [2],
}


def _slab():
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0)
    slab.center(vacuum=4.0, axis=2)
    return slab


def _template(comp, slab=None):
    tmpl = Atoms(symbols=comp, positions=np.zeros((len(comp), 3)), pbc=False)
    if slab is not None:
        tmpl.set_cell(slab.get_cell())
        tmpl.pbc = True
    return tmpl


def _check_consistency(
    system_type,
    comp,
    *,
    n_slab=0,
    adsorbate_definition=None,
):
    slab = _slab() if n_slab > 0 else None
    tmpl = _template(comp, slab)
    blmin = build_blmin_from_zs(tmpl.numbers, ratio=0.7)
    fragment_template = (
        [tmpl[-len(adsorbate_definition.adsorbate_symbols) :]]
        if adsorbate_definition is not None
        else None
    )
    ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=len(comp),
        blmin=blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        system_type=system_type,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=fragment_template,
    )
    weights, _ = calculate_system_type_weights(
        system_type,
        comp,
        adsorbate_definition=adsorbate_definition,
    )
    adaptive = get_adaptive_mutation_config(
        comp,
        use_adaptive=True,
        system_type=system_type,
        adsorbate_definition=adsorbate_definition,
    )
    # Every table key must resolve to a registered operator (typo / drift
    # guard). Registered operators without table mass are allowed: bare
    # slab-target searches deliberately keep in_plane_rotate registered at
    # zero weight for future weight schedules.
    for candidate in (weights, adaptive["operator_weights"]):
        assert unmatched_operator_weight_keys(candidate, name_map) == []


@pytest.mark.parametrize("comp", [["Pt"] * 3, ["Au", "Au", "Pt", "Pt"]])
def test_gas_cluster_tables_match_registry(comp):
    _check_consistency("gas_cluster", comp)


@pytest.mark.parametrize("comp", [["Pt"] * 4, ["Au", "Au", "Pt", "Pt"]])
def test_surface_cluster_tables_match_registry(comp):
    _check_consistency("surface_cluster", comp, n_slab=len(_slab()))


@pytest.mark.parametrize(
    "system_type",
    ["gas_cluster_adsorbate", "surface_cluster_adsorbate"],
)
def test_adsorbate_tables_match_registry(system_type):
    ads = AdsorbateDefinition(**_OH)
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    n_slab = len(_slab()) if system_type.startswith("surface") else 0
    _check_consistency(system_type, comp, n_slab=n_slab, adsorbate_definition=ads)


@pytest.mark.parametrize("comp", [["Pt"] * 4, ["Au", "Au", "Pt", "Pt"]])
def test_bare_surface_tables_match_registry(comp):
    _check_consistency("surface", comp, n_slab=len(_slab()) + len(comp))


def test_surface_adsorbate_table_matches_registry():
    ads = AdsorbateDefinition(**_OH)
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    _check_consistency(
        "surface_adsorbate", comp, n_slab=len(_slab()), adsorbate_definition=ads
    )


def test_unmatched_helper_reports_unknown_keys():
    name_map = {"rattle": 0}
    assert unmatched_operator_weight_keys({"rattle": 1.0}, name_map) == []
    assert unmatched_operator_weight_keys(
        {"rattle": 1.0, "flattning": 0.5}, name_map
    ) == ["flattning"]
