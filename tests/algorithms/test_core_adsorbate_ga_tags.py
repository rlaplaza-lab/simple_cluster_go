"""Core vs adsorbate partition: tags and operator wiring for two-block mobile GA."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.algorithms.ga_common import (
    apply_mobile_core_ads_tags,
    core_adsorbate_partition_counts,
    create_ga_pairing,
    create_mutation_operators,
    maybe_apply_mobile_core_ads_tags,
)
from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.system_types import AdsorbateDefinition


def test_apply_mobile_core_ads_tags() -> None:
    a = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H"],
        positions=np.zeros((5, 3)),
        pbc=False,
    )
    apply_mobile_core_ads_tags(a, n_slab=2, n_core=1, n_ads=2)
    assert list(a.get_tags()) == [0, 0, 0, 1, 1]


@pytest.mark.parametrize(
    ("ads", "composition", "expected"),
    [
        (
            {"core_symbols": ["Pt", "Pt"], "adsorbate_symbols": ["O", "H"]},
            ["Pt", "Pt", "O", "H"],
            (2, 2),
        ),
        (
            {"core_symbols": ["Pt", "Pt", "O", "H"], "adsorbate_symbols": []},
            ["Pt", "Pt", "O", "H"],
            None,
        ),
    ],
)
def test_core_adsorbate_partition_counts(
    ads: AdsorbateDefinition,
    composition: list[str],
    expected: tuple[int, int] | None,
) -> None:
    assert (
        core_adsorbate_partition_counts("gas_cluster_adsorbate", composition, ads)
        == expected
    )


def test_create_ga_pairing_use_tags_for_two_block() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
    }
    at = Atoms(symbols=comp, positions=np.zeros((4, 3)), cell=[20, 20, 20], pbc=False)
    p = create_ga_pairing(
        at,
        4,
        np.random.default_rng(0),
        system_type="gas_cluster_adsorbate",
        composition=comp,
        adsorbate_definition=ads,
        exploratory_crossover_probability=0.0,
    )
    assert isinstance(p, CutAndSplicePairing)
    assert p.use_tags is True


def test_create_mutation_operators_two_block_tags_omit_distort() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((4, 3)), pbc=False)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, [0, 1, 2, 3]), ratio_of_covalent_radii=0.7
    )
    ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=4,
        blmin=blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
    )
    assert "flattening" not in name_map and "breathing" not in name_map
    assert ops[name_map["rattle"]].use_tags is True
    assert ops[name_map["anisotropic_rattle"]].use_tags is True


def test_maybe_apply_skips_monolithic_ads_def() -> None:
    a = Atoms("H2", [[0, 0, 0], [0, 0, 0.78]], pbc=False)
    maybe_apply_mobile_core_ads_tags(
        a,
        0,
        ["H", "H"],
        {"core_symbols": ["H", "H"], "adsorbate_symbols": []},
        "gas_cluster_adsorbate",
    )
    assert np.all(a.get_tags() == 0)
