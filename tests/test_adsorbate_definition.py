"""Boundary tests for :class:`~scgo.system_types.AdsorbateDefinition` helpers.

Pure validation/coercion (no EMT/GA). Complements
``tests/cluster_adsorbate/test_general_adsorbate.py``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.system_types import (
    AdsorbateDefinition,
    as_adsorbate_definition,
    extract_adsorbate_definition_from_params,
    flatten_adsorbate_symbols,
    normalize_adsorbates_input,
    resolve_adsorbate_run_composition,
    validate_adsorbate_definition,
)


def _oh() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])


def test_adsorbate_definition_frozen_and_effective_lengths() -> None:
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[],
    )
    assert ads.n_core == 2
    assert ads.n_adsorbate == 2
    assert ads.effective_fragment_lengths == [2]
    assert AdsorbateDefinition(core_symbols=["Pt"]).effective_fragment_lengths == []
    with pytest.raises(FrozenInstanceError):
        ads.core_symbols = ["Au"]  # type: ignore[misc]


def test_from_dict_defaults_and_fields() -> None:
    empty = AdsorbateDefinition.from_dict({})
    assert empty.core_symbols == []
    assert empty.adsorbate_symbols == []
    assert empty.adsorbate_fragment_lengths == []

    ads = AdsorbateDefinition.from_dict(
        {
            "core_symbols": ["Pt", "Pt"],
            "adsorbate_symbols": ["O", "H"],
            "adsorbate_fragment_lengths": [2],
        }
    )
    assert ads.core_symbols == ["Pt", "Pt"]
    assert ads.adsorbate_symbols == ["O", "H"]
    assert ads.adsorbate_fragment_lengths == [2]
    assert ads.fragment_anchor_index is None
    assert ads.fragment_bond_axis is None


@pytest.mark.parametrize(
    "data,match",
    [
        ({"core_symbols": "Pt"}, "core_symbols"),
        ({"adsorbate_symbols": ("O", "H")}, "adsorbate_symbols"),
        ({"adsorbate_fragment_lengths": [2.5]}, "adsorbate_fragment_lengths"),
        ({"fragment_anchor_index": 1.0}, "fragment_anchor_index"),
        ({"fragment_bond_axis": [0]}, "fragment_bond_axis"),
        ({"fragment_bond_axis": [0, 1, 2]}, "fragment_bond_axis"),
        ({"fragment_bond_axis": ["a", "b"]}, "fragment_bond_axis"),
    ],
)
def test_from_dict_rejects_bad_types(data: dict, match: str) -> None:
    with pytest.raises(SCGOValidationError, match=match):
        AdsorbateDefinition.from_dict(data)


def test_as_adsorbate_definition_coercion() -> None:
    assert as_adsorbate_definition(None) is None
    ads = AdsorbateDefinition(core_symbols=["Pt"], adsorbate_symbols=["O", "H"])
    assert as_adsorbate_definition(ads) is ads
    coerced = as_adsorbate_definition(
        {"core_symbols": ["Pt"], "adsorbate_symbols": ["O", "H"]}
    )
    assert isinstance(coerced, AdsorbateDefinition)
    assert coerced.core_symbols == ["Pt"]
    with pytest.raises(SCGOValidationError, match="dict or AdsorbateDefinition"):
        as_adsorbate_definition(["Pt", "O", "H"])


def test_extract_adsorbate_definition_from_params() -> None:
    assert extract_adsorbate_definition_from_params(None) is None
    assert extract_adsorbate_definition_from_params({}) is None
    ads = extract_adsorbate_definition_from_params(
        {
            "adsorbate_definition": {
                "core_symbols": ["Pt"],
                "adsorbate_symbols": ["O", "H"],
            }
        }
    )
    assert ads is not None
    assert ads.core_symbols == ["Pt"]


def test_normalize_and_flatten_adsorbates() -> None:
    oh = _oh()
    one = normalize_adsorbates_input(oh, context="test")
    assert len(one) == 1
    assert one[0] is not oh
    assert list(one[0].get_chemical_symbols()) == ["O", "H"]
    assert len(normalize_adsorbates_input([oh, oh.copy()], context="test")) == 2

    co = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    assert flatten_adsorbate_symbols([oh, co]) == ["O", "H", "C", "O"]

    with pytest.raises(SCGOValidationError, match="required"):
        normalize_adsorbates_input(None, context="test")
    with pytest.raises(SCGOValidationError, match="at least one"):
        normalize_adsorbates_input([], context="test")
    with pytest.raises(SCGOValidationError, match="ase.Atoms"):
        normalize_adsorbates_input([{"symbols": ["O"]}], context="test")  # type: ignore[list-item]
    with pytest.raises(SCGOValidationError, match="must not be empty"):
        normalize_adsorbates_input(Atoms(), context="test")


def test_resolve_adsorbate_run_composition() -> None:
    oh = _oh()
    with pytest.raises(SCGOValidationError, match="does not accept adsorbates"):
        resolve_adsorbate_run_composition(
            system_type="gas_cluster",
            composition=["Pt", "Pt"],
            adsorbates=oh,
            preset_adsorbate_definition=None,
            context="test",
        )
    with pytest.raises(
        SCGOValidationError, match="does not accept adsorbate_definition"
    ):
        resolve_adsorbate_run_composition(
            system_type="gas_cluster",
            composition=["Pt", "Pt"],
            adsorbates=None,
            preset_adsorbate_definition=AdsorbateDefinition(
                core_symbols=["Pt"], adsorbate_symbols=["O", "H"]
            ),
            context="test",
        )

    ads_def, frags, full = resolve_adsorbate_run_composition(
        system_type="gas_cluster_adsorbate",
        composition=["Pt", "Pt"],
        adsorbates=oh,
        preset_adsorbate_definition=None,
        context="test",
    )
    assert ads_def is not None
    assert ads_def.core_symbols == ["Pt", "Pt"]
    assert ads_def.adsorbate_symbols == ["O", "H"]
    assert frags is not None and len(frags) == 1
    assert full == ["Pt", "Pt", "O", "H"]


def test_validate_adsorbate_definition_fragment_length_sum_mismatch() -> None:
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[1],
    )
    with pytest.raises(SCGOValidationError, match="must sum to the adsorbate"):
        validate_adsorbate_definition(
            system_type="gas_cluster_adsorbate",
            composition=["Pt", "Pt", "O", "H"],
            adsorbate_definition=ads,
            context="test",
        )


def test_validate_adsorbate_definition_rejects_metal_core_on_slab_search() -> None:
    """``surface_adsorbate`` searches the slab top layer; metal cores are invalid."""
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
    )
    with pytest.raises(SCGOValidationError, match="does not support metal cores"):
        validate_adsorbate_definition(
            system_type="surface_adsorbate",
            composition=["Pt", "Pt", "O", "H"],
            adsorbate_definition=ads,
            context="test",
        )


def test_validate_adsorbate_definition_allows_adsorbate_only_slab_search() -> None:
    validate_adsorbate_definition(
        system_type="surface_adsorbate",
        composition=["O", "H"],
        adsorbate_definition=AdsorbateDefinition(
            core_symbols=[],
            adsorbate_symbols=["O", "H"],
            adsorbate_fragment_lengths=[2],
        ),
        context="test",
    )


def test_validate_adsorbate_definition_keeps_gas_cores_valid() -> None:
    """Gas adsorbate types keep their metal-core support."""
    validate_adsorbate_definition(
        system_type="gas_cluster_adsorbate",
        composition=["Pt", "Pt", "O", "H"],
        adsorbate_definition=AdsorbateDefinition(
            core_symbols=["Pt", "Pt"],
            adsorbate_symbols=["O", "H"],
            adsorbate_fragment_lengths=[2],
        ),
        context="test",
    )
