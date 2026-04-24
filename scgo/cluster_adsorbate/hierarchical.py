"""Hierarchical (core + rigid fragment) gas-phase cluster building for GA seeds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ase import Atoms

if TYPE_CHECKING:
    from numpy.random import Generator

    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
    from scgo.system_types import AdsorbateDefinition


def reorder_cluster_to_composition(
    cluster: Atoms, composition: Sequence[str]
) -> Atoms:
    """Reorder generated cluster atoms to match requested symbol sequence."""
    desired = list(composition)
    current = cluster.get_chemical_symbols()
    if current == desired:
        return cluster

    by_symbol: dict[str, list[int]] = {}
    for idx, sym in enumerate(current):
        by_symbol.setdefault(sym, []).append(idx)

    selection: list[int] = []
    for sym in desired:
        matching = by_symbol.get(sym)
        if not matching:
            raise ValueError(
                "Generated cluster symbols do not match requested composition."
            )
        selection.append(matching.pop(0))
    return cluster[selection].copy()


def build_hierarchical_core_fragment_cluster(
    full_composition: Sequence[str],
    adsorbate_definition: "AdsorbateDefinition",
    rng: "Generator",
    previous_search_glob: str,
    fragment_template: Atoms | None,
    cluster_adsorbate_config: "ClusterAdsorbateConfig | None",
    *,
    cluster_init_vacuum: float = 8.0,
    init_mode: str = "smart",
    max_placement_attempts: int = 200,
) -> Atoms | None:
    """Build core cluster, place fragment, return gas-phase core+fragment (no slab).

    Requires a non-empty ``core_symbols`` in ``adsorbate_definition`` and
    :func:`scgo.system_types.validate_adsorbate_definition` to have been satisfied
    for ``deposition_layout=core_then_fragment`` (and ordered ``composition``).
    """
    from scgo.cluster_adsorbate.combine import combine_core_adsorbate
    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
    from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
    from scgo.initialization import create_initial_cluster
    from scgo.surface.fragment_templates import build_default_fragment_template

    core_list = [str(s) for s in adsorbate_definition["core_symbols"]]
    ads_list = [str(s) for s in adsorbate_definition["adsorbate_symbols"]]
    if not core_list:
        raise ValueError(
            "build_hierarchical_core_fragment_cluster requires non-empty core_symbols; "
            "use monolithic init when the mobile region is molecular-only (empty core)."
        )
    ca = cluster_adsorbate_config or ClusterAdsorbateConfig()
    expected_mobile = list(core_list) + list(ads_list)
    if list(full_composition) != expected_mobile:
        raise ValueError(
            "Hierarchical init requires the mobile composition to be "
            "core_symbols (in order) then adsorbate_symbols (in order). "
            f"Got {list(full_composition)!r}, expected {expected_mobile!r}."
        )

    tmpl = fragment_template.copy() if fragment_template is not None else None
    if tmpl is None:
        tmpl = build_default_fragment_template(ads_list)
    if tmpl is None:
        raise ValueError(
            "No adsorbate_fragment_template provided and no default template for "
            f"adsorbate_symbols={ads_list!r}. Pass adsorbate_fragment_template=."
        )
    if list(tmpl.get_chemical_symbols()) != ads_list:
        raise ValueError(
            "adsorbate_fragment_template symbols must match "
            f"adsorbate_definition['adsorbate_symbols'] in order, got {tmpl.get_chemical_symbols()!r} "
            f"vs {ads_list!r}"
        )
    if len(ads_list) != len(tmpl):
        raise ValueError("len(adsorbate_fragment_template) must match adsorbate_symbols")
    anchor = int(adsorbate_definition.get("fragment_anchor_index", 0))
    fba = adsorbate_definition.get("fragment_bond_axis")
    bond_axis: tuple[int, int] | None = None
    if fba is not None:
        bond_axis = (int(fba[0]), int(fba[1]))
    for _ in range(max_placement_attempts):
        core = create_initial_cluster(
            list(core_list),
            vacuum=cluster_init_vacuum,
            rng=rng,
            previous_search_glob=previous_search_glob,
            mode=init_mode,
        )
        core = reorder_cluster_to_composition(core, core_list)
        frag = place_fragment_on_cluster(
            core, tmpl, rng, ca, anchor_index=anchor, bond_axis=bond_axis
        )
        if frag is None:
            continue
        return combine_core_adsorbate(core, frag)
    return None
