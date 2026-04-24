"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from ase import Atoms

from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.validation import (
    validate_supported_cluster_deposit,
    validate_surface_config_slab_prefix,
)

SystemType = Literal[
    "gas_cluster",
    "surface_cluster",
    "gas_cluster_adsorbate",
    "surface_cluster_adsorbate",
]

DepositionLayout = Literal["monolithic", "core_then_fragment"]


class AdsorbateDefinition(TypedDict, total=False):
    """Role and layout for ``*_adsorbate`` system types (gas or surface mobile region).

    Both ``core_symbols`` and ``adsorbate_symbols`` must be set (use ``[]`` for
    the side that is empty). They must form an **ordered** partition of the run
    ``composition`` such that
    ``composition == core_symbols + adsorbate_symbols`` (list equality, same
    length and order for the mobile atoms). The slab, if any, is *not* part of
    ``composition``.

    **Empty core** (``core_symbols=[]``): all mobile atoms are in
    ``adsorbate_symbols``. **Hierarchical** (``deposition_layout=core_then_fragment``)
    requires a non-empty core; use monolithic for pure-molecular seeds.

    **Monolithic (default)**: one gas-phase cluster for the full mobile
    ``composition`` (or hierarchical rules do not apply). On surfaces, the
    cluster is then placed on the slab.

    **Hierarchical** (``deposition_layout=\"core_then_fragment\"``): build a
    core cluster, place rigid fragment(s) with
    :func:`scgo.cluster_adsorbate.place_fragment_on_cluster`, then (for
    surface) deposit. Requires a fragment template or
    :func:`scgo.surface.fragment_templates.build_default_fragment_template`
    for supported symbol lists.
    """

    core_symbols: list[str]
    adsorbate_symbols: list[str]
    deposition_layout: DepositionLayout
    fragment_anchor_index: NotRequired[int]
    fragment_bond_axis: NotRequired[list[int]]


@dataclass(frozen=True)
class SystemPolicy:
    """Behavior flags for a concrete system type."""

    system_type: SystemType
    uses_surface: bool
    has_adsorbate: bool
    requires_slab_prefix_validation: bool
    needs_supported_deposit_validation: bool
    neb_force_mic: bool
    neb_disable_alignment: bool
    constrain_adsorbate_moves: bool
    adsorbate_move_scale: float
    allow_composition_permutations: bool


SYSTEM_TYPE_POLICIES: dict[SystemType, SystemPolicy] = {
    "gas_cluster": SystemPolicy(
        system_type="gas_cluster",
        uses_surface=False,
        has_adsorbate=False,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
        neb_force_mic=False,
        neb_disable_alignment=False,
        constrain_adsorbate_moves=False,
        adsorbate_move_scale=1.0,
        allow_composition_permutations=True,
    ),
    "surface_cluster": SystemPolicy(
        system_type="surface_cluster",
        uses_surface=True,
        has_adsorbate=False,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
        neb_force_mic=True,
        neb_disable_alignment=True,
        constrain_adsorbate_moves=False,
        adsorbate_move_scale=1.0,
        allow_composition_permutations=True,
    ),
    "gas_cluster_adsorbate": SystemPolicy(
        system_type="gas_cluster_adsorbate",
        uses_surface=False,
        has_adsorbate=True,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
        neb_force_mic=False,
        neb_disable_alignment=False,
        constrain_adsorbate_moves=True,
        adsorbate_move_scale=0.6,
        allow_composition_permutations=False,
    ),
    "surface_cluster_adsorbate": SystemPolicy(
        system_type="surface_cluster_adsorbate",
        uses_surface=True,
        has_adsorbate=True,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
        neb_force_mic=True,
        neb_disable_alignment=True,
        constrain_adsorbate_moves=True,
        adsorbate_move_scale=0.6,
        allow_composition_permutations=False,
    ),
}


def get_system_policy(system_type: SystemType) -> SystemPolicy:
    """Return centralized behavior policy for one explicit system type."""
    return SYSTEM_TYPE_POLICIES[system_type]


def validate_system_type_settings(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
) -> None:
    """Validate system-type companion settings."""
    surface_type = get_system_policy(system_type).uses_surface
    if surface_type and surface_config is None:
        raise ValueError(
            f"system_type={system_type!r} requires surface_config to be provided."
        )
    if not surface_type and surface_config is not None:
        raise ValueError(
            f"system_type={system_type!r} does not allow surface_config. "
            "Use surface_cluster or surface_cluster_adsorbate."
        )


def uses_surface(system_type: SystemType) -> bool:
    return get_system_policy(system_type).uses_surface


def validate_mobile_symbols_match_adsorbate_definition(
    atoms: Atoms,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition,
) -> None:
    """Ensure ``atoms`` mobile slice matches ``core_symbols + adsorbate_symbols`` in order.

    The mobile region is ``atoms[n_slab:]`` (entire system for gas ``n_slab=0``;
    after the slab prefix for supported systems). Used at runtime so structures
    cannot pass GA/TS with scrambled core/adsorbate ordering when a definition
    is provided.
    """
    if "core_symbols" not in adsorbate_definition:
        raise ValueError(
            "adsorbate_definition['core_symbols'] is required for mobile symbol "
            "validation (use [] for molecular-only mobile region)."
        )
    if "adsorbate_symbols" not in adsorbate_definition:
        raise ValueError(
            "adsorbate_definition['adsorbate_symbols'] is required for mobile symbol "
            "validation (use [] for core-only mobile region)."
        )
    cr = adsorbate_definition["core_symbols"]
    ad = adsorbate_definition["adsorbate_symbols"]
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise ValueError(
            "adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )
    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]
    expected = core_list + ads_list
    n = len(atoms)
    if n_slab < 0 or n_slab > n:
        raise ValueError(
            f"Invalid n_slab={n_slab} for len(atoms)={n} in mobile symbol validation."
        )
    mobile = atoms.get_chemical_symbols()[n_slab:]
    if len(mobile) != len(expected):
        raise ValueError(
            "Mobile region length does not match adsorbate definition: "
            f"len(mobile)={len(mobile)} vs len(core_symbols)+len(adsorbate_symbols)="
            f"{len(expected)} (n_slab={n_slab}, len(atoms)={n})."
        )
    if mobile != list(expected):
        def _head(syms: list[str], k: int = 12) -> str:
            h = syms[:k]
            return str(h) + ("..." if len(syms) > k else "")

        raise ValueError(
            "Mobile chemical symbols do not match adsorbate_definition ordered partition "
            f"(core then adsorbate). Expected: {_head(expected)}; got: {_head(mobile)}."
        )


def validate_structure_for_system_type(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> None:
    """Apply system-type-specific structural validation.

    When ``adsorbate_definition`` is set for a ``*_adsorbate`` system type, the
    mobile region must match ``core_symbols + adsorbate_symbols`` in order (after
    the slab prefix for surface systems).
    """
    policy = get_system_policy(system_type)
    if policy.uses_surface:
        if surface_config is None:
            raise ValueError(
                "surface_config is required for surface system validation."
            )
        if policy.requires_slab_prefix_validation:
            validate_surface_config_slab_prefix(atoms, surface_config)
        if policy.needs_supported_deposit_validation:
            n_slab_eff = int(n_slab if n_slab is not None else len(surface_config.slab))
            ok, msg = validate_supported_cluster_deposit(
                atoms,
                n_slab_eff,
                surface_normal_axis=surface_config.surface_normal_axis,
                use_mic=bool(surface_config.comparator_use_mic),
            )
            if not ok:
                raise ValueError(msg)
    elif policy.has_adsorbate:
        ok, msg = validate_combined_cluster_structure(atoms)
        if not ok:
            raise ValueError(msg)

    if policy.has_adsorbate and adsorbate_definition is not None:
        if policy.uses_surface:
            if surface_config is None:
                raise ValueError(
                    "surface_config is required for surface adsorbate mobile symbol validation."
                )
            n_mobile_slab = int(
                n_slab if n_slab is not None else len(surface_config.slab)
            )
        else:
            n_mobile_slab = 0
        validate_mobile_symbols_match_adsorbate_definition(
            atoms, n_mobile_slab, adsorbate_definition
        )


def validate_composition_against_adsorbate(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], list[str]]:
    """Check ordered partition and return ``(core_list, ads_list)`` as ``list[str]``.

    Both ``core_symbols`` and ``adsorbate_symbols`` must be present (use ``[]`` if
    empty). The run ``composition`` must equal ``core_symbols + adsorbate_symbols``
    in order.

    Raises:
        ValueError: If keys are missing, both sides are empty for non-empty
            composition, or the ordered partition does not match ``composition``.
    """
    prefix = f"{context}: " if context else ""
    if "core_symbols" not in adsorbate_definition:
        raise ValueError(
            f"{prefix}adsorbate_definition['core_symbols'] is required (use [] for "
            "molecular-only mobile region)."
        )
    if "adsorbate_symbols" not in adsorbate_definition:
        raise ValueError(
            f"{prefix}adsorbate_definition['adsorbate_symbols'] is required (use [] for "
            "a core-only mobile region)."
        )
    cr = adsorbate_definition["core_symbols"]
    ad = adsorbate_definition["adsorbate_symbols"]
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise ValueError(
            f"{prefix}adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] "
            "must be lists of str."
        )
    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]
    if not composition and not core_list and not ads_list:
        return core_list, ads_list
    if len(core_list) == 0 and len(ads_list) == 0:
        raise ValueError(
            f"{prefix}core_symbols and adsorbate_symbols cannot both be empty unless "
            "composition is also empty."
        )
    if list(composition) != core_list + ads_list:
        raise ValueError(
            f"{prefix}Run composition must equal core_symbols + adsorbate_symbols in "
            f"order. composition={list(composition)!r}, expected "
            f"core+ads={core_list + ads_list!r}."
        )
    if len(core_list) and len(ads_list):
        core_set = set(Counter(core_list).keys())
        ads_set = set(Counter(ads_list).keys())
        if core_set & ads_set:
            raise ValueError(
                f"{prefix}adsorbate_definition core_symbols and adsorbate_symbols use "
                "overlapping element types while both non-empty; lists must be disjoint "
                f"by atom role (overlapping symbols: {sorted(core_set & ads_set)})."
            )
    return core_list, ads_list


def validate_adsorbate_definition(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    context: str,
) -> None:
    """Validate explicit adsorbate role definition for high-level runners."""
    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        if adsorbate_definition is not None:
            raise ValueError(
                f"{context} received adsorbate_definition for non-adsorbate "
                f"system_type={system_type!r}."
            )
        return

    if adsorbate_definition is None:
        raise ValueError(
            f"{context} requires adsorbate_definition when system_type={system_type!r}."
        )

    layout_raw = adsorbate_definition.get("deposition_layout", "monolithic")
    if layout_raw not in ("monolithic", "core_then_fragment"):
        raise ValueError(
            "adsorbate_definition['deposition_layout'] must be 'monolithic' or "
            f"'core_then_fragment', got {layout_raw!r}"
        )
    layout: DepositionLayout = layout_raw  # type: ignore[assignment]

    core_list, _ads_list = validate_composition_against_adsorbate(
        composition, adsorbate_definition, context=context
    )

    if layout == "core_then_fragment" and len(core_list) == 0:
        raise ValueError(
            f"{context}: deposition_layout='core_then_fragment' requires a non-empty "
            "adsorbate_definition['core_symbols'] (fragment placement needs a core). "
            "Use deposition_layout='monolithic' for molecular-only mobile (empty core) "
            "or pick system_type without adsorbate roles."
        )

    fba = adsorbate_definition.get("fragment_bond_axis")
    if fba is not None:
        if not isinstance(fba, list) or len(fba) != 2:
            raise ValueError(
                "adsorbate_definition['fragment_bond_axis'] must be a list of two int "
                f"indices or omitted, got {fba!r}"
            )
        if not all(isinstance(x, int) for x in fba):
            raise ValueError("adsorbate_definition['fragment_bond_axis'] must be int indices")
    ai = adsorbate_definition.get("fragment_anchor_index")
    if ai is not None and not isinstance(ai, int):
        raise ValueError(
            "adsorbate_definition['fragment_anchor_index'] must be int or omitted, "
            f"got {ai!r}"
        )
