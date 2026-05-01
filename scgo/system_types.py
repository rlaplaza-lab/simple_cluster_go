"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from ase import Atoms

from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
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

DepositionLayout = Literal["core_then_fragment"]


class AdsorbateDefinition(TypedDict, total=False):
    """Role and layout for ``*_adsorbate`` system types (gas or surface mobile region).

    Both ``core_symbols`` and ``adsorbate_symbols`` must be set (use ``[]`` for
    the side that is empty). They must form an **ordered** partition of the run
    ``composition`` such that
    ``composition == core_symbols + adsorbate_symbols`` (list equality, same
    length and order for the mobile atoms). The slab, if any, is *not* part of
    ``composition``.

    **Empty core** (``core_symbols=[]``): all mobile atoms are in
    ``adsorbate_symbols``.

    **Monolithic (default)**: one gas-phase cluster for the full mobile
    ``composition`` (or hierarchical rules do not apply). On surfaces, the
    cluster is then placed on the slab.

    **Hierarchical** (``deposition_layout="core_then_fragment"``): build a
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


AdsorbatesInput = Atoms | list[Atoms]


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
    """Ensure ``atoms`` mobile slice matches ``core_symbols + adsorbate_symbols`` in order."""
    cr = adsorbate_definition.get("core_symbols", [])
    ad = adsorbate_definition.get("adsorbate_symbols", [])
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
            f"Mobile region length mismatch: len(mobile)={len(mobile)} vs expected={len(expected)}"
        )

    if mobile != expected:

        def _head(syms: list[str], k: int = 12) -> str:
            h = syms[:k]
            return str(h) + ("..." if len(syms) > k else "")

        raise ValueError(
            f"Mobile symbols mismatch. Expected: {_head(expected)}; got: {_head(mobile)}."
        )


def validate_structure_for_system_type(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: float | None = None,
) -> None:
    """Apply system-type-specific structural validation.

    When ``adsorbate_definition`` is set for a ``*_adsorbate`` system type, the
    mobile region must match ``core_symbols + adsorbate_symbols`` in order (after
    the slab prefix for surface systems).

    Args:
        atoms: The Atoms object to validate
        system_type: The system type
        surface_config: Surface configuration (for surface systems)
        n_slab: Number of slab atoms (for surface systems)
        adsorbate_definition: Adsorbate definition (for adsorbate systems)
        connectivity_factor: Connectivity factor to use for cluster connectivity
            validation. If None, defaults to CONNECTIVITY_FACTOR from config.
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
                connectivity_factor=connectivity_factor,
            )
            if not ok:
                raise ValueError(msg)
    elif policy.has_adsorbate:
        # Use the provided connectivity_factor, or default to CONNECTIVITY_FACTOR
        cf = (
            connectivity_factor
            if connectivity_factor is not None
            else CONNECTIVITY_FACTOR
        )
        ok, msg = validate_combined_cluster_structure(atoms, connectivity_factor=cf)
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

    cr = adsorbate_definition.get("core_symbols", [])
    ad = adsorbate_definition.get("adsorbate_symbols", [])
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise ValueError(
            f"{prefix}adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )

    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]

    if not composition and not core_list and not ads_list:
        return core_list, ads_list
    if len(core_list) == 0 and len(ads_list) == 0:
        raise ValueError(
            f"{prefix}core_symbols and adsorbate_symbols cannot both be empty unless composition is also empty."
        )

    expected = core_list + ads_list
    if list(composition) != expected:
        raise ValueError(
            f"{prefix}composition must equal core_symbols + adsorbate_symbols. Got {composition}, expected {expected}."
        )

    if core_list and ads_list:
        core_set = set(core_list)
        ads_set = set(ads_list)
        if core_set & ads_set:
            raise ValueError(
                f"{prefix}core_symbols and adsorbate_symbols must be disjoint. Overlapping: {sorted(core_set & ads_set)}."
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

    if (
        adsorbate_definition.get("deposition_layout", "core_then_fragment")
        != "core_then_fragment"
    ):
        raise ValueError(
            "SCGO now supports hierarchical adsorbate initialization only. Set deposition_layout='core_then_fragment'."
        )

    core_list, _ads_list = validate_composition_against_adsorbate(
        composition, adsorbate_definition, context=context
    )

    fba = adsorbate_definition.get("fragment_bond_axis")
    if fba is not None and (
        not isinstance(fba, list)
        or len(fba) != 2
        or not all(isinstance(x, int) for x in fba)
    ):
        raise ValueError(
            f"adsorbate_definition['fragment_bond_axis'] must be a list of two int indices or omitted, got {fba!r}"
        )

    ai = adsorbate_definition.get("fragment_anchor_index")
    if ai is not None and not isinstance(ai, int):
        raise ValueError(
            f"adsorbate_definition['fragment_anchor_index'] must be int or omitted, got {ai!r}"
        )


def normalize_adsorbates_input(
    adsorbates: AdsorbatesInput | None, *, context: str
) -> list[Atoms]:
    prefix = f"{context}: " if context else ""
    if adsorbates is None:
        raise ValueError(f"{prefix}adsorbates is required for adsorbate system types.")

    items = adsorbates if isinstance(adsorbates, list) else [adsorbates]
    out: list[Atoms] = []

    for idx, item in enumerate(items):
        if not isinstance(item, Atoms):
            raise TypeError(
                f"{prefix}adsorbates[{idx}] must be ase.Atoms, got {type(item).__name__}."
            )
        if len(item) == 0:
            raise ValueError(f"{prefix}adsorbates[{idx}] must not be empty.")
        out.append(item.copy())

    if not out:
        raise ValueError(f"{prefix}adsorbates must contain at least one fragment.")
    return out


def flatten_adsorbate_symbols(adsorbates: list[Atoms]) -> list[str]:
    symbols: list[str] = []
    for frag in adsorbates:
        symbols.extend([str(s) for s in frag.get_chemical_symbols()])
    return symbols


def combine_adsorbates_to_template(adsorbates: list[Atoms]) -> Atoms:
    if not adsorbates:
        raise ValueError("adsorbates must contain at least one fragment")
    combined = adsorbates[0].copy()
    for frag in adsorbates[1:]:
        combined += frag.copy()
    return combined


def build_adsorbate_definition_from_inputs(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbates: AdsorbatesInput | None,
    context: str,
) -> tuple[AdsorbateDefinition | None, Atoms | None, list[str]]:
    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        if adsorbates is not None:
            raise ValueError(
                f"{context} does not accept adsorbates for system_type={system_type!r}."
            )
        return None, None, list(composition)
    core_list = [str(s) for s in composition]
    fragments = normalize_adsorbates_input(adsorbates, context=context)
    ads_list = flatten_adsorbate_symbols(fragments)
    full_mobile_composition = list(core_list) + list(ads_list)
    ads_def: AdsorbateDefinition = {
        "core_symbols": core_list,
        "adsorbate_symbols": ads_list,
        "deposition_layout": "core_then_fragment",
    }
    validate_adsorbate_definition(
        system_type=system_type,
        composition=full_mobile_composition,
        adsorbate_definition=ads_def,
        context=context,
    )
    return ads_def, combine_adsorbates_to_template(fragments), full_mobile_composition
