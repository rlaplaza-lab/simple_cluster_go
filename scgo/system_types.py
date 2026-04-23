"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, TypedDict

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


class AdsorbateDefinition(TypedDict, total=False):
    """Optional explicit role definition for adsorbate workflows."""

    adsorbate_symbols: list[str]
    core_symbols: list[str]


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


def validate_structure_for_system_type(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
) -> None:
    """Apply system-type-specific structural validation."""
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

    adsorbate_symbols_raw = adsorbate_definition.get("adsorbate_symbols")
    if not isinstance(adsorbate_symbols_raw, list) or not adsorbate_symbols_raw:
        raise ValueError(
            "adsorbate_definition['adsorbate_symbols'] must be a non-empty list[str]."
        )
    adsorbate_symbols = [str(s) for s in adsorbate_symbols_raw]

    composition_counts = Counter(composition)
    adsorbate_counts = Counter(adsorbate_symbols)
    missing = {
        symbol: count
        for symbol, count in adsorbate_counts.items()
        if composition_counts.get(symbol, 0) < count
    }
    if missing:
        raise ValueError(
            "adsorbate_definition['adsorbate_symbols'] exceeds available composition "
            f"counts. Missing requirements: {missing}. "
            f"Composition={dict(composition_counts)}"
        )

    core_symbols_raw = adsorbate_definition.get("core_symbols")
    if core_symbols_raw is None:
        return
    if not isinstance(core_symbols_raw, list) or not core_symbols_raw:
        raise ValueError(
            "adsorbate_definition['core_symbols'] must be a non-empty list[str] when set."
        )
    core_symbols = {str(s) for s in core_symbols_raw}
    overlap = core_symbols.intersection(set(adsorbate_symbols))
    if overlap:
        raise ValueError(
            "adsorbate_definition core_symbols and adsorbate_symbols must be disjoint; "
            f"overlap={sorted(overlap)}"
        )
