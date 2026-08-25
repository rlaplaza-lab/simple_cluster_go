"""Type-aware uniqueness geometry for structure de-duplication.

Resolves *what* to compare (role blocks) and *how strongly* each part counts
(weights, tolerances) for a given system type:

- ``surface`` / ``surface_adsorbate`` (slab-as-target): the mobile top layers
  are the region of interest and keep full weight.
- ``surface_cluster*`` (supported deposits): relaxed support layers are
  included in the fingerprint at a reduced default weight
  (:data:`scgo.constants.DEFAULT_SUPPORTED_SLAB_WEIGHT`) so near-rigid lattice
  motion cannot dilute deposit/adsorbate discrimination, and the geometry
  gates default to tighter values (:data:`SUPPORTED_CLUSTER_COMPARATOR_TOL`,
  :data:`SUPPORTED_CLUSTER_PAIR_COR_MAX`) because block-aware fingerprints no
  longer dilute genuine differences.
- Gas-phase clusters compare their deposit/adsorbate blocks directly.

Explicit user knobs always win: ``comparator_component_weights`` /
``comparator_cross_weight`` merge over the defaults, and tightened tolerances
apply only while the effective value still equals the generic default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_PAIR_COR_MAX,
    DEFAULT_SUPPORTED_SLAB_WEIGHT,
    SUPPORTED_CLUSTER_COMPARATOR_TOL,
    SUPPORTED_CLUSTER_PAIR_COR_MAX,
)
from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.layers import _layer_indices_by_clustering
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.system_types.composition import as_adsorbate_definition
from scgo.system_types.policy import SystemType, get_system_policy
from scgo.utils.comparators import (
    ComparatorBlock,
    ComparatorBlocks,
    PureInteratomicDistanceComparator,
    UniquenessSettings,
    _validate_block_role,
)
from scgo.utils.logging import get_logger

__all__ = [
    "ResolvedUniquenessGeometry",
    "resolve_uniqueness_geometry",
]

logger = get_logger(__name__)

_SUPPORTED_CLUSTER_TYPES = frozenset({"surface_cluster", "surface_cluster_adsorbate"})


@dataclass(frozen=True)
class ResolvedUniquenessGeometry:
    """Everything a uniqueness comparator needs, resolved for one system type."""

    settings: UniquenessSettings
    blocks: ComparatorBlocks | None
    component_weights: dict[str, float]
    cross_weight: float

    def build_comparator(
        self,
        *,
        n_top: int,
        mic: bool = False,
    ) -> PureInteratomicDistanceComparator:
        """Construct a comparator with this geometry's blocks and merged weights."""
        return PureInteratomicDistanceComparator(
            n_top=n_top,
            tol=self.settings.comparator_tol,
            pair_cor_max=self.settings.comparator_pair_cor_max,
            mic=mic,
            blocks=self.blocks,
            component_weights=self.component_weights if self.blocks else None,
            cross_weight=self.cross_weight,
        )


def _mobile_support_indices(config: SurfaceSystemConfig) -> tuple[int, ...]:
    """Indices of relaxable support atoms, mirroring ``attach_slab_constraints``."""
    slab = config.slab
    n_slab = len(slab)
    positions = np.asarray(slab.get_positions())
    axis = int(config.surface_normal_axis)

    if config.n_relax_top_slab_layers is not None:
        mobile = _layer_indices_by_clustering(
            positions,
            axis,
            n_layers=int(config.n_relax_top_slab_layers),
            from_top=True,
        )
    elif config.n_fix_bottom_slab_layers is not None:
        fixed = _layer_indices_by_clustering(
            positions,
            axis,
            n_layers=int(config.n_fix_bottom_slab_layers),
            from_top=False,
        )
        mobile = set(range(n_slab)) - fixed
    else:
        # fix_all_slab_atoms=False without a layer policy freezes nothing.
        mobile = set(range(n_slab))

    return tuple(sorted(int(i) for i in mobile))


def _merge_component_weights(
    defaults: dict[str, float],
    overrides: dict[str, float] | None,
    present_roles: set[str],
) -> dict[str, float]:
    merged = {role: w for role, w in defaults.items() if role in present_roles}
    for role, weight in (overrides or {}).items():
        _validate_block_role(role)
        weight_f = float(weight)
        if not np.isfinite(weight_f) or weight_f < 0.0:
            raise SCGOValidationError(
                f"comparator_component_weights[{role!r}] must be a non-negative "
                f"finite float, got {weight!r}."
            )
        if role not in present_roles:
            raise SCGOValidationError(
                f"comparator_component_weights[{role!r}] given but this system "
                f"has no {role!r} block (present: {sorted(present_roles)})."
            )
        merged[role] = weight_f
    return merged


def resolve_uniqueness_geometry(
    *,
    system_type: SystemType,
    n_atoms: int,
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: Any = None,
    counts: tuple[int, int] | None = None,
    settings: UniquenessSettings | None = None,
) -> ResolvedUniquenessGeometry:
    """Resolve role blocks, weights, and tolerances for one system type.

    Args:
        system_type: Resolved system type.
        n_atoms: Total atom count of the structures being compared. Must match
            the stored layout: ``[slab][deposit][adsorbates]``, or
            ``[fixed bottom][mobile top layers][adsorbates]`` for slab-as-target
            types.
        surface_config: Required for every surface system type.
        adsorbate_definition: Adsorbate definition used to split the
            deposit/adsorbate suffixes (may be ``None`` when ``counts`` given).
        counts: Optional ``(n_deposit, n_adsorbate)`` override for callers that
            already know the split (e.g. TS mobile dims).
        settings: User geometry settings; ``component_weights`` /
            ``cross_weight`` override the type-aware defaults, and tightened
            supported-cluster tolerances apply only when the tolerance values
            still equal the generic defaults.

    Returns:
        :class:`ResolvedUniquenessGeometry`. Its ``blocks`` is ``None`` when
        fewer than two roles exist (the legacy trailing window is exactly right).
    """
    policy = get_system_policy(system_type)
    user_settings = settings or UniquenessSettings()

    if policy.slab_is_search_target:
        ranges = _slab_target_ranges(
            system_type=system_type,
            surface_config=surface_config,
            adsorbate_definition=adsorbate_definition,
            counts=counts,
            n_atoms=n_atoms,
        )
        defaults: dict[str, float] = {"mobile_slab": 1.0, "adsorbate": 1.0}
        tighten = False
    elif policy.uses_surface:
        ranges, defaults = _supported_ranges(
            system_type=system_type,
            surface_config=surface_config,
            adsorbate_definition=adsorbate_definition,
            counts=counts,
            n_atoms=n_atoms,
        )
        tighten = system_type in _SUPPORTED_CLUSTER_TYPES
    else:
        ranges = _gas_ranges(
            system_type=system_type,
            adsorbate_definition=adsorbate_definition,
            counts=counts,
            n_atoms=n_atoms,
        )
        defaults = {"deposit": 1.0, "adsorbate": 1.0}
        tighten = False

    present_roles = {role for role, _idx in ranges}
    blocks: ComparatorBlocks | None = None
    if len(present_roles) >= 2:
        blocks = ComparatorBlocks(
            blocks=tuple(
                ComparatorBlock(role=role, indices=idx) for role, idx in ranges
            )
        )

    resolved_tol = float(user_settings.comparator_tol)
    resolved_pair_cor = float(user_settings.comparator_pair_cor_max)
    tightened = False
    if tighten and blocks is not None:
        if resolved_tol == DEFAULT_COMPARATOR_TOL:
            resolved_tol = SUPPORTED_CLUSTER_COMPARATOR_TOL
            tightened = True
        if resolved_pair_cor == DEFAULT_PAIR_COR_MAX:
            resolved_pair_cor = SUPPORTED_CLUSTER_PAIR_COR_MAX
            tightened = True
    if tightened:
        logger.info(
            "Tighter %s uniqueness gates applied: tol=%.4g pair_cor_max=%.3g "
            "(block-aware fingerprints)",
            system_type,
            resolved_tol,
            resolved_pair_cor,
        )

    weights = _merge_component_weights(
        defaults, user_settings.component_weights, present_roles
    )
    cross_weight = float(user_settings.cross_weight)

    return ResolvedUniquenessGeometry(
        settings=UniquenessSettings(
            comparator_tol=resolved_tol,
            comparator_pair_cor_max=resolved_pair_cor,
            component_weights=user_settings.component_weights,
            cross_weight=cross_weight,
        ),
        blocks=blocks,
        component_weights=weights,
        cross_weight=cross_weight,
    )


def _slab_target_ranges(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: Any,
    counts: tuple[int, int] | None,
    n_atoms: int,
) -> list[tuple[str, tuple[int, ...]]]:
    if surface_config is None:
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config."
        )
    # Idempotent: identity when the slab is already [fixed|mobile] ordered.
    _, partition = prepare_slab_search_surface_config(surface_config)
    n_slab = len(surface_config.slab)
    n_fixed = int(partition.n_fixed)
    n_ads = _adsorbate_count(system_type, adsorbate_definition, counts)

    ranges: list[tuple[str, tuple[int, ...]]] = []
    if n_fixed < n_slab:
        ranges.append(("mobile_slab", tuple(range(n_fixed, n_slab))))
    if n_ads > 0:
        if n_slab + n_ads != n_atoms:
            raise SCGOValidationError(
                f"system_type={system_type!r}: expected [slab][adsorbates] "
                f"layout (n_slab={n_slab}, n_ads={n_ads}, n_atoms={n_atoms})."
            )
        ranges.append(("adsorbate", tuple(range(n_slab, n_atoms))))
    if not ranges:
        raise SCGOValidationError(
            f"system_type={system_type!r} resolved an empty uniqueness partition."
        )
    return ranges


def _supported_ranges(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: Any,
    counts: tuple[int, int] | None,
    n_atoms: int,
) -> tuple[list[tuple[str, tuple[int, ...]]], dict[str, float]]:
    if surface_config is None:
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config."
        )
    n_slab = len(surface_config.slab)
    n_ads = _adsorbate_count(system_type, adsorbate_definition, counts)
    n_deposit = n_atoms - n_slab - n_ads
    if n_deposit < 0 or (n_deposit == 0 and n_ads == 0):
        raise SCGOValidationError(
            f"system_type={system_type!r}: no mobile atoms after the slab "
            f"(n_atoms={n_atoms}, n_slab={n_slab}, n_ads={n_ads})."
        )

    defaults: dict[str, float] = {"deposit": 1.0, "adsorbate": 1.0}
    ranges: list[tuple[str, tuple[int, ...]]] = []
    if n_deposit > 0:
        ranges.append(("deposit", tuple(range(n_slab, n_slab + n_deposit))))
    if n_ads > 0:
        first_ads = n_slab + n_deposit
        ranges.append(("adsorbate", tuple(range(first_ads, n_atoms))))

    if not bool(surface_config.fix_all_slab_atoms):
        support = _mobile_support_indices(surface_config)
        if support:
            ranges.insert(0, ("mobile_slab", support))
            defaults["mobile_slab"] = DEFAULT_SUPPORTED_SLAB_WEIGHT
    return ranges, defaults


def _gas_ranges(
    *,
    system_type: SystemType,
    adsorbate_definition: Any,
    counts: tuple[int, int] | None,
    n_atoms: int,
) -> list[tuple[str, tuple[int, ...]]]:
    n_ads = _adsorbate_count(system_type, adsorbate_definition, counts)
    n_deposit = n_atoms - n_ads
    if n_deposit < 0 or (n_deposit == 0 and n_ads == 0):
        raise SCGOValidationError(
            f"No mobile atoms in gas-phase structure (n_atoms={n_atoms}, "
            f"n_ads={n_ads})."
        )
    ranges = []
    if n_deposit > 0:
        ranges.append(("deposit", tuple(range(n_deposit))))
    if n_ads > 0:
        ranges.append(("adsorbate", tuple(range(n_deposit, n_atoms))))
    return ranges


def _adsorbate_count(
    system_type: SystemType,
    adsorbate_definition: Any,
    counts: tuple[int, int] | None,
) -> int:
    if counts is not None:
        return int(counts[1])
    if not get_system_policy(system_type).has_adsorbate:
        return 0
    ads_def = as_adsorbate_definition(adsorbate_definition)
    if ads_def is None:
        raise SCGOValidationError(
            "Adsorbate system types require adsorbate_definition (or explicit "
            "counts) to resolve the uniqueness geometry."
        )
    return len(list(ads_def.adsorbate_symbols))
