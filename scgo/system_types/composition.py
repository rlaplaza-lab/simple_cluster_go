"""Adsorbate role partitions and mobile-composition reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types.policy import SystemType, get_system_policy
from scgo.utils.composition import get_composition_counts

AdsorbatesInput = Atoms | list[Atoms]
AdsorbateFragmentInput = Atoms | list[Atoms]


@dataclass(frozen=True)
class AdsorbateDefinition:
    """Role and layout for ``*_adsorbate`` system types (gas or surface mobile region).

    Both ``core_symbols`` and ``adsorbate_symbols`` must be set (use ``[]`` for
    the side that is empty). They must form an **ordered** partition of the run
    ``composition`` such that
    ``composition == core_symbols + adsorbate_symbols`` (list equality, same
    length and order for the mobile atoms). Element symbols may appear in both
    lists (e.g. oxide cores with O-containing adsorbates). The slab, if any, is *not* part of
    ``composition``.

    **Empty core** (``core_symbols=[]``): all mobile atoms are in
    ``adsorbate_symbols``.

    Build a core cluster, place rigid fragment(s) with
    :func:`~scgo.cluster_adsorbate.placement.place_fragment_on_cluster` (one site per fragment),
    then (for surface) deposit. Pass ``adsorbates`` as ``list[Atoms]`` with one
    entry per fragment at the runner API; ``adsorbate_fragment_lengths`` must match.
    """

    core_symbols: list[str] = field(default_factory=list)
    adsorbate_symbols: list[str] = field(default_factory=list)
    adsorbate_fragment_lengths: list[int] = field(default_factory=list)
    fragment_anchor_index: int | None = None
    fragment_bond_axis: list[int] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdsorbateDefinition:
        """Build a validated :class:`AdsorbateDefinition` from a plain mapping.

        Used at the runtime boundary where ``adsorbate_definition`` arrives as a
        user-supplied dict (e.g. inside ``params``). Missing keys default to an
        empty list / ``None`` so this mirrors the previous ``total=False``
        TypedDict semantics.
        """
        core = data.get("core_symbols")
        ads = data.get("adsorbate_symbols")
        lengths = data.get("adsorbate_fragment_lengths")
        anchor = data.get("fragment_anchor_index")
        bond_axis = data.get("fragment_bond_axis")
        if core is not None and not isinstance(core, list):
            raise SCGOValidationError(
                "adsorbate_definition['core_symbols'] must be a list[str]."
            )
        if ads is not None and not isinstance(ads, list):
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_symbols'] must be a list[str]."
            )
        if lengths is not None and not (
            isinstance(lengths, list) and all(isinstance(x, int) for x in lengths)
        ):
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_fragment_lengths'] must be a list[int]."
            )
        if anchor is not None and not isinstance(anchor, int):
            raise SCGOValidationError(
                "adsorbate_definition['fragment_anchor_index'] must be int or omitted."
            )
        if bond_axis is not None and not (
            isinstance(bond_axis, list)
            and len(bond_axis) == 2
            and all(isinstance(x, int) for x in bond_axis)
        ):
            raise SCGOValidationError(
                "adsorbate_definition['fragment_bond_axis'] must be a list of two int indices."
            )
        return cls(
            core_symbols=[str(s) for s in (core or [])],
            adsorbate_symbols=[str(s) for s in (ads or [])],
            adsorbate_fragment_lengths=[int(x) for x in (lengths or [])],
            fragment_anchor_index=anchor,
            fragment_bond_axis=list(bond_axis) if bond_axis is not None else None,
        )

    @property
    def n_core(self) -> int:
        return len(self.core_symbols)

    @property
    def n_adsorbate(self) -> int:
        return len(self.adsorbate_symbols)

    @property
    def effective_fragment_lengths(self) -> list[int]:
        """Fragment lengths, defaulting to one fragment spanning all adsorbate atoms."""
        lengths = list(self.adsorbate_fragment_lengths)
        if lengths:
            return lengths
        return [len(self.adsorbate_symbols)] if self.adsorbate_symbols else []


def as_adsorbate_definition(
    obj: Any,
) -> AdsorbateDefinition | None:
    """Coerce a boundary value into an :class:`AdsorbateDefinition` or ``None``.

    Accepts ``None``, a plain dict (via :meth:`AdsorbateDefinition.from_dict`), or
    an already-built :class:`AdsorbateDefinition`.
    """
    if obj is None:
        return None
    if isinstance(obj, AdsorbateDefinition):
        return obj
    if isinstance(obj, dict):
        return AdsorbateDefinition.from_dict(obj)
    raise SCGOValidationError(
        f"adsorbate_definition must be a dict or AdsorbateDefinition, got "
        f"{type(obj).__name__}."
    )


def _reject_adsorbate_inputs_for_non_adsorbate(
    *,
    system_type: SystemType,
    adsorbates: AdsorbatesInput | None,
    adsorbate_definition: AdsorbateDefinition | None,
    context: str,
) -> None:
    if adsorbates is not None:
        raise SCGOValidationError(
            f"{context} does not accept adsorbates for system_type={system_type!r}."
        )
    if adsorbate_definition is not None:
        raise SCGOValidationError(
            f"{context} does not accept adsorbate_definition for "
            f"system_type={system_type!r}."
        )


def resolve_search_mobile_composition(
    *,
    system_type: SystemType,
    composition: list[str],
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> list[str]:
    """Return the GA/BH search-mobile symbol list for algorithm sizing and operators.

    For cluster modes this is ``composition`` (adsorbate-reconciled). For
    slab-as-target modes it is top-layer slab symbols plus any adsorbate symbols.
    """
    policy = get_system_policy(system_type)
    if not policy.slab_is_search_target:
        return list(composition)

    if surface_config is None:
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to resolve "
            "search-mobile composition."
        )
    from scgo.surface.partition import resolve_slab_search_partition

    part = resolve_slab_search_partition(surface_config)
    mobile = list(part.mobile_slab_symbols)
    if policy.has_adsorbate:
        if adsorbate_definition is not None:
            mobile.extend(str(s) for s in adsorbate_definition.adsorbate_symbols)
        elif composition:
            # Adsorbate-only composition input (no core).
            mobile.extend(str(s) for s in composition)
    return mobile


def _strip_adsorbate_symbols(
    composition: list[str],
    adsorbate_symbols: list[str],
) -> list[str] | None:
    """Return ``composition`` minus one of each adsorbate symbol, or ``None``."""
    remaining = list(composition)
    for symbol in adsorbate_symbols:
        try:
            remaining.remove(symbol)
        except ValueError:
            return None
    return remaining


def resolve_mobile_composition(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], AdsorbateDefinition]:
    """Return ``(reconciled mobile composition, reconciled adsorbate definition)``.

    Does NOT mutate the input ``adsorbate_definition``. When a full mobile formula
    is reconciled by stripping known ``adsorbate_symbols``, a copy with the
    corrected ``core_symbols`` is returned as the second element; otherwise the
    original (unmodified) definition is returned. Callers must use the returned
    definition rather than relying on in-place mutation.
    """
    prefix = f"{context}: " if context else ""
    core_list = list(adsorbate_definition.core_symbols)
    ads_list = list(adsorbate_definition.adsorbate_symbols)
    expected = core_list + ads_list
    comp = [str(s) for s in composition]

    if comp == expected:
        return expected, adsorbate_definition

    comp_counts = get_composition_counts(comp)
    exp_counts = get_composition_counts(expected)
    if comp_counts == exp_counts or (
        ads_list and comp_counts == get_composition_counts(core_list)
    ):
        return expected, adsorbate_definition

    if ads_list:
        derived_core = _strip_adsorbate_symbols(comp, ads_list)
        if derived_core is not None:
            reconciled = replace(adsorbate_definition, core_symbols=derived_core)
            return derived_core + ads_list, reconciled

    raise SCGOValidationError(
        f"{prefix}composition must match core_symbols + adsorbate_symbols: "
        f"got counts {dict(comp_counts)}, expected {dict(exp_counts)}."
    )


def extract_adsorbate_definition_from_params(
    params: dict[str, Any] | None,
) -> AdsorbateDefinition | None:
    if not params:
        return None
    return as_adsorbate_definition(params.get("adsorbate_definition"))


def resolve_adsorbate_run_composition(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbates: AdsorbatesInput | None,
    preset_adsorbate_definition: AdsorbateDefinition | None,
    context: str,
) -> tuple[AdsorbateDefinition | None, list[Atoms] | None, list[str]]:
    """Build or reconcile mobile composition for adsorbate runs (gas or surface).

    Uses explicit ``adsorbates`` when provided; otherwise reconciles ``composition``
    against a preset ``adsorbate_definition`` from params.
    """
    policy = get_system_policy(system_type)
    comp = [str(s) for s in composition]

    if not policy.has_adsorbate:
        _reject_adsorbate_inputs_for_non_adsorbate(
            system_type=system_type,
            adsorbates=adsorbates,
            adsorbate_definition=preset_adsorbate_definition,
            context=context,
        )
        return None, None, comp

    if adsorbates is not None:
        return build_adsorbate_definition_from_inputs(
            system_type=system_type,
            composition=comp,
            adsorbates=adsorbates,
            context=context,
        )

    if preset_adsorbate_definition is not None:
        full_comp, ads_def = resolve_mobile_composition(
            comp, preset_adsorbate_definition, context=context
        )
        validate_adsorbate_definition(
            system_type=system_type,
            composition=full_comp,
            adsorbate_definition=ads_def,
            context=context,
        )
        return ads_def, None, full_comp

    return build_adsorbate_definition_from_inputs(
        system_type=system_type,
        composition=comp,
        adsorbates=None,
        context=context,
    )


def validate_composition_against_adsorbate(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], list[str]]:
    """Check composition against the core/adsorbate partition; return both lists.

    ``composition`` may match ``core_symbols + adsorbate_symbols`` exactly, share
    the same element counts in a different order, list only ``core_symbols``, or
    be a full mobile formula from which ``adsorbate_symbols`` are stripped.
    """
    prefix = f"{context}: " if context else ""
    core_list = list(adsorbate_definition.core_symbols)
    ads_list = list(adsorbate_definition.adsorbate_symbols)

    if not composition and not core_list and not ads_list:
        return core_list, ads_list
    if len(core_list) == 0 and len(ads_list) == 0:
        raise SCGOValidationError(
            f"{prefix}core_symbols and adsorbate_symbols cannot both be empty unless composition is also empty."
        )

    _, adsorbate_definition = resolve_mobile_composition(
        list(composition), adsorbate_definition, context=context
    )
    return list(adsorbate_definition.core_symbols), list(
        adsorbate_definition.adsorbate_symbols
    )


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
            raise SCGOValidationError(
                f"{context} received adsorbate_definition for non-adsorbate "
                f"system_type={system_type!r}."
            )
        return

    if adsorbate_definition is None:
        raise SCGOValidationError(
            f"{context} requires adsorbate_definition when system_type={system_type!r}."
        )

    core_list, _ads_list = validate_composition_against_adsorbate(
        composition, adsorbate_definition, context=context
    )

    if policy.slab_is_search_target and core_list:
        raise SCGOValidationError(
            f"{context}: system_type={system_type!r} does not support metal cores "
            "(slab top layers are the search core). Pass adsorbates only."
        )

    fba = adsorbate_definition.fragment_bond_axis
    if fba is not None and (len(fba) != 2 or not all(isinstance(x, int) for x in fba)):
        raise SCGOValidationError(
            f"adsorbate_definition['fragment_bond_axis'] must be a list of two int indices or omitted, got {fba!r}"
        )

    ai = adsorbate_definition.fragment_anchor_index
    if ai is not None and not isinstance(ai, int):
        raise SCGOValidationError(
            f"adsorbate_definition['fragment_anchor_index'] must be int or omitted, got {ai!r}"
        )

    frag_lengths = adsorbate_definition.adsorbate_fragment_lengths
    if frag_lengths:
        if any(int(x) <= 0 for x in frag_lengths):
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_fragment_lengths'] values must be positive."
            )
        expected_ads_len = len(composition) - len(core_list)
        if sum(int(x) for x in frag_lengths) != expected_ads_len:
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_fragment_lengths'] must sum to the adsorbate "
                f"length ({expected_ads_len}), got {sum(int(x) for x in frag_lengths)}."
            )


def resolve_adsorbate_fragments(
    templates: AdsorbateFragmentInput | None,
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> list[Atoms]:
    """Normalize fragment templates and validate them against the adsorbate definition."""
    from scgo.cluster_adsorbate.helpers import parse_positive_fragment_lengths

    prefix = f"{context}: " if context else ""
    if templates is None:
        raise SCGOValidationError(
            f"{prefix}adsorbate fragment template(s) are required."
        )

    fragments = (
        [templates.copy()]
        if isinstance(templates, Atoms)
        else [frag.copy() for frag in templates]
    )
    if not fragments:
        raise SCGOValidationError(
            f"{prefix}adsorbate fragment template(s) must not be empty."
        )

    lengths = adsorbate_definition.adsorbate_fragment_lengths
    if not lengths:
        parsed_lengths: list[int] = []
    else:
        parsed_lengths = parse_positive_fragment_lengths(lengths)
    lengths = parsed_lengths
    ads_symbols = list(adsorbate_definition.adsorbate_symbols)
    if not lengths and ads_symbols:
        lengths = [len(ads_symbols)]

    if len(fragments) != len(lengths):
        if (
            len(fragments) == 1
            and len(fragments[0]) == sum(lengths)
            and len(lengths) > 1
        ):
            raise SCGOValidationError(
                f"{prefix}found one combined adsorbate template for "
                f"{len(lengths)} fragments. Pass adsorbates as list[Atoms] "
                "with one entry per fragment."
            )
        raise SCGOValidationError(
            f"{prefix}fragment template count ({len(fragments)}) must match "
            f"adsorbate_fragment_lengths ({len(lengths)})."
        )

    offset = 0
    for idx, (frag, frag_len) in enumerate(zip(fragments, lengths, strict=True)):
        if len(frag) != frag_len:
            raise SCGOValidationError(
                f"{prefix}adsorbate fragment {idx} has len={len(frag)}, "
                f"expected {frag_len}."
            )
        expected_symbols = ads_symbols[offset : offset + frag_len]
        if expected_symbols and list(frag.get_chemical_symbols()) != expected_symbols:
            raise SCGOValidationError(
                f"{prefix}adsorbate fragment {idx} symbols "
                f"{frag.get_chemical_symbols()!r} do not match expected "
                f"{expected_symbols!r}."
            )
        offset += frag_len
    return fragments


def normalize_adsorbates_input(
    adsorbates: AdsorbatesInput | None, *, context: str
) -> list[Atoms]:
    prefix = f"{context}: " if context else ""
    if adsorbates is None:
        raise SCGOValidationError(
            f"{prefix}adsorbates is required for adsorbate system types."
        )

    items = adsorbates if isinstance(adsorbates, list) else [adsorbates]
    out: list[Atoms] = []

    for idx, item in enumerate(items):
        if not isinstance(item, Atoms):
            raise SCGOValidationError(
                f"{prefix}adsorbates[{idx}] must be ase.Atoms, got {type(item).__name__}."
            )
        if len(item) == 0:
            raise SCGOValidationError(f"{prefix}adsorbates[{idx}] must not be empty.")
        out.append(item.copy())

    if not out:
        raise SCGOValidationError(
            f"{prefix}adsorbates must contain at least one fragment."
        )
    return out


def flatten_adsorbate_symbols(adsorbates: list[Atoms]) -> list[str]:
    symbols: list[str] = []
    for frag in adsorbates:
        symbols.extend([str(s) for s in frag.get_chemical_symbols()])
    return symbols


def build_adsorbate_definition_from_inputs(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbates: AdsorbatesInput | None,
    context: str,
) -> tuple[AdsorbateDefinition | None, list[Atoms] | None, list[str]]:
    from scgo.cluster_adsorbate.feasibility import (
        validate_adsorbate_placement_feasibility,
    )
    from scgo.system_types.validation import (
        _validate_input_adsorbate_fragments_connected,
    )

    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        _reject_adsorbate_inputs_for_non_adsorbate(
            system_type=system_type,
            adsorbates=adsorbates,
            adsorbate_definition=None,
            context=context,
        )
        return None, None, list(composition)
    core_list = [str(s) for s in composition]
    fragments = normalize_adsorbates_input(adsorbates, context=context)
    _validate_input_adsorbate_fragments_connected(fragments, context=context)
    ads_list = flatten_adsorbate_symbols(fragments)
    full_mobile_composition = list(core_list) + list(ads_list)
    ads_def = AdsorbateDefinition(
        core_symbols=core_list,
        adsorbate_symbols=ads_list,
        adsorbate_fragment_lengths=[len(frag) for frag in fragments],
    )
    validate_adsorbate_definition(
        system_type=system_type,
        composition=full_mobile_composition,
        adsorbate_definition=ads_def,
        context=context,
    )
    validate_adsorbate_placement_feasibility(
        core_list,
        ads_def.adsorbate_fragment_lengths,
        fragments,
        context=context,
    )
    return ads_def, fragments, full_mobile_composition
