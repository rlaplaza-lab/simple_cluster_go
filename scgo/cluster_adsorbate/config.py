"""Configuration for placing adsorbates on gas-phase metal clusters."""

from __future__ import annotations

from dataclasses import dataclass

from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.utils.validation import validate_positive


@dataclass(frozen=True)
class ClusterAdsorbateConfig:
    """Shared placement / relaxation settings for any small adsorbate fragment.

    Attributes:
        height_min: Minimum offset (Å) along the sampled outward normal from the
            outermost core atom in that direction to the fragment anchor atom.
        height_max: Maximum such offset (Å).
        max_placement_attempts: Retries with new random directions (and spins).
        blmin_ratio: Passed to ASE ``closest_distances_generator``.
        cell_margin: Extra padding (Å) for the cubic cell before relaxation.
        random_spin_about_normal: If True and ``bond_axis`` is used in placement,
            apply a random rotation about the surface normal after aligning the
            bond to that normal (does nothing for monatomics or fully random
            molecular orientations).
        validate_combined_structure: If True, run structure checks on core+adsorbate
            after placement and before/after local relaxation (connectivity by
            default; clashes optional).
        structure_min_distance_factor: Scale covalent radii for clash detection
            (same meaning as in :func:`~scgo.initialization.geometry_helpers.validate_cluster_structure`).
        structure_connectivity_factor: Edge threshold ``(r_i+r_j) * factor`` for
            connectivity (same as cluster initialization).
        structure_check_clashes: Whether to reject clashing combined structures.
        structure_check_connectivity: Whether to require a single connected component.
    """

    height_min: float = 0.9
    height_max: float = 2.2
    max_placement_attempts: int = 80
    blmin_ratio: float = 0.7
    cell_margin: float = 6.0
    random_spin_about_normal: bool = True
    validate_combined_structure: bool = True
    structure_min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR
    structure_check_clashes: bool = True
    structure_check_connectivity: bool = True

    def __post_init__(self) -> None:
        validate_positive("height_min", self.height_min, strict=True)
        validate_positive("height_max", self.height_max, strict=True)
        if self.height_max < self.height_min:
            raise ValueError("height_max must be >= height_min")
        if self.max_placement_attempts < 1:
            raise ValueError("max_placement_attempts must be positive")
        validate_positive("blmin_ratio", self.blmin_ratio, strict=True)
        validate_positive("cell_margin", self.cell_margin, strict=True)
        validate_positive(
            "structure_min_distance_factor",
            self.structure_min_distance_factor,
            strict=True,
        )
        validate_positive(
            "structure_connectivity_factor",
            self.structure_connectivity_factor,
            strict=True,
        )


@dataclass(frozen=True)
class ClusterOHConfig(ClusterAdsorbateConfig):
    """OH-specific defaults: gas-phase O–H bond length used to build the template."""

    oh_bond_length: float = 0.96

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_positive("oh_bond_length", self.oh_bond_length, strict=True)
