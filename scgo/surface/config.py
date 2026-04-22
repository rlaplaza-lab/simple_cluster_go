"""Configuration for cluster-on-surface (adsorbate + slab) workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms

from scgo.utils.logging import get_logger
from scgo.utils.validation import validate_positive

logger = get_logger(__name__)


@dataclass(frozen=True)
class SurfaceSystemConfig:
    """Describe a fixed slab plus a movable adsorbate cluster for GA.

    Atom ordering in combined systems must be ``slab`` atoms first, then the
    ``len(composition)`` adsorbate atoms (matching ASE GA patches: ``n_top``
    trailing atoms are optimized). Pass the same instance to TS search
    (``get_ts_search_params(..., surface_config=...)`` or
    ``run_ts_search(..., surface_config=..., ts_kwargs={...})``) so NEB uses the
    identical slab ``FixAtoms`` policy as local relaxation.
    At runtime, :func:`scgo.surface.validation.validate_surface_config_slab_prefix`
    checks that combined systems still begin with ``slab``'s symbols in order.

    **Slab motion during local relaxation** (three common modes; ``L`` is the
    number of distinct slab coordinate layers along ``surface_normal_axis``):

    ================================  ============================================
    Intent                            Settings
    ================================  ============================================
    Full slab frozen                  ``fix_all_slab_atoms=True`` (default)
    Frozen except top N slab layers   ``fix_all_slab_atoms=False`` and either
                                      ``n_relax_top_slab_layers=N``, or
                                      ``n_fix_bottom_slab_layers=L - N``
    Nothing on the slab frozen        ``fix_all_slab_atoms=False``,
                                      ``n_fix_bottom_slab_layers=None``,
                                      ``n_relax_top_slab_layers=None``
    ================================  ============================================

    For a typical slab with vacuum along ``z``, the adsorbate sits on the
    high-``z`` side; fixing the bottom ``L - N`` distinct layers is the
    same as leaving only the top ``N`` layers free to relax.

    Do not set ``n_relax_top_slab_layers`` together with
    ``n_fix_bottom_slab_layers``, or together with ``fix_all_slab_atoms=True``.

    Attributes:
        slab: Frozen substrate (positions copied at use sites). Should use a
            cell and ``pbc`` appropriate for the slab (often periodic in-plane).
        adsorption_height_min: Minimum distance (Å) from the slab extreme along
            the surface normal to the adsorbate's closest atom along that axis.
        adsorption_height_max: Maximum such distance (Å).
        surface_normal_axis: Cartesian axis index (0, 1, or 2) along which the
            surface normal is assumed for height and layer logic. Slabs from
            ``ase.build.fcc111`` etc. are typically oriented with vacuum along ``z``
            (axis ``2``).
        fix_all_slab_atoms: If True, apply ``FixAtoms`` to all slab atoms during
            local relaxation.
        n_fix_bottom_slab_layers: If set (and ``fix_all_slab_atoms`` is False),
            fix only the bottom N distinct coordinate layers among slab atoms
            along ``surface_normal_axis``. Mutually exclusive with
            ``n_relax_top_slab_layers``.
        n_relax_top_slab_layers: If set (and ``fix_all_slab_atoms`` is False),
            fix every slab atom **except** those in the top N distinct layers
            along ``surface_normal_axis`` (the layers nearest the adsorbate for a
            typical orientation). Mutually exclusive with
            ``n_fix_bottom_slab_layers``.
        comparator_use_mic: If True, duplicate detection uses MIC for pairwise
            distances on the adsorbate (useful when ``pbc`` is True in-plane).
        cluster_init_vacuum: Vacuum (Å) for isolated cluster seed generation.
        init_mode: Mode passed to :func:`scgo.initialization.create_initial_cluster`.
        max_placement_attempts: Max tries per structure for valid placement.
    """

    slab: Atoms
    adsorption_height_min: float = 1.2
    adsorption_height_max: float = 3.0
    surface_normal_axis: int = 2
    fix_all_slab_atoms: bool = True
    n_fix_bottom_slab_layers: int | None = None
    n_relax_top_slab_layers: int | None = None
    comparator_use_mic: bool = False
    cluster_init_vacuum: float = 8.0
    init_mode: str = "smart"
    max_placement_attempts: int = 200

    def __post_init__(self) -> None:
        if self.surface_normal_axis not in (0, 1, 2):
            raise ValueError("surface_normal_axis must be 0, 1, or 2")
        validate_positive(
            "adsorption_height_min", self.adsorption_height_min, strict=True
        )
        validate_positive(
            "adsorption_height_max", self.adsorption_height_max, strict=True
        )
        if self.adsorption_height_min > self.adsorption_height_max:
            raise ValueError(
                "adsorption_height_min must be <= adsorption_height_max, "
                f"got {self.adsorption_height_min} and {self.adsorption_height_max}"
            )
        if len(self.slab) == 0:
            raise ValueError("slab must contain at least one atom")

        if not any(self.slab.pbc):
            raise ValueError("Slab must have at least one periodic dimension.")

        if not all(self.slab.pbc):
            logger.warning("Extending slab periodicity to 3D for VASP compatibility.")
            self.slab.pbc = [True, True, True]

        vacuum_length = self.slab.cell.lengths()[self.surface_normal_axis]
        if vacuum_length < 10.0:
            logger.warning(
                f"Slab vacuum size ({vacuum_length:.2f} A) on axis {self.surface_normal_axis} "
                "might be too small to prevent periodic interaction.",
            )

        if (
            self.n_fix_bottom_slab_layers is not None
            and self.n_fix_bottom_slab_layers < 1
        ):
            raise ValueError("n_fix_bottom_slab_layers must be >= 1 when set")
        if (
            self.n_relax_top_slab_layers is not None
            and self.n_relax_top_slab_layers < 1
        ):
            raise ValueError("n_relax_top_slab_layers must be >= 1 when set")
        if self.fix_all_slab_atoms and self.n_relax_top_slab_layers is not None:
            raise ValueError(
                "n_relax_top_slab_layers is incompatible with fix_all_slab_atoms=True"
            )
        if (
            self.n_fix_bottom_slab_layers is not None
            and self.n_relax_top_slab_layers is not None
        ):
            raise ValueError(
                "set at most one of n_fix_bottom_slab_layers and "
                "n_relax_top_slab_layers"
            )
