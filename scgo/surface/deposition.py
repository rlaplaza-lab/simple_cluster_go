"""Place gas-phase cluster seeds onto a slab for global optimization setup."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as ase_atomic_numbers
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import (
    ClusterAdsorbateConfig,
    resolve_cluster_adsorbate_config,
)
from scgo.cluster_adsorbate.helpers import resolve_fragment_anchor_and_bond_axis
from scgo.cluster_adsorbate.hierarchical import (
    _stamp_site_metadata,
    build_hierarchical_core_fragment_cluster,
)
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.cluster_adsorbate.sites import (
    SiteType,
    SurfaceSiteCandidate,
    count_site_candidates,
    filter_sites_to_outward,
    get_or_compute_planar_layer_site_candidates,
    get_or_compute_surface_site_candidates,
)
from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.initialization import (
    BatchInitPlan,
    create_initial_cluster,
    emit_init_diagnostics,
    plan_batch_initialization,
    reset_init_diagnostics,
)
from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import _generate_rotation_matrix
from scgo.metadata.atoms import get_tag
from scgo.surface.validation import validate_supported_cluster_deposit
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    normalize_connectivity_factor,
    resolve_adsorbate_fragments,
    resolve_connectivity_factor,
)
from scgo.system_types.connectivity_factor import max_connectivity_scale
from scgo.utils.combine_atoms import (
    concatenate_inherit_cell_pbc,
    random_rotation_matrix,
    top_layer_indices,
)
from scgo.utils.combine_atoms import (
    slab_surface_extreme as _shared_slab_surface_extreme,
)
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs, resolve_n_jobs_for_tasks
from scgo.utils.phase_logging import format_count_summary
from scgo.utils.site_counts import increment_site_type_count

if TYPE_CHECKING:
    from numpy.random import Generator

    from scgo.surface.config import SurfaceSystemConfig

logger = get_logger(__name__)


def _slab_surface_layer(slab: Atoms, axis: int, thickness: float = 2.5) -> Atoms:
    """Return slab atoms in the top ``thickness`` Å along the surface normal."""
    pos = slab.get_positions()
    if len(pos) == 0:
        return slab.copy()
    indices = top_layer_indices(pos, axis, thickness=thickness)
    return slab[indices].copy()


def _outward_slab_site_candidates(
    site_core: Atoms, axis: int
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Hull sites of a slab top-layer slice, restricted to outward-facing ones.

    The 3D convex hull of a slab slice also yields downward/sideways normals,
    which point into the bulk. Those are filtered out (never in place: the hull
    dict is cached by positions hash). Planar layers, or slices whose sites are
    all filtered away, fall back to :func:`planar_layer_site_candidates`.
    """
    sites = get_or_compute_surface_site_candidates(site_core)
    if count_site_candidates(sites) > 0:
        top_layer_z_min = float(np.min(site_core.get_positions()[:, axis]))
        sites = filter_sites_to_outward(
            sites, axis=axis, top_layer_z_min=top_layer_z_min
        )
    if count_site_candidates(sites) == 0:
        # Planar top layers (graphene/graphite) have no 3D convex hull; reuse the
        # cached planar-layer getter so the Voronoi build happens once per geometry.
        sites = get_or_compute_planar_layer_site_candidates(
            site_core, surface_normal_axis=axis
        )
    return sites


def _stamp_site_types_on_combined(combined: Atoms, site_types: list[str]) -> None:
    """Stamp adsorbate site-type tags without aliasing another structure's tag bag."""
    if not site_types:
        return
    existing = combined.info.get("key_value_pairs")
    if isinstance(existing, dict):
        combined.info["key_value_pairs"] = dict(existing)
    _stamp_site_metadata(combined, site_types)


def _site_types_from_structure(source: Atoms) -> list[str]:
    """Read back the site-type list stamped on ``source`` (empty when absent)."""
    site_types = get_tag(source, "adsorbate_site_types_json")
    if isinstance(site_types, list) and site_types:
        return [str(x) for x in site_types]
    site_type = get_tag(source, "adsorbate_site_type")
    if isinstance(site_type, str) and site_type:
        return [site_type]
    return []


def _build_adsorbate_fragments_on_slab(
    slab: Atoms,
    fragments: list[Atoms],
    adsorbate_definition: AdsorbateDefinition,
    rng: Generator,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
    batch_site_counts: dict[str, int] | None,
    axis: int,
    max_placement_attempts: int,
) -> Atoms | None:
    """Place molecular fragments on slab top-layer hull sites (no metal core)."""
    if not fragments:
        return None

    ca = resolve_cluster_adsorbate_config(cluster_adsorbate_config)
    # Whole-structure connectivity is suppressed for surface systems: a multi-layer
    # slab (e.g. graphite) is not covalently connected through the stack, so the
    # per-subgroup slab-contact rule in validate_connectivity_policy owns surface
    # connectivity instead. place_fragment_on_cluster receives uses_surface=True,
    # which makes its connectivity pre-check a no-op for the slab+adsorbate combo.
    site_core = _slab_surface_layer(slab, axis)
    precomputed_sites = _outward_slab_site_candidates(site_core, axis)
    anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(adsorbate_definition)
    within_structure_site_counts: dict[str, int] = {}

    for _ in range(max_placement_attempts):
        mobile = Atoms()
        mobile.set_cell(slab.get_cell())
        mobile.set_pbc(slab.get_pbc())
        site_types: list[str] = []
        all_ok = True
        for frag_tmpl in fragments:
            clash_target = slab if len(mobile) == 0 else slab + mobile
            frag_metadata: dict[str, str] = {}
            placed = place_fragment_on_cluster(
                site_core,
                frag_tmpl,
                rng,
                ca,
                anchor_index=anchor,
                bond_axis=bond_axis,
                site_core=site_core,
                clash_atoms=clash_target,
                within_structure_site_counts=within_structure_site_counts,
                batch_site_counts=batch_site_counts,
                placement_metadata=frag_metadata,
                site_candidates=precomputed_sites,
                uses_surface=True,
            )
            if placed is None:
                all_ok = False
                break
            site_types.append(frag_metadata.get("site_type", "directional_fallback"))
            mobile = combine_core_adsorbate(mobile, placed) if len(mobile) else placed
        if all_ok and len(mobile) > 0:
            combined = combine_slab_adsorbate(slab, mobile)
            _stamp_site_types_on_combined(combined, site_types)
            return combined
    return None


def _near_surface_rotation_matrix(rng: Generator, axis: int) -> np.ndarray:
    """Mostly in-plane rotation with a small independent tilt off the normal."""
    in_plane_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    tilt = float(rng.uniform(-0.35, 0.35))
    tilt_azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    if axis == 0:
        rot_axis = np.array([0.0, np.cos(tilt_azimuth), np.sin(tilt_azimuth)])
    elif axis == 1:
        rot_axis = np.array([np.cos(tilt_azimuth), 0.0, np.sin(tilt_azimuth)])
    else:
        rot_axis = np.array([np.cos(tilt_azimuth), np.sin(tilt_azimuth), 0.0])
    rot_axis /= max(np.linalg.norm(rot_axis), 1e-12)
    return _generate_rotation_matrix(rot_axis, tilt) @ _generate_rotation_matrix(
        normal, in_plane_angle
    )


def slab_surface_extreme(slab: Atoms, axis: int, *, upper: bool = True) -> float:
    """Return max (or min) Cartesian coordinate of slab atoms along ``axis``."""
    return _shared_slab_surface_extreme(slab, axis, upper=upper)


def _in_plane_translation_near_slab_atom(
    slab: Atoms, axis: int, rng: Generator, cluster_radius: float
) -> np.ndarray:
    """In-plane translation onto a random slab atom, with a small random offset.

    With an empty slab, falls back to a random fractional in-plane shift.
    """
    cell = slab.get_cell()
    slab_positions = slab.get_positions()

    n_slab = len(slab)
    if n_slab == 0:
        u, v = rng.random(), rng.random()
        if axis == 0:
            return np.asarray(u * cell[1] + v * cell[2], dtype=float)
        elif axis == 1:
            return np.asarray(u * cell[0] + v * cell[2], dtype=float)
        else:
            return np.asarray(u * cell[0] + v * cell[1], dtype=float)

    atom_idx = rng.integers(0, n_slab)
    atom_pos = slab_positions[atom_idx]

    offset_scale = cluster_radius * 0.1

    if axis == 0:
        angle = rng.uniform(0, 2 * np.pi)
        dy = offset_scale * np.cos(angle)
        dz = offset_scale * np.sin(angle)
        return np.asarray([0, atom_pos[1] + dy, atom_pos[2] + dz], dtype=float)
    elif axis == 1:
        angle = rng.uniform(0, 2 * np.pi)
        dx = offset_scale * np.cos(angle)
        dz = offset_scale * np.sin(angle)
        return np.asarray([atom_pos[0] + dx, 0, atom_pos[2] + dz], dtype=float)
    else:
        angle = rng.uniform(0, 2 * np.pi)
        dx = offset_scale * np.cos(angle)
        dy = offset_scale * np.sin(angle)
        return np.asarray([atom_pos[0] + dx, atom_pos[1] + dy, 0], dtype=float)


def _in_plane_translation(
    slab: Atoms, axis: int, rng: Generator, cluster_radius: float | None = None
) -> np.ndarray:
    """Random fractional shift along the two cell directions not dominated by ``axis``.

    For ``axis == 2``, uses ``cell[0]`` and ``cell[1]``. Uses ``[0, 1)`` fractions.
    If ``cluster_radius`` is positive, the shift is centered on a random slab atom
    instead.

    Args:
        slab: The slab atoms
        axis: Surface normal axis
        rng: Random number generator
        cluster_radius: Approximate radius of the cluster (optional)

    Returns:
        Shift vector
    """
    if cluster_radius is not None and cluster_radius > 0:
        return _in_plane_translation_near_slab_atom(slab, axis, rng, cluster_radius)

    cell = slab.get_cell()
    u, v = rng.random(), rng.random()
    if axis == 0:
        shift = u * cell[1] + v * cell[2]
    elif axis == 1:
        shift = u * cell[0] + v * cell[2]
    else:
        shift = u * cell[0] + v * cell[1]
    return np.asarray(shift, dtype=float)


def combine_slab_adsorbate(slab: Atoms, adsorbate: Atoms) -> Atoms:
    """Concatenate slab and adsorbate; adsorbate cell/pbc are replaced by slab's."""
    return concatenate_inherit_cell_pbc(slab, adsorbate)


def _place_cluster_above_slab(
    cluster_positions: np.ndarray,
    slab: Atoms,
    slab_top: float,
    axis: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    cluster_atomic_numbers: np.ndarray | None = None,
    *,
    prefer_surface_normal: bool = False,
) -> np.ndarray:
    """Rotate/translate centered cluster positions into a deposited position.

    Args:
        cluster_positions: Centered cluster positions (will be rotated/translated).
        slab: The slab atoms.
        slab_top: Maximum coordinate of slab atoms along ``axis``.
        axis: Surface normal axis index (0, 1, or 2).
        rng: Random number generator.
        config: Surface system configuration.
        cluster_atomic_numbers: Atomic numbers of cluster atoms. If provided,
            used to calculate covalent radii for connectivity-based placement.
        prefer_surface_normal: Use a mostly in-plane rotation with a small tilt
            instead of a uniformly random rotation.

    Returns:
        Rotated and translated cluster positions.
    """
    rotation = (
        _near_surface_rotation_matrix(rng, axis)
        if prefer_surface_normal
        else random_rotation_matrix(rng)
    )
    rotated_positions = cluster_positions @ rotation.T
    cluster_radius = float(np.max(np.linalg.norm(rotated_positions, axis=1)))

    defect_pos = slab.info.get("vacancy_cartesian_angstrom")
    if (
        config.defect_bias_probability > 0.0
        and defect_pos is not None
        and rng.random() < config.defect_bias_probability
    ):
        # Bias the in-plane center onto the vacancy; the z height is still set
        # below by the shared slab_top + effective_height logic.
        in_plane = [i for i in range(3) if i != axis]
        shift = np.zeros(3, dtype=float)
        shift[in_plane[0]] = defect_pos[in_plane[0]]
        shift[in_plane[1]] = defect_pos[in_plane[1]]

        offset_scale = cluster_radius * 0.1
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = offset_scale * np.cos(angle)
        dy = offset_scale * np.sin(angle)
        if axis == 0:
            shift[1] += dx
            shift[2] += dy
        elif axis == 1:
            shift[0] += dx
            shift[2] += dy
        else:
            shift[0] += dx
            shift[1] += dy

        translated_positions = rotated_positions + shift
    else:
        translated_positions = rotated_positions + _in_plane_translation(
            slab, axis, rng, cluster_radius
        )

    # Cap the sampled height so the cluster bottom stays within bonding
    # distance of the slab top.
    cf = config.structure_connectivity_factor

    slab_symbols = slab.get_chemical_symbols()
    slab_radius = get_covalent_radius(slab_symbols[0]) if slab_symbols else 1.36

    if cluster_atomic_numbers is not None and len(cluster_atomic_numbers) > 0:
        number_to_symbol = {v: k for k, v in ase_atomic_numbers.items()}

        unique_atomic_numbers = set(cluster_atomic_numbers)
        cluster_radii = [
            get_covalent_radius(number_to_symbol.get(int(z), str(int(z))))
            for z in unique_atomic_numbers
        ]
        cluster_radius_est = max(cluster_radii) if cluster_radii else 1.36
    else:
        cluster_radius_est = 1.36

    connectivity_threshold = max_connectivity_scale(
        normalize_connectivity_factor(cf)
    ) * (slab_radius + cluster_radius_est)

    cluster_min = float(np.min(rotated_positions[:, axis]))

    # Truncated uniform on [h_min, min(h_max, connectivity_threshold)].
    h_min = config.adsorption_height_min
    h_max = config.adsorption_height_max
    hi = min(h_max, max(h_min, connectivity_threshold))
    effective_height = float(rng.uniform(h_min, hi)) if hi > h_min else h_min

    translated_positions[:, axis] += slab_top + effective_height - cluster_min
    return translated_positions


def create_deposited_cluster(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    rng: Generator,
    config: SurfaceSystemConfig,
    previous_search_glob: str = "**/*.db",
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    batch_site_counts: dict[str, int] | None = None,
    *,
    plan: BatchInitPlan | None = None,
    allocation: tuple[str, int | None] | None = None,
    emit_diagnostics: bool = True,
    verbosity: int = 1,
) -> Atoms | None:
    """One adsorbate+slab structure, or None if placement fails.

    For non-adsorbate runs: build one gas-phase cluster for ``composition``, then
    place above slab. For adsorbate runs with core symbols: build hierarchical
    core+fragment first. For adsorbate runs without core symbols: place the
    fragments directly on slab surface sites.

    Args:
        plan: Pre-computed :class:`~scgo.initialization.BatchInitPlan` (discovery + allocation) so the
            noisy DB scan runs once per batch instead of once per candidate.
        allocation: Override the ``(strategy, template_index)`` allocation for
            this single cluster (used with ``plan`` to preserve diversity).
        emit_diagnostics: When ``False``, suppress the per-call diagnostic
            summary (the batch owner emits the aggregate summary).
        verbosity: Verbosity for initialization diagnostic summaries (0-3).

    Raises:
        SCGOValidationError: If ``adsorbate_definition`` is given without
            ``adsorbate_fragment_template``.
    """
    cluster_adsorbate_config = resolve_cluster_adsorbate_config(
        cluster_adsorbate_config
    )
    axis = config.surface_normal_axis
    slab_top = slab_surface_extreme(slab, axis, upper=True)

    # Callers may pass a plain dict (runtime boundary) or an already-constructed
    # ``AdsorbateDefinition`` (e.g. the GA acceptance tests). Normalize to the
    # dataclass so the body and the downstream helpers all use attribute access.
    if adsorbate_definition is not None and not isinstance(
        adsorbate_definition, AdsorbateDefinition
    ):
        adsorbate_definition = AdsorbateDefinition.from_dict(dict(adsorbate_definition))

    for _ in range(config.max_placement_attempts):
        if adsorbate_definition is None:
            cluster_seed = create_initial_cluster(
                list(composition),
                vacuum=config.cluster_init_vacuum,
                rng=rng,
                previous_search_glob=previous_search_glob,
                mode=config.init_mode,
                plan=plan,
                allocation=allocation,
                emit_diagnostics=emit_diagnostics,
                verbosity=verbosity,
            )
        else:
            if adsorbate_fragment_template is None:
                raise SCGOValidationError(
                    "create_deposited_cluster requires adsorbate_fragment_template "
                    "for hierarchical adsorbate initialization."
                )
            core_symbols = [str(s) for s in adsorbate_definition.core_symbols]
            if len(core_symbols) == 0:
                fragments = resolve_adsorbate_fragments(
                    adsorbate_fragment_template,
                    adsorbate_definition,
                    context="create_deposited_cluster",
                )
                combined = _build_adsorbate_fragments_on_slab(
                    slab,
                    fragments,
                    adsorbate_definition,
                    rng,
                    cluster_adsorbate_config,
                    batch_site_counts,
                    axis,
                    max_placement_attempts=1,
                )
                if combined is None:
                    continue
                mobile = combined[len(slab) :]
                if atoms_too_close(mobile, blmin, use_tags=False):
                    continue
                if atoms_too_close_two_sets(mobile, slab, blmin):
                    continue
                return combined
            else:
                cluster_seed = build_hierarchical_core_fragment_cluster(
                    adsorbate_definition,
                    rng,
                    previous_search_glob,
                    adsorbate_fragment_template,
                    cluster_adsorbate_config,
                    cluster_init_vacuum=config.cluster_init_vacuum,
                    init_mode=config.init_mode,
                    max_placement_attempts=1,
                    batch_site_counts=batch_site_counts,
                    plan=plan,
                    allocation=allocation,
                    emit_diagnostics=emit_diagnostics,
                    verbosity=verbosity,
                )
            if cluster_seed is None:
                continue
        atomic_numbers = cluster_seed.get_atomic_numbers()
        cluster_positions = cluster_seed.get_positions().copy()

        cluster_positions -= np.mean(cluster_positions, axis=0)
        prefer_surface = adsorbate_definition is not None
        deposited_positions = _place_cluster_above_slab(
            cluster_positions=cluster_positions,
            slab=slab,
            slab_top=slab_top,
            axis=axis,
            rng=rng,
            config=config,
            cluster_atomic_numbers=atomic_numbers,
            prefer_surface_normal=prefer_surface,
        )

        adsorbate = Atoms(
            numbers=atomic_numbers,
            positions=deposited_positions,
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
        )

        if atoms_too_close(adsorbate, blmin, use_tags=False):
            continue
        if atoms_too_close_two_sets(adsorbate, slab, blmin):
            continue

        combined = combine_slab_adsorbate(slab, adsorbate)
        # The bare Atoms rebuild above drops the site metadata stamped on the
        # gas-phase seed; re-stamp it so batch anti-repetition can see it.
        _stamp_site_types_on_combined(
            combined, _site_types_from_structure(cluster_seed)
        )
        n_slab = len(slab)
        connectivity_factor = resolve_connectivity_factor(
            None,
            cluster_adsorbate_config=cluster_adsorbate_config,
            surface_config=config,
        )

        ok, err = validate_supported_cluster_deposit(
            combined,
            n_slab,
            surface_normal_axis=config.surface_normal_axis,
            use_mic=bool(config.comparator_use_mic),
            connectivity_factor=connectivity_factor,
        )
        if not ok:
            logger.debug(
                "Rejected deposited structure by supported-cluster check: %s", err
            )
            continue
        return combined

    logger.warning(
        "Exhausted max_placement_attempts=%s in create_deposited_cluster",
        config.max_placement_attempts,
    )
    return None


def _plan_deposition_batch(
    composition: Sequence[str],
    n_structures: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    *,
    previous_search_glob: str,
    adsorbate_definition: AdsorbateDefinition | None,
) -> BatchInitPlan | None:
    """Resolve the one-per-batch initialization plan for the deposited cores.

    Returns ``None`` when there is no metal core to build (adsorbate-only
    deposition places fragments straight onto slab sites, so no previous-search
    discovery or strategy allocation is involved).
    """
    if adsorbate_definition is not None:
        core_symbols = [str(s) for s in adsorbate_definition.core_symbols]
        if not core_symbols:
            return None
        plan_composition: list[str] = core_symbols
    elif composition:
        plan_composition = [str(s) for s in composition]
    else:
        return None

    return plan_batch_initialization(
        plan_composition,
        n_structures,
        rng,
        vacuum=config.cluster_init_vacuum,
        previous_search_glob=previous_search_glob,
        mode=config.init_mode,
    )


def _record_batch_site_type(
    structure: Atoms,
    shared_site_counts: dict[str, int] | None,
    site_counts_lock: Lock | None = None,
) -> None:
    site_type = get_tag(structure, "adsorbate_site_type")
    increment_site_type_count(shared_site_counts, site_type, site_counts_lock)


def create_deposited_cluster_batch(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    n_structures: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    *,
    previous_search_glob: str = "**/*.db",
    n_jobs: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    batch_site_counts: dict[str, int] | None = None,
    verbosity: int = 1,
) -> list[Atoms]:
    """Generate multiple deposited structures (sequential or threaded).

    Discovery (the previous-search DB scan) and strategy allocation run exactly
    once for the whole batch via a shared :class:`~scgo.initialization.BatchInitPlan`; each produced
    structure uses its own planned allocation so strategy diversity is
    preserved. Per-candidate initialization diagnostics are suppressed and this
    function emits the single aggregate summary.

    Args:
        n_jobs: Parallelism for structure generation; ``None`` uses the project
            default (single worker). ``1`` keeps the deterministic
            sequential path; opt in with -1/-2 for parallelism.
        verbosity: Verbosity for the aggregate initialization summary (0-3).

    Raises:
        SCGORuntimeError: If fewer than ``n_structures`` structures can be built.
    """
    if n_structures <= 0:
        return []

    n_jobs = resolve_n_jobs(n_jobs)
    max_attempts = max(n_structures * 50, config.max_placement_attempts)
    reset_init_diagnostics()
    plan = _plan_deposition_batch(
        composition,
        n_structures,
        rng,
        config,
        previous_search_glob=previous_search_glob,
        adsorbate_definition=adsorbate_definition,
    )

    def _emit_batch_summary() -> None:
        site_summary = (
            format_count_summary(batch_site_counts) if batch_site_counts else ""
        )
        emit_init_diagnostics(
            n_structures,
            verbosity=verbosity,
            extra=f"site types {site_summary}" if site_summary else "",
        )

    if n_jobs == 1:
        out: list[Atoms] = []
        attempts = 0
        shared_site_counts = batch_site_counts

        while len(out) < n_structures and attempts < max_attempts:
            attempts += 1
            child_rng = np.random.default_rng(
                rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            allocation = plan.allocation_for(len(out)) if plan is not None else None
            struct = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
                batch_site_counts=shared_site_counts,
                plan=plan,
                allocation=allocation,
                emit_diagnostics=False,
                verbosity=verbosity,
            )
            if struct is not None:
                _record_batch_site_type(struct, shared_site_counts)
                out.append(struct)
        if len(out) < n_structures:
            raise SCGORuntimeError(
                f"Could only generate {len(out)} of {n_structures} deposited structures; "
                "try widening height range or increasing max_placement_attempts."
            )
        _emit_batch_summary()
        return out

    # Parallel: precompute deterministic per-task seeds on the main thread.
    per_worker_limit = max(config.max_placement_attempts, 50)
    task_seeds = [
        int(rng.integers(0, 2**63 - 1, dtype=np.int64)) for _ in range(n_structures)
    ]
    shared_site_counts = batch_site_counts
    site_counts_lock = Lock() if shared_site_counts is not None else None

    def _build_structure_with_seed(task_seed: int, task_idx: int) -> Atoms:
        task_rng = np.random.default_rng(task_seed)
        allocation = plan.allocation_for(task_idx) if plan is not None else None
        for _ in range(per_worker_limit):
            child_rng = np.random.default_rng(
                task_rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            structure = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
                batch_site_counts=shared_site_counts,
                plan=plan,
                allocation=allocation,
                emit_diagnostics=False,
                verbosity=verbosity,
            )
            if structure is not None:
                _record_batch_site_type(structure, shared_site_counts, site_counts_lock)
                return structure
        raise SCGORuntimeError(
            "Could not generate deposited structure in parallel worker; "
            "try widening height range or increasing max_placement_attempts."
        )

    workers = resolve_n_jobs_for_tasks(n_jobs, n_structures)
    ordered_results: list[Atoms | None] = [None] * n_structures
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="scgo_deposit"
    ) as ex:
        futures = {
            ex.submit(_build_structure_with_seed, seed, idx): idx
            for idx, seed in enumerate(task_seeds)
        }
        first_failure: BaseException | None = None
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ordered_results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 - failures are aggregated and re-raised as SCGORuntimeError below
                failures[idx] = f"{type(exc).__name__}: {exc}"
                if first_failure is None:
                    first_failure = exc
    if failures:
        raise SCGORuntimeError(
            f"Failed to generate {len(failures)} deposited structure(s) "
            f"(indexes {sorted(failures.keys())}): {failures}"
        ) from first_failure
    if any(result is None for result in ordered_results):
        raise SCGORuntimeError("Parallel batch returned too few structures")
    _emit_batch_summary()
    return [result for result in ordered_results if result is not None]
