"""Place molecular fragments near the surface of a gas-phase cluster."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase_ga.utilities import atoms_too_close_two_sets, closest_distances_generator
from numpy.random import Generator

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig, ClusterOHConfig
from scgo.cluster_adsorbate.geometry import (
    outermost_point_along_normal,
    random_rotation_matrix,
    random_spin_about_normal,
    random_unit_vector,
    rotation_matrix_a_to_b,
)
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def blmin_for_core_and_fragment(
    core: Atoms, fragment: Atoms, blmin_ratio: float
) -> dict:
    """Minimum interatomic distances for all element pairs in core ∪ fragment."""
    zs = {int(z) for z in core.numbers}
    zs.update(int(z) for z in fragment.numbers)
    return closest_distances_generator(list(zs), ratio_of_covalent_radii=blmin_ratio)


def place_fragment_on_cluster(
    core: Atoms,
    fragment_template: Atoms,
    rng: Generator,
    config: ClusterAdsorbateConfig | None = None,
    *,
    anchor_index: int = 0,
    bond_axis: tuple[int, int] | None = None,
) -> Atoms | None:
    """Rigidly place a gas-phase fragment with random orientation near the cluster.

    The fragment geometry is copied from ``fragment_template``. Positions are
    expressed relative to ``anchor_index``, optionally rotated, then the anchor
    atom is placed at ``surface_point + height * n`` with ``n`` a random outward
    direction and ``height`` uniform in ``[height_min, height_max]``.

    Args:
        core: Bare metal cluster.
        fragment_template: Equilibrium adsorbate (any size ≥ 1). Symbols and
            geometry define the rigid body; only positions are transformed.
        rng: Random number generator.
        config: Placement parameters.
        anchor_index: Fragment atom placed along the height offset from the
            outermost core atom in direction ``n``.
        bond_axis: If ``(i, j)``, rotate the fragment so the vector from atom
            ``i`` to ``j`` aligns with the outward normal before optional spin.
            Use for diatomics (e.g. OH, CO). If ``None``, apply a uniform random
            rotation (for multi-atom molecules such as H2O).

    Returns:
        The positioned fragment only, or ``None`` if placement fails.

    When ``config.validate_combined_structure`` is True (default), candidates
    that fail the same connectivity / clash checks as cluster initialization
    are discarded and retried.
    """
    if config is None:
        config = ClusterAdsorbateConfig()
    if len(core) == 0:
        raise ValueError("core must contain at least one atom")
    n_frag = len(fragment_template)
    if n_frag == 0:
        raise ValueError("fragment_template must contain at least one atom")
    if not (0 <= anchor_index < n_frag):
        raise ValueError(
            f"anchor_index={anchor_index} invalid for fragment with {n_frag} atoms"
        )
    if bond_axis is not None:
        i, j = bond_axis
        if not (0 <= i < n_frag and 0 <= j < n_frag) or i == j:
            raise ValueError(f"bond_axis={bond_axis} invalid for this fragment")

    core_pos = core.get_positions()
    com = np.mean(core_pos, axis=0)
    relative_core_pos = core_pos - com
    blmin = blmin_for_core_and_fragment(core, fragment_template, config.blmin_ratio)
    symbols = fragment_template.get_chemical_symbols()

    # Precompute base geometry relative to anchor
    base_frag_pos = fragment_template.get_positions().astype(float).copy()
    base_frag_pos -= base_frag_pos[anchor_index]

    for _ in range(config.max_placement_attempts):
        n_dir = random_unit_vector(rng)
        anchor_surf = outermost_point_along_normal(core_pos, relative_core_pos, n_dir)
        h_off = float(rng.uniform(config.height_min, config.height_max))
        target = anchor_surf + h_off * n_dir

        pos = base_frag_pos.copy()

        if n_frag > 1:
            if bond_axis is not None:
                ia, ja = bond_axis
                v = pos[ja] - pos[ia]
                vn = float(np.linalg.norm(v))
                if vn < 1e-10:
                    continue
                v = v / vn
                r_align = rotation_matrix_a_to_b(v, n_dir)
                pos = (r_align @ pos.T).T
                if config.random_spin_about_normal:
                    r_spin = random_spin_about_normal(rng, n_dir)
                    pos = (r_spin @ pos.T).T
            else:
                r_rand = random_rotation_matrix(rng)
                pos = (r_rand @ pos.T).T

        pos += target

        frag = Atoms(
            symbols=symbols,
            positions=pos,
            cell=core.get_cell(),
            pbc=core.get_pbc(),
        )

        if atoms_too_close_two_sets(frag, core, blmin):
            continue

        if config.validate_combined_structure:
            trial = combine_core_adsorbate(core, frag)
            ok, _msg = validate_combined_cluster_structure(
                trial,
                min_distance_factor=config.structure_min_distance_factor,
                connectivity_factor=config.structure_connectivity_factor,
                check_clashes=config.structure_check_clashes,
                check_connectivity=config.structure_check_connectivity,
            )
            if not ok:
                continue

        return frag

    logger.warning(
        "place_fragment_on_cluster: exceeded max_placement_attempts=%s",
        config.max_placement_attempts,
    )
    return None


def place_oh_on_cluster(
    core: Atoms,
    rng: Generator,
    config: ClusterOHConfig | None = None,
) -> Atoms | None:
    """Build an OH fragment positioned near the cluster (O = anchor, O–H ∥ outward)."""
    if config is None:
        config = ClusterOHConfig()
    d = config.oh_bond_length
    tmpl = Atoms(
        symbols=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=float),
        cell=core.get_cell(),
        pbc=core.get_pbc(),
    )
    return place_fragment_on_cluster(
        core,
        tmpl,
        rng,
        config,
        anchor_index=0,
        bond_axis=(0, 1),
    )
