"""End-to-end uniqueness plumbing: minima filtering, diversity, TS similarity."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types.dedup_geometry import resolve_uniqueness_geometry
from scgo.ts_search.transition_state import calculate_structure_similarity
from scgo.utils.comparators import (
    ComparatorBlocks,
    PureInteratomicDistanceComparator,
)
from scgo.utils.diversity_scorer import DiversityScorer
from scgo.utils.helpers import filter_unique_minima
from tests.utils.test_comparator_blocks import _pt_support_plus_tetramer


def _tagged(energy: float, atoms: Atoms) -> tuple[float, Atoms]:
    atoms.info["raw_score"] = -energy  # GA convention used by filter_unique_minima
    return (energy, atoms)


def test_filter_unique_minima_block_aware_keeps_support_distinct_deposit() -> None:
    """Same-element support + deposit: legacy window merges, blocks do not."""
    noise = np.random.default_rng(5).normal(0, 0.02, (9, 3))
    a, b = _pt_support_plus_tetramer(noise, tetramer_shift=0.5)

    cfg_blocks = ComparatorBlocks.from_ranges(
        [("mobile_slab", 0, 9), ("deposit", 9, 13)]
    )
    kept_legacy = filter_unique_minima(
        [_tagged(-1.0, a.copy()), _tagged(-1.001, b.copy())],
        energy_tolerance=0.02,
        n_top=13,
    )
    assert len(kept_legacy) == 1

    kept_blocks = filter_unique_minima(
        [_tagged(-1.0, a.copy()), _tagged(-1.001, b.copy())],
        energy_tolerance=0.02,
        n_top=13,
        blocks=cfg_blocks,
        component_weights={"mobile_slab": 1.0},
        cross_weight=1.0,
    )
    assert len(kept_blocks) == 2


def test_filter_unique_minima_resolved_geometry_supported_cluster() -> None:
    cfg = SurfaceSystemConfig(slab=fcc111("Pt", size=(2, 2, 2), vacuum=8.0))
    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster",
        n_atoms=len(cfg.slab) + 2,
        surface_config=cfg,
        counts=(2, 0),
    )
    rng = np.random.default_rng(4)
    base = np.vstack([cfg.slab.get_positions(), rng.uniform(0, 3, (2, 3)) + [0, 0, 10]])
    other = base.copy()
    other[-1] += [0.8, 0.0, 0.0]
    cell, pbc = cfg.slab.cell, cfg.slab.pbc
    m1 = Atoms("Pt10", positions=base, cell=cell, pbc=pbc)
    m2 = Atoms("Pt10", positions=other, cell=cell, pbc=pbc)

    kept = filter_unique_minima(
        [_tagged(-2.0, m1), _tagged(-2.005, m2)],
        energy_tolerance=0.02,
        n_top=len(m1),
        mic=True,
        blocks=geo.blocks,
        comparator_tol=geo.settings.comparator_tol,
        comparator_pair_cor_max=geo.settings.comparator_pair_cor_max,
    )
    assert len(kept) == 2


def test_diversity_scorer_weighted_slab_slices() -> None:
    ref_a, ref_b = _pt_support_plus_tetramer(None, tetramer_shift=0.0)
    noise = np.random.default_rng(2).normal(0, 0.15, (9, 3))
    moved_a = ref_a.copy()
    moved_b = ref_b.copy()
    for moved in (moved_a, moved_b):
        pos = moved.get_positions()
        pos[:9] += noise
        moved.set_positions(pos)

    blocks = ComparatorBlocks.from_ranges([("mobile_slab", 0, 9), ("deposit", 9, 13)])
    scorer_uniform = DiversityScorer(
        [ref_a],
        PureInteratomicDistanceComparator(n_top=13, blocks=blocks),
    )
    scorer_low = DiversityScorer(
        [ref_a],
        PureInteratomicDistanceComparator(
            n_top=13,
            blocks=blocks,
            component_weights={"mobile_slab": 0.0},
            cross_weight=0.0,
        ),
    )

    score_moved_uniform = scorer_uniform.score(moved_b)
    score_moved_low = scorer_low.score(moved_b)
    # Support-only differences contribute less when the slab is downweighted.
    assert score_moved_low < score_moved_uniform
    # Deposit-identical structure scores exactly zero at weight zero.
    assert scorer_low.score(ref_b) == pytest.approx(0.0)


def test_diversity_scorer_block_descriptors_match_comparator() -> None:
    _, ref_b = _pt_support_plus_tetramer(None, tetramer_shift=1.0)
    blocks = ComparatorBlocks.from_ranges([("mobile_slab", 0, 9), ("deposit", 9, 13)])
    comparator = PureInteratomicDistanceComparator(n_top=13, blocks=blocks)
    scorer = DiversityScorer([ref_b], comparator)

    assert comparator.get_differences(ref_b, ref_b) == (0.0, 0.0)
    # Descriptor extraction must not raise and must stay length-stable.
    desc = scorer._atoms_to_descriptor(ref_b)
    assert len(desc) == len(scorer._atoms_to_descriptor(ref_b.copy()))


def test_calculate_structure_similarity_with_blocks_skips_slicing() -> None:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    top = slab.positions[:, 2].max()
    site = [slab.positions[0, 0], slab.positions[0, 1], top + 1.3]
    site_shift = [slab.positions[0, 0] + 1.42, slab.positions[0, 1], top + 1.3]
    a1 = slab.copy() + Atoms("O", positions=[site])
    a2 = slab.copy() + Atoms("O", positions=[site_shift])
    for atoms in (a1, a2):
        atoms.cell = slab.cell
        atoms.pbc = slab.pbc

    blocks = ComparatorBlocks.from_ranges(
        [("mobile_slab", 0, n_slab), ("adsorbate", n_slab, n_slab + 1)]
    )
    cum_l, mx_l, similar_l = calculate_structure_similarity(
        a1,
        a2,
        tolerance=DEFAULT_COMPARATOR_TOL,
        pair_cor_max=DEFAULT_PAIR_COR_MAX,
        use_mic=True,
        n_slab=n_slab,
    )
    assert mx_l == pytest.approx(0.0)
    assert similar_l is True

    _cum, mx, similar = calculate_structure_similarity(
        a1,
        a2,
        tolerance=DEFAULT_COMPARATOR_TOL,
        pair_cor_max=DEFAULT_PAIR_COR_MAX,
        use_mic=True,
        n_slab=n_slab,
        blocks=blocks,
    )
    # Legacy mobile-only slicing compares the O atom alone (vacuous); cross
    # terms must see the registry difference.
    assert mx > 0.1
    assert similar is False
