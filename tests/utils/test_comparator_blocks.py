"""Block-aware, component-weighted uniqueness comparator tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.constants import DEFAULT_COMPARATOR_TOL, DEFAULT_PAIR_COR_MAX
from scgo.exceptions import SCGOValidationError
from scgo.utils.comparators import (
    ComparatorBlock,
    ComparatorBlocks,
    PureInteratomicDistanceComparator,
    get_block_distance_units,
)


def _blocks_for(n_slab: int, n_ads: int, *, n_support: int = 0):
    ranges = []
    if n_support:
        ranges.append(("mobile_slab", 0, n_support))
    if n_ads:
        ranges.append(("adsorbate", n_slab, n_slab + n_ads))
    return ComparatorBlocks.from_ranges(ranges)


# --- Validation ---------------------------------------------------------------


def test_invalid_role_rejected() -> None:
    with pytest.raises(SCGOValidationError, match="Unknown comparator block role"):
        ComparatorBlock(role="mantle", indices=(0, 1))


def test_non_monotonic_indices_rejected() -> None:
    with pytest.raises(SCGOValidationError, match="strictly increasing"):
        ComparatorBlock(role="deposit", indices=(2, 1))


def test_overlapping_blocks_rejected() -> None:
    with pytest.raises(SCGOValidationError, match="overlap"):
        ComparatorBlocks(
            blocks=(
                ComparatorBlock(role="deposit", indices=(0, 1, 2)),
                ComparatorBlock(role="adsorbate", indices=(2, 3)),
            )
        )


def test_weights_require_blocks() -> None:
    with pytest.raises(SCGOValidationError, match="requires blocks"):
        PureInteratomicDistanceComparator(component_weights={"deposit": 0.5})


def test_unknown_weight_role_rejected_at_comparator() -> None:
    blocks = ComparatorBlocks.from_ranges([("deposit", 0, 2)])
    with pytest.raises(SCGOValidationError, match="not among the provided blocks"):
        PureInteratomicDistanceComparator(
            blocks=blocks,
            component_weights={"adsorbate": 0.5},
        )


def test_negative_weight_rejected_at_comparator() -> None:
    blocks = ComparatorBlocks.from_ranges([("deposit", 0, 2)])
    with pytest.raises(SCGOValidationError, match="non-negative"):
        PureInteratomicDistanceComparator(
            blocks=blocks, component_weights={"deposit": -0.5}
        )


def test_block_indices_out_of_range_rejected_at_compare_time() -> None:
    blocks = ComparatorBlocks.from_ranges([("deposit", 0, 5)])
    comparator = PureInteratomicDistanceComparator(blocks=blocks)
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(SCGOValidationError, match="out of range"):
        comparator.get_differences(atoms, atoms.copy())


# --- Single-block equivalence with the legacy window --------------------------


def test_single_block_matches_legacy_math() -> None:
    rng = np.random.default_rng(7)
    pos = rng.uniform(0, 5, (6, 3))
    a1 = Atoms("Pt4O2", positions=pos)
    pos2 = pos + rng.normal(0, 0.05, pos.shape)
    a2 = Atoms("Pt4O2", positions=pos2)

    legacy = PureInteratomicDistanceComparator(n_top=6)
    single_block = PureInteratomicDistanceComparator(
        n_top=6, blocks=ComparatorBlocks.from_ranges([("deposit", 0, 6)])
    )
    assert legacy.get_differences(a1, a2) == pytest.approx(
        single_block.get_differences(a1, a2)
    )


# --- Cross-block pairs expose binding geometry --------------------------------


def test_cross_pairs_separate_binding_sites_legacy_cannot() -> None:
    """A monatomic adsorbate has no intra-element pairs; only cross terms can
    distinguish its registry on the surface."""
    slab = fcc111("Pt", size=(2, 4, 1), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    top = slab.positions[:, 2].max()
    # Two inequivalent hollow sites ~1.4 A apart in-plane.
    site_a = [slab.positions[0, 0] + 1.42, slab.positions[0, 1], top + 1.3]
    site_b = [slab.positions[0, 0] + 2.84, slab.positions[0, 1], top + 1.3]
    a1 = slab.copy() + Atoms("O", positions=[site_a])
    a2 = slab.copy() + Atoms("O", positions=[site_b])
    for atoms in (a1, a2):
        atoms.cell = slab.cell
        atoms.pbc = slab.pbc

    legacy = PureInteratomicDistanceComparator(n_top=1)  # window = O alone
    cum_l, max_l = legacy.get_differences(a1[[-1]], a2[[-1]])
    assert max_l == pytest.approx(0.0)

    blocky = PureInteratomicDistanceComparator(
        n_top=len(a1),
        mic=True,
        blocks=_blocks_for(n_slab=n_slab, n_ads=1, n_support=n_slab),
    )
    _cum, max_diff = blocky.get_differences(a1, a2)
    assert max_diff > 0.3
    assert not blocky.looks_like(a1, a2)


# --- Dilution: same-element support swamps the deposit in one shared bucket ---


def _pt_support_plus_tetramer(
    slab_top_noise: np.ndarray | None,
    tetramer_shift: float,
) -> tuple[Atoms, Atoms]:
    """A two-layer Pt support + Pt4 deposit; all atoms share one element.

    Layout: [9 support Pt][4 deposit Pt], mimicking a Pt cluster on a Pt slab
    whose top layers only relax.
    """
    # Bottom 9 atoms on a flat lattice; deposit as a relaxed tetramer above.
    base = [
        [0, 0, 0],
        [2.8, 0, 0],
        [0, 2.8, 0],
        [2.8, 2.8, 0],
        [1.4, 1.4, 0],
        [2.8, 0, 2.8],
        [0, 2.8, 2.8],
        [2.8, 2.8, 2.8],
        [1.4, 1.4, 2.8],
    ]
    tetra = [[1.4, 1.4, 4.6], [3.4, 1.4, 5.4], [1.4, 3.4, 5.4], [2.4, 2.4, 6.4]]
    pos_a = np.array(base + tetra, dtype=float)
    pos_b = pos_a.copy()
    if slab_top_noise is not None:
        pos_a[:9] += slab_top_noise
        pos_b[:9] += slab_top_noise
    pos_b[-1] += np.array([tetramer_shift, 0.0, 0.0])
    a = Atoms("Pt13", positions=pos_a)
    b = Atoms("Pt13", positions=pos_b)
    return a, b


def test_block_aware_separates_deposit_legacy_merges() -> None:
    """The motivating case: a globally optimized deposit on a barely-moving
    same-element support. One shared-bucket window merges distinct deposits;
    role blocks do not."""
    noise = np.random.default_rng(5).normal(0, 0.02, (9, 3))
    a, b = _pt_support_plus_tetramer(noise, tetramer_shift=0.5)

    blocks = ComparatorBlocks.from_ranges([("mobile_slab", 0, 9), ("deposit", 9, 13)])
    legacy = PureInteratomicDistanceComparator(n_top=13)  # single shared bucket
    cum_l, max_l = legacy.get_differences(a, b)
    assert cum_l < DEFAULT_COMPARATOR_TOL
    assert max_l < DEFAULT_PAIR_COR_MAX

    blocky = PureInteratomicDistanceComparator(n_top=13, blocks=blocks)
    cum_b, _max_b = blocky.get_differences(a, b)
    # The same differences read far larger once the deposit normalizes alone.
    assert not blocky.looks_like(a, b)
    assert cum_b > cum_l


def test_mobile_slab_weight_controls_reconstruction_sensitivity() -> None:
    """Support-only differences count at weight 1.0 and vanish at weight 0."""
    rng = np.random.default_rng(8)
    deposit = rng.uniform(0, 5, (4, 3)) + np.array([0.0, 0.0, 10.0])
    support_a = rng.uniform(0, 8, (9, 3))
    support_b = support_a + np.random.default_rng(9).normal(0, 0.25, (9, 3))

    x = Atoms("Pt13", positions=np.vstack([support_a, deposit]))
    y = Atoms("Pt13", positions=np.vstack([support_b, deposit]))

    blocks = ComparatorBlocks.from_ranges([("mobile_slab", 0, 9), ("deposit", 9, 13)])
    full_w = PureInteratomicDistanceComparator(n_top=13, blocks=blocks)
    zero_w = PureInteratomicDistanceComparator(
        n_top=13, blocks=blocks, component_weights={"mobile_slab": 0.0}
    )

    cum_full, _ = full_w.get_differences(x, y)
    cum_zero, max_zero = zero_w.get_differences(x, y)
    assert full_w.looks_like(x, y) is False
    assert cum_zero == pytest.approx(0.0)
    assert max_zero == pytest.approx(0.0)
    assert cum_full > DEFAULT_COMPARATOR_TOL


# --- Composition / cache behavior ---------------------------------------------


def test_per_block_composition_mismatch_returns_inf() -> None:
    a1 = Atoms("Pt2O", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    a2 = Atoms("PtO2", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    blocks = ComparatorBlocks.from_ranges([("deposit", 0, 2), ("adsorbate", 2, 3)])
    comparator = PureInteratomicDistanceComparator(blocks=blocks)
    cum, mx = comparator.get_differences(a1, a2)
    assert math.isinf(cum) and math.isinf(mx)


def test_block_fingerprint_cache_isolation_and_invalidation() -> None:
    atoms = Atoms(
        "Pt3O", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0.75, 0.75, 2.0]]
    )
    b_small = ComparatorBlocks.from_ranges([("deposit", 0, 3)])
    u1 = get_block_distance_units(atoms, mic=False, blocks=b_small)
    assert set(u1.keys()) == {("intra", 0, 78)}

    b_split = ComparatorBlocks.from_ranges([("deposit", 0, 3), ("adsorbate", 3, 4)])
    u2 = get_block_distance_units(atoms, mic=False, blocks=b_split)
    assert ("cross", 0, 1, 78, 8) in u2
    # The earlier slot must survive untouched.
    u1_again = get_block_distance_units(atoms, mic=False, blocks=b_small)
    np.testing.assert_array_equal(u1[("intra", 0, 78)], u1_again[("intra", 0, 78)])

    pos = atoms.get_positions()
    pos[0, 0] += 0.2
    atoms.set_positions(pos)
    u1_new = get_block_distance_units(atoms, mic=False, blocks=b_small)
    assert not np.allclose(u1_new[("intra", 0, 78)], u1[("intra", 0, 78)])


def test_blocks_signature_stable_and_distinct() -> None:
    a = ComparatorBlocks.from_ranges([("deposit", 0, 3), ("adsorbate", 3, 4)])
    b = ComparatorBlocks.from_ranges([("deposit", 0, 3), ("adsorbate", 3, 4)])
    c = ComparatorBlocks.from_ranges([("adsorbate", 3, 4), ("deposit", 0, 3)])
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()
