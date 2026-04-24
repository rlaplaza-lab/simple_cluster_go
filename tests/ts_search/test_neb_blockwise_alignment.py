"""NEB: blockwise endpoint matching (slab + core + adsorbate)."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.ts_search.transition_state import (
    _align_endpoints_blockwise,
    interpolate_path,
)


def test_blockwise_reorders_adsorbate_block_to_reactant() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.0, 0.0, 0.1],
            [1.2, 0.7, 0.3],
        ],
    )
    react = Atoms(symbols=["Pt", "Pt", "O", "H"], positions=pos, pbc=False)
    prod = Atoms(
        symbols=["Pt", "Pt", "H", "O"],
        positions=np.vstack([pos[:2], pos[3:4], pos[2:3]]),
        pbc=False,
    )
    _align_endpoints_blockwise(react, prod, n_slab=1, n_core=1, n_ads=2)
    np.testing.assert_array_almost_equal(
        prod.get_positions()[2:4], react.get_positions()[2:4]
    )


def test_interpolate_path_accepts_block_dims_for_gas_adsorbate() -> None:
    sym = ["Pt", "Pt", "H"]
    pos = np.random.default_rng(0).random((3, 3))
    a1 = Atoms(symbols=sym, positions=pos, pbc=False, cell=[20, 20, 20])
    a2 = Atoms(symbols=sym, positions=pos.copy(), pbc=False, cell=[20, 20, 20])
    out = interpolate_path(
        a1,
        a2,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=2,
        n_adsorbate_mobile=1,
    )
    assert len(out) == 2 + 2
    assert len(out[0]) == 3 and len(out[-1]) == 3
