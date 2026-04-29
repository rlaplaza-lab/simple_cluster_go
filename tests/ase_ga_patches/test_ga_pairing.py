import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
)
from tests.test_utils import create_paired_rngs


def test_cut_and_splice_preserves_stoichiometry_and_is_deterministic(au2pt2_atoms, rng):
    # Prepare two parent structures
    p1 = au2pt2_atoms.copy()
    p2 = au2pt2_atoms.copy()
    p1.info["confid"] = "p1"
    p2.info["confid"] = "p2"

    # Use identical seeds to test determinism across different operator instances
    rng1, rng2 = create_paired_rngs(123)

    # minimal bond-length dict to avoid KeyError in atoms_too_close
    pt = 78
    au = 79
    blmin = {(pt, pt): 0.1, (pt, au): 0.1, (au, au): 0.1}
    op1 = CutAndSplicePairing(
        slab=Atoms(), n_top=4, blmin=blmin, system_type="gas_cluster", rng=rng1
    )
    op2 = CutAndSplicePairing(
        slab=Atoms(), n_top=4, blmin=blmin, system_type="gas_cluster", rng=rng2
    )

    child1 = op1.cross(p1, p2)
    child2 = op2.cross(p1, p2)

    assert child1 is not None
    assert child2 is not None

    # Stoichiometry (element counts) should be preserved
    assert sorted(child1.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())
    assert sorted(child2.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())

    # Deterministic for identical seeds
    assert np.allclose(child1.get_positions(), child2.get_positions())


def test_dual_cut_and_splice_returns_offspring(pt3_atoms):
    n_top = len(pt3_atoms)
    blmin = closest_distances_generator(
        get_all_atom_types(pt3_atoms, range(n_top)),
        ratio_of_covalent_radii=0.7,
    )
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())
    primary = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        minfrac=0.3,
        system_type="gas_cluster",
        rng=np.random.default_rng(11),
    )
    exploratory = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        minfrac=0.15,
        system_type="gas_cluster",
        rng=np.random.default_rng(22),
    )
    dual = DualCutAndSplicePairing(
        primary,
        exploratory,
        0.5,
        rng=np.random.default_rng(99),
    )
    p1 = pt3_atoms.copy()
    p2 = pt3_atoms.copy()
    p1.info["confid"] = "a"
    p2.info["confid"] = "b"
    child, _desc = dual.get_new_individual([p1, p2])
    assert child is not None
    assert len(child) == n_top
    assert child.get_chemical_symbols() == pt3_atoms.get_chemical_symbols()


def test_create_ga_pairing_returns_single_operator_when_explore_probability_zero(
    pt3_atoms,
):
    from numpy.random import default_rng

    from scgo.algorithms.ga_common import create_ga_pairing

    pairing = create_ga_pairing(
        pt3_atoms,
        len(pt3_atoms),
        default_rng(0),
        exploratory_crossover_probability=0.0,
    )
    assert isinstance(pairing, CutAndSplicePairing)


def test_cut_and_splice_constructor_rejects_legacy_randomstate():
    import numpy as _np

    with pytest.raises(TypeError):
        # Legacy RandomState should be rejected after enforcing Generator-only policy
        CutAndSplicePairing(
            slab=Atoms(),
            n_top=2,
            blmin={},
            system_type="gas_cluster",
            rng=_np.random.RandomState(42),
        )
