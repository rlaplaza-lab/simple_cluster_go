import numpy as np
import pytest
from ase import Atoms

from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
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
    op1 = CutAndSplicePairing(slab=Atoms(), n_top=4, blmin=blmin, rng=rng1)
    op2 = CutAndSplicePairing(slab=Atoms(), n_top=4, blmin=blmin, rng=rng2)

    child1 = op1.cross(p1, p2)
    child2 = op2.cross(p1, p2)

    assert child1 is not None
    assert child2 is not None

    # Stoichiometry (element counts) should be preserved
    assert sorted(child1.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())
    assert sorted(child2.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())

    # Deterministic for identical seeds
    assert np.allclose(child1.get_positions(), child2.get_positions())


def test_cut_and_splice_constructor_rejects_legacy_randomstate():
    import numpy as _np

    with pytest.raises(TypeError):
        # Legacy RandomState should be rejected after enforcing Generator-only policy
        CutAndSplicePairing(
            slab=Atoms(), n_top=2, blmin={}, rng=_np.random.RandomState(42)
        )
