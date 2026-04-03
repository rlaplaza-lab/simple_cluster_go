import pytest
from ase import Atoms

from scgo.ase_ga_patches.population import Population
from scgo.database.metadata import add_metadata
from tests.test_utils import create_paired_rngs


class FakeDC:
    def __init__(self, candidates):
        self._cands = candidates

    def get_all_relaxed_candidates(self, only_new=False, use_extinct=False):
        return list(self._cands)

    def get_participation_in_pairing(self):
        # No history for this simple fake
        return ({}, set())


def _make_candidate(symbols, raw_score, confid, relax_id):
    a = Atoms(symbols)
    a.set_cell([10.0, 10.0, 10.0])
    a.set_pbc(False)
    # Add metadata used by get_raw_score
    add_metadata(a, raw_score=raw_score)
    a.info["confid"] = confid
    a.info["relax_id"] = relax_id
    return a


def test_get_two_candidates_is_deterministic_with_seeded_rng():
    # Create three candidates with different raw scores
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)

    dc = FakeDC([c1, c2, c3])

    rng1, rng2 = create_paired_rngs(1234)

    pop1 = Population(dc, population_size=3, rng=rng1)
    pop2 = Population(dc, population_size=3, rng=rng2)

    pair1 = pop1.get_two_candidates()
    pair2 = pop2.get_two_candidates()

    assert pair1 is not None and pair2 is not None
    ids1 = tuple(sorted([pair1[0].info["confid"], pair1[1].info["confid"]]))
    ids2 = tuple(sorted([pair2[0].info["confid"], pair2[1].info["confid"]]))

    assert ids1 == ids2


def test_population_constructor_rejects_legacy_randomstate():
    import numpy as _np

    dc = FakeDC([])
    with pytest.raises(TypeError):
        Population(dc, population_size=2, rng=_np.random.RandomState(1))
