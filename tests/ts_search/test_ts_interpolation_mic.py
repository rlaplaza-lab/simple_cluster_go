"""NEB interpolation MIC flag (`neb_interpolation_mic`) for periodic endpoints."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.ts_search.transition_state import find_transition_state


def test_find_transition_state_records_neb_interpolation_mic(
    h2_reactant, h2_product, tmp_path
):
    """Result dict includes `neb_interpolation_mic`; default matches gas-phase (False)."""
    out = str(tmp_path / "neb_mic")
    result = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=out,
        pair_id="mic_default",
        n_images=3,
        fmax=0.5,
        neb_steps=5,
        verbosity=0,
    )
    assert result.get("neb_interpolation_mic") is False

    result_mic = find_transition_state(
        h2_reactant,
        h2_product,
        calculator=EMT(),
        output_dir=out,
        pair_id="mic_true",
        n_images=3,
        fmax=0.5,
        neb_steps=5,
        verbosity=0,
        neb_interpolation_mic=True,
    )
    assert result_mic.get("neb_interpolation_mic") is True


@pytest.mark.slow
def test_find_transition_state_slab_emt_runs_with_mic(
    tmp_path,
):
    """Small slab + one Pt adsorbate: two lateral positions, EMT NEB runs to completion."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    z0 = slab.get_positions()[:, 2].max() + 1.5

    a = slab.copy() + Atoms("Pt", positions=[[1.0, 1.0, z0]])
    a.calc = EMT()

    b = slab.copy() + Atoms("Pt", positions=[[2.0, 2.0, z0]])
    b.calc = EMT()

    out = tmp_path / "neb_slab"
    out.mkdir(parents=True, exist_ok=True)

    result = find_transition_state(
        a,
        b,
        calculator=EMT(),
        output_dir=str(out),
        pair_id="slab_pt",
        n_images=3,
        fmax=0.3,
        neb_steps=30,
        verbosity=0,
        neb_interpolation_mic=True,
        climb=False,
    )

    assert "status" in result
    assert result.get("neb_interpolation_mic") is True
    assert result.get("n_images") == 3
