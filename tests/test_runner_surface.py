"""Tests for generic surface-runner helpers (scgo.runner_surface)."""

import numpy as np
from ase import Atoms
from ase.build import fcc111, graphene

from scgo.runner_surface import make_surface_config
from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.surface.config import SurfaceSystemConfig
from scgo.utils.helpers import get_cluster_formula

# ---------------------------------------------------------------------------
# make_surface_config
# ---------------------------------------------------------------------------


def _graphene_slab() -> Atoms:
    slab = graphene(size=(3, 3, 1), vacuum=12.0)
    slab.pbc = True
    return slab


def _pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 3), vacuum=10.0)


class TestMakeSurfaceConfig:
    def test_returns_surface_system_config(self) -> None:
        cfg = make_surface_config(_graphene_slab())
        assert isinstance(cfg, SurfaceSystemConfig)

    def test_slab_is_stored(self) -> None:
        slab = _graphene_slab()
        cfg = make_surface_config(slab)
        assert len(cfg.slab) == len(slab)
        np.testing.assert_array_equal(cfg.slab.get_positions(), slab.get_positions())

    def test_defaults(self) -> None:
        cfg = make_surface_config(_graphene_slab())
        assert cfg.fix_all_slab_atoms is True
        assert cfg.comparator_use_mic is True
        assert cfg.adsorption_height_min == 2.0
        assert cfg.adsorption_height_max == 3.5
        assert cfg.max_placement_attempts == 500

    def test_custom_kwargs(self) -> None:
        cfg = make_surface_config(
            _graphene_slab(),
            adsorption_height_min=1.5,
            adsorption_height_max=4.0,
            fix_all_slab_atoms=False,
            comparator_use_mic=False,
            max_placement_attempts=200,
        )
        assert cfg.adsorption_height_min == 1.5
        assert cfg.adsorption_height_max == 4.0
        assert cfg.fix_all_slab_atoms is False
        assert cfg.comparator_use_mic is False
        assert cfg.max_placement_attempts == 200

    def test_works_with_metal_slab(self) -> None:
        cfg = make_surface_config(_pt_slab())
        assert isinstance(cfg, SurfaceSystemConfig)
        assert "Pt" in cfg.slab.get_chemical_symbols()


# ---------------------------------------------------------------------------
# full_adsorbate_slab_composition
# ---------------------------------------------------------------------------


def test_full_adsorbate_slab_composition_matches_ga_template_order() -> None:
    """Same ordering as ga_go surface Atoms: slab symbols then adsorbate."""
    slab = _graphene_slab()
    cfg = make_surface_config(slab)
    adsorbate = ["Pt", "Pt", "Pt"]
    full = full_adsorbate_slab_composition(adsorbate, cfg)
    from_ga_style = list(slab.get_chemical_symbols()) + adsorbate
    assert full == from_ga_style
    assert get_cluster_formula(full) == get_cluster_formula(from_ga_style)
