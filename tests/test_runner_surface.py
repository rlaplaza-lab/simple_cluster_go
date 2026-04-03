"""Tests for generic surface-runner helpers (scgo.runner_surface)."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, graphene
from ase.io import write

from scgo.runner_surface import (
    make_surface_config,
    read_full_composition_from_first_xyz,
)
from scgo.surface.config import SurfaceSystemConfig

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
# read_full_composition_from_first_xyz
# ---------------------------------------------------------------------------


class TestReadFullComposition:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            read_full_composition_from_first_xyz(tmp_path / "nope")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "final_unique_minima"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No final_unique_minima"):
            read_full_composition_from_first_xyz(empty)

    def test_happy_path(self, tmp_path: Path) -> None:
        d = tmp_path / "final_unique_minima"
        d.mkdir()
        atoms = Atoms("Cu4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        write(str(d / "min_01.xyz"), atoms)
        result = read_full_composition_from_first_xyz(d)
        assert result == ["Cu", "Cu", "Cu", "Cu"]
