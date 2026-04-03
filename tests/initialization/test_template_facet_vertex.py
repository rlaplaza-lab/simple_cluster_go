"""Tests for facet-based template growth and vertex-based template deletion."""

import pytest
from ase import Atoms
from ase.cluster import Icosahedron

from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
)
from scgo.initialization.templates import (
    grow_template_via_facets,
    remove_atoms_from_vertices,
)
from tests.test_utils import assert_cluster_valid


class TestRemoveAtomsFromVertices:
    """Tests for vertex-based bulk removal."""

    def test_remove_exact_match(self, rng):
        """Remove few atoms with target composition (single round)."""
        base = Icosahedron("Pt", 2)  # 13 atoms
        base.center()
        comp = ["Pt"] * 10
        result = remove_atoms_from_vertices(
            base,
            3,
            target_composition=comp,
            connectivity_factor=CONNECTIVITY_FACTOR,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            rng=rng,
        )
        assert result is not None
        assert_cluster_valid(result, comp)

    def test_remove_with_composition(self, rng):
        """Vertex removal preserves exact composition."""
        base = Icosahedron("Pt", 3)  # 55 atoms
        base.center()
        comp = ["Pt"] * 50
        result = remove_atoms_from_vertices(
            base,
            5,
            target_composition=comp,
            connectivity_factor=CONNECTIVITY_FACTOR,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            rng=rng,
        )
        assert result is not None
        assert_cluster_valid(result, comp)

    def test_remove_multi_round(self, rng):
        """Multi-round removal when n_remove exceeds vertices (55 -> 35)."""
        base = Icosahedron("Pt", 3)
        base.center()
        comp = ["Pt"] * 35
        result = remove_atoms_from_vertices(
            base,
            20,
            target_composition=comp,
            connectivity_factor=CONNECTIVITY_FACTOR,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 35

    def test_remove_no_composition(self, rng):
        """Vertex removal without target_composition."""
        base = Icosahedron("Pt", 2)
        base.center()
        result = remove_atoms_from_vertices(
            base,
            3,
            target_composition=None,
            connectivity_factor=CONNECTIVITY_FACTOR,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 10

    def test_remove_zero_returns_copy(self, rng):
        """Removing zero atoms returns copy of cluster."""
        base = Icosahedron("Pt", 2)
        base.center()
        result = remove_atoms_from_vertices(
            base, 0, connectivity_factor=CONNECTIVITY_FACTOR, rng=rng
        )
        assert result is not None
        assert len(result) == len(base)
        assert result is not base

    def test_remove_fewer_than_four_atoms_returns_none(self, rng):
        """Cluster with <4 atoms cannot use vertex removal (no hull)."""
        base = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 1.73, 0]])
        result = remove_atoms_from_vertices(
            base,
            1,
            connectivity_factor=CONNECTIVITY_FACTOR,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            rng=rng,
        )
        assert result is None

    def test_remove_too_many_raises(self, rng):
        """Removing >= len(cluster) raises ValueError."""
        base = Icosahedron("Pt", 2)
        base.center()
        with pytest.raises(ValueError, match="Cannot remove"):
            remove_atoms_from_vertices(
                base,
                13,
                connectivity_factor=CONNECTIVITY_FACTOR,
                rng=rng,
            )


class TestGrowTemplateViaFacets:
    """Tests for facet-based bulk template growth."""

    def test_grow_single_round(self, rng):
        """Add atoms in single round when facets suffice (13 -> 20)."""
        base = Icosahedron("Pt", 2)
        base.center()
        base.set_cell([30.0, 30.0, 30.0])
        comp = ["Pt"] * 20
        result = grow_template_via_facets(
            base,
            comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=30.0,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )
        assert result is not None
        assert_cluster_valid(result, comp)

    def test_grow_multi_round(self, rng):
        """Multi-round growth when more atoms than facets (13 -> 30)."""
        base = Icosahedron("Pt", 2)
        base.center()
        base.set_cell([30.0, 30.0, 30.0])
        comp = ["Pt"] * 30
        result = grow_template_via_facets(
            base,
            comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=30.0,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )
        if result is not None:
            assert len(result) == 30

    def test_grow_fallback_under_four_atoms(self, rng):
        """Seed with <4 atoms falls back to grow_from_seed."""
        base = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        base.set_cell([30.0, 30.0, 30.0])
        comp = ["Pt"] * 6
        result = grow_template_via_facets(
            base,
            comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=30.0,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )
        if result is not None:
            assert len(result) == 6

    def test_grow_nothing_to_add_returns_base(self, rng):
        """When target equals seed composition, return updated base."""
        base = Icosahedron("Pt", 2)
        base.center()
        base.set_cell([30.0, 30.0, 30.0])
        comp = ["Pt"] * 13
        result = grow_template_via_facets(
            base,
            comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=30.0,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )
        assert result is not None
        assert len(result) == 13
