"""Tests for scgo.optimization.algorithm_select."""

from scgo.optimization.algorithm_select import select_scgo_minima_algorithm


def test_select_matches_surface_trimer_uses_bh() -> None:
    assert select_scgo_minima_algorithm(3, "surface_cluster") == "bh"


def test_select_surface_five_uses_ga() -> None:
    assert select_scgo_minima_algorithm(5, "surface_cluster") == "ga"


def test_select_gas_dimer_uses_simple() -> None:
    assert select_scgo_minima_algorithm(2, "gas_cluster") == "simple"
