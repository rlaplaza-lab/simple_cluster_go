"""WP6 small-fix tests: F3 MIC-knob warning + F12/F13 screen override plumbing."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from ase import Atoms

from scgo.ts_search import transition_state_run as ts_run_mod
from tests.ts_search.test_ts_dedupe_partition import (
    _capture_pairing_minima,
    _prepared_search_config,
    _run_ts_until_pairing,
)

LOGGER_NAME = "scgo.ts_search.transition_state_run"


class _ListHandler(logging.Handler):
    """Direct handler attachment; ``configure_logging`` disables propagation."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_logger_records(name: str) -> tuple[_ListHandler, logging.Logger]:
    handler = _ListHandler()
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    return handler, logger


def test_comparator_mic_false_warns_on_surface_ts_runs(monkeypatch, tmp_path):
    """``comparator_use_mic=False`` must warn: TS forces MIC for surface types."""
    cfg, _n_fixed, _n_slab = _prepared_search_config()
    cfg_no_mic = replace(cfg, comparator_use_mic=False)
    formula = "Pt8OH"
    _capture_pairing_minima(monkeypatch, {formula: []})
    handler, logger = _capture_logger_records(LOGGER_NAME)
    try:
        _run_ts_until_pairing(
            composition=["O", "H"],
            system_type="surface_adsorbate",
            surface_config=cfg_no_mic,
            adsorbate_definition=None,
            tmp_path=tmp_path,
        )
    finally:
        logger.removeHandler(handler)

    assert any(
        "comparator_use_mic=False affects GO comparators only" in r.getMessage()
        for r in handler.records
        if r.levelno >= logging.WARNING
    )


def test_comparator_mic_true_does_not_warn(monkeypatch, tmp_path):
    cfg, _n_fixed, _n_slab = _prepared_search_config()
    _capture_pairing_minima(monkeypatch, {"Pt8OH": []})
    handler, logger = _capture_logger_records(LOGGER_NAME)
    try:
        _run_ts_until_pairing(
            composition=["O", "H"],
            system_type="surface_adsorbate",
            surface_config=cfg,
            adsorbate_definition=None,
            tmp_path=tmp_path,
        )
    finally:
        logger.removeHandler(handler)

    warnings = [
        r.getMessage()
        for r in handler.records
        if r.levelno >= logging.WARNING and "comparator_use_mic" in r.getMessage()
    ]
    assert warnings == []


def test_idpp_screen_threads_prominence_barrier_and_bond_tolerance(monkeypatch):
    """Custom prominence/barrier/bond-tolerance reach the screen gates (F12/F13)."""
    captured: dict = {}

    def _fake_interpolate(*_args, **kwargs):
        captured["interpolate"] = kwargs
        return []

    def _fake_profile(_energies, **kwargs):
        captured["profile"] = kwargs

    def _fake_priority(_energies, *, min_saddle_prominence=0.40):
        captured["priority_prominence"] = min_saddle_prominence
        return (2, 0.9, 0.9)

    monkeypatch.setattr(ts_run_mod, "interpolate_path", _fake_interpolate)
    monkeypatch.setattr(ts_run_mod, "validate_initial_neb_path", lambda *a, **k: None)
    monkeypatch.setattr(
        ts_run_mod, "validate_initial_neb_energy_profile", _fake_profile
    )
    monkeypatch.setattr(ts_run_mod, "idpp_band_optimization_priority", _fake_priority)
    monkeypatch.setattr(
        ts_run_mod,
        "_evaluate_bands_in_chunks",
        lambda bands, _relaxer, **_k: [[0.0, 0.9, 0.2, 0.9, 0.0] for _ in bands],
    )

    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    kept = ts_run_mod._prioritize_adsorbate_pairs_by_idpp(
        [(0, 1)],
        [(0.0, h2), (0.3, h2.copy())],
        max_pairs=1,
        relaxer=object(),
        neb_n_images=3,
        neb_interpolation_method="linear",
        neb_interpolation_mic=False,
        neb_align_endpoints=False,
        neb_perturb_sigma=0.0,
        rng=None,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=None,
        n_adsorbate_mobile=None,
        adsorbate_fragment_lengths=None,
        neb_surface_cell_remap=False,
        neb_surface_lattice_rotation=False,
        neb_surface_max_lattice_shift=0,
        max_endpoint_mismatch=1.25,
        neb_prescreen_clash_distance=0.7,
        min_saddle_prominence=0.63,
        neb_max_spurious_barrier=4.2,
        neb_interpolation_bond_tolerance_a=0.33,
        parallel_neb_max_batch_atoms=None,
        parallel_neb_max_bands=None,
        logger=logging.getLogger(LOGGER_NAME),
        verbosity=0,
    )

    assert kept == [(0, 1)]
    assert captured["profile"]["min_saddle_prominence"] == pytest.approx(0.63)
    assert captured["profile"]["max_spurious_barrier"] == pytest.approx(4.2)
    assert captured["priority_prominence"] == pytest.approx(0.63)
    assert captured["interpolate"][
        "neb_interpolation_bond_tolerance_a"
    ] == pytest.approx(0.33)
