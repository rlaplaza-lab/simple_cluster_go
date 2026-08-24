"""Regression tests: run-level ``verbosity`` must be forwarded, not defaulted.

The single-run path used to drop ``verbosity``, so ``run_trials`` (and every
algorithm below it) fell back to the default ``verbosity=1`` even when the
caller asked for a quiet run. These tests also cover gas batch init and NEB
helpers that previously inferred verbosity from the logger.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms.ga_common import ClusterStartGenerator
from scgo.exceptions import SCGOValidationError
from scgo.runner_api import run_go, run_go_campaign
from scgo.ts_search.parallel_neb import run_parallel_neb_search
from scgo.ts_search.transition_state import find_transition_state
from scgo.utils.logging import log_info_v
from scgo.utils.ts_runner_kwargs import NebRunConfig

CORE_LOGGER_NAME = "scgo.minima_search.core"


class _RecordingHandler(logging.Handler):
    """Collect every record that reaches it, regardless of level."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _gas_neb_cfg(**overrides) -> NebRunConfig:
    """Minimal gas-cluster NebRunConfig for verbosity forwarding stubs."""
    kwargs: dict = {
        "neb_n_images": 3,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": 1,
        "neb_climb": False,
        "neb_interpolation_method": "linear",
        "neb_align_endpoints": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_mic": False,
        "neb_tangent_method": "aseneb",
        "neb_surface_cell_remap": True,
        "neb_surface_lattice_rotation": True,
        "neb_surface_max_lattice_shift": 1,
        "n_slab": 0,
        "n_core_mobile": None,
        "n_adsorbate_mobile": None,
        "adsorbate_fragment_lengths": None,
        "max_endpoint_mismatch": None,
        "neb_prescreen_clash_distance": 1.0,
        "min_saddle_prominence": 0.10,
        "neb_max_spurious_barrier": 8.0,
        "layer_cluster_threshold_ang": 0.4,
        "neb_interpolation_bond_tolerance_a": 0.5,
        "adsorbate_definition": None,
        "connectivity_factor": None,
        "allow_cluster_fragmentation": False,
        "allow_adsorbate_surface_detachment": False,
        "enforce_adsorbate_subgraph_integrity": True,
        "system_type": "gas_cluster",
        "surface_config": None,
        "torchsim_params": {},
    }
    kwargs.update(overrides)
    return NebRunConfig(**kwargs)


@pytest.fixture
def captured_run_trials(monkeypatch):
    """Replace ``run_trials`` with a stub that records its keyword arguments."""
    captured: dict[str, object] = {}

    def _fake_run_trials(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("scgo.runner_go.run_trials", _fake_run_trials)
    monkeypatch.setattr("scgo.runner_go.get_calculator_class", lambda name: EMT)
    return captured


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_run_go_forwards_verbosity_to_run_trials(
    captured_run_trials, tmp_path, verbosity
):
    run_go(
        ["Pt", "Pt"],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=verbosity,
        output_dir=tmp_path,
    )

    assert captured_run_trials["verbosity"] == verbosity


def test_run_go_campaign_forwards_verbosity_to_run_trials(
    captured_run_trials, tmp_path
):
    run_go_campaign(
        [["Pt", "Pt"]],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=0,
        output_dir=tmp_path,
    )

    assert captured_run_trials["verbosity"] == 0


@pytest.mark.parametrize(
    "verbosity,expect_info",
    [(0, False), (1, True)],
)
def test_run_go_verbosity_gates_run_trials_info_logs(
    monkeypatch, tmp_path, verbosity, expect_info
):
    """``verbosity=0`` must silence INFO logging emitted from ``run_trials``."""
    handler = _RecordingHandler()
    core_logger = logging.getLogger(CORE_LOGGER_NAME)

    def _fake_run_trials(**kwargs):
        # ``configure_logging`` already ran inside the runner; re-enable INFO on
        # the root logger so this assertion isolates the verbosity gate itself
        # instead of the global log level.
        root = logging.getLogger()
        previous_level = root.level
        root.setLevel(logging.INFO)
        core_logger.addHandler(handler)
        try:
            log_info_v(
                core_logger,
                "run_trials info message",
                verbosity=kwargs.get("verbosity", 1),
            )
        finally:
            core_logger.removeHandler(handler)
            root.setLevel(previous_level)
        return []

    monkeypatch.setattr("scgo.runner_go.run_trials", _fake_run_trials)
    monkeypatch.setattr("scgo.runner_go.get_calculator_class", lambda name: EMT)

    run_go(
        ["Pt", "Pt"],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=verbosity,
        output_dir=tmp_path,
    )

    info_messages = [
        record.getMessage()
        for record in handler.records
        if record.levelno >= logging.INFO
    ]
    assert bool(info_messages) is expect_info


@pytest.mark.parametrize("verbosity", [0, 2])
def test_cluster_start_generator_forwards_verbosity_to_batch(monkeypatch, verbosity):
    """Gas ClusterStartGenerator must pass run verbosity into batch init."""
    captured: dict[str, object] = {}

    def _fake_batch(**kwargs):
        captured.update(kwargs)
        n = int(kwargs["n_structures"])
        return [Atoms("Pt2") for _ in range(n)]

    monkeypatch.setattr(
        "scgo.algorithms.ga_common.create_initial_cluster_batch", _fake_batch
    )

    ClusterStartGenerator(
        composition=["Pt", "Pt"],
        vacuum=8.0,
        rng=np.random.default_rng(0),
        population_size=3,
        verbosity=verbosity,
    )

    assert captured["verbosity"] == verbosity


@pytest.mark.parametrize("verbosity", [0, 2])
def test_find_transition_state_forwards_verbosity_to_interpolate_path(
    monkeypatch, tmp_path, verbosity
):
    """Serial NEB must thread run verbosity into interpolate_path."""
    captured: dict[str, object] = {}
    reactant = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    product = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    reactant.calc = EMT()
    product.calc = EMT()

    def _fake_interpolate(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after interpolate_path")

    monkeypatch.setattr(
        "scgo.ts_search.transition_state.interpolate_path", _fake_interpolate
    )

    result = find_transition_state(
        reactant,
        product,
        EMT(),
        output_dir=str(tmp_path),
        pair_id="0_1",
        verbosity=verbosity,
        neb_steps=1,
        n_images=3,
        align_endpoints=False,
    )

    assert captured["verbosity"] == verbosity
    assert result["error"] == "stop after interpolate_path"


@pytest.mark.parametrize("verbosity", [0, 2])
def test_parallel_neb_forwards_verbosity_to_interpolate_and_save(
    monkeypatch, tmp_path, verbosity
):
    """Parallel NEB must thread run verbosity into interpolate_path and save."""
    interpolate_captured: dict[str, object] = {}
    save_captured: dict[str, object] = {}

    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    b = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    minima = [(0.0, a), (1.0, b)]
    pairs = [(0, 1)]

    def _fake_prepare(reactant, product, neb_cfg):
        return reactant.copy(), product.copy()

    def _fake_interpolate(*args, **kwargs):
        interpolate_captured.update(kwargs)
        n_images = int(kwargs.get("n_images", 3))
        return [a.copy() for _ in range(n_images + 2)]

    def _fake_save(*args, **kwargs):
        save_captured.update(kwargs)

    def _fake_validate(*args, **kwargs):
        raise SCGOValidationError("reject path after interpolate")

    monkeypatch.setattr(
        "scgo.ts_search.parallel_neb.prepare_neb_endpoints", _fake_prepare
    )
    monkeypatch.setattr(
        "scgo.ts_search.parallel_neb.interpolate_path", _fake_interpolate
    )
    monkeypatch.setattr("scgo.ts_search.parallel_neb.save_neb_result", _fake_save)
    monkeypatch.setattr(
        "scgo.ts_search.parallel_neb.validate_initial_neb_path", _fake_validate
    )

    results, _meta = run_parallel_neb_search(
        pairs,
        minima,
        neb_cfg=_gas_neb_cfg(),
        run_dir=Path(tmp_path),
        rng=None,
        relaxer=MagicMock(),
        verbosity=verbosity,
    )

    assert interpolate_captured["verbosity"] == verbosity
    assert save_captured["verbosity"] == verbosity
    assert len(results) == 1
    assert results[0]["status"] == "skipped"
