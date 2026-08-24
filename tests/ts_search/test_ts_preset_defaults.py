"""Direct-call TS runs resolve omitted NEB knobs from per-system presets (F9).

``run_transition_state_search`` used to carry hardcoded signature defaults
(k=0.1, ``neb_steps="auto"``, ...) so callers bypassing ``coerce`` never saw
the preset values. Omitted knobs now resolve from ``get_ts_defaults``.
"""

from __future__ import annotations

from typing import Any

import pytest
from ase import Atoms

from scgo.system_types import AdsorbateDefinition
from scgo.ts_search import transition_state_run as ts_run_mod
from scgo.utils.helpers import get_cluster_formula
from tests.ts_search.test_ts_dedupe_partition import _prepared_search_config


def _stub_ts_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    minima_by_formula: dict[str, list[tuple[float, Atoms]]],
    *,
    system_type: str,
) -> dict[str, Any]:
    """Stub minima loading/pairing; capture the NebRunConfig reaching NEB."""
    captured: dict[str, Any] = {}

    def _fake_load(_minima_dir, composition, prefer_final_unique=True):
        return dict(minima_by_formula)

    def _fake_select_pairs(_minima, **_kwargs):
        return [(0, 1)]

    def _fake_find_transition_state(
        reactant, product, calculator, *, neb_cfg=None, **_kwargs
    ):
        captured["neb_cfg"] = neb_cfg
        return {
            "status": "failed",
            "pair_id": _kwargs.get("pair_id", "stub"),
            "error": "stub",
            "neb_converged": False,
        }

    monkeypatch.setattr(ts_run_mod, "load_minima_by_composition", _fake_load)
    monkeypatch.setattr(ts_run_mod, "select_structure_pairs", _fake_select_pairs)
    monkeypatch.setattr(
        ts_run_mod, "find_transition_state", _fake_find_transition_state
    )
    monkeypatch.setattr(ts_run_mod, "save_neb_result", lambda *a, **k: None)
    monkeypatch.setattr(
        ts_run_mod, "save_transition_state_results", lambda *a, **k: None
    )
    monkeypatch.setattr(ts_run_mod, "save_ts_network_metadata", lambda *a, **k: None)
    return captured


def test_gas_adsorbate_direct_call_resolves_adsorbate_presets(monkeypatch, tmp_path):
    """No NEB kwargs: gas_cluster_adsorbate gets spring 0.5 / steps 4000."""
    composition = ["Pt", "Pt", "O", "H"]
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    formula = get_cluster_formula(composition)
    atoms = Atoms(
        "Pt2OH",
        positions=[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [3.7, 1.2, 0.0], [3.7, 1.2, 1.0]],
    )
    captured = _stub_ts_pipeline(
        monkeypatch,
        {formula: [(0.0, atoms), (0.3, atoms.copy())]},
        system_type="gas_cluster_adsorbate",
    )

    ts_run_mod.run_transition_state_search(
        composition=composition,
        system_type="gas_cluster_adsorbate",
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        adsorbate_definition=ads_def,
        verbosity=0,
        use_torchsim=False,
        use_parallel_neb=False,
        dedupe_ts=False,
        tag_ts_in_db=False,
    )

    cfg = captured["neb_cfg"]
    assert cfg.neb_spring_constant == pytest.approx(0.5)
    assert cfg.neb_steps == 4000
    assert cfg.neb_n_images == 7
    assert cfg.neb_climb is True
    assert cfg.neb_interpolation_mic is False


def test_surface_adsorbate_direct_call_keeps_registry_safe_rotation(
    monkeypatch, tmp_path
):
    """surface_adsorbate resolves MIC on / remap on / lattice rotation off."""
    cfg_surf, _n_fixed, _n_slab = _prepared_search_config()
    combined = cfg_surf.slab.copy() + Atoms(
        "OH",
        positions=[
            [cfg_surf.slab.positions[0, 0], cfg_surf.slab.positions[0, 1], 9.0],
            [cfg_surf.slab.positions[0, 0], cfg_surf.slab.positions[0, 1], 9.97],
        ],
    )
    formula = get_cluster_formula(
        list(cfg_surf.slab.get_chemical_symbols()) + ["O", "H"]
    )
    captured = _stub_ts_pipeline(
        monkeypatch,
        {formula: [(0.0, combined), (0.3, combined.copy())]},
        system_type="surface_adsorbate",
    )
    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )

    ts_run_mod.run_transition_state_search(
        composition=["O", "H"],
        system_type="surface_adsorbate",
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=cfg_surf,
        adsorbate_definition=ads_def,
        verbosity=0,
        use_torchsim=False,
        use_parallel_neb=False,
        dedupe_ts=False,
        tag_ts_in_db=False,
    )

    cfg = captured["neb_cfg"]
    assert cfg.neb_spring_constant == pytest.approx(0.5)
    assert cfg.neb_steps == 4000
    assert cfg.neb_interpolation_mic is True
    assert cfg.neb_surface_cell_remap is True
    # Preset AND policy both disable free in-plane rotation for adsorbates.
    assert cfg.neb_surface_lattice_rotation is False
