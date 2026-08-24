"""Partition-aware pre-pair minima dedupe for slab-search TS runs (F1/F2).

For ``slab_is_search_target`` system types the dedupe window must cover the
mobile partition ``[fixed | top layers | adsorbate]`` tail only — matching the
GO-phase ``search_mobile_count`` contract — so distinct top-layer registries
survive and frozen-slab geometry cannot dilute the comparison.
"""

from __future__ import annotations

from typing import Any

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import AdsorbateDefinition
from scgo.ts_search import transition_state_run as ts_run_mod
from scgo.utils.helpers import get_cluster_formula


def _prepared_search_config() -> tuple[SurfaceSystemConfig, int, int]:
    """Return a slab-search config ordered [fixed...][mobile...] plus sizes."""
    from scgo.surface.partition import prepare_slab_search_surface_config

    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    return cfg, int(part.n_fixed), len(cfg.slab)


def _capture_pairing_minima(
    monkeypatch: pytest.MonkeyPatch,
    minima_by_formula: dict[str, list[tuple[float, Atoms]]],
) -> dict[str, Any]:
    """Stub minima loading; capture the deduped minima reaching pair selection."""
    captured: dict[str, Any] = {}

    def _fake_load(_minima_dir, composition, prefer_final_unique=True):
        captured["loaded_composition"] = list(composition)
        return dict(minima_by_formula)

    def _fake_select_pairs(minima, **_kwargs):
        captured["pairing_minima"] = list(minima)
        return []

    monkeypatch.setattr(ts_run_mod, "load_minima_by_composition", _fake_load)
    monkeypatch.setattr(ts_run_mod, "select_structure_pairs", _fake_select_pairs)
    return captured


def _run_ts_until_pairing(
    *,
    composition: list[str],
    system_type: str,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: AdsorbateDefinition | None,
    tmp_path,
) -> None:
    ts_run_mod.run_transition_state_search(
        composition=composition,
        system_type=system_type,  # type: ignore[arg-type]
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
        verbosity=0,
        use_torchsim=False,
        use_parallel_neb=False,
    )


def test_surface_adsorbate_dedupe_keeps_distinct_top_layer_registries(
    monkeypatch, tmp_path
):
    """Adsorbate-identical minima with distinct mobile top layers stay unique."""
    cfg, n_fixed, n_slab = _prepared_search_config()
    z_top = float(cfg.slab.positions[n_fixed:, 2].max())
    xy = cfg.slab.positions[n_fixed, :2].copy()
    oh_z = z_top + 1.6
    oh = [[xy[0], xy[1], oh_z], [xy[0], xy[1], oh_z + 0.97]]

    def _combined(shift_first_mobile: bool) -> Atoms:
        atoms = cfg.slab.copy() + Atoms("OH", positions=oh)
        if shift_first_mobile:
            pos = atoms.get_positions()
            pos[n_fixed, 0] += 0.5
            atoms.set_positions(pos)
        return atoms

    m_a = (0.0, _combined(False))
    m_b = (0.005, _combined(True))
    m_dup = (0.002, m_a[1].copy())

    formula = get_cluster_formula(list(cfg.slab.get_chemical_symbols()) + ["O", "H"])
    captured = _capture_pairing_minima(monkeypatch, {formula: [m_dup, m_b, m_a]})

    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    _run_ts_until_pairing(
        composition=["O", "H"],
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads_def,
        tmp_path=tmp_path,
    )

    kept = captured["pairing_minima"]
    assert len(kept) == 2
    energies = sorted(e for e, _ in kept)
    assert energies[0] == pytest.approx(0.0, abs=1e-9)
    assert energies[1] == pytest.approx(0.005, abs=1e-9)


def test_surface_adsorbate_dedupe_still_collapses_true_duplicates(
    monkeypatch, tmp_path
):
    """Exact copies within the energy window collapse as before."""
    cfg, n_fixed, _n_slab = _prepared_search_config()
    z_top = float(cfg.slab.positions[n_fixed:, 2].max())
    xy = cfg.slab.positions[n_fixed, :2].copy()
    oh = [[xy[0], xy[1], z_top + 1.6], [xy[0], xy[1], z_top + 2.57]]

    def _combined(shift_first_mobile: bool) -> Atoms:
        atoms = cfg.slab.copy() + Atoms("OH", positions=oh)
        if shift_first_mobile:
            pos = atoms.get_positions()
            pos[n_fixed, 0] += 0.5
            atoms.set_positions(pos)
        return atoms

    formula = get_cluster_formula(list(cfg.slab.get_chemical_symbols()) + ["O", "H"])
    captured = _capture_pairing_minima(
        monkeypatch,
        {
            formula: [
                (0.0, _combined(False)),
                (0.002, _combined(False)),
                (0.01, _combined(False)),
                # Different energy bin: always kept, keeps >= 2 minima for pairing.
                (5.0, _combined(True)),
            ]
        },
    )

    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    _run_ts_until_pairing(
        composition=["O", "H"],
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads_def,
        tmp_path=tmp_path,
    )

    kept = captured["pairing_minima"]
    assert len(kept) == 2
    energies = sorted(e for e, _ in kept)
    assert energies == pytest.approx([0.0, 5.0])


def test_bare_surface_dedupe_ignores_frozen_slab_and_keeps_top_layers(
    monkeypatch, tmp_path
):
    """Bare ``surface`` compares the mobile tail, not the full frozen slab."""
    cfg, n_fixed, n_slab = _prepared_search_config()

    def _slab_only(shift_first_mobile: bool) -> Atoms:
        atoms = cfg.slab.copy()
        if shift_first_mobile:
            pos = atoms.get_positions()
            pos[n_fixed, 1] += 0.5
            atoms.set_positions(pos)
        return atoms

    m_a = (0.0, _slab_only(False))
    m_b = (0.004, _slab_only(True))
    m_dup = (0.001, _slab_only(False))

    formula = get_cluster_formula(list(cfg.slab.get_chemical_symbols()))
    captured = _capture_pairing_minima(monkeypatch, {formula: [m_b, m_dup, m_a]})

    _run_ts_until_pairing(
        composition=[],
        system_type="surface",
        surface_config=cfg,
        adsorbate_definition=None,
        tmp_path=tmp_path,
    )

    kept = captured["pairing_minima"]
    assert len(kept) == 2
    energies = sorted(e for e, _ in kept)
    assert energies[0] == pytest.approx(0.0, abs=1e-9)
    assert energies[1] == pytest.approx(0.004, abs=1e-9)
