"""Tunable defaults for NEB minima pair selection.

Hard gates and ranking scales used by
``scgo.ts_search.transition_state_io.select_structure_pairs``. Override via
``ts_params`` (see docs ``parameters.rst``). Values depend on whether the system
uses a surface and/or an adsorbate.

Regime mapping (deliberate, see ``pair_selection_param_defaults``):

- ``gas_cluster`` / ``surface_cluster`` -> BARE regime: a supported metal
  cluster IS the whole mobile part, so bare distinctness + energy-gap ranking
  applies; there is no separate adsorbate block to gate.
- ``gas_cluster_adsorbate`` / ``surface_cluster_adsorbate`` /
  ``surface_adsorbate`` -> ADSORBATE regime with the core-RMS hard gate. For
  ``surface_adsorbate`` on a searchable top layer the runner infers the mobile
  top-layer block as the "core" (see ``run_transition_state_search``), so
  ``pair_core_rms_max`` bounds top-layer registry drift between paired minima;
  whole-layer registry variants that exceed it are intentionally filtered
  before NEB budget is spent.
"""

from __future__ import annotations

from typing import Any

# Hard gate: max core RMS (Å) for adsorbate+core pairing.
DEFAULT_PAIR_CORE_RMS_MAX_GAS = 1.5
DEFAULT_PAIR_CORE_RMS_MAX_SURFACE = 2.0

# Soft ranking scales (Å or eV) and weights. Adsorbate path prefers similar
# cores + mid energy gap + some adsorbate site displacement; bare path prefers
# structural distinctness + mid gap.
_ADSORBATE_GAS: dict[str, float] = {
    "pair_score_gap_center": 0.50,
    "pair_score_gap_width": 0.45,
    "pair_score_cum_scale": 0.08,
    "pair_score_mismatch_scale": 0.35,
    "pair_score_core_rms_scale": 0.35,
    "pair_score_w_gap": 0.25,
    "pair_score_w_distinct": 0.20,
    "pair_score_w_mismatch": 0.25,
    "pair_score_w_core": 0.30,
}

_ADSORBATE_SURFACE: dict[str, float] = {
    "pair_score_gap_center": 0.55,
    "pair_score_gap_width": 0.50,
    "pair_score_cum_scale": 0.10,
    "pair_score_mismatch_scale": 0.45,
    "pair_score_core_rms_scale": 0.45,
    "pair_score_w_gap": 0.25,
    "pair_score_w_distinct": 0.20,
    "pair_score_w_mismatch": 0.25,
    "pair_score_w_core": 0.30,
}

_BARE_GAS: dict[str, float] = {
    "pair_score_gap_center": 0.30,
    "pair_score_gap_width": 0.40,
    "pair_score_cum_scale": 0.09,
    "pair_score_mismatch_scale": 0.35,
    "pair_score_core_rms_scale": 0.35,
    "pair_score_w_gap": 0.50,
    "pair_score_w_distinct": 0.35,
    "pair_score_w_mismatch": 0.15,
    "pair_score_w_core": 0.0,
}

_BARE_SURFACE: dict[str, float] = {
    "pair_score_gap_center": 0.45,
    "pair_score_gap_width": 0.55,
    "pair_score_cum_scale": 0.12,
    "pair_score_mismatch_scale": 0.45,
    "pair_score_core_rms_scale": 0.45,
    "pair_score_w_gap": 0.50,
    "pair_score_w_distinct": 0.35,
    "pair_score_w_mismatch": 0.15,
    "pair_score_w_core": 0.0,
}


def pair_selection_param_defaults(
    *,
    surface_aware: bool,
    adsorbate_aware: bool,
) -> dict[str, Any]:
    """Return default pair-selection knobs for a system regime."""
    if adsorbate_aware:
        score = dict(_ADSORBATE_SURFACE if surface_aware else _ADSORBATE_GAS)
        score["pair_core_rms_max"] = (
            DEFAULT_PAIR_CORE_RMS_MAX_SURFACE
            if surface_aware
            else DEFAULT_PAIR_CORE_RMS_MAX_GAS
        )
        return score
    score = dict(_BARE_SURFACE if surface_aware else _BARE_GAS)
    score["pair_core_rms_max"] = None
    return score
