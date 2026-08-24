"""Strict CPU EMT end-to-end coverage for all six ``system_type`` values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.param_presets import get_testing_params, get_ts_search_params
from scgo.runner_api import run_go, run_go_ts
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.presets import (
    make_defected_graphite_surface_config,
    make_n_doped_graphite_surface_config,
)
from scgo.system_types import SystemType, get_system_policy
from scgo.utils.helpers import get_cluster_formula
from tests.constants import PT4_EMT_BARRIER_EV
from tests.helpers import (
    assert_e2e_go_ts_summary,
    assert_e2e_minima_list,
    assert_supported_cluster_binding,
)

SEED = 42
CONNECTIVITY = 1.8


def _adsorbates_oh(*, n: int = 1) -> list[Atoms]:
    out: list[Atoms] = []
    for i in range(n):
        shift = float(2.2 * i)
        out.append(
            Atoms(
                symbols=["O", "H"],
                positions=[[shift, 0.0, 0.0], [shift, 0.0, 0.96]],
            )
        )
    return out


def _pt111_surface_config() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )


def _defected_graphite() -> SurfaceSystemConfig:
    return make_defected_graphite_surface_config(
        slab_layers=2, slab_repeat_xy=1, n_vacancies=1, seed=0
    )


def _n_doped_graphite() -> SurfaceSystemConfig:
    return make_n_doped_graphite_surface_config(
        slab_layers=3, slab_repeat_xy=2, n_dopants=1, seed=0
    )


@dataclass(frozen=True)
class EmtE2eCase:
    system_type: SystemType
    mode: Literal["go", "go_ts"]
    composition: list[str]
    surface_config: SurfaceSystemConfig | None = None
    adsorbates: list[Atoms] | None = None
    expected_mobile_atoms: int = 0
    n_core_mobile: int | None = None
    adsorbate_fragment_lengths: list[int] | None = None
    expected_formula: str | None = None
    connectivity_factor: float | None = None
    ga_overrides: dict = field(default_factory=dict)
    ts_overrides: dict = field(default_factory=dict)
    tag_final_minima: bool = False
    expect_xyz_export: bool = False
    freeze_adsorbate_internal_geometry: bool = False
    check_supported_binding: bool = False
    require_ts_candidates: bool = False
    barrier_range: tuple[float, float] | None = None
    seed: int = SEED


EMT_E2E_CASES = [
    EmtE2eCase(
        system_type="gas_cluster",
        mode="go_ts",
        composition=["Pt", "Pt", "Pt", "Pt"],
        expected_mobile_atoms=4,
        expected_formula="Pt4",
        ga_overrides={
            "niter": 2,
            "population_size": 4,
            "niter_local_relaxation": 5,
        },
        ts_overrides={
            "neb_steps": 80,
            "max_pairs": 1,
            "neb_n_images": 5,
        },
        # Gas-cluster CI budget legitimately yields no interior saddle for the
        # selected pair (its highest-energy image is an endpoint), so this case
        # keeps require_ts_candidates=False — matching the GPU matrix gas cases.
        # Only surface-cluster cases opt into the "trial of fire" success bar.
        require_ts_candidates=False,
        barrier_range=PT4_EMT_BARRIER_EV,
    ),
    EmtE2eCase(
        system_type="surface_cluster",
        mode="go",
        composition=["Pt", "Pt"],
        surface_config=_pt111_surface_config(),
        expected_mobile_atoms=2,
        n_core_mobile=2,
        ga_overrides={
            "niter": 1,
            "population_size": 2,
            "offspring_fraction": 0.5,
            "niter_local_relaxation": 30,
            "batch_size": 2,
            "n_jobs_population_init": 1,
            "early_stopping_niter": 0,
        },
        check_supported_binding=True,
    ),
    EmtE2eCase(
        system_type="gas_cluster_adsorbate",
        mode="go",
        composition=["Pt", "Pt", "Pt"],
        adsorbates=_adsorbates_oh(n=1),
        expected_mobile_atoms=5,
        n_core_mobile=3,
        adsorbate_fragment_lengths=[2],
        connectivity_factor=CONNECTIVITY,
        ga_overrides={"niter": 1, "population_size": 2, "niter_local_relaxation": 5},
        tag_final_minima=True,
        expect_xyz_export=True,
    ),
    EmtE2eCase(
        system_type="surface_cluster_adsorbate",
        mode="go",
        composition=["Pt", "Pt"],
        surface_config=_pt111_surface_config(),
        adsorbates=_adsorbates_oh(n=1),
        expected_mobile_atoms=4,
        n_core_mobile=2,
        adsorbate_fragment_lengths=[2],
        connectivity_factor=CONNECTIVITY,
        ga_overrides={
            "niter": 2,
            "population_size": 4,
            "niter_local_relaxation": 50,
            "batch_size": 2,
            "n_jobs_population_init": 1,
            "early_stopping_niter": 0,
        },
        freeze_adsorbate_internal_geometry=True,
        check_supported_binding=True,
    ),
    EmtE2eCase(
        system_type="surface",
        mode="go",
        composition=[],
        surface_config=_defected_graphite(),
        expected_mobile_atoms=0,
        n_core_mobile=0,
        connectivity_factor=CONNECTIVITY,
        ga_overrides={"niter": 1, "population_size": 4, "niter_local_relaxation": 5},
        seed=0,
    ),
    EmtE2eCase(
        system_type="surface_adsorbate",
        mode="go_ts",
        composition=[],
        surface_config=_n_doped_graphite(),
        adsorbates=_adsorbates_oh(n=1),
        expected_mobile_atoms=2,
        n_core_mobile=0,
        adsorbate_fragment_lengths=[2],
        expected_formula="HO",
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        ga_overrides={
            # EMT poorly describes the H-O bond and relaxes it apart during the
            # adsorbate-only GO; the post-relaxation cluster validation ("Cluster
            # is not connected") then rejects every candidate, emptying the GA
            # population (0 minima). Freezing the adsorbate internal geometry
            # keeps OH intact (mirrors the sibling surface_cluster_adsorbate
            # case). The budget bump is secondary safety margin only.
            "niter": 2,
            "population_size": 8,
            "niter_local_relaxation": 10,
        },
        ts_overrides={
            "max_pairs": 1,
            "neb_steps": 15,
            "write_timing_json": False,
            "connectivity_factor": CONNECTIVITY,
        },
        seed=0,
    ),
]


def _build_go_params(case: EmtE2eCase) -> dict:
    params = get_testing_params()
    if case.connectivity_factor is not None:
        params["connectivity_factor"] = case.connectivity_factor
    if case.tag_final_minima:
        params["tag_final_minima"] = True
    if case.freeze_adsorbate_internal_geometry:
        params["freeze_adsorbate_internal_geometry"] = True
    if case.surface_config is not None:
        params["surface_config"] = case.surface_config
    params["optimizer_params"]["ga"].update(case.ga_overrides)
    return params


def _build_ts_params(case: EmtE2eCase) -> dict:
    params = {
        **get_ts_search_params(
            system_type=case.system_type,
            surface_config=case.surface_config,
            calculator="EMT",
            calculator_kwargs={},
            seed=case.seed,
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
        "max_pairs": 1,
        "neb_steps": 50,
    }
    params.update(case.ts_overrides)
    if case.connectivity_factor is not None:
        params.setdefault("connectivity_factor", case.connectivity_factor)
    return params


def _expected_formula(case: EmtE2eCase) -> str:
    if case.expected_formula is not None:
        return case.expected_formula
    if case.composition:
        symbols = list(case.composition)
        if case.adsorbates is not None:
            for ads in case.adsorbates:
                symbols.extend(ads.get_chemical_symbols())
        return get_cluster_formula(symbols)
    if case.surface_config is not None:
        return case.surface_config.name or "surface"
    return ""


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case", EMT_E2E_CASES, ids=lambda c: c.system_type)
def test_run_go_e2e_system_type_matrix(tmp_path: Path, case: EmtE2eCase) -> None:
    """Public EMT ``run_go`` / ``run_go_ts`` for every ``system_type``."""
    output_dir = tmp_path / f"emt_{case.system_type}"
    go_params = _build_go_params(case)
    policy = get_system_policy(case.system_type)
    n_slab = len(case.surface_config.slab) if case.surface_config is not None else 0

    if case.mode == "go":
        minima = run_go(
            case.composition,
            params=go_params,
            seed=case.seed,
            verbosity=0,
            output_dir=str(output_dir),
            system_type=case.system_type,
            surface_config=case.surface_config,
            adsorbates=case.adsorbates,
        )
        best = assert_e2e_minima_list(
            minima,
            expected_n_atoms=n_slab + case.expected_mobile_atoms,
            output_dir=output_dir,
            expect_final_tag=case.tag_final_minima,
            expect_xyz_export=case.expect_xyz_export,
        )
        if case.check_supported_binding and policy.needs_supported_deposit_validation:
            assert case.surface_config is not None
            assert_supported_cluster_binding(
                best,
                case.surface_config,
                n_core_mobile=case.n_core_mobile,
                adsorbate_fragment_lengths=case.adsorbate_fragment_lengths,
                connectivity_factor=case.connectivity_factor
                or go_params.get("connectivity_factor"),
            )
        return

    if case.mode == "go_ts":
        summary = run_go_ts(
            case.composition,
            go_params=go_params,
            ts_params=_build_ts_params(case),
            seed=case.seed,
            verbosity=0,
            output_dir=str(output_dir),
            system_type=case.system_type,
            surface_config=case.surface_config,
            adsorbates=case.adsorbates,
            log_summary=False,
        )
        assert_e2e_go_ts_summary(
            summary,
            expected_formula=_expected_formula(case),
            expected_mobile_atoms=case.expected_mobile_atoms,
            output_dir=output_dir,
            surface_config=case.surface_config,
            n_core_mobile=case.n_core_mobile,
            adsorbate_fragment_lengths=case.adsorbate_fragment_lengths,
            connectivity_factor=case.connectivity_factor
            or go_params.get("connectivity_factor"),
            check_supported_binding=(
                case.check_supported_binding
                and policy.needs_supported_deposit_validation
            ),
            require_ts_candidates=case.require_ts_candidates,
            barrier_range=case.barrier_range,
        )
        return

    raise AssertionError(f"Unhandled e2e mode: {case.mode!r}")


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_pt2_produces_tagged_minima(tmp_path: Path) -> None:
    """Gas GO tags final minima and exports XYZ (negative-control companion)."""
    params = get_testing_params()
    params["tag_final_minima"] = True
    output_dir = tmp_path / "pt2_go"
    minima = run_go(
        ["Pt", "Pt"],
        params=params,
        seed=SEED,
        verbosity=0,
        output_dir=str(output_dir),
        system_type="gas_cluster",
    )
    assert_e2e_minima_list(
        minima,
        expected_n_atoms=2,
        output_dir=output_dir,
        expect_final_tag=True,
        expect_xyz_export=True,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_ts_h2_has_no_ts_pairs(tmp_path: Path) -> None:
    """H2 GO+TS finds no candidate pairs (negative control)."""
    go_params = get_testing_params()
    go_params["optimizer_params"]["simple"].update(
        {"niter": 2, "niter_local_relaxation": 8}
    )
    ts_params = {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
        "max_pairs": 3,
        "neb_steps": 200,
    }
    summary = run_go_ts(
        ["H", "H"],
        go_params=go_params,
        ts_params=ts_params,
        seed=SEED,
        verbosity=0,
        output_dir=str(tmp_path / "h2_go_ts"),
        system_type="gas_cluster",
        log_summary=False,
    )
    assert_e2e_go_ts_summary(
        summary,
        expected_formula="H2",
        expected_mobile_atoms=2,
        output_dir=tmp_path / "h2_go_ts",
        expect_zero_ts=True,
    )
