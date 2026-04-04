"""End-to-end integration test for Cu4 with MACE calculator and database persistence."""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch_sim.neighbors as _ts_nl
from ase import Atoms
from ase.optimize import FIRE
from ase_ga.data import DataConnection
from mace.calculators import mace_mp

from scgo.ts_search.transition_state_io import load_minima_by_composition
from scgo.ts_search.transition_state_run import run_transition_state_search
from scgo.utils.helpers import extract_energy_from_atoms, validate_pair_id
from tests.test_utils import create_preparedb, mark_test_minima_as_final

logger = logging.getLogger(__name__)

_VESIN_SKIP_MSG = (
    "TorchSim is using vesin (broken periodic/tensor API). "
    "Use conda env scgo or ensure nvalchemi-toolkit-ops is installed; "
    "uninstall vesin and vesin-torch if present."
)

# Shared geometries for MACE integration tests (descriptions + positions in Å)
CU4_MACE_CONFIGS_FULL: list[tuple[str, list[list[float]]]] = [
    ("Cu4_tet", [[0, 0, 0], [2.55, 0, 0], [0, 2.55, 0], [0, 0, 2.55]]),
    ("Cu4_sq", [[0, 0, 0], [2.55, 0, 0], [2.55, 2.55, 0], [0, 2.55, 0]]),
    (
        "Cu4_lin",
        [[0, 0, 0], [2.55, 0, 0], [5.1, 0, 0], [7.65, 0, 0]],
    ),
    (
        "Cu4_dist",
        [[0, 0, 0], [2.55, 0.3, 0], [0.2, 2.55, 0.1], [0.1, -0.1, 2.55]],
    ),
]

CU4_MACE_CONFIGS_REPRO: list[tuple[str, list[list[float]]]] = [
    ("Cu4_tet", [[0, 0, 0], [2.55, 0, 0], [0, 2.55, 0], [0, 0, 2.55]]),
    ("Cu4_sq", [[0, 0, 0], [2.55, 0, 0], [2.55, 2.55, 0], [0, 2.55, 0]]),
]


def _skip_if_torchsim_uses_vesin() -> None:
    if _ts_nl.default_batched_nl.__name__ == "vesin_nl_ts":
        pytest.skip(_VESIN_SKIP_MSG)


def _prepare_cu4_mace_minima_db(
    run_dir: Path,
    mace_calc: Any,
    configs: list[tuple[str, list[list[float]]]],
    *,
    population_size: int = 50,
    fire_fmax: float = 0.05,
    fire_steps: int = 200,
) -> tuple[Path, list[Atoms]]:
    """Create ``candidates.db`` under ``run_dir``, relax all candidates with MACE."""
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "candidates.db"
    db = create_preparedb(Atoms("Cu4"), db_path, population_size=population_size)

    for i, (desc, pos) in enumerate(configs):
        atoms = Atoms("Cu4", positions=pos)
        atoms.center(vacuum=6.0)
        atoms.calc = mace_calc
        atoms.info["key_value_pairs"] = {"raw_score": -10.0}
        atoms.info["confid"] = i + 1
        db.add_unrelaxed_candidate(atoms, description=desc)

    da = DataConnection(str(db_path))
    minima_atoms_list: list[Atoms] = []
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.calc = mace_calc
        opt = FIRE(a, logfile=None)
        opt.run(fmax=fire_fmax, steps=fire_steps)
        a.info["key_value_pairs"]["raw_score"] = -a.get_potential_energy()
        da.add_relaxed_step(a)
        minima_atoms_list.append(a.copy())

    return db_path, minima_atoms_list


@pytest.mark.slow
def test_full_workflow_cu4_mace_database_persistence():
    """End-to-end Cu4 MACE integration test (slow)."""
    _skip_if_torchsim_uses_vesin()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        run_dir = tmpdir_path / "run_20260203_150000"
        mace_calc = mace_mp(model="small", default_dtype="float32")
        db_path, minima_atoms_list = _prepare_cu4_mace_minima_db(
            run_dir, mace_calc, CU4_MACE_CONFIGS_FULL
        )

        assert db_path.exists()
        assert len(minima_atoms_list) == 4

        mark_test_minima_as_final(db_path)

        minima = load_minima_by_composition(
            tmpdir, composition=["Cu", "Cu", "Cu", "Cu"]
        )
        assert "Cu4" in minima
        assert len(minima["Cu4"]) == 4
        energies = [e for e, _ in minima["Cu4"]]
        assert energies == sorted(energies)

        logger.debug(
            "MINIMA ANALYSIS: %d minima; energies=%s; range=%.4f eV",
            len(minima["Cu4"]),
            [f"{e:.4f}" for e in energies],
            energies[-1] - energies[0],
        )

        ts_params = {
            "calculator": "MACE",
            "calculator_kwargs": {
                "model_name": "mace_matpes_0",
                "default_dtype": "float32",
            },
        }

        torchsim_params = {
            "mace_model_name": "mace_matpes_0",
            "autobatch_strategy": "binning",
            "max_steps": 120,
        }

        ts_results = run_transition_state_search(
            composition=["Cu", "Cu", "Cu", "Cu"],
            base_dir=tmpdir,
            params=ts_params,
            verbosity=0,
            max_pairs=3,
            neb_n_images=5,
            neb_fmax=0.1,
            neb_steps=120,
            neb_perturb_sigma=0.03,
            use_torchsim=True,
            torchsim_params=torchsim_params,
        )

        assert isinstance(ts_results, list)
        assert len(ts_results) > 0

        logger.debug("TRANSITION STATE SEARCH RESULTS: total_runs=%d", len(ts_results))

        successful_results = [r for r in ts_results if r["status"] == "success"]
        failed_results = [r for r in ts_results if r["status"] != "success"]

        logger.debug(
            "NEB outcomes: successful=%d, failed=%d",
            len(successful_results),
            len(failed_results),
        )

        if failed_results:
            failure_reasons = {}
            for r in failed_results:
                reason = r.get("status", "unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            logger.debug("Failure reasons: %s", failure_reasons)

            for i, r in enumerate(failed_results, 1):
                logger.debug(
                    "Failed run #%d: pair=%s status=%s error=%s neb_converged=%s n_images=%s",
                    i,
                    r.get("pair_id", "N/A"),
                    r.get("status", "unknown"),
                    r.get("error", "No error message"),
                    r.get("neb_converged"),
                    r.get("n_images"),
                )

        barriers = []
        for result in successful_results:
            barrier = result.get("barrier_height")
            if barrier is not None:
                barriers.append(barrier)

        logger.debug("ENERGY BARRIER ANALYSIS: count=%d", len(barriers))
        if barriers:
            logger.debug(
                "Barriers (eV): %s; min=%.4f max=%.4f mean=%.4f",
                [f"{b:.4f}" for b in sorted(barriers)],
                min(barriers),
                max(barriers),
                sum(barriers) / len(barriers),
            )

        if successful_results:
            logger.debug("DETAILED TS RESULTS: count=%d", len(successful_results))
            for i, result in enumerate(successful_results, 1):
                logger.debug(
                    "TS #%d: pair=%s reactant=%.4f product=%.4f ts=%.4f barrier=%.4f neb_converged=%s n_images=%d",
                    i,
                    result.get("pair_id", "N/A"),
                    result.get("reactant_energy", 0),
                    result.get("product_energy", 0),
                    result.get("ts_energy", 0),
                    result.get("barrier_height", 0),
                    result.get("neb_converged", False),
                    result.get("n_images", 0),
                )

        logger.debug("End of TS search summary")

        assert len(ts_results) == 3, (
            f"Expected 3 NEB runs (max_pairs=3), got {len(ts_results)}"
        )
        if not successful_results:
            errs = [str(r.get("error") or "").lower() for r in ts_results]
            if all("out of memory" in e or "outofmemory" in e for e in errs if e):
                pytest.skip(
                    "GPU out-of-memory in this environment; cannot validate MACE TS results"
                )
            if any("endpoint" in e for e in errs if e):
                pytest.skip(
                    "All NEB runs were endpoint-detected; no successful TS to validate"
                )
            if any(
                "periodic" in e and "bool" in e and "tensor" in e for e in errs if e
            ):
                pytest.skip(
                    "vesin periodic/tensor type mismatch in this environment; "
                    "use torch-sim with alchemiops (nvalchemi-toolkit-ops) or avoid vesin"
                )
            assert False, (
                f"No successful TS runs and unexpected failure reasons: {errs}"
            )

        assert len(barriers) > 0, (
            "Expected at least one energy barrier to be calculated"
        )

        for barrier in barriers:
            if barrier > 5.0:
                logger.warning(
                    "Barrier %.4f eV exceeds typical range; may indicate difficult path or CI variance",
                    barrier,
                )
            assert 0 <= barrier <= 10.0, (
                f"Barrier {barrier:.4f} eV is outside reasonable range [0, 10] eV"
            )

        ts_result_dir = tmpdir_path / "ts_results_Cu4"
        assert ts_result_dir.exists()

        summary_file = ts_result_dir / "ts_search_summary_Cu4.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["formula"] == "Cu4"
        assert "statistics" in summary and isinstance(summary["statistics"], dict)

        for result in successful_results:
            barrier = result.get("barrier_height")
            if barrier is not None:
                assert barrier >= -0.01

            reactant_e = result.get("reactant_energy")
            product_e = result.get("product_energy")
            ts_e = result.get("ts_energy")

            if all(x is not None for x in [reactant_e, product_e, ts_e]):
                min_e = min(reactant_e, product_e)
                assert ts_e >= min_e - 0.1

            assert "neb_converged" in result
            assert "n_images" in result
            assert "pair_id" in result

            pair = result.get("pair_id")
            assert pair is not None and re.fullmatch(r"\d+_\d+", pair), (
                f"Malformed pair_id: {pair!r}"
            )

        network_file = ts_result_dir / "ts_network_metadata_Cu4.json"
        assert network_file.exists(), "Expected TS network metadata file to exist"
        with open(network_file) as f:
            network_meta = json.load(f)
        assert network_meta["formula"] == "Cu4"
        assert "ts_connections" in network_meta
        assert "statistics" in network_meta and isinstance(
            network_meta["statistics"], dict
        )
        assert network_meta["statistics"]["total_ts_found"] == summary["num_successful"]
        assert network_meta["statistics"] == summary["statistics"]

        da_verify = DataConnection(str(db_path))

        all_relaxed = da_verify.get_all_relaxed_candidates()

        expected_descriptions = {d for d, _ in CU4_MACE_CONFIGS_FULL}
        found_descriptions = {
            r.info.get("key_value_pairs", {}).get("description") for r in all_relaxed
        }

        for desc in expected_descriptions:
            assert desc in found_descriptions, (
                f"Minima description '{desc}' not found in DB"
            )

        minima_entries = [
            r
            for r in all_relaxed
            if r.info.get("key_value_pairs", {}).get("description")
            in expected_descriptions
        ]
        assert len(minima_entries) >= 4, "Expected at least 4 minima entries in the DB"

        ts_entries = []
        final_ts_summary = (
            ts_result_dir / "final_unique_ts" / "final_unique_ts_summary_Cu4.json"
        )
        assert final_ts_summary.exists(), (
            "Expected final_unique_ts summary to be present"
        )
        with open(final_ts_summary) as f:
            final_summary = json.load(f)
        unique_ts = final_summary.get("unique_ts", [])

        if summary.get("num_successful", 0) > 0:
            assert len(unique_ts) > 0, (
                "Expected at least one unique TS when num_successful > 0"
            )

        for item in unique_ts:
            ts_e = float(item.get("ts_energy"))
            pair_id = item.get("pair_id")
            matched = False
            for r in all_relaxed:
                kv = r.info.get("key_value_pairs", {})

                if kv.get("is_transition_state") and (
                    r.info.get("metadata", {}).get("ts_pair_id") == pair_id
                    or r.info.get("provenance", {}).get("pair_id") == pair_id
                ):
                    matched = True
                    ts_entries.append(r)
                    break

                raw = kv.get("raw_score")
                energy = extract_energy_from_atoms(r)
                if raw is not None and abs(raw + ts_e) < 1e-4:
                    matched = True
                    ts_entries.append(r)
                    break
                if energy is not None and abs(energy - ts_e) < 1e-4:
                    matched = True
                    ts_entries.append(r)
                    break

            assert matched, (
                f"TS with energy {ts_e:.6f} eV (pair {pair_id}) not found in DB"
            )

        assert len(ts_entries) == len(unique_ts)

        for retrieved in minima_entries + ts_entries:
            energy = extract_energy_from_atoms(retrieved)
            assert energy is not None
            assert math.isfinite(energy)

        with open(summary_file) as f:
            summary_reload = json.load(f)

        assert summary_reload["formula"] == "Cu4"
        assert len(summary_reload["results"]) > 0

        for result in summary_reload["results"]:
            assert "status" in result
            assert "pair_id" in result
            assert re.fullmatch(r"\d+_\d+", result.get("pair_id")), (
                f"Malformed pair_id in summary: {result.get('pair_id')!r}"
            )
            validate_pair_id(result["pair_id"])

        stats = summary.get("statistics", {})
        if stats:
            assert stats["total_ts_found"] == summary["num_successful"]
            assert stats["converged_ts"] <= stats["total_ts_found"]

            if stats["converged_ts"] > 0:
                assert "min_barrier" in stats
                assert "max_barrier" in stats
                assert stats["min_barrier"] >= -0.01
                assert stats["max_barrier"] >= stats["min_barrier"]


@pytest.mark.slow
def test_ts_search_reproducibility_with_mace():
    """Test that TS search with MACE produces reproducible results.

    Verifies that identical inputs (same minima, same parameters) produce
    identical transition state search results, including barrier heights.
    """
    _skip_if_torchsim_uses_vesin()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mace_calc = mace_mp(model="small", default_dtype="float32")

        ts_params = {
            "calculator": "MACE",
            "calculator_kwargs": {
                "model_name": "mace_matpes_0",
                "default_dtype": "float32",
            },
        }

        torchsim_params = {
            "mace_model_name": "mace_matpes_0",
            "autobatch_strategy": "binning",
            "max_steps": 50,
        }

        def run_ts_workflow(run_name: str) -> list:
            run_dir = tmpdir_path / run_name
            _prepare_cu4_mace_minima_db(run_dir, mace_calc, CU4_MACE_CONFIGS_REPRO)
            return run_transition_state_search(
                composition=["Cu", "Cu", "Cu", "Cu"],
                base_dir=run_dir,
                params=ts_params,
                verbosity=0,
                max_pairs=1,
                neb_n_images=5,
                neb_fmax=0.1,
                neb_steps=50,
                use_torchsim=True,
                torchsim_params=torchsim_params,
            )

        results1 = run_ts_workflow("run1")
        results2 = run_ts_workflow("run2")

        assert len(results1) == len(results2)

        for r1, r2 in zip(results1, results2, strict=False):
            assert r1["status"] == r2["status"]

            if r1["status"] == "success" and r2["status"] == "success":
                barrier1 = r1.get("barrier_height")
                barrier2 = r2.get("barrier_height")

                if barrier1 is not None and barrier2 is not None:
                    assert abs(barrier1 - barrier2) < 0.01, (
                        f"Barrier heights differ: {barrier1:.6f} vs {barrier2:.6f}"
                    )

                for key in ["reactant_energy", "product_energy", "ts_energy"]:
                    e1 = r1.get(key)
                    e2 = r2.get(key)
                    if e1 is not None and e2 is not None:
                        assert abs(e1 - e2) < 0.01, (
                            f"{key} differs: {e1:.6f} vs {e2:.6f}"
                        )
