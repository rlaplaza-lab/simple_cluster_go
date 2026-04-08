#!/usr/bin/env python3
"""Pt5-focused NEB sweep (TorchSim+MACE) for gas and supported clusters.

Supported (slab + cluster) sweeps default to a **larger** NEB budget than bare
``auto_niter_ts`` (see ``--supported-neb-step-factor``) and vary ``n_images``
(3/5/7) so bands can resolve interior maxima instead of always landing on endpoints.
Use ``--supported-grid compact`` for a faster 2-cell align True/False comparison
at ``n_images=3`` only.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from ase import Atoms

from benchmark.benchmark_common import PT_SURFACE_NIO_RESULTS_DIR
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
)
from scgo.runner_surface import read_full_composition_from_first_xyz
from scgo.ts_search.transition_state import (
    calculate_structure_similarity,
    find_transition_state,
)
from scgo.ts_search.transition_state_io import (
    load_minima_by_composition,
    select_structure_pairs,
)
from scgo.utils.helpers import auto_niter_ts, filter_unique_minima, get_cluster_formula
from scgo.utils.logging import configure_logging, get_logger


def resolve_searches_composition(
    searches_dir: Path,
    composition: list[str] | None,
) -> list[str]:
    if composition is not None:
        return list(composition)
    final_minima = searches_dir / "final_unique_minima"
    if final_minima.is_dir() and any(final_minima.glob("*.xyz")):
        return read_full_composition_from_first_xyz(final_minima)
    raise SystemExit(
        f"Could not infer composition for {searches_dir}; expected final_unique_minima/*.xyz"
    )


def _parse_models(models_csv: str | None, default_name: str) -> list[str]:
    """Split comma-separated model names for sweeps (used by tests and CLI)."""
    if models_csv is None:
        return [default_name]
    stripped = models_csv.strip()
    if not stripped:
        return [default_name]
    return [p.strip() for p in stripped.split(",") if p.strip()]


def _barrier_suspicious(record: dict[str, Any], *, max_barrier_ev: float) -> bool:
    """True when NEB reports converged but the barrier is unphysically large."""
    if not record.get("neb_converged"):
        return False
    bh = record.get("barrier_height")
    if bh is None:
        return False
    return float(bh) > max_barrier_ev


@dataclass(frozen=True)
class EndpointBundle:
    composition: list[str]
    neb_steps: int
    minima: list[tuple[float, Atoms]]
    pairs: list[tuple[int, int]]
    meta: dict[str, Any]


def load_bundle_from_searches(
    searches_dir: Path,
    composition: list[str] | None,
    *,
    max_pairs: int,
    energy_gap_threshold: float,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
    surface_aware: bool,
    pair_priority_mode: str,
    minima_energy_tolerance: float,
    neb_steps_override: int | None,
    neb_steps_scale: float = 1.0,
) -> EndpointBundle:
    comp = resolve_searches_composition(searches_dir, composition)
    formula = get_cluster_formula(comp)
    minima_by_formula = load_minima_by_composition(
        str(searches_dir.resolve()),
        composition=comp,
        prefer_final_unique=True,
    )
    minima = minima_by_formula.get(formula, [])
    minima = filter_unique_minima(minima, minima_energy_tolerance)
    if len(minima) < 2:
        raise SystemExit(
            f"Need >=2 minima for {formula} in {searches_dir}; got {len(minima)}."
        )
    pairs = select_structure_pairs(
        minima,
        max_pairs=max_pairs,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
        pair_priority_mode=pair_priority_mode,
        surface_aware=surface_aware,
    )
    if not pairs:
        raise SystemExit(f"No candidate pairs found under {searches_dir}.")
    if neb_steps_override is not None:
        neb_steps = int(neb_steps_override)
    else:
        neb_steps = int(round(auto_niter_ts(comp) * float(neb_steps_scale)))
    return EndpointBundle(
        composition=comp,
        neb_steps=neb_steps,
        minima=minima,
        pairs=pairs,
        meta={
            "formula": formula,
            "n_minima_loaded": len(minima),
            "n_pairs_selected": len(pairs),
            "max_pairs_requested": max_pairs,
            "energy_gap_threshold_ev": energy_gap_threshold,
            "pair_priority_mode": pair_priority_mode,
            "surface_aware": surface_aware,
            "neb_steps": neb_steps,
            "searches_dir": str(searches_dir.resolve()),
        },
    )


@dataclass(frozen=True)
class SweepCell:
    regime: str
    cell_id: str
    neb_interpolation_method: str
    neb_interpolation_mic: bool
    n_images: int
    neb_climb: bool
    spring_constant: float
    fmax: float
    neb_steps: int
    neb_tangent_method: str
    align_endpoints: bool = True
    perturb_sigma: float = 0.0
    neb_retry_on_endpoint: bool = True


def _supported_neb_steps_for_cell(base_steps: int, n_images: int) -> int:
    """Extra optimizer steps for wider bands (more images)."""
    band_factor = {3: 1.0, 5: 1.2, 7: 1.45}.get(n_images, 1.0)
    return max(1, int(round(base_steps * band_factor)))


def gas_grid(neb_steps: int) -> list[SweepCell]:
    return [
        SweepCell(
            "gas",
            "gas_ref",
            "idpp",
            False,
            3,
            True,
            0.1,
            0.05,
            neb_steps,
            DEFAULT_NEB_TANGENT_METHOD,
            True,
            0.0,
            True,
        ),
        SweepCell(
            "gas",
            "gas_noclimb",
            "idpp",
            False,
            3,
            False,
            0.1,
            0.05,
            neb_steps,
            DEFAULT_NEB_TANGENT_METHOD,
            True,
            0.0,
            True,
        ),
    ]


def supported_grid(neb_steps: int, mode: str) -> list[SweepCell]:
    """Supported-cluster sweep cells. ``mode`` is ``compact`` or ``full``."""
    if mode == "compact":
        return [
            SweepCell(
                "supported",
                "sup_af_n3",
                "idpp",
                True,
                3,
                False,
                0.1,
                0.05,
                _supported_neb_steps_for_cell(neb_steps, 3),
                DEFAULT_NEB_TANGENT_METHOD,
                False,
                0.0,
                True,
            ),
            SweepCell(
                "supported",
                "sup_at_n3",
                "idpp",
                True,
                3,
                False,
                0.1,
                0.05,
                _supported_neb_steps_for_cell(neb_steps, 3),
                DEFAULT_NEB_TANGENT_METHOD,
                True,
                0.0,
                True,
            ),
        ]
    return [
        SweepCell(
            "supported",
            "sup_af_n3",
            "idpp",
            True,
            3,
            False,
            0.1,
            0.05,
            _supported_neb_steps_for_cell(neb_steps, 3),
            DEFAULT_NEB_TANGENT_METHOD,
            False,
            0.0,
            True,
        ),
        SweepCell(
            "supported",
            "sup_af_n5",
            "idpp",
            True,
            5,
            False,
            0.1,
            0.05,
            _supported_neb_steps_for_cell(neb_steps, 5),
            DEFAULT_NEB_TANGENT_METHOD,
            False,
            0.0,
            True,
        ),
        SweepCell(
            "supported",
            "sup_af_n7",
            "idpp",
            True,
            7,
            False,
            0.1,
            0.05,
            _supported_neb_steps_for_cell(neb_steps, 7),
            DEFAULT_NEB_TANGENT_METHOD,
            False,
            0.0,
            True,
        ),
        SweepCell(
            "supported",
            "sup_at_n5",
            "idpp",
            True,
            5,
            False,
            0.1,
            0.05,
            _supported_neb_steps_for_cell(neb_steps, 5),
            DEFAULT_NEB_TANGENT_METHOD,
            True,
            0.0,
            True,
        ),
        SweepCell(
            "supported",
            "sup_af_n5_climb",
            "idpp",
            True,
            5,
            True,
            0.1,
            0.05,
            _supported_neb_steps_for_cell(neb_steps, 5),
            DEFAULT_NEB_TANGENT_METHOD,
            False,
            0.0,
            True,
        ),
    ]


def row_for_result(
    *,
    cell: SweepCell,
    bundle: EndpointBundle,
    pair_rank: int,
    pair_indices: tuple[int, int],
    pair_similarity_cum_diff: float,
    pair_similarity_max_diff: float,
    model_name: str,
    wall_s: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    err = result.get("error")
    if err is not None and not isinstance(err, str):
        err = str(err)
    i, j = pair_indices
    e_i = float(bundle.minima[i][0])
    e_j = float(bundle.minima[j][0])
    return {
        "regime": cell.regime,
        "cell_id": cell.cell_id,
        "searches_dir": bundle.meta["searches_dir"],
        "formula": bundle.meta["formula"],
        "pair_rank": int(pair_rank),
        "pair_indices": [int(i), int(j)],
        "reactant_energy_ev": float(e_i),
        "product_energy_ev": float(e_j),
        "energy_gap_ev": float(abs(e_j - e_i)),
        "pair_priority_mode": bundle.meta["pair_priority_mode"],
        "pair_similarity_cum_diff": float(pair_similarity_cum_diff),
        "pair_similarity_max_diff": float(pair_similarity_max_diff),
        "n_minima_loaded": int(bundle.meta["n_minima_loaded"]),
        "n_pairs_selected": int(bundle.meta["n_pairs_selected"]),
        "backend": "torchsim_mace",
        "mace_model_name": model_name,
        "spring_constant": cell.spring_constant,
        "fmax": cell.fmax,
        "neb_steps": cell.neb_steps,
        "neb_interpolation_method": cell.neb_interpolation_method,
        "neb_interpolation_mic": cell.neb_interpolation_mic,
        "neb_climb": cell.neb_climb,
        "n_images": cell.n_images,
        "neb_tangent_method": cell.neb_tangent_method,
        "neb_align_endpoints": cell.align_endpoints,
        "neb_perturb_sigma": cell.perturb_sigma,
        "neb_retry_on_endpoint": cell.neb_retry_on_endpoint,
        "wall_s": wall_s,
        "status": result.get("status"),
        "neb_converged": bool(result.get("neb_converged", False)),
        "final_fmax": result.get("final_fmax"),
        "steps_taken": result.get("steps_taken"),
        "barrier_height": result.get("barrier_height"),
        "ts_image_index": result.get("ts_image_index"),
        "error": err,
    }


def _error_is_endpoint_ts(row: dict[str, Any]) -> bool:
    err = str(row.get("error") or "").lower()
    return "endpoint as ts" in err


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["cell_id"]))].append(row)
    table: list[dict[str, Any]] = []
    for (regime, cell_id), entries in grouped.items():
        converged = sum(1 for r in entries if bool(r["neb_converged"]))
        finals = [
            float(r["final_fmax"])
            for r in entries
            if r.get("final_fmax") is not None and bool(r["neb_converged"])
        ]
        finals_all = [
            float(r["final_fmax"]) for r in entries if r.get("final_fmax") is not None
        ]
        endpoint_ts = sum(1 for r in entries if _error_is_endpoint_ts(r))
        times = [float(r["wall_s"]) for r in entries]
        cfg = entries[0]
        table.append(
            {
                "regime": regime,
                "cell_id": cell_id,
                "runs": len(entries),
                "converged": converged,
                "success_rate": converged / len(entries) if entries else 0.0,
                "endpoint_as_ts_count": endpoint_ts,
                "median_final_fmax": median(finals) if finals else None,
                "median_final_fmax_all": median(finals_all) if finals_all else None,
                "median_wall_s": median(times) if times else None,
                "config": {
                    "neb_interpolation_method": cfg["neb_interpolation_method"],
                    "neb_interpolation_mic": bool(cfg["neb_interpolation_mic"]),
                    "neb_n_images": int(cfg["n_images"]),
                    "neb_climb": bool(cfg["neb_climb"]),
                    "neb_k": float(cfg["spring_constant"]),
                    "neb_fmax": float(cfg["fmax"]),
                    "neb_tangent_method": cfg["neb_tangent_method"],
                    "neb_align_endpoints": bool(cfg["neb_align_endpoints"]),
                    "neb_perturb_sigma": float(cfg["neb_perturb_sigma"]),
                    "neb_retry_on_endpoint": bool(cfg["neb_retry_on_endpoint"]),
                },
            }
        )
    table.sort(
        key=lambda t: (
            t["regime"],
            -float(t["success_rate"]),
            int(t["endpoint_as_ts_count"]),
            float("inf")
            if t["median_final_fmax_all"] is None
            else float(t["median_final_fmax_all"]),
            float("inf")
            if t["median_final_fmax"] is None
            else float(t["median_final_fmax"]),
            float("inf") if t["median_wall_s"] is None else float(t["median_wall_s"]),
        )
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("gas", "supported", "all"), default="all")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--energy-gap-threshold", type=float, default=1.0)
    parser.add_argument(
        "--pair-priority-mode", choices=("legacy", "physics"), default="physics"
    )
    parser.add_argument(
        "--similarity-tolerance", type=float, default=DEFAULT_COMPARATOR_TOL
    )
    parser.add_argument("--similarity-pair-cor-max", type=float, default=0.1)
    parser.add_argument("--gas-searches-dir", type=Path, default=Path("Pt5_searches"))
    parser.add_argument(
        "--supported-searches-dir",
        type=Path,
        default=PT_SURFACE_NIO_RESULTS_DIR / "Pt5_searches",
    )
    parser.add_argument("--model", default="mace_matpes_0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--neb-steps",
        type=int,
        default=None,
        help="Override NEB steps for both regimes (legacy; prefer --gas-neb-steps / --supported-neb-steps)",
    )
    parser.add_argument(
        "--gas-neb-steps",
        type=int,
        default=None,
        help="Override NEB steps for gas only (default: auto_niter_ts)",
    )
    parser.add_argument(
        "--supported-neb-steps",
        type=int,
        default=None,
        help="Override NEB steps for supported base budget before per-image scaling "
        "(default: auto_niter_ts × --supported-neb-step-factor)",
    )
    parser.add_argument(
        "--supported-neb-step-factor",
        type=float,
        default=2.0,
        help="Multiply auto_niter_ts for supported when --supported-neb-steps is omitted (default: 2.0)",
    )
    parser.add_argument(
        "--supported-grid",
        choices=("compact", "full"),
        default="full",
        help="compact: 2 cells (align F/T, n_images=3); full: n_images 3/5/7 align F, n5 align T, n5 climb",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("benchmark/neb_pt5_mace")
    )
    args = parser.parse_args()

    configure_logging(1)
    logger = get_logger(__name__)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    regimes = ["gas", "supported"] if args.regime == "all" else [args.regime]
    dev = torch.device(args.device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Expected a CUDA device for this benchmark.")

    gas_neb_override = (
        args.gas_neb_steps if args.gas_neb_steps is not None else args.neb_steps
    )
    sup_neb_override = (
        args.supported_neb_steps
        if args.supported_neb_steps is not None
        else args.neb_steps
    )
    sup_scale = (
        float(args.supported_neb_step_factor) if sup_neb_override is None else 1.0
    )

    gas_bundle: EndpointBundle | None = None
    sup_bundle: EndpointBundle | None = None
    if "gas" in regimes:
        gas_bundle = load_bundle_from_searches(
            args.gas_searches_dir,
            composition=None,
            max_pairs=args.max_pairs,
            energy_gap_threshold=args.energy_gap_threshold,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            surface_aware=False,
            pair_priority_mode=args.pair_priority_mode,
            minima_energy_tolerance=DEFAULT_ENERGY_TOLERANCE,
            neb_steps_override=gas_neb_override,
            neb_steps_scale=1.0,
        )
        logger.info("Gas bundle: %s", gas_bundle.meta)
    if "supported" in regimes:
        sup_bundle = load_bundle_from_searches(
            args.supported_searches_dir,
            composition=None,
            max_pairs=args.max_pairs,
            energy_gap_threshold=args.energy_gap_threshold,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            surface_aware=True,
            pair_priority_mode=args.pair_priority_mode,
            minima_energy_tolerance=DEFAULT_ENERGY_TOLERANCE,
            neb_steps_override=sup_neb_override,
            neb_steps_scale=sup_scale,
        )
        logger.info("Supported bundle: %s", sup_bundle.meta)

    jobs: list[tuple[SweepCell, EndpointBundle, tuple[int, int], int]] = []
    if gas_bundle is not None:
        for cell in gas_grid(gas_bundle.neb_steps):
            for rank, pair in enumerate(gas_bundle.pairs):
                jobs.append((cell, gas_bundle, pair, rank))
    if sup_bundle is not None:
        for cell in supported_grid(sup_bundle.neb_steps, args.supported_grid):
            for rank, pair in enumerate(sup_bundle.pairs):
                jobs.append((cell, sup_bundle, pair, rank))
    logger.info("Total jobs: %d", len(jobs))
    if args.dry_run:
        return

    rows: list[dict[str, Any]] = []
    out_jsonl = args.output_prefix.with_suffix(".jsonl")
    if out_jsonl.exists():
        out_jsonl.unlink()

    for idx, (cell, bundle, pair, pair_rank) in enumerate(jobs, 1):
        i, j = pair
        atoms_a = bundle.minima[i][1].copy()
        atoms_b = bundle.minima[j][1].copy()
        pair_cum_diff, pair_max_diff, _ = calculate_structure_similarity(
            atoms_a,
            atoms_b,
            tolerance=args.similarity_tolerance,
            pair_cor_max=args.similarity_pair_cor_max,
        )
        torchsim_params: dict[str, Any] = {
            "mace_model_name": args.model,
            "device": dev,
            "dtype": torch.float32,
            "force_tol": cell.fmax,
            "max_steps": cell.neb_steps,
        }
        rng = np.random.default_rng(42) if cell.perturb_sigma > 0.0 else None
        with tempfile.TemporaryDirectory(prefix="neb_pt5_sweep_") as tmp:
            t0 = time.perf_counter()
            result = find_transition_state(
                atoms_a,
                atoms_b,
                calculator=None,
                output_dir=tmp,
                pair_id=f"{cell.regime}_{cell.cell_id}_{i}_{j}",
                rng=rng,
                n_images=cell.n_images,
                spring_constant=cell.spring_constant,
                fmax=cell.fmax,
                neb_steps=cell.neb_steps,
                verbosity=0,
                use_torchsim=True,
                torchsim_params=torchsim_params,
                climb=cell.neb_climb,
                interpolation_method=cell.neb_interpolation_method,
                align_endpoints=cell.align_endpoints,
                perturb_sigma=cell.perturb_sigma,
                neb_interpolation_mic=cell.neb_interpolation_mic,
                neb_retry_on_endpoint=cell.neb_retry_on_endpoint,
                neb_tangent_method=cell.neb_tangent_method,
            )
            wall_s = time.perf_counter() - t0

        row = row_for_result(
            cell=cell,
            bundle=bundle,
            pair_rank=pair_rank,
            pair_indices=pair,
            pair_similarity_cum_diff=pair_cum_diff,
            pair_similarity_max_diff=pair_max_diff,
            model_name=args.model,
            wall_s=wall_s,
            result=result,
        )
        rows.append(row)
        with out_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[{idx}/{len(jobs)}] {cell.regime} {cell.cell_id} pair={pair_rank} conv={row['neb_converged']} status={row['status']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    agg_path = args.output_prefix.with_name(args.output_prefix.name + "_aggregate.json")
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    defaults: dict[str, dict[str, Any]] = {}
    rec_notes: dict[str, str] = {}
    for regime in ("gas", "supported"):
        ranked = [x for x in aggregate if x["regime"] == regime]
        if not ranked:
            continue
        if all(float(x["success_rate"]) == 0.0 for x in ranked):
            if regime == "supported":
                best = ranked[0]
                defaults[regime] = dict(best["config"])
                rec_notes[regime] = (
                    "No neb_converged successes; ranked cells by fewest "
                    "'endpoint as TS' errors, then lower median final_fmax (all runs). "
                    f"Best proxy cell: {best['cell_id']} "
                    f"(endpoint_as_ts={best['endpoint_as_ts_count']}/{best['runs']}). "
                    "Validate with chemistry-specific runs before changing global surface presets."
                )
            else:
                defaults[regime] = dict(ranked[0]["config"])
                rec_notes[regime] = (
                    "No runs had neb_converged in this sweep; first grid cell after "
                    "proxy ranking kept."
                )
        else:
            defaults[regime] = ranked[0]["config"]
    rec_path = args.output_prefix.with_name(
        args.output_prefix.name + "_recommended_defaults.json"
    )
    rec_payload: dict[str, Any] = dict(defaults)
    if rec_notes:
        rec_payload["_recommendation_notes"] = rec_notes
    rec_path.write_text(json.dumps(rec_payload, indent=2), encoding="utf-8")
    logger.info("Wrote rows: %s", out_jsonl)
    logger.info("Wrote aggregate: %s", agg_path)
    logger.info("Wrote defaults: %s", rec_path)


if __name__ == "__main__":
    main()
