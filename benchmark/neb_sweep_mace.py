#!/usr/bin/env python3
"""Pt5 NEB sweep (TorchSim+MACE) across all NEB knobs.

This script is intentionally focused on two benchmark search directories:
- gas: benchmark/results/pt_cluster/Pt5_searches
- supported: benchmark/results/pt_surface_nio/Pt5_searches

It evaluates the first N selected NEB paths per regime (default N=5), sweeps all
NEB knobs that are exposed by TS search, and writes:
- per-run rows: <output_prefix>.jsonl
- aggregated ranking: <output_prefix>_aggregate.json
- suggested defaults: <output_prefix>_recommended_defaults.json

Overnight example (runs in background and logs to file):
    conda run -n scgo nohup python benchmark/neb_sweep_mace.py \
      --regime all \
      --output-prefix benchmark/neb_pt5_knob_sweep_overnight \
      > benchmark/neb_pt5_knob_sweep_overnight.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from ase import Atoms

from benchmark.benchmark_common import (
    PT_CLUSTER_RESULTS_DIR,
    PT_SURFACE_NIO_RESULTS_DIR,
)
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


@dataclass(frozen=True)
class EndpointBundle:
    composition: list[str]
    neb_steps_base: int
    minima: list[tuple[float, Atoms]]
    pairs_all: list[tuple[int, int]]
    pairs_sampled: list[tuple[int, int]]
    meta: dict[str, Any]


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


def _parse_models(models_csv: str | None, default_name: str) -> list[str]:
    if models_csv is None:
        return [default_name]
    stripped = models_csv.strip()
    if not stripped:
        return [default_name]
    return [p.strip() for p in stripped.split(",") if p.strip()]


def _barrier_suspicious(record: dict[str, Any], *, max_barrier_ev: float) -> bool:
    if not record.get("neb_converged"):
        return False
    bh = record.get("barrier_height")
    if bh is None:
        return False
    return float(bh) > max_barrier_ev


def load_bundle_from_searches(
    searches_dir: Path,
    composition: list[str] | None,
    *,
    max_pairs: int,
    sample_paths: int,
    energy_gap_threshold: float,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
    surface_aware: bool,
    minima_energy_tolerance: float,
    neb_steps_override: int | None,
    neb_steps_scale: float,
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

    pairs_all = select_structure_pairs(
        minima,
        max_pairs=max_pairs,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
        surface_aware=surface_aware,
    )
    if not pairs_all:
        raise SystemExit(f"No candidate pairs found under {searches_dir}.")
    pairs_sampled = pairs_all[:sample_paths]
    if not pairs_sampled:
        raise SystemExit(
            f"No sampled pairs found under {searches_dir} with sample_paths={sample_paths}."
        )

    if neb_steps_override is not None:
        neb_steps = int(neb_steps_override)
    else:
        neb_steps = int(round(auto_niter_ts(comp) * neb_steps_scale))

    return EndpointBundle(
        composition=comp,
        neb_steps_base=neb_steps,
        minima=minima,
        pairs_all=pairs_all,
        pairs_sampled=pairs_sampled,
        meta={
            "formula": formula,
            "searches_dir": str(searches_dir.resolve()),
            "n_minima_loaded": len(minima),
            "n_pairs_selected_total": len(pairs_all),
            "n_pairs_sampled": len(pairs_sampled),
            "sample_paths_requested": sample_paths,
            "max_pairs_requested": max_pairs,
            "energy_gap_threshold_ev": energy_gap_threshold,
            "surface_aware": surface_aware,
            "neb_steps_base": neb_steps,
        },
    )


def _supported_neb_steps_for_cell(base_steps: int, n_images: int) -> int:
    band_factor = {3: 1.0, 5: 1.2, 7: 1.45}.get(n_images, 1.0)
    return max(1, int(round(base_steps * band_factor)))


def _build_knob_sweep(
    *,
    regime: str,
    base: SweepCell,
    base_steps: int,
    include_n_images_7: bool,
) -> list[SweepCell]:
    values_by_knob: dict[str, list[Any]] = {
        "n_images": [3, 5, 7] if include_n_images_7 else [3, 5],
        "neb_climb": [False, True],
        "spring_constant": [0.05, 0.1, 0.2],
        "fmax": [0.03, 0.05, 0.08],
        "neb_interpolation_method": ["idpp", "linear"],
        "neb_interpolation_mic": [False, True],
        "neb_tangent_method": [DEFAULT_NEB_TANGENT_METHOD, "aseneb"],
        "align_endpoints": [False, True],
        "perturb_sigma": [0.0, 0.02],
        "neb_retry_on_endpoint": [False, True],
        "neb_steps_multiplier": [1.0, 1.5, 2.0],
    }

    cells: list[SweepCell] = [base]

    for knob, values in values_by_knob.items():
        for value in values:
            if knob == "neb_steps_multiplier":
                steps = max(1, int(round(base_steps * float(value))))
                if regime == "supported":
                    steps = _supported_neb_steps_for_cell(steps, base.n_images)
                cell = replace(
                    base, cell_id=f"{regime}_{knob}_{value}", neb_steps=steps
                )
            elif knob == "n_images":
                n_images = int(value)
                steps = base_steps
                if regime == "supported":
                    steps = _supported_neb_steps_for_cell(base_steps, n_images)
                cell = replace(
                    base,
                    cell_id=f"{regime}_{knob}_{n_images}",
                    n_images=n_images,
                    neb_steps=steps,
                )
            else:
                cell = replace(
                    base, cell_id=f"{regime}_{knob}_{value}", **{knob: value}
                )
            if cell not in cells:
                cells.append(cell)

    return cells


def gas_grid(base_steps: int, include_n_images_7: bool) -> list[SweepCell]:
    base = SweepCell(
        regime="gas",
        cell_id="gas_baseline",
        neb_interpolation_method="idpp",
        neb_interpolation_mic=False,
        n_images=3,
        neb_climb=False,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=base_steps,
        neb_tangent_method=DEFAULT_NEB_TANGENT_METHOD,
        align_endpoints=True,
        perturb_sigma=0.0,
        neb_retry_on_endpoint=True,
    )
    return _build_knob_sweep(
        regime="gas",
        base=base,
        base_steps=base_steps,
        include_n_images_7=include_n_images_7,
    )


def supported_grid(base_steps: int, include_n_images_7: bool) -> list[SweepCell]:
    base = SweepCell(
        regime="supported",
        cell_id="supported_baseline",
        neb_interpolation_method="idpp",
        neb_interpolation_mic=True,
        n_images=5,
        neb_climb=True,
        spring_constant=0.1,
        fmax=0.05,
        neb_steps=_supported_neb_steps_for_cell(base_steps, 5),
        neb_tangent_method=DEFAULT_NEB_TANGENT_METHOD,
        align_endpoints=False,
        perturb_sigma=0.0,
        neb_retry_on_endpoint=True,
    )
    return _build_knob_sweep(
        regime="supported",
        base=base,
        base_steps=base_steps,
        include_n_images_7=include_n_images_7,
    )


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
        "reactant_energy_ev": e_i,
        "product_energy_ev": e_j,
        "energy_gap_ev": float(abs(e_j - e_i)),
        "pair_similarity_cum_diff": float(pair_similarity_cum_diff),
        "pair_similarity_max_diff": float(pair_similarity_max_diff),
        "n_minima_loaded": int(bundle.meta["n_minima_loaded"]),
        "n_pairs_selected_total": int(bundle.meta["n_pairs_selected_total"]),
        "n_pairs_sampled": int(bundle.meta["n_pairs_sampled"]),
        "sample_paths_requested": int(bundle.meta["sample_paths_requested"]),
        "backend": "torchsim_mace",
        "mace_model_name": model_name,
        "spring_constant": float(cell.spring_constant),
        "fmax": float(cell.fmax),
        "neb_steps": int(cell.neb_steps),
        "neb_interpolation_method": cell.neb_interpolation_method,
        "neb_interpolation_mic": bool(cell.neb_interpolation_mic),
        "neb_climb": bool(cell.neb_climb),
        "n_images": int(cell.n_images),
        "neb_tangent_method": cell.neb_tangent_method,
        "neb_align_endpoints": bool(cell.align_endpoints),
        "neb_perturb_sigma": float(cell.perturb_sigma),
        "neb_retry_on_endpoint": bool(cell.neb_retry_on_endpoint),
        "wall_s": float(wall_s),
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


def aggregate_rows(
    rows: list[dict[str, Any]], max_reasonable_barrier_ev: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["cell_id"]))].append(row)

    table: list[dict[str, Any]] = []
    for (regime, cell_id), entries in grouped.items():
        converged = sum(1 for r in entries if bool(r["neb_converged"]))
        endpoint_ts = sum(1 for r in entries if _error_is_endpoint_ts(r))
        suspicious = sum(
            1
            for r in entries
            if _barrier_suspicious(r, max_barrier_ev=max_reasonable_barrier_ev)
        )
        finals = [
            float(r["final_fmax"])
            for r in entries
            if r.get("final_fmax") is not None and bool(r["neb_converged"])
        ]
        finals_all = [
            float(r["final_fmax"]) for r in entries if r.get("final_fmax") is not None
        ]
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
                "suspicious_barrier_count": suspicious,
                "median_final_fmax": median(finals) if finals else None,
                "median_final_fmax_all": median(finals_all) if finals_all else None,
                "median_wall_s": median(times) if times else None,
                "config": {
                    "neb_interpolation_method": cfg["neb_interpolation_method"],
                    "neb_interpolation_mic": bool(cfg["neb_interpolation_mic"]),
                    "neb_n_images": int(cfg["n_images"]),
                    "neb_climb": bool(cfg["neb_climb"]),
                    "neb_spring_constant": float(cfg["spring_constant"]),
                    "neb_fmax": float(cfg["fmax"]),
                    "neb_steps": int(cfg["neb_steps"]),
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
            int(t["suspicious_barrier_count"]),
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


def _timestamped_prefix(prefix: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return prefix.with_name(f"{prefix.name}_{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("gas", "supported", "all"), default="all")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--sample-paths", type=int, default=5)
    parser.add_argument("--energy-gap-threshold", type=float, default=1.0)
    parser.add_argument(
        "--similarity-tolerance", type=float, default=DEFAULT_COMPARATOR_TOL
    )
    parser.add_argument("--similarity-pair-cor-max", type=float, default=0.1)
    parser.add_argument(
        "--gas-searches-dir",
        type=Path,
        default=PT_CLUSTER_RESULTS_DIR / "Pt5_searches",
    )
    parser.add_argument(
        "--supported-searches-dir",
        type=Path,
        default=PT_SURFACE_NIO_RESULTS_DIR / "Pt5_searches",
    )
    parser.add_argument("--model", default="mace_matpes_0")
    parser.add_argument("--models", default=None, help="Comma-separated model names.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neb-steps", type=int, default=None)
    parser.add_argument("--gas-neb-steps", type=int, default=None)
    parser.add_argument("--supported-neb-steps", type=int, default=None)
    parser.add_argument("--supported-neb-step-factor", type=float, default=2.0)
    parser.add_argument(
        "--max-reasonable-barrier-ev",
        type=float,
        default=5.0,
        help="Mark converged runs above this barrier as suspicious.",
    )
    parser.add_argument(
        "--include-n-images-7",
        action="store_true",
        help="Include n_images=7 cells in the knob sweep.",
    )
    parser.add_argument(
        "--max-cells-per-regime",
        type=int,
        default=None,
        help="Optional cap for quick smoke/calibration runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmark/neb_pt5_knob_sweep"),
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Append datetime suffix to output prefix.",
    )
    args = parser.parse_args()

    configure_logging(1)
    logger = get_logger(__name__)

    output_prefix = (
        _timestamped_prefix(args.output_prefix)
        if args.timestamp_output
        else args.output_prefix
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    models = _parse_models(args.models, args.model)
    regimes = ["gas", "supported"] if args.regime == "all" else [args.regime]

    dev = torch.device(args.device)
    if not args.dry_run and (dev.type != "cuda" or not torch.cuda.is_available()):
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
            sample_paths=args.sample_paths,
            energy_gap_threshold=args.energy_gap_threshold,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            surface_aware=False,
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
            sample_paths=args.sample_paths,
            energy_gap_threshold=args.energy_gap_threshold,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            surface_aware=True,
            minima_energy_tolerance=DEFAULT_ENERGY_TOLERANCE,
            neb_steps_override=sup_neb_override,
            neb_steps_scale=sup_scale,
        )
        logger.info("Supported bundle: %s", sup_bundle.meta)

    jobs: list[tuple[SweepCell, EndpointBundle, tuple[int, int], int, str]] = []
    if gas_bundle is not None:
        gas_cells = gas_grid(gas_bundle.neb_steps_base, args.include_n_images_7)
        if args.max_cells_per_regime is not None:
            gas_cells = gas_cells[: max(0, args.max_cells_per_regime)]
        for cell in gas_cells:
            for rank, pair in enumerate(gas_bundle.pairs_sampled):
                for model_name in models:
                    jobs.append((cell, gas_bundle, pair, rank, model_name))
    if sup_bundle is not None:
        sup_cells = supported_grid(sup_bundle.neb_steps_base, args.include_n_images_7)
        if args.max_cells_per_regime is not None:
            sup_cells = sup_cells[: max(0, args.max_cells_per_regime)]
        for cell in sup_cells:
            for rank, pair in enumerate(sup_bundle.pairs_sampled):
                for model_name in models:
                    jobs.append((cell, sup_bundle, pair, rank, model_name))

    logger.info("Total jobs: %d", len(jobs))
    if args.dry_run:
        return

    rows: list[dict[str, Any]] = []
    out_jsonl = output_prefix.with_suffix(".jsonl")
    if out_jsonl.exists():
        out_jsonl.unlink()

    for idx, (cell, bundle, pair, pair_rank, model_name) in enumerate(jobs, 1):
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
            "mace_model_name": model_name,
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
            model_name=model_name,
            wall_s=wall_s,
            result=result,
        )
        rows.append(row)
        with out_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[{idx}/{len(jobs)}] {cell.regime} {cell.cell_id} model={model_name} "
            f"pair={pair_rank} conv={row['neb_converged']} status={row['status']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows, args.max_reasonable_barrier_ev)
    agg_path = output_prefix.with_name(output_prefix.name + "_aggregate.json")
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    defaults: dict[str, dict[str, Any]] = {}
    rec_notes: dict[str, str] = {}
    for regime in ("gas", "supported"):
        ranked = [x for x in aggregate if x["regime"] == regime]
        if not ranked:
            continue
        best = ranked[0]
        defaults[regime] = dict(best["config"])
        if float(best["success_rate"]) == 0.0:
            rec_notes[regime] = (
                "No converged runs; recommendation is proxy-ranked by endpoint-as-TS, "
                "suspicious barriers, and final_fmax."
            )

    rec_path = output_prefix.with_name(
        output_prefix.name + "_recommended_defaults.json"
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
