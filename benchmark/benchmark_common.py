"""Shared benchmark utilities for Pt cluster regression suites.

Constants, helpers, evaluation logic, and parameter setup used by
benchmark_Pt and benchmark variants.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import Hartree

from scgo.param_presets import get_torchsim_ga_params, get_uma_ga_benchmark_params
from scgo.runner_api import run_go_campaign
from scgo.utils.atoms_helpers import parse_energy_from_xyz_comment
from scgo.utils.comparators import (
    PureInteratomicDistanceComparator as InteratomicDistanceComparator,
)

# ---------------------------------------------------------------------------
# Constants & simple configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_CLUSTERS = [f"Pt{n}" for n in range(4, 12)]
# Structural recall vs 50 reference minima per cluster (comparator tol 0.05).
# Perfect 50/50 is ideal but not required for every size; 0.88 (=44/50) marks
# "very close" recovery for regression and CLI summary.
MIN_RECOVERY_RATE = 0.88
NUM_GROUND_TRUTH = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "benchmark_files_Pt"

# Campaign outputs (used by benchmark_Pt and surface benchmark scripts).
# Each suite writes under ``results/<suite>/``; per-formula runs use ``<Formula>_searches``.
BENCHMARK_RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results"
PT_SURFACE_GRAPHITE_RESULTS_DIR = BENCHMARK_RESULTS_ROOT / "pt_surface_graphite"


def _slug(value: str) -> str:
    """Normalize names used in benchmark output directory names."""
    return value.strip().lower().replace("/", "-").replace(" ", "-").replace("__", "_")


def default_pt_cluster_benchmark_output_dir(
    cluster_formula: str,
    params: dict,
) -> Path:
    """Destination for gas-phase Pt benchmarks (per composition + backend/model)."""
    calculator = str(params.get("calculator", "MACE")).upper()
    calculator_kwargs = params.get("calculator_kwargs", {}) or {}
    model_name = str(calculator_kwargs.get("model_name", "default"))
    prefix = _slug(cluster_formula)
    if calculator == "UMA":
        task_name = str(calculator_kwargs.get("task_name", "default"))
        dirname = f"{prefix}_uma_{_slug(model_name)}-{_slug(task_name)}"
    else:
        dirname = f"{prefix}_mace_{_slug(model_name)}"
    return (BENCHMARK_RESULTS_ROOT / dirname).resolve()


def resolve_pt_cluster_benchmark_output_dir(
    cluster_formula: str,
    params: dict,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve and create the output root for a Pt cluster benchmark campaign."""
    if output_dir is not None:
        path = Path(output_dir).resolve()
    else:
        path = default_pt_cluster_benchmark_output_dir(cluster_formula, params)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_atom_count(cluster_formula: str) -> int:
    """Extract the integer atom count from a cluster formula like 'Pt7'."""
    digits = "".join(filter(str.isdigit, cluster_formula))
    if not digits:
        raise ValueError(f"Cluster formula '{cluster_formula}' missing atom count.")
    return int(digits)


def add_common_benchmark_cli(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by all Pt benchmark scripts."""
    parser.add_argument(
        "--backend",
        choices=("mace", "uma"),
        default=os.environ.get("SCGO_BENCHMARK_BACKEND", "mace"),
        help=(
            "mace: TorchSim GA + MACE (GPU-friendly default for ML potentials); "
            "uma: ASE GA + UMA (install scgo[uma] in a separate env)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override calculator model_name (default UMA: uma-s-1p2; MACE: preset default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uma-task",
        default="oc25",
        help="UMA task (only used when --backend uma).",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        metavar="FORMULA",
        help="Optional subset (e.g. Pt4 Pt5). Default: full DEFAULT_CLUSTERS list.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=10,
        help="GA generations per benchmark campaign.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="GA population size per benchmark campaign.",
    )


@dataclass
class BenchmarkResult:
    """Container summarising the outcome of a single cluster benchmark."""

    cluster_formula: str
    seed: int
    total_ground_truth: int
    matched_indices: list[int] = field(default_factory=list)
    unmatched_indices: list[int] = field(default_factory=list)
    match_details: dict[int, dict[str, float]] = field(default_factory=dict)
    relative_order_matches: bool | None = None
    recovery_rate: float = 0.0
    found_ground_state: bool = False
    messages: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


_CAMPAIGN_RESULTS_CACHE: dict[str, dict[str, list[tuple[float, Atoms]]]] = {}


def _params_digest(params: dict) -> str:
    """Return a stable string representation for caching purposes."""
    return json.dumps(params, sort_keys=True, default=str)


def _campaign_cache_key(
    element: str,
    min_atoms: int,
    max_atoms: int,
    seed: int,
    params: dict,
    output_dir: str | Path | None,
) -> str:
    digest = _params_digest(params)
    out = str(Path(output_dir).resolve()) if output_dir is not None else ""
    return f"{element}|{min_atoms}|{max_atoms}|{seed}|{digest}|{out}"


def load_latest_ga_profile(
    output_dir: str | Path,
    cluster_formula: str,
) -> dict | None:
    """Load ``timing.json`` from the latest run."""
    root = Path(output_dir)
    run_dirs = sorted(
        (root / f"{cluster_formula}_searches").glob("run_*"),
        key=lambda p: p.name,
    )
    if not run_dirs:
        return None
    latest = run_dirs[-1]
    trial_dir = latest / "trial_1"
    profile_path = trial_dir / "timing.json"
    if not profile_path.exists():
        return None
    try:
        return json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def format_ga_profile_lines(
    profile: dict,
    *,
    detailed: bool = True,
    max_entries: int = 8,
) -> list[str]:
    """Return human-readable profiling lines from a GA profile payload."""
    timings = profile.get("timings_s", {})
    backend = str(profile.get("backend", "ga"))
    total = float(timings.get("total_wall_s", 0.0))
    if "local_relaxation_s" in timings:
        relax = float(timings.get("local_relaxation_s", 0.0))
    elif "relax_batch_s" in timings:
        relax = float(timings.get("relax_batch_s", 0.0))
    else:
        relax = float(timings.get("initial_local_relaxation_s", 0.0)) + float(
            timings.get("offspring_local_relaxation_s", 0.0)
        )
    cpu = float(timings.get("cpu_non_relax_s", max(0.0, total - relax)))
    db_io = float(
        timings.get("db_read_s", 0.0)
        + timings.get("db_write_s", 0.0)
        + timings.get("offspring_db_io_s", 0.0)
        + timings.get("initial_unrelaxed_insert_s", 0.0)
        + timings.get("initial_relaxed_write_s", 0.0)
        + timings.get("offspring_unrelaxed_insert_s", 0.0)
        + timings.get("offspring_relaxed_write_s", 0.0)
    )
    lines = [
        (
            f"Profiling ({backend}): total={total:.1f}s, relax={relax:.1f}s, "
            f"cpu_non_relax={cpu:.1f}s, db_io={db_io:.1f}s"
        )
    ]
    if not detailed:
        return lines

    timing_items: list[tuple[str, float]] = []
    for key, value in timings.items():
        if key == "total_wall_s":
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration <= 0.0:
            continue
        timing_items.append((key, duration))

    if not timing_items:
        lines.append("Profiling details: no non-zero timing phases captured.")
        return lines

    timing_items.sort(key=lambda item: item[1], reverse=True)
    total_for_pct = total if total > 0.0 else sum(v for _, v in timing_items)
    lines.append(
        f"Profiling details (top {min(max_entries, len(timing_items))} phases by wall time):"
    )
    for name, duration in timing_items[:max_entries]:
        pct = (duration / total_for_pct * 100.0) if total_for_pct > 0.0 else 0.0
        lines.append(f"  - {name}: {duration:.2f}s ({pct:.1f}%)")

    counters = profile.get("counters")
    if isinstance(counters, dict) and counters:
        parts: list[str] = []
        for key in sorted(counters):
            value = counters[key]
            parts.append(f"{key}={value}")
        lines.append(f"Counters: {', '.join(parts)}")

    per_gen = profile.get("per_generation")
    if isinstance(per_gen, list) and per_gen:
        timing_rows: list[dict[str, float]] = []
        for row in per_gen:
            if not isinstance(row, dict):
                continue
            t = row.get("timings_s")
            if isinstance(t, dict):
                parsed: dict[str, float] = {}
                for k, v in t.items():
                    try:
                        parsed[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                if parsed:
                    timing_rows.append(parsed)

        if timing_rows:
            keys: set[str] = set()
            for row in timing_rows:
                keys.update(row.keys())

            def _series(key: str) -> np.ndarray:
                return np.asarray(
                    [row.get(key, 0.0) for row in timing_rows], dtype=float
                )

            ordered_keys = [
                "parent_select_s",
                "crossover_s",
                "mutation_s",
                "db_unrelaxed_insert_s",
                "relax_s",
                "db_relaxed_write_s",
                "torchsim_db_read_s",
                "torchsim_relax_s",
                "torchsim_db_write_s",
                "population_update_s",
            ]
            remaining = sorted(k for k in keys if k not in ordered_keys)
            ordered_all = [k for k in ordered_keys if k in keys] + remaining

            # Rank phases by mean time per generation (descending), then show the most relevant ones.
            ranked: list[tuple[str, float]] = []
            for key in ordered_all:
                s = _series(key)
                mean = float(np.mean(s))
                if mean <= 0.0:
                    continue
                ranked.append((key, mean))
            ranked.sort(key=lambda kv: kv[1], reverse=True)

            lines.append(
                f"Per-generation timing summary (N={len(timing_rows)} generations):"
            )
            for key, _mean in ranked[:12]:
                s = _series(key)
                mean = float(np.mean(s))
                median = float(np.median(s))
                p90 = float(np.quantile(s, 0.9))
                lines.append(
                    f"  - {key}: mean={mean:.3f}s, median={median:.3f}s, p90={p90:.3f}s"
                )

    return lines


def _get_campaign_results(
    element: str,
    min_atoms: int,
    max_atoms: int,
    seed: int,
    params: dict,
    output_dir: str | Path,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run (or reuse) a SCGO campaign for the requested configuration."""
    key = _campaign_cache_key(element, min_atoms, max_atoms, seed, params, output_dir)
    cached = _CAMPAIGN_RESULTS_CACHE.get(key)
    if cached is not None:
        return cached

    params_copy = deepcopy(params)
    compositions = [[element] * n for n in range(min_atoms, max_atoms + 1)]
    results = run_go_campaign(
        compositions,
        params=params_copy,
        seed=seed,
        output_dir=output_dir,
        system_type="gas_cluster",
    )
    _CAMPAIGN_RESULTS_CACHE[key] = results
    return results


def get_benchmark_params(
    seed: int,
    model_name: str | None = None,
    *,
    backend: str = "mace",
    uma_task: str = "oc25",
) -> dict:
    """Build benchmark params: TorchSim GA + MACE (``backend=mace``) or ASE GA + UMA."""
    if backend == "uma":
        mn = model_name or "uma-s-1p2"
        return get_uma_ga_benchmark_params(seed, model_name=mn, uma_task=uma_task)
    params = get_torchsim_ga_params(seed=seed, model_name=model_name)
    params["calculator"] = "MACE"
    return params


def apply_ga_benchmark_overrides(
    params: dict,
    *,
    niter: int,
    population_size: int,
    surface_config: Any | None = None,
    n_jobs_population_init: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """Return params with standard GA benchmark overrides applied."""
    params_copy = deepcopy(params)
    ga_params = params_copy.setdefault("optimizer_params", {}).setdefault("ga", {})
    ga_params["niter"] = niter
    ga_params["population_size"] = population_size
    if surface_config is not None:
        ga_params["surface_config"] = surface_config
    if n_jobs_population_init is not None:
        ga_params["n_jobs_population_init"] = n_jobs_population_init
    if batch_size is not None:
        ga_params["batch_size"] = batch_size
    return params_copy


def load_ground_truth_minima(
    benchmark_file: Path,
    num_to_load: int = NUM_GROUND_TRUTH,
) -> list[tuple[float, Atoms]] | None:
    """Load ground truth minima from a benchmark XYZ file."""
    if not benchmark_file.is_file():
        return None

    raw_gt_minima = read(benchmark_file, index=":")
    if not isinstance(raw_gt_minima, list):
        raw_gt_minima = [raw_gt_minima]

    ground_truth_minima: list[tuple[float, Atoms]] = []
    for atoms_obj in raw_gt_minima:
        if not isinstance(atoms_obj, Atoms):
            continue
        energy_hartree = parse_energy_from_xyz_comment(atoms_obj.info)
        if energy_hartree is None:
            raise ValueError(f"Could not parse energy from comment in {benchmark_file}")
        energy_ev = energy_hartree * Hartree
        ground_truth_minima.append((energy_ev, atoms_obj))

    ground_truth_minima.sort(key=lambda x: x[0])
    return ground_truth_minima[:num_to_load]


def find_best_match(
    gt_atoms: Atoms,
    found_minima: list[tuple[float, Atoms]],
    comparator: InteratomicDistanceComparator,
) -> tuple[dict | None, float]:
    """Find best structural match for a ground truth structure."""
    best_match = None
    min_diff = float("inf")
    for found_energy, found_atoms in found_minima:
        diff = comparator.get_differences(gt_atoms, found_atoms)[0]
        if diff < min_diff:
            min_diff = diff
            best_match = {"energy": found_energy, "atoms": found_atoms, "diff": diff}

    return best_match, min_diff


def evaluate_cluster(
    cluster_formula: str,
    params: dict,
    seed: int,
    comparator_tolerance: float = 0.05,
    num_ground_truth: int = NUM_GROUND_TRUTH,
    element: str = "Pt",
    campaign_results: dict[str, list[tuple[float, Atoms]]] | None = None,
    verbose: bool = False,
    output_dir: str | Path | None = None,
    profile_detail: bool = True,
    profile_top_n: int = 8,
) -> BenchmarkResult:
    """Run SCGO for a single cluster and evaluate against the benchmark set."""
    lines: list[str] = []
    result = BenchmarkResult(
        cluster_formula=cluster_formula,
        seed=seed,
        total_ground_truth=0,
    )

    n_atoms = parse_atom_count(cluster_formula)
    benchmark_file = BENCHMARK_DIR / f"{cluster_formula}.xyz"
    ground_truth_minima = load_ground_truth_minima(
        benchmark_file,
        num_to_load=num_ground_truth,
    )

    if ground_truth_minima is None:
        result.skipped = True
        result.skip_reason = f"Benchmark file not found: {benchmark_file}"
        lines.append(result.skip_reason)
        result.messages = lines
        if verbose:
            for line in lines:
                print(line)
        return result

    result.total_ground_truth = len(ground_truth_minima)
    lines.append(
        f"--- Running SCGO for {cluster_formula} (seed={seed}) against {result.total_ground_truth} ground truth minima ---",
    )

    resolved_out: Path | None = None
    if campaign_results is None:
        resolved_out = resolve_pt_cluster_benchmark_output_dir(
            cluster_formula, params, output_dir
        )
        campaign_results = _get_campaign_results(
            element=element,
            min_atoms=n_atoms,
            max_atoms=n_atoms,
            seed=seed,
            params=params,
            output_dir=resolved_out,
        )
    found_minima_tuples = campaign_results.get(cluster_formula)

    if not found_minima_tuples:
        result.skipped = True
        result.skip_reason = f"SCGO campaign returned no minima for {cluster_formula}."
        lines.append(result.skip_reason)
        result.messages = lines
        if verbose:
            for line in lines:
                print(line)
        return result

    comparator = InteratomicDistanceComparator(tol=comparator_tolerance, mic=False)
    lines.append("Evaluating structural matches...")

    matches: dict[int, dict[str, float]] = {}
    for idx, (gt_energy, gt_atoms) in enumerate(ground_truth_minima):
        best_match, min_diff = find_best_match(
            gt_atoms,
            found_minima_tuples,
            comparator,
        )
        if best_match and min_diff < comparator.tol:
            found_energy = best_match["energy"]
            matches[idx] = {
                "gt_energy": gt_energy,
                "found_energy": found_energy,
                "delta_energy": found_energy - gt_energy,
                "structural_diff": best_match["diff"],
            }
            lines.append(
                f"  - GT #{idx + 1:02d} (E_ref={gt_energy:9.4f} eV) -> match (E={found_energy:9.4f} eV, dE={found_energy - gt_energy:+7.4f} eV, diff={best_match['diff']:.4f})",
            )
        else:
            lines.append(
                f"  - GT #{idx + 1:02d} (E_ref={gt_energy:9.4f} eV) -> no match (closest diff={min_diff:.4f}, tol={comparator.tol})",
            )

    matched_indices = sorted(matches.keys())
    unmatched_indices = [i for i in range(len(ground_truth_minima)) if i not in matches]
    result.matched_indices = matched_indices
    result.unmatched_indices = unmatched_indices
    result.match_details = matches

    num_matched = len(matched_indices)
    num_total = len(ground_truth_minima)
    result.recovery_rate = num_matched / num_total if num_total else 0.0
    result.found_ground_state = 0 in matched_indices if num_total else False

    lines.append(
        f"Recovered {num_matched}/{num_total} minima (recovery rate {result.recovery_rate:.2f}).",
    )
    if result.found_ground_state:
        lines.append("Ground state recovered.")
    else:
        lines.append("Ground state missing.")

    if num_matched > 1:
        matched_gt_energies = [matches[i]["gt_energy"] for i in matched_indices]
        matched_found_energies = [matches[i]["found_energy"] for i in matched_indices]
        gt_order = [int(x) for x in np.argsort(np.argsort(matched_gt_energies))]
        found_order = [int(x) for x in np.argsort(np.argsort(matched_found_energies))]
        result.relative_order_matches = gt_order == found_order
        lines.append(
            "Relative ordering "
            + ("matches." if result.relative_order_matches else "differs."),
        )
        if not result.relative_order_matches:
            lines.append(f"  - Reference order: {gt_order}")
            lines.append(f"  - Found order:     {found_order}")
    else:
        result.relative_order_matches = None
        lines.append("Not enough matches to assess relative ordering.")

    if result.recovery_rate < MIN_RECOVERY_RATE:
        lines.append(
            f"Recovery rate {result.recovery_rate:.2f} is below threshold {MIN_RECOVERY_RATE:.2f}.",
        )

    profile_root = output_dir or resolved_out
    if profile_root is not None:
        profile = load_latest_ga_profile(profile_root, cluster_formula)
        if profile:
            lines.extend(
                format_ga_profile_lines(
                    profile,
                    detailed=profile_detail,
                    max_entries=profile_top_n,
                )
            )

    result.messages = lines

    if verbose:
        for line in lines:
            print(line)

    return result


def run_benchmark_suite(
    clusters: list[str] | None = None,
    seed: int = 42,
    params: dict | None = None,
    model_name: str | None = None,
    verbose: bool = True,
    output_dir: str | Path | None = None,
    *,
    backend: str = "mace",
    uma_task: str = "oc25",
    profile_detail: bool = True,
    profile_top_n: int = 8,
) -> list[BenchmarkResult]:
    """Execute the Pt benchmark suite for a set of cluster formulas."""
    clusters = clusters or DEFAULT_CLUSTERS
    if params is None:
        params = get_benchmark_params(
            seed,
            model_name=model_name,
            backend=backend,
            uma_task=uma_task,
        )

    if not clusters:
        return []

    element = "Pt"

    results: list[BenchmarkResult] = []
    for cluster_formula in clusters:
        n_atoms = parse_atom_count(cluster_formula)
        resolved_out = resolve_pt_cluster_benchmark_output_dir(
            cluster_formula, params, output_dir
        )
        per_cluster_results = _get_campaign_results(
            element=element,
            min_atoms=n_atoms,
            max_atoms=n_atoms,
            seed=seed,
            params=params,
            output_dir=resolved_out,
        )
        result = evaluate_cluster(
            cluster_formula,
            params=params,
            seed=seed,
            comparator_tolerance=0.05,
            num_ground_truth=NUM_GROUND_TRUTH,
            element=element,
            campaign_results=per_cluster_results,
            verbose=verbose,
            output_dir=resolved_out,
            profile_detail=profile_detail,
            profile_top_n=profile_top_n,
        )
        results.append(result)

    if verbose:
        successful = sum(
            1
            for res in results
            if not res.skipped and res.recovery_rate >= MIN_RECOVERY_RATE
        )
        attempted = sum(1 for res in results if not res.skipped)
        print(
            f"\n=== Benchmark summary: {successful}/{attempted} clusters met the recovery threshold ({MIN_RECOVERY_RATE:.2f}). ===",
        )

    return results
