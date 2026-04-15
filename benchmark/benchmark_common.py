"""Shared benchmark utilities for Pt cluster regression suites.

Constants, helpers, evaluation logic, and parameter setup used by
benchmark_Pt, benchmark_Pt_PBE, and seed-specific variants.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import Hartree

from scgo.param_presets import get_torchsim_ga_params, get_uma_ga_benchmark_params
from scgo.run_minima import run_scgo_campaign_one_element
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

# Campaign outputs (used by benchmark_Pt, benchmark_Pt_PBE, benchmark_Pt_surface_nio).
# Each suite writes under ``results/<suite>/``; per-formula runs use ``<Formula>_searches``.
BENCHMARK_RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results"
PT_CLUSTER_RESULTS_DIR = BENCHMARK_RESULTS_ROOT / "pt_cluster"
PT_CLUSTER_PBE_RESULTS_DIR = BENCHMARK_RESULTS_ROOT / "pt_cluster_pbe"
PT_CLUSTER_UMA_RESULTS_DIR = BENCHMARK_RESULTS_ROOT / "pt_cluster_uma"
PT_SURFACE_NIO_RESULTS_DIR = BENCHMARK_RESULTS_ROOT / "pt_surface_nio"


def default_pt_cluster_benchmark_output_dir(
    model_name: str | None,
    *,
    calculator: str = "MACE",
) -> Path:
    """Destination for gas-phase Pt benchmark campaigns (split by backend / MACE variant)."""
    if calculator == "UMA":
        return PT_CLUSTER_UMA_RESULTS_DIR.resolve()
    if model_name == "large":
        return PT_CLUSTER_PBE_RESULTS_DIR.resolve()
    return PT_CLUSTER_RESULTS_DIR.resolve()


def resolve_pt_cluster_benchmark_output_dir(
    params: dict,
    output_dir: str | Path | None = None,
) -> Path:
    """Resolve and create the output root for a Pt cluster benchmark campaign."""
    if output_dir is not None:
        path = Path(output_dir).resolve()
    else:
        model_name = params.get("calculator_kwargs", {}).get("model_name")
        calculator = str(params.get("calculator", "MACE"))
        path = default_pt_cluster_benchmark_output_dir(
            model_name, calculator=calculator
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_atom_count(cluster_formula: str) -> int:
    """Extract the integer atom count from a cluster formula like 'Pt7'."""
    digits = "".join(filter(str.isdigit, cluster_formula))
    if not digits:
        raise ValueError(f"Cluster formula '{cluster_formula}' missing atom count.")
    return int(digits)


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
    results = run_scgo_campaign_one_element(
        element,
        min_atoms,
        max_atoms,
        params=params_copy,
        seed=seed,
        output_dir=output_dir,
    )
    _CAMPAIGN_RESULTS_CACHE[key] = results
    return results


def get_benchmark_params(
    seed: int,
    model_name: str | None = None,
    *,
    backend: str = "mace",
    uma_task_name: str = "omat",
) -> dict:
    """Build benchmark params: TorchSim GA + MACE (``backend=mace``) or ASE GA + UMA."""
    if backend == "uma":
        mn = model_name or "uma-s-1p1"
        return get_uma_ga_benchmark_params(seed, model_name=mn, task_name=uma_task_name)
    params = get_torchsim_ga_params(seed=seed)
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name
    return params


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
) -> BenchmarkResult:
    """Run SCGO for a single cluster and evaluate against the benchmark set."""
    lines: list[str] = []
    result = BenchmarkResult(
        cluster_formula=cluster_formula,
        seed=seed,
        total_ground_truth=0,
    )

    n_atoms = _parse_atom_count(cluster_formula)
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

    if campaign_results is None:
        resolved_out = resolve_pt_cluster_benchmark_output_dir(params, output_dir)
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
    uma_task_name: str = "omat",
) -> list[BenchmarkResult]:
    """Execute the Pt benchmark suite for a set of cluster formulas."""
    clusters = clusters or DEFAULT_CLUSTERS
    if params is None:
        params = get_benchmark_params(
            seed,
            model_name=model_name,
            backend=backend,
            uma_task_name=uma_task_name,
        )

    if not clusters:
        return []

    atom_counts = [_parse_atom_count(c) for c in clusters]
    min_atoms = min(atom_counts)
    max_atoms = max(atom_counts)
    element = "Pt"

    resolved_out = resolve_pt_cluster_benchmark_output_dir(params, output_dir)
    shared_campaign_results = _get_campaign_results(
        element=element,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        seed=seed,
        params=params,
        output_dir=resolved_out,
    )

    results: list[BenchmarkResult] = []
    for cluster_formula in clusters:
        result = evaluate_cluster(
            cluster_formula,
            params=params,
            seed=seed,
            comparator_tolerance=0.05,
            num_ground_truth=NUM_GROUND_TRUTH,
            element=element,
            campaign_results=shared_campaign_results,
            verbose=verbose,
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
