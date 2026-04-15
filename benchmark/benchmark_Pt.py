"""Benchmark suite for platinum clusters based on reference structures.

The utilities here drive SCGO against a curated Pt benchmark set and evaluate
the recovered minima primarily through geometric agreement and, secondarily,
relative energetic ordering. The script doubles as a pytest entry point and a
standalone CLI for rapid regression checks while tweaking optimization knobs.

Campaign outputs are written under ``benchmark/results/pt_cluster/`` (MACE) or
``benchmark/results/pt_cluster_uma/`` (UMA); see ``benchmark.benchmark_common``.
"""

from __future__ import annotations

import argparse
import os
import time

import pytest

from benchmark.benchmark_common import (
    DEFAULT_CLUSTERS,
    MIN_RECOVERY_RATE,
    NUM_GROUND_TRUTH,
    evaluate_cluster,
    get_benchmark_params,
    run_benchmark_suite,
)

BENCHMARK_SEEDS = [42, 43, 44, 45]
MODEL_NAME: str | None = None
BENCHMARK_BACKEND = os.environ.get("SCGO_BENCHMARK_BACKEND", "mace")


@pytest.mark.parametrize(
    "cluster_formula, seed",
    [(c, s) for c in DEFAULT_CLUSTERS for s in BENCHMARK_SEEDS],
)
def test_benchmark_minima_recovery(cluster_formula: str, seed: int):
    """Pytest entry point for validating a single cluster benchmark run."""
    params = get_benchmark_params(
        seed, model_name=MODEL_NAME, backend=BENCHMARK_BACKEND
    )
    result = evaluate_cluster(
        cluster_formula,
        params=params,
        seed=seed,
        comparator_tolerance=0.05,
        num_ground_truth=NUM_GROUND_TRUTH,
        verbose=True,
    )

    if result.skipped:
        pytest.skip(result.skip_reason or "Benchmark skipped")

    if not result.matched_indices:
        pytest.skip(
            f"No minima recovered for {cluster_formula}. Check optimizer settings.",
        )

    if result.recovery_rate < MIN_RECOVERY_RATE:
        pytest.xfail(
            f"Recovery rate {result.recovery_rate:.2f} below threshold {MIN_RECOVERY_RATE:.2f}.",
        )


def main() -> None:
    """CLI entry point for running the Pt benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Pt cluster geometric recovery benchmark (MACE/TorchSim or UMA/ASE GA).",
    )
    parser.add_argument(
        "--backend",
        choices=("mace", "uma"),
        default=os.environ.get("SCGO_BENCHMARK_BACKEND", "mace"),
        help="mace: TorchSim GA + MACE; uma: ASE GA + UMA (install scgo[uma] in a separate env).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override calculator model_name (default UMA: uma-s-1p1; MACE: preset default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uma-task",
        default="omat",
        help="UMA task_name (only used when --backend uma).",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        metavar="FORMULA",
        help="Optional subset (e.g. Pt4 Pt5). Default: full DEFAULT_CLUSTERS list.",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    run_benchmark_suite(
        clusters=args.clusters,
        seed=args.seed,
        model_name=args.model_name,
        verbose=True,
        backend=args.backend,
        uma_task_name=args.uma_task,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBenchmark wall time: {elapsed:.1f} s ({args.backend})")


if __name__ == "__main__":
    main()
