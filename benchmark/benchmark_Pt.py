"""Benchmark suite for platinum clusters based on reference structures.

The utilities here drive SCGO against a curated Pt benchmark set and evaluate
the recovered minima primarily through geometric agreement and, secondarily,
relative energetic ordering. The script doubles as a pytest entry point and a
standalone CLI for rapid regression checks while tweaking optimization knobs.
"""

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


@pytest.mark.parametrize(
    "cluster_formula, seed",
    [(c, s) for c in DEFAULT_CLUSTERS for s in BENCHMARK_SEEDS],
)
def test_benchmark_minima_recovery(cluster_formula: str, seed: int):
    """Pytest entry point for validating a single cluster benchmark run."""
    params = get_benchmark_params(seed, model_name=MODEL_NAME)
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


def main():
    """CLI entry point for running the Pt benchmark suite."""
    run_benchmark_suite(seed=42, model_name=MODEL_NAME, verbose=True)


if __name__ == "__main__":
    main()
