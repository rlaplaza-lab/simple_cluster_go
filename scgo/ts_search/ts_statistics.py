"""Shared transition-state statistics helpers."""

from __future__ import annotations

from typing import Any, TypedDict


class TransitionStateStatistics(TypedDict):
    """Common TS statistics payload shared across output artifacts."""

    total_ts_found: int
    converged_ts: int
    successful_ts: int
    min_barrier: float | None
    max_barrier: float | None
    avg_barrier: float | None


def compute_ts_statistics(
    ts_results: list[dict[str, Any]],
) -> TransitionStateStatistics:
    """Compute consistent success/convergence/barrier statistics."""
    successful_results = [
        result for result in ts_results if result.get("status", "success") == "success"
    ]
    barriers = [
        float(result["barrier_height"])
        for result in successful_results
        if result.get("barrier_height") is not None
    ]
    successful_count = len(successful_results)
    converged_count = sum(
        1 for result in successful_results if result.get("neb_converged")
    )
    return {
        "total_ts_found": successful_count,
        "converged_ts": converged_count,
        "successful_ts": successful_count,
        "min_barrier": float(min(barriers)) if barriers else None,
        "max_barrier": float(max(barriers)) if barriers else None,
        "avg_barrier": float(sum(barriers) / len(barriers)) if barriers else None,
    }
