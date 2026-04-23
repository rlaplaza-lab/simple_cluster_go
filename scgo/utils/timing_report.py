"""Timing summary logging and ``timing.json`` for GO, basin hopping, NEB/TS, and GO+TS."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

TIMING_JSON_FILENAME = "timing.json"

_DB_IO_SUM_KEYS: tuple[str, ...] = (
    "db_read_s",
    "db_write_s",
    "offspring_db_io_s",
    "initial_unrelaxed_insert_s",
    "initial_relaxed_write_s",
    "offspring_unrelaxed_insert_s",
    "offspring_relaxed_write_s",
    "unrelaxed_insert_s",
    "relaxed_write_s",
)


def relax_seconds_from_timings(timings: dict[str, float]) -> float:
    if "go_phase_s" in timings or "ts_neb_sum_s" in timings:
        return float(timings.get("go_phase_s", 0.0)) + float(
            timings.get("ts_neb_sum_s", 0.0)
        )
    if "neb_optimization_s" in timings:
        return float(timings.get("neb_optimization_s", 0.0))
    if "local_relaxation_s" in timings and "relax_batch_s" not in timings:
        return float(timings.get("local_relaxation_s", 0.0))
    if "relax_batch_s" in timings:
        return float(timings.get("relax_batch_s", 0.0))
    return float(timings.get("initial_local_relaxation_s", 0.0)) + float(
        timings.get("offspring_local_relaxation_s", 0.0)
    )


def log_timing_summary(
    logger: logging.Logger,
    backend: str,
    timings_s: dict[str, float],
    *,
    verbosity: int,
) -> None:
    if verbosity < 1:
        return
    total = float(timings_s.get("total_wall_s", 0.0))
    relax = relax_seconds_from_timings(timings_s)
    cpu = float(timings_s.get("cpu_non_relax_s", max(0.0, total - relax)))
    db_io = sum(float(timings_s.get(k, 0.0)) for k in _DB_IO_SUM_KEYS)
    logger.info(
        "Timing (%s): total=%.1fs, relax=%.1fs, non_relax=%.1fs, db_io=%.1fs",
        backend,
        total,
        relax,
        cpu,
        db_io,
    )


def write_timing_file(
    output_dir: str,
    payload: dict[str, Any],
    *,
    filename: str | None = None,
) -> None:
    name = filename if filename is not None else TIMING_JSON_FILENAME
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sum_neb_seconds_from_ts_results(
    ts_results: list[dict[str, Any]],
) -> float:
    return sum(
        float((r.get("timings_s") or {}).get("neb_optimization_s", 0.0))
        for r in ts_results
    )
