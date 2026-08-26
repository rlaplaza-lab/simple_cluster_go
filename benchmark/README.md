# SCGO MLIP regression benchmarks

This directory holds **long-running** scripts that sweep cluster sizes (or surface models) and compare recovered minima to reference data. They are separate from the fast unit tests of the benchmark helpers in [`tests/benchmark/`](../tests/benchmark/).

## Dependencies

- **`[mace]`** (default in these scripts): TorchSim GA + MACE. Install with `pip install "scgo[mace]"` from PyPI or `pip install -e ".[mace]"` from the repository root.
- **`[uma]`** (optional): pass `--backend uma` where supported; use a separate environment from MACE to avoid conflicting extras (see main [`README.md`](../README.md)).

## Output layout

Per [`benchmark_common.py`](benchmark_common.py), campaign outputs go under `benchmark/results/`:

- **Gas-phase Pt sweeps** — `benchmark/results/{formula}_{backend}_{model}/` (for example `pt5_mace_mace_matpes_0/`), then per run:

  ```
  Pt5_searches/
  ├── run_<timestamp>_<microseconds>/
  │   ├── metadata.json
  │   ├── timing.json
  │   └── ga_go.db
  ├── results_summary.json
  └── final_unique_minima/
  ```

- **Surface Pt-on-graphite** — flat root `benchmark/results/pt_surface_graphite/` ([`benchmark_Pt_surface_graphite.py`](benchmark_Pt_surface_graphite.py)), same `{path_key}_searches/` layout (e.g. `Pt5_graphite_searches/`).

TS runs add sibling `{path_key}_ts_results/` trees with the same run-oriented layout (`run_*/`, `results_summary.json`, deduplicated export); pair artifacts use `pair_*` subdirectories. See [`docs/source/output_layout.rst`](../docs/source/output_layout.rst) (*On-disk layout*).

Benchmark GA presets enable `write_timing_json` and `detailed_timing` so profiling lines in CLI output match `{run_dir}/timing.json` on disk.

## Entry points

| Script | Purpose |
|--------|---------|
| [`benchmark_Pt.py`](benchmark_Pt.py) | Gas-phase `Pt4`–`Pt11` recovery vs reference minima; CLI and pytest hooks. |
| [`benchmark_Pt_surface_graphite.py`](benchmark_Pt_surface_graphite.py) | Same size sweep for Pt on the bundled graphite surface (`make_graphite_surface_config`). |
| [`benchmark_parallel_neb.py`](benchmark_parallel_neb.py) | Serial vs parallel NEB wall-time comparison on existing Pt5 minima. |

Run with:

```bash
python -m benchmark.benchmark_Pt --help
python -m benchmark.benchmark_Pt_surface_graphite --help
python -m benchmark.benchmark_parallel_neb --help
python -m benchmark.benchmark_parallel_neb \
    --searches-dir /path/to/results/pt5_mace_mace_matpes_0/Pt5_searches \
    --max-pairs 5
```

From the repository root, ensure the package is on `PYTHONPATH` (editable install) so `import benchmark` resolves.

## Pytest

[`pytest.ini`](../pytest.ini) sets `testpaths = tests`, so `benchmark/` is **not** collected by default `pytest tests/`. MLIP regression hooks also require CUDA (`@pytest.mark.requires_cuda` on `benchmark_Pt.py`). To run them explicitly:

```bash
pytest benchmark/ -m "slow and requires_cuda"
```

## Environment

`SCGO_BENCHMARK_BACKEND` defaults to `mace` (see `add_common_benchmark_cli` in `benchmark_common.py`).
