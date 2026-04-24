# SCGO MLIP regression benchmarks

This directory holds **long-running** scripts that sweep cluster sizes (or surface models) and compare recovered minima to reference data. They are **not** the same as the fast EMT checks in [`tests/benchmarks/`](../tests/benchmarks/).

## Dependencies

- **`[mace]`** (default in these scripts): TorchSim GA + MACE. Install with `pip install -e ".[mace]"` (or the conda env from the project root).
- **`[uma]`** (optional): pass `--backend uma` where supported; use a separate environment from MACE to avoid conflicting extras (see main [`README.md`](../README.md)).

## Output layout

Per [`benchmark_common.py`](benchmark_common.py), campaign outputs go under:

- `benchmark/results/` — gas-phase Pt sweeps default here (subfolders per formula/backend/model).
- `benchmark/results/pt_surface_graphite/` — [`benchmark_Pt_surface_graphite.py`](benchmark_Pt_surface_graphite.py) default root.

Each run creates `Formula_searches` trees under the chosen output directory, consistent with other SCGO GO campaigns.

## Entry points

| Script | Purpose |
|--------|---------|
| [`benchmark_Pt.py`](benchmark_Pt.py) | Gas-phase `Pt4`–`Pt11` recovery vs reference minima; CLI and pytest hooks. |
| [`benchmark_Pt_surface_graphite.py`](benchmark_Pt_surface_graphite.py) | Same size sweep for Pt on the bundled graphite surface (`make_graphite_surface_config`). |

Run with:

```bash
python -m benchmark.benchmark_Pt --help
python -m benchmark.benchmark_Pt_surface_graphite --help
```

From the repository root, ensure the package is on `PYTHONPATH` (editable install) so `import benchmark` resolves.

## Environment

`SCGO_BENCHMARK_BACKEND` defaults to `mace` (see `add_common_benchmark_cli` in `benchmark_common.py`).
