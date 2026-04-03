# Test and validation runtimes (reference)

Machine-local timings vary with CPU/GPU and cache state. Values below are from a full validation run on this repository (conda env `scgo`, single NVIDIA GPU).

## Pytest

| Command | Approx. wall time | Notes |
|--------|-------------------|--------|
| `pytest tests/ -m "not slow"` | ~8–9 min | CI-style fast subset |
| `pytest tests/` (full) | ~52 min | Includes `slow` + `integration`; ~1964 tests |

Slowest tests (examples from `--durations=40`): multiprocess reproducibility (~200 s), large template scan (~170 s), runner emulation + Cu4 MACE TS integration (~100–125 s each).

## Benchmark and example runners

| Step | Approx. wall time |
|------|-------------------|
| `python benchmark/benchmark_Pt.py` | ~35 min |
| `python runners/run_scgo_with_ts_search.py` | (varies; GPU-heavy) |
| `python runners/run_scgo_with_ts_search_graphene.py` | (varies; GPU-heavy) |
| `python runners/run_scgo_with_ts_search_graphene_with_oh.py` | (varies; GPU-heavy) |

Run GPU jobs **one at a time** on a single-GPU machine.

## Dependency pins (PyPI)

As of the last refresh: `mace-torch==0.3.15`, `torch-sim-atomistic[mace]==0.5.2`, `e3nn==0.4.4` (see `pyproject.toml`).

## Pytest warning filters

`pytest.ini` filters unavoidable third-party deprecations (e.g. `torch.jit.*` while MACE/torch_sim still use those paths). Remaining warnings from MACE/e3nn/torch are often environmental (e.g. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`).
