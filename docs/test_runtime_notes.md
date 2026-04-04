# Test and validation runtimes (reference)

Machine-local timings vary with CPU/GPU and cache state. Values below are from a full validation run on this repository (conda env `scgo`, single NVIDIA GPU).

## Pytest

| Command | Approx. wall time | Notes |
|--------|-------------------|--------|
| `pytest tests/ -m "not slow"` | ~8–9 min | CI-style fast subset |
| `pytest tests/ -n 2` (full) | (faster than serial) | CI uses **2 workers** (`ubuntu-latest` has 2 vCPUs). Requires `pytest-xdist` (`pip install -e ".[dev]"`). |
| `pytest tests/ -n auto` (full) | (best on multi-core) | Local full suite; scales with CPU cores. |
| `pytest tests/` (full, serial) | ~52 min (historical) | Includes `slow` + `integration`; ~1964 tests |

**Parallel runs (`-n`):** GA tests with `n_jobs_population_init=-2` spawn their own worker processes; with many xdist workers you can oversubscribe CPUs—use a modest `-n` or cap library thread pools if you see thrashing. Heavy MACE tests in `tests/ts_search/test_ts_integration_cu4_mace.py` can contend on a **single GPU**; on GPU boxes you may prefer serial pytest for that file or a lower worker count.

Slowest tests (examples from `--durations=40`): multiprocess reproducibility (~200 s), large template scan (~170 s), runner emulation + Cu4 MACE TS integration (~100–125 s each).

**Suite trim (same strictness):** runner emulation merged to one GA campaign; GA multiprocess repro uses two runs instead of four; `test_near_match_templates_larger_sizes` uses 2 seeds × 4 sizes (was 3 × 4); `test_find_valid_types_very_large` uses `n=600` (was 1000). Re-check coverage after large changes.

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
