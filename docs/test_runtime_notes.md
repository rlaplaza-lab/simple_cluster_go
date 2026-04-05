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

### MACE NEB preset sweep (TorchSim, local GPU)

From the repo root, conda env `scgo`, single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/neb_sweep_mace.py --output benchmark/neb_mace_primary.jsonl
```

**Surface-adsorbed nanoparticles (real minima, recommended):** pass the same SCGO directory you would use for TS search (contains `run_*/` databases and usually `final_unique_minima/*.xyz`). Composition is read from the first xyz if you omit `--surface-composition`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/neb_sweep_mace.py \
  --regime surface \
  --surface-searches-dir ts_search_graphene_results/Cu4_searches \
  --output benchmark/neb_mace_surface_real.jsonl
```

Without `--surface-searches-dir`, `--regime surface` **exits** unless you pass `--allow-synthetic-surface` (toy slab pair; not useful for tuning). For `--regime all` without a surface dir, use `--allow-synthetic-surface` or provide `--surface-searches-dir`.

**Gas-phase real minima (optional):** `--gas-searches-dir PATH` with optional `--gas-composition`.

**Graphene + Cu₄OH (extra validation campaign):** after `runners/run_scgo_with_ts_search_graphene_with_oh.py`, point `--surface-searches-dir` at `ts_search_graphene_with_oh_results/..._searches`. Presets still use a single `regime="surface"` unless a dedicated OH sweep drives different defaults.

Other flags:

- `--regime gas|surface|all` — which regimes to sweep (default `all`).
- `--pair-index N` — starting pair index from `select_structure_pairs` (default `0`); with `--primary-pair-count` / `--extended-pair-count` >1, consecutive pairs `N, N+1, …` are used.
- `--primary-pair-count K` — with `--grid primary` and real searches dirs, repeat the full primary factorial over `K` consecutive pairs (default `1`). Improves representativeness vs tuning on a single pair only.
- `--no-energy-gap-filter` — pair minima regardless of energy gap.
- `--max-cells N` — truncate the grid for smoke tests.
- `--summarize FILE.jsonl` — convergence-ranked summary (no GPU); optional `--summarize-physics-check` flags converged rows with `barrier_height` above `--summarize-max-barrier-ev` (default 3 eV).
- `--list-followup-sweeps` — print suggested axes for future sweeps (pairing, ML dtype, step budget, etc.).
- `--grid extended` — fixed “winner” NEB settings × `align_endpoints` × `perturb_sigma` × tangent × `--extended-pair-count` pairs (needs real `--gas-searches-dir` / `--surface-searches-dir`). Optional `--sweep-retry-on-endpoint` adds `neb_retry_on_endpoint` True/False (doubles extended cell count per pair).
- `--models id1,id2` — repeat the entire job list once per MACE model id (large wall-time multiplier; use with `--max-cells` for smoke).
- `--dry-run` — prints total NEB count, per-regime cell counts, model replication factor, and each bundle’s endpoint metadata (no GPU).

Sweep output is gitignored (`benchmark/neb_mace*.jsonl`); keep a local copy for tuning `get_ts_search_params(regime=...)`.

A full **synthetic** ``--regime all`` run with ``--allow-synthetic-surface`` is **96** NEB jobs per model (order-of-magnitude **~1 h wall** on a mid-range GPU with `mace_matpes_0`). Multiplying by `--primary-pair-count` or `--models` scales linearly. Real minima use `auto_niter_ts` for step budget unless `--neb-steps` is set; wall time scales with system size. JSONL rows include `pair_index`, `n_atoms`, `pbc`, and `neb_retry_on_endpoint` for auditing.

### MACE sweep → `get_ts_search_params` (reference runs, Apr 2026)

Local JSONL outputs (gitignored): `benchmark/neb_mace_gas_Cu4.jsonl`, `benchmark/neb_mace_surface_graphene.jsonl`.

| Regime | Minima source | Outcome | Preset takeaways |
|--------|---------------|---------|------------------|
| Gas | `ts_search_results/Cu4_searches` (pairs 0–1 in reference JSONL) | 24/24 converged | Keep **idpp**, **n_images=3**, **climb=True**, **k=0.1**, **fmax=0.05**; idpp and linear tied on fmax. Re-check with `--primary-pair-count`. |
| Surface | `ts_search_graphene_results/Cu4_searches` (C18Cu4, pair 0–1) | 36/72 converged | **climb=False** critical (33/36 successes vs 3/36 with climb); **fmax=0.08** gave 12/12 converged when climb=False vs 9/12 at 0.05; best fmax row: **idpp**, **nimg=3**, **k=0.1**. |
| Surface (extended) | same dirs, 3 pairs, `--grid extended` | 14/24 surface cells | **`neb_align_endpoints=False`**: 10/12 converged vs 4/12 with True; barriers ~0.17 eV vs ~7 eV (Kabsch + MIC/slab mismatch). Gas: keep **align=True** (12/12 vs 8/12). |

Re-run sweeps after changing chemistry or MACE model; tighten surface **fmax** toward 0.05 only if most NEBs converge.

**Presets aligned with the runner:** `similarity_pair_cor_max` is **0.1 Å** (same as `run_transition_state_search`), not `DEFAULT_PAIR_COR_MAX`. Default `neb_spring_constant` on the runner and on `find_transition_state` is **0.1 eV/Å²** (MACE-tuned gas/surface TS presets).

**Further sweeps to consider** (see `python benchmark/neb_sweep_mace.py --list-followup-sweeps`): several `pair-index` values per campaign; pairing thresholds; `aseneb` vs `improvedtangent`; `align_endpoints` / small `perturb_sigma`; MACE model and float32 vs float64; NEB step budget vs `auto_niter_ts`; optional frequency validation after NEB.

Run GPU jobs **one at a time** on a single-GPU machine.

## Dependency pins (PyPI)

As of the last refresh: `mace-torch==0.3.15`, `torch-sim-atomistic[mace]==0.5.2`, `e3nn==0.4.4` (see `pyproject.toml`).

## Pytest warning filters

`pytest.ini` filters unavoidable third-party deprecations (e.g. `torch.jit.*` while MACE/torch_sim still use those paths). Remaining warnings from MACE/e3nn/torch are often environmental (e.g. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`).
