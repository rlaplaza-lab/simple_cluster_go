<div align="center">
  <img src="docs/_static/scgo_logo.svg" alt="SCGO Logo" width="300">
</div>

# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A compact toolkit for global optimization of small atomic clusters using ASE. SCGO provides a focused API for Basin Hopping (BH) and Genetic Algorithm (GA) workflows with practical defaults.

**Documentation**: Comprehensive API documentation is available in the `docs/` directory. For online documentation, see [ReadTheDocs](https://scgo.readthedocs.io/).

## Install

SCGO has a small core dependency set plus two mutually exclusive MLIP extras:

- `[mace]` for MACE + TorchSim + `nvalchemi-toolkit-ops`
- `[uma]` for `fairchem-core` UMA checkpoints

Install only one of `[mace]` or `[uma]` per environment.

Conda (recommended):

```bash
git clone https://github.com/rlaplaza-lab/simple_cluster_go.git
cd simple_cluster_go
conda env create -f environment.yml
conda activate scgo
```

`environment.yml` installs the package editable with **`[mace,dev]`** (MACE/TorchSim + test/lint tooling). For a runtime-only editable install, use `pip install -e .` instead.

The conda env uses `torch-sim-atomistic[mace]` with `nvalchemi-toolkit-ops` for TorchSim neighbor lists. Do not install `vesin` or `vesin-torch`—they conflict with the TorchSim stack we use.

Note: SCGO requires SQLite with the JSON1 extension (for `json_extract` and related functions). If you installed using conda, ensure `sqlite` from `conda-forge` is available in your environment (e.g., `conda install -c conda-forge sqlite`). If you use pip-only installs, consider installing `pysqlite3-binary` (e.g., `pip install pysqlite3-binary`) so that the Python `sqlite3` module exposes JSON1. This repository's CI enforces JSON1 availability.

Sella is not required by the core SCGO package and has been removed from the default pip constraints to avoid heavy native builds during dependency resolution. If you need Sella for advanced optimization features, install it manually (it builds C extensions and may require a C toolchain and Cython).

pip (alternative):

```bash
git clone https://github.com/rlaplaza-lab/simple_cluster_go.git
cd simple_cluster_go
pip install -e ".[mace]"   # or: pip install -e ".[uma]"
```

For pip installs, the same TorchSim stack applies: ensure `nvalchemi-toolkit-ops` is available; uninstall `vesin` and `vesin-torch` if you see TorchSim-related errors.

Dependency note: SCGO now allows `scipy>=1.14,<3` so the UMA/fairchem extra can resolve cleanly with `torch-sim-atomistic[fairchem]` (which constrains SciPy to `<1.17` in current releases).

For development with tests and linting (after a **runtime-only** `pip install -e .`):

```bash
pip install -e ".[mace,dev]"   # or UMA: pip install -e ".[uma,dev]"
pre-commit install
```

### Running on HPC (Slurm, shared filesystem)

- **SQLite**: SCGO keeps WAL mode off by default (fewer `-wal`/`-shm` issues on Lustre/GPFS/NFS). Prefer writing active `*.db` files under job-local scratch (`$SLURM_TMPDIR` or site-specific scratch) when you can, then copying results back to project storage.
- **Registry**: Discovery may write `.scgo_db_registry.json` and `.scgo_db_registry.lock` (with `flock` on Linux) for fast DB listing. When your run lives under a directory whose name ends in `_searches`, the index is kept at that parent only (not beside every `trial_*` folder). If your filesystem does not honor `flock`, use separate output directories per job or avoid parallel registry updates.
- **Logging**: Batch-friendly defaults suppress noisy third-party loggers. For local debugging, set `SCGO_LOCAL_DEV=1` or call `configure_logging(..., hpc_mode=False)`.

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Installation**: `docs/source/installation.rst` - Setup instructions for conda and pip
- **Quick Start**: `docs/source/quickstart.rst` - Basic usage examples
- **API Reference**: 
  - `docs/source/api/runner_api.rst` - High-level API entry points
  - `docs/source/api/param_presets.rst` - Parameter presets
  - `docs/source/api/system_types.rst` - System type definitions
- **Advanced Topics**: `docs/source/advanced/` - Adsorbates, surface systems, customization

To build the documentation:

```bash
cd docs
pip install -r requirements.txt
make html
```

The built documentation will be available in `docs/build/html/index.html`.

---

## Quick start

```python
from scgo import run_go
from scgo.param_presets import get_testing_params
results = run_go(
    ["Pt"] * 4,
    params=get_testing_params(),
    seed=42,
    system_type="gas_cluster",
)
```

- `results` is a list of `(energy, Atoms)` for unique minima (sorted by energy by default).
- Sequential multi-composition GO uses `run_go_campaign([...], system_type=...)` from [`scgo.runner_api`](scgo/runner_api.py) (also re-exported from `scgo`).

### Explicit system types

SCGO supports exactly four explicit `system_type` values:

- `gas_cluster`: gas-phase cluster (no slab, no extra adsorbate constraints)
- `surface_cluster`: cluster supported on a slab (`surface_config` required)
- `gas_cluster_adsorbate`: gas-phase cluster that includes adsorbate-like species (no slab)
- `surface_cluster_adsorbate`: supported cluster + adsorbate species (`surface_config` required)

`system_type` must be passed to each `run_*` API call. System-definition keys are intentionally rejected from preset dicts (`go_params` / `ts_params`) to keep one canonical source of truth at the API boundary.
For adsorbate system types (`gas_cluster_adsorbate`, `surface_cluster_adsorbate`),
high-level runners require core-only `composition` and `adsorbates` (one ASE `Atoms`
fragment or a list of fragments). SCGO flattens adsorbate symbols in provided fragment
order and constructs the full mobile composition as
`core_composition + flattened_adsorbate_symbols` (mobile region after any slab).
Hierarchical initialization is the only supported adsorbate layout.

---

## What to expect on disk (output)

When you run a search for composition `Pt4`, SCGO writes into `Pt4_searches/` with the following structure:

### Global optimization (`{formula}_searches/`)

- `Pt4_searches/run_<YYYYMMDD_HHMMSS_ffffff>/trial_<N>/`
  - `bh_go.db` or `ga_go.db` (ASE database with candidates and relaxed structures)
  - `population.log` (GA runs)
- `Pt4_searches/results_summary.json` — campaign-level snapshot after the latest run. Top-level keys include:
  - **Provenance** (same convention as other SCGO JSON sidecars): `schema_version` (currently **3**), `scgo_version`, `created_at` (UTC ISO8601), `python_version`
  - **Run summary**: `composition` (formula string, e.g. `"Pt4"`), `total_unique_minima`, `minima_by_run` (map of `run_id` → count), `current_run_id`, `params` (JSON-safe snapshot aligned with `run_*/metadata.json`), `run_metadata_relpath` (e.g. `run_<id>/metadata.json`)
- `Pt4_searches/final_unique_minima/` — final XYZ files, named like `Pt4_minimum_01_run_YYYYMMDD_HHMMSS_ffffff_trial_1.xyz`
- `Pt4_searches/run_<...>/metadata.json` — per-run record: provenance header above plus `run_id`, `timestamp`, `composition` (symbol list), `formula`, `params`, and related run fields
- `Pt4_searches/validation/` — optional; created when `validate_with_hessian=True` to run vibrational checks
- `Pt4_searches/.scgo_db_registry.json` and `.scgo_db_registry.lock` — optional DB index and lock (see *Running on HPC* above)

Notes:
- If `clean=False`, SCGO will merge previous runs by scanning `run_*` directories and `trial_*/` DB files.
- `.db` files are ignored by the project `.gitignore`.

### Transition state search (`ts_results_{formula}/`)

`run_ts_search` (from `scgo`, wrapping [`scgo.runner_api`](scgo/runner_api.py)) reads minima from `{formula}_searches/` (or the `output_dir` you pass) and writes **under the same tree** into a dedicated folder:

- `{formula}_searches/ts_results_{formula}/`
  - **Per pair** `pair_id` (e.g. `0_1`): `ts_{pair_id}.xyz`, `reactant_{pair_id}.xyz`, `product_{pair_id}.xyz` (when geometries exist), and `neb_{pair_id}_metadata.json`
  - **`ts_search_summary_{formula}.json`** — full run: provenance header, NEB settings (`calculator_name`, `neb_fmax`, `neb_steps_resolved`, `neb_backend` `torchsim` or `ase`, `use_parallel_neb`, climb/interpolation flags, image count, spring constant, etc.), `composition`, `formula`, `num_total_pairs`, `num_successful`, `num_converged`, `results` (list of per-pair records), and `statistics` (`total_ts_found`, `converged_ts`, `successful_ts`, `min_barrier` / `max_barrier` / `avg_barrier` over successes)
  - **`ts_network_metadata_{formula}.json`** — graph-oriented view: `ts_connections[]` (each edge: `pair_id`, `minima_indices`, energies, `barrier_height`, optional `barrier_forward` / `barrier_reverse`, `neb_converged`, `n_images`, optional `minima_provenance`), `num_minima`, `statistics`, optional `minima_base_dir`
  - **`final_unique_ts/`** — after deduplication: `final_unique_ts_summary_{formula}.json` (provenance + `unique_ts[]` with `connected_edges`, `connected_minima`, `filename`, energies, etc.) and one `.xyz` per deduplicated TS (names may include `pair_…` when a single edge maps to that file)

**Per-pair entries** in `ts_search_summary_*.json` (and overlapping fields in `neb_*_metadata.json`) typically include: `pair_id`, `status` (`success` / `failed`), `neb_converged`, `n_images`, `spring_constant`, `reactant_energy`, `product_energy`, `ts_energy`, `barrier_height`, `error`, and on success `ts_image_index`. When traceability is available, `minima_indices` and **`minima_provenance`** appear: each endpoint lists `run_id`, `trial_id`, `source_db`, `source_db_relpath`, `systems_row_id`, `confid`, `gaid`, `unique_id`, `final_id`, `energy` (see `scgo/ts_search/transition_state_io.py`).

**`neb_{pair_id}_metadata.json`** merges the provenance header with pair fields above plus, when present: `final_fmax`, `steps_taken`, `force_calls`, and NEB-parameter echoes (`use_torchsim`, `neb_backend`, `interpolation_method`, `climb`, `align_endpoints`, `perturb_sigma`, `neb_interpolation_mic`, `fmax`, `neb_steps`, etc.).

---

## Key options (short)

- **Global optimization (`params` for `run_go` / `run_go_campaign`)** is merged with `get_default_params()` via `initialize_params`: any preset that omits keys inherits defaults. Common entry points: `get_default_params()`, `get_minimal_ga_params()`, `get_testing_params()`, `get_high_energy_params()`, `get_diversity_params()`, `get_default_uma_params()` (fairchem UMA), and `get_torchsim_ga_params(seed, model_name=...)` (MACE + TorchSim GA benchmark stack; requires `scgo[mace]`).
- **Transition-state search (`ts_params` for `run_ts_search` / `run_go_ts`)** is **not** merged with GO defaults. Build a flat dict with `get_ts_search_params(...)` or `get_ts_search_params_uma(...)` and pass it explicitly alongside `go_params` when using `run_go_ts` / `run_go_ts_campaign`.

Preset-vs-runtime split in `runner_api`:

- Put scientific/tuning knobs in preset dicts (`go_params`/`ts_params`): calculator choice, optimizer settings, NEB settings, pairing thresholds, etc.
- Keep run-control knobs on the `run_*` call itself: `verbosity`, `output_dir`, `output_root`, `output_stem`, `seed`, `log_summary`, `write_timing_json`, `profile_ga`.
- Keep system-definition inputs on the `run_*` call itself: `system_type`, and when required by system type, `surface_config`, core-only `composition`, and `adsorbates`.

Inspect -> edit -> run pattern:

```python
from scgo import run_go_ts
from scgo.param_presets import get_default_params, get_ts_search_params

go_params = get_default_params()
ts_params = get_ts_search_params(system_type="gas_cluster")

print(go_params["optimizer_params"]["ga"]["niter"])
go_params["optimizer_params"]["ga"]["niter"] = 8
ts_params["max_pairs"] = 12

summary = run_go_ts(
    "Pt5",
    go_params=go_params,
    ts_params=ts_params,
    system_type="gas_cluster",
    seed=7,
    verbosity=1,
)
```

Tagging final minima in databases: ✅ After writing final XYZ files, SCGO can optionally tag the corresponding database records with metadata ("final_unique_minimum": true, "final_rank", and "final_written") so downstream tools can find final minima without re-scanning databases. This behaviour is enabled by default and can be disabled by setting `params['tag_final_minima'] = False` when calling `run_go(...)`.
- `fitness_strategy`: `low_energy` (default), `high_energy`, `diversity`.
- `validate_with_hessian` (bool): run force + Hessian checks (uses ASE vibrational analysis).
- **GA backend**: MLIPs use TorchSim batched GA; classical calculators use ASE GA.

Important: `diversity` requires a `diversity_reference_db` glob (e.g. `"Pt*_searches/**/*.db"`).

---

## Surface workflows (supported clusters)

SCGO can run **genetic-algorithm** global optimization for a small **adsorbate cluster** on a periodic **slab**. The GA explores the adsorbate degrees of freedom (`composition`); the slab supplies the cell and controls which substrate atoms move during **local** relaxations via [`SurfaceSystemConfig`](scgo/surface/config.py) (`FixAtoms` under the hood, including on the TorchSim GA path).

### How to run

Build (or load) any ASE `Atoms` slab and pass it through the generic surface helper:

```python
from ase.build import fcc111
from scgo.runner_surface import make_surface_config

slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
surface_config = make_surface_config(slab)
```

For the **graphite preset** used in example runners, use [`scgo.surface.make_graphite_surface_config`](scgo/surface/presets.py) (or `from scgo import make_graphite_surface_config`) instead of building a slab by hand.

Then wire `surface_config` into the GA parameters:

```python
from scgo.param_presets import get_minimal_ga_params

params = get_minimal_ga_params(seed=42)
params["optimizer_params"]["ga"]["surface_config"] = surface_config
```

- **Direct API** (any adsorbate size): `from scgo import ga_go, SurfaceSystemConfig` and pass `surface_config=...`.
- **`run_go`**: pass `surface_config=...` directly to `run_go(...)`; it is copied into each **present** `optimizer_params` entry among `simple` / `bh` / `ga` so the active algorithm sees the slab. The high-level runner only selects GA when `len(composition) >= 4`, so use **at least four adsorbate atoms** if you rely on automatic algorithm choice; for dimers/trimers, call `ga_go` directly.
- For slab workflows, choose `system_type="surface_cluster"` (supported cluster only) or `system_type="surface_cluster_adsorbate"` (supported cluster with explicit adsorbate-mode policies). Use `scgo.runner_surface.make_surface_config` for a custom ASE slab; use `scgo.surface.make_graphite_surface_config` for the bundled graphite template.

**Adsorbate inputs and initial structures:** For both `gas_cluster_adsorbate` and `surface_cluster_adsorbate`, pass core-only `composition` plus `adsorbates` (one `Atoms` or list of `Atoms`). SCGO derives a strict mobile partition in order (`core_symbols == composition`, then flattened adsorbate symbols); slab atoms are not part of `composition`. SCGO uses hierarchical initialization only: build the core, place the rigid fragment(s), then (for surface) deposit the combined cluster on the slab. Optional: `run_go(..., cluster_adsorbate_config=ClusterAdsorbateConfig(...))` for fragment height and validation. Use `scgo.surface.describe_surface_config` to log effective slab and height settings. GA and basin-hopping attach `n_core_atoms` and per-role symbol JSON in metadata for round-trip checks. When adsorbate metadata is present, [`validate_structure_for_system_type`](scgo/system_types.py) also asserts that the mobile region’s chemical symbols match `core_symbols + adsorbate_symbols` in order (in addition to geometry checks). The helper [`validate_mobile_symbols_match_adsorbate_definition`](scgo/system_types.py) is available for the same symbol order check alone.

### Slab motion during local relaxation

| Mode | Settings |
|------|----------|
| Entire slab frozen | `fix_all_slab_atoms=True` (default) |
| Relax only the top **N** slab layers (along `surface_normal_axis`) | `fix_all_slab_atoms=False`, `n_relax_top_slab_layers=N` |
| Same intent, using bottom layer count | `fix_all_slab_atoms=False`, `n_fix_bottom_slab_layers=L - N` where `L` is the number of distinct slab layers along that axis |
| Slab fully free to relax | `fix_all_slab_atoms=False` and leave `n_relax_top_slab_layers` and `n_fix_bottom_slab_layers` unset (`None`) |

Do not set `n_relax_top_slab_layers` together with `n_fix_bottom_slab_layers`, or together with `fix_all_slab_atoms=True`. For typical `ase.build.fcc111` slabs with vacuum along **z**, use `surface_normal_axis=2` (the default).

Run metadata records a JSON-safe summary of these flags (no embedded `Atoms`) under the sanitized `surface_config` key.

---

## Testing

```bash
# Fast default
pytest tests/ -m "not slow"

# Integration-only
pytest tests/ -m integration

# Slow-only
pytest tests/ -m slow
```

Fast EMT “benchmark” smoke tests (initialization and dimers) live under [`tests/benchmarks/`](tests/benchmarks/); long MLIP regression sweeps live under the top-level [`benchmark/`](benchmark/) package (see [`benchmark/README.md`](benchmark/README.md)).

For long GA/TorchSim tests, run in foreground with live output (`-s`) and an explicit timeout to avoid “looks stalled” sessions:

```bash
timeout 5400 pytest tests/ -m "not slow" -vv -s
```

---

## High-Level API

Canonical workflow entry points are defined in [`scgo/runner_api.py`](scgo/runner_api.py) and imported from the `scgo` package. Composition arguments may be a **formula string** (`"Pt3Au"`), a **symbol list**, or **`ase.Atoms`** (only symbols are used for GO).

### Global optimization

#### `run_go(composition, params=None, seed=None, ...)`

Single composition; returns a list of `(energy, Atoms)` unique minima.

```python
from scgo import run_go
from scgo.param_presets import get_default_params

results = run_go(
    ["Pt", "Pt", "Pt", "Pt"],
    params=get_default_params(),
    seed=42,
    verbosity=1,
    clean=False,
    output_dir=None,
    system_type="gas_cluster",
)
```

**Algorithm selection** (unchanged): 1–2 atoms → simple; 3 → basin hopping; 4+ → genetic algorithm.

#### `run_go_campaign(compositions, ..., system_type=...)`

Run GO for each composition **sequentially**; returns `dict[formula, list[(energy, Atoms)]]`.

For element or binary scans, build compositions explicitly and pass them to `run_go_campaign`.

### Transition state search

`run_ts_search` and `run_ts_campaign` take a **flat `ts_params` dict** from [`get_ts_search_params`](scgo/param_presets.py) (or edit a copy). TorchSim use is resolved from the calculator; pass `system_type` on the `run_*` call.

```python
from scgo import run_ts_search
from scgo.param_presets import get_ts_search_params

ts_params = get_ts_search_params(system_type="gas_cluster")
ts_results = run_ts_search(
    ["Pt", "Pt", "Pt"],
    output_dir="Pt3_searches",
    params={"calculator": "MACE"},
    seed=42,
    ts_params=ts_params,
    system_type="gas_cluster",
)
```

`run_ts_campaign` forwards the same `ts_params` to each composition.

### GO then TS

`run_go_ts` / `run_go_ts_campaign` use **`go_params=`** (merged like other GO runs) and **`ts_params=`** (same flat shape as above; **not** deep-merged with `get_default_params()`). For **slab + adsorbate**, pass a `SurfaceSystemConfig` directly to `run_go_ts(..., surface_config=...)` / `run_go_ts_campaign(..., surface_config=...)`; the `composition` argument is **adsorbate symbols only** (the full system for loading minima is built as slab + adsorbate, matching GA). For MACE + TorchSim GA, start from [`get_torchsim_ga_params`](scgo/param_presets.py) with a `seed` (optional `model_name=` so the TorchSim relaxer matches the calculator), set `go_params["calculator"] = "MACE"` and `optimizer_params["ga"]` as needed; pair with `get_ts_search_params(...)` and set `ts_params["max_pairs"]`, etc. For UMA NEB defaults, you can use `get_ts_search_params_uma`. See `runners/example_pt5_gas.py` for a minimal end-to-end example. Default output if `output_dir` is omitted is under `scgo_runs/<stem>_<mace|uma>/` (set `output_root` / `output_stem` to change).

Benchmarks comparing MACE vs UMA on the same GA structure can use [`get_uma_ga_benchmark_params`](scgo/param_presets.py) (re-exported from `scgo`).

### Advanced / internals

- `from scgo.run_minima import run_scgo_trials`, `run_scgo_campaign_arbitrary_compositions`, …
- `from scgo.ts_search.transition_state_run import run_transition_state_search` for a flat keyword API without the `ts_params` dict.

---

## Notes

- TorchSim is an optional tool that provides GPU-accelerated batched optimization when available; SCGO works with EMT (CPU) out of the box for quick tests.
- For reproducible results, pass `seed=` to the workflow functions above.
- Optional scripts in [`runners/`](runners/) are minimal, no-CLI examples that call [`run_go_ts`](scgo/runner_api.py). Each is tuned for MACE + TorchSim (edit calculator in the script if needed):

| Script | `system_type` | Notes |
|--------|----------------|-------|
| [`runners/example_pt5_gas.py`](runners/example_pt5_gas.py) | `gas_cluster` | Gas-phase `Pt5` only |
| [`runners/example_pt5_graphite.py`](runners/example_pt5_graphite.py) | `surface_cluster` | `Pt5` on preset graphite |
| [`runners/example_pt5_oh_gas.py`](runners/example_pt5_oh_gas.py) | `gas_cluster_adsorbate` | core-only `Pt5` composition + one `adsorbates` OH fragment |
| [`runners/example_pt5_2oh_graphite.py`](runners/example_pt5_2oh_graphite.py) | `surface_cluster_adsorbate` | core-only `Pt5` composition + two `adsorbates` OH fragments |

  For multi-size MLIP sweeps, see [`benchmark/`](benchmark/) (e.g. [`benchmark_Pt.py`](benchmark/benchmark_Pt.py), [`benchmark_Pt_surface_graphite.py`](benchmark/benchmark_Pt_surface_graphite.py)), not `tests/benchmarks/`.
- See `tests/` for concrete usage patterns.

---

MIT License — see `LICENSE`.
