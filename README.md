# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A compact toolkit for global optimization of small atomic clusters using ASE. SCGO provides simple scripts and a small API for running Basin Hopping (BH) and Genetic Algorithm (GA) searches, with sensible defaults and helpers for common workflows.

## Install (minimal)

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

## Quick start (3 lines) 💡

```python
from scgo import run_scgo_trials
from scgo.param_presets import get_testing_params
results = run_scgo_trials(["Pt"]*4, params=get_testing_params(), seed=42)
```

- `results` is a list of `(energy, Atoms)` for unique minima (sorted by energy by default).
- The public helpers include `run_scgo_campaign_one_element(...)` and `run_scgo_campaign_two_elements(...)` for campaigns.

---

## What to expect on disk (output) 📂

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

`run_transition_state_search(...)` reads minima from `{formula}_searches/` (or the `base_dir` you pass) and writes **under the same tree** into a dedicated folder:

- `{formula}_searches/ts_results_{formula}/`
  - **Per pair** `pair_id` (e.g. `0_1`): `ts_{pair_id}.xyz`, `reactant_{pair_id}.xyz`, `product_{pair_id}.xyz` (when geometries exist), and `neb_{pair_id}_metadata.json`
  - **`ts_search_summary_{formula}.json`** — full run: provenance header, NEB settings (`calculator_name`, `neb_fmax`, `neb_steps_resolved`, `neb_backend` `torchsim` or `ase`, `use_parallel_neb`, climb/interpolation flags, image count, spring constant, etc.), `composition`, `formula`, `num_total_pairs`, `num_successful`, `num_converged`, `results` (list of per-pair records), and `statistics` (`total_ts_found`, `converged_ts`, `successful_ts`, `min_barrier` / `max_barrier` / `avg_barrier` over successes)
  - **`ts_network_metadata_{formula}.json`** — graph-oriented view: `ts_connections[]` (each edge: `pair_id`, `minima_indices`, energies, `barrier_height`, optional `barrier_forward` / `barrier_reverse`, `neb_converged`, `n_images`, optional `minima_provenance`), `num_minima`, `statistics`, optional `minima_base_dir`
  - **`final_unique_ts/`** — after deduplication: `final_unique_ts_summary_{formula}.json` (provenance + `unique_ts[]` with `connected_edges`, `connected_minima`, `filename`, energies, etc.) and one `.xyz` per deduplicated TS (names may include `pair_…` when a single edge maps to that file)

**Per-pair entries** in `ts_search_summary_*.json` (and overlapping fields in `neb_*_metadata.json`) typically include: `pair_id`, `status` (`success` / `failed`), `neb_converged`, `n_images`, `spring_constant`, `reactant_energy`, `product_energy`, `ts_energy`, `barrier_height`, `error`, and on success `ts_image_index`. When traceability is available, `minima_indices` and **`minima_provenance`** appear: each endpoint lists `run_id`, `trial`, `source_db`, `source_db_relpath`, `systems_row_id`, `confid`, `gaid`, `unique_id`, `final_id`, `energy` (see `scgo/ts_search/transition_state_io.py`).

**`neb_{pair_id}_metadata.json`** merges the provenance header with pair fields above plus, when present: `final_fmax`, `steps_taken`, `force_calls`, and NEB-parameter echoes (`use_torchsim`, `neb_backend`, `interpolation_method`, `climb`, `align_endpoints`, `perturb_sigma`, `neb_interpolation_mic`, `fmax`, `neb_steps`, etc.).

---

## Key options (short)

- `params` come from presets: `get_default_params()`, `get_testing_params()`, `get_high_energy_params()`, `get_diversity_params()`.

Tagging final minima in databases: ✅ After writing final XYZ files, SCGO can optionally tag the corresponding database records with metadata ("final_unique_minimum": true, "final_rank", and "final_written") so downstream tools can find final minima without re-scanning databases. This behaviour is enabled by default and can be disabled by setting `params['tag_final_minima'] = False` when calling `run_scgo_trials(...)`.
- `fitness_strategy`: `low_energy` (default), `high_energy`, `diversity`.
- `validate_with_hessian` (bool): run force + Hessian checks (uses ASE vibrational analysis).
- **GA backend**: MLIPs use TorchSim batched GA; classical calculators use ASE GA.

Important: `diversity` requires a `diversity_reference_db` glob (e.g. `"Pt*_searches/**/*.db"`).

---

## Adsorbate on a surface (supported-cluster GA)

SCGO can run **genetic-algorithm** global optimization for a small **adsorbate cluster** on a periodic **slab**. The GA explores the adsorbate degrees of freedom (`composition`); the slab supplies the cell and controls which substrate atoms move during **local** relaxations via [`SurfaceSystemConfig`](scgo/surface/config.py) (`FixAtoms` under the hood, including on the TorchSim GA path).

### How to run

Build (or load) any ASE `Atoms` slab and pass it through the generic surface helper:

```python
from ase.build import fcc111
from scgo.runner_surface import make_surface_config

slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
surface_config = make_surface_config(slab)
```

Then wire `surface_config` into the GA parameters:

```python
from scgo.param_presets import get_minimal_ga_params

params = get_minimal_ga_params(seed=42)
params["optimizer_params"]["ga"]["surface_config"] = surface_config
```

- **Direct API** (any adsorbate size): `from scgo import ga_go, SurfaceSystemConfig` and pass `surface_config=...`.
- **`run_scgo_trials`**: set `params["optimizer_params"]["ga"]["surface_config"]` to a `SurfaceSystemConfig` instance. The high-level runner only selects GA when `len(composition) >= 4`, so use **at least four adsorbate atoms** if you rely on automatic algorithm choice; for dimers/trimers, call `ga_go` directly.
- **Example runner scripts** in `runners/` show the full workflow (minima search + TS search) on graphene slabs. Replace the inline `_make_slab()` helper with any other slab to run on a different surface.

Surface workflows use `from scgo.runner_surface import make_surface_config` with your ASE slab.

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
pytest tests/ -m "not slow"
```

---

## High-Level API 📚

### Global Optimization

#### `run_scgo_trials(composition, params=None, seed=None, ...)`

Run global optimization for a single composition.

```python
from scgo import run_scgo_trials
from scgo.param_presets import get_default_params

results = run_scgo_trials(
    composition=["Pt", "Pt", "Pt", "Pt"],
    params=get_default_params(),
    seed=42,
    verbosity=1,
    clean=False,
    output_dir=None
)
# Returns: list of (energy, Atoms) tuples for unique minima
```

**Parameters:**
- `composition`: List of atomic symbols, e.g., `["Pt", "Pt", "Au"]`
- `params`: Parameter dict from `get_default_params()` or `get_testing_params()` (default: `get_default_params()`)
- `seed`: Random seed for reproducibility (default: None = random)
- `verbosity`: 0=quiet, 1=normal, 2=debug, 3=trace (default: 1)
- `clean`: If True, ignore previous runs (default: False)
- `output_dir`: Base output directory (default: `{formula}_searches`)

**Algorithm Selection:**
- 1-2 atoms: Simple optimization
- 3 atoms: Basin Hopping
- 4+ atoms: Genetic Algorithm

#### `run_scgo_campaign_one_element(element, min_atoms, max_atoms, ...)`

Systematic search across cluster sizes for a single element.

```python
from scgo import run_scgo_campaign_one_element

results = run_scgo_campaign_one_element(
    element="Pt",
    min_atoms=3,
    max_atoms=10,
    params=get_default_params(),
    seed=42
)
# Returns: dict[formula, list[(energy, Atoms)]]
```

#### `run_scgo_campaign_two_elements(element1, element2, min_atoms, max_atoms, ...)`

Systematic search for bimetallic clusters across all compositions.

```python
from scgo import run_scgo_campaign_two_elements

results = run_scgo_campaign_two_elements(
    element1="Au",
    element2="Pt",
    min_atoms=3,
    max_atoms=6,
    params=get_default_params(),
    seed=42
)
# Returns: dict[formula, list[(energy, Atoms)]]
```

### Transition State Search

#### `run_transition_state_search(composition, base_dir=None, ...)`

Find transition states connecting minima from previous global optimization.

```python
from scgo import run_transition_state_search

# First run global optimization to get minima
results_go = run_scgo_trials(["Pt", "Pt", "Pt"], seed=42)

# Then find transition states (minima live under Pt3_searches/ by default)
ts_results = run_transition_state_search(
    composition=["Pt", "Pt", "Pt"],
    base_dir="Pt3_searches",  # Same root as GO output: run_*/trial_*/*.db
    params={"calculator": "MACE"},
    seed=42,
    max_pairs=10,
    neb_fmax=0.05,
    neb_steps=500,
    use_torchsim=True  # Optional: TorchSim batched NEB (CPU or GPU)
)
# Returns: list of dicts with TS info (barrier, TS structure, etc.)
# On disk: see "Transition state search (ts_results_{formula}/)" above.
```

**Key Parameters:**
- `composition`: Same composition as the minima you want to connect
- `base_dir`: Directory containing `run_*/` folders with previous optimization results (default: `{formula}_searches`). TS artifacts are written to `base_dir/ts_results_{formula}/`.
- `max_pairs`: Maximum number of structure pairs to evaluate
- `energy_gap_threshold`: Only pair structures with energy gap below threshold (eV)
- `neb_n_images`: Number of NEB images (default: 3)
- `neb_spring_constant`: Spring constant for NEB band (default: 0.1)
- `neb_perturb_sigma`: Interior-image perturbation (Å) applied after interpolation (default: 0.0)
- `neb_fmax`: Force convergence criterion (default: 0.05 eV/Å)
- `neb_steps`: Max optimization steps (default: 500)
- `use_torchsim` (bool): Use TorchSim for batched NEB (works on CPU or GPU). Default: False.
- `torchsim_params` (dict): Parameters for TorchSim relaxer (if use_torchsim=True).

**Returns:** List of dictionaries (one per pair). In-memory dicts mirror the JSON-friendly fields saved under `ts_results_{formula}/` (plus live `Atoms` objects where applicable):

- `"pair_id"`, `"status"` (`"success"` / `"failed"`), `"neb_converged"`, `"n_images"`, `"spring_constant"`
- `"reactant_energy"`, `"product_energy"`, `"ts_energy"`, `"barrier_height"`, `"error"`
- `"transition_state"`: TS `Atoms` on success (not serialized in summary JSON)
- `"reactant_structure"`, `"product_structure"`: endpoint `Atoms` when kept on the result
- `"minima_indices"`, `"minima_provenance"`: traceability back to GO databases when attached
- `"ts_image_index"`, `"final_fmax"`, `"steps_taken"`, `"force_calls"`: NEB diagnostics when present
- `"use_torchsim"`, `"neb_backend"`, interpolation/climb/MIC flags: echo NEB configuration

#### `run_transition_state_campaign(compositions, output_dir=None, ...)`

Run transition state search across multiple compositions in sequence.

```python
from scgo import run_transition_state_campaign

# Run TS search for multiple compositions
compositions = [["Pt", "Pt"], ["Pt", "Pt", "Pt"], ["Au", "Pt"]]

results = run_transition_state_campaign(
    compositions=compositions,
    output_dir="ts_searches",  # Optional: base dir for all compositions
    params={"calculator": "MACE"},
    seed=42,
    neb_params={
        "max_pairs": 10,
        "neb_n_images": 3,
        "neb_fmax": 0.05,
        "neb_steps": 500,
    }
)
# Returns: dict[formula, list[dict]] mapping each formula to its TS results
```

**Key Parameters:**
- `compositions`: List of compositions (each a list of atomic symbols)
- `output_dir`: Base directory for all searches (default: individual `{formula}_ts_searches` dirs)
- `params`: Global optimization parameters with "calculator" field
- `neb_params`: Dictionary with NEB-specific parameters (same as `run_transition_state_search`)

---

## Notes

- TorchSim is an optional tool that provides GPU-accelerated batched optimization when available; SCGO works with EMT (CPU) out of the box for quick tests.
- For reproducible results, pass `seed=` to `run_scgo_trials` or campaign helpers.
- See `runners/` directory for complete working examples and the `tests/` directory for usage patterns.

---

MIT License — see `LICENSE`.
