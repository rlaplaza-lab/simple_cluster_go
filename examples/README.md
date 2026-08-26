# SCGO Examples

Most scripts are `run_go_ts` smoke runs for the supported system types
(MACE + TorchSim). Each builds params from `get_low_effort_torchsim_ga_params` /
`get_low_effort_ts_search_params` for its `system_type`, overriding only
`max_pairs` (the NEB budget) plus a few example-specific knobs. Those presets
are reduced-budget (~25% of production) variants of `get_torchsim_ga_params` /
`get_ts_search_params`: calculator, TorchSim relaxer and NEB physics knobs are
inherited unchanged; only GA and NEB step budgets shrink (floored so bands still
converge). The Kaggle GPU matrix in
`tests/integration/test_gpu_examples_integration.py` builds its params from the
same two presets, so CI cannot drift from these scripts. For a full-strength
campaign, swap in the production presets.

Adsorbate examples set `connectivity_factor=1.8` and
`freeze_adsorbate_internal_geometry=True`; the bare surface example also uses
`connectivity_factor=1.8` for slab validation. Adsorbate examples often use
fewer `max_pairs` because IDPP screening is heavier and their bands run
two-stage climb over 7 images.

| Script | System type | `max_pairs` | `neb_steps` | TS preset highlights |
|--------|-------------|-------------|-------------|----------------------|
| `example_pt5_gas.py` | `gas_cluster` | 6 | 1000 | no climb, shared `neb_fmax=0.20`, 5 images, parallel NEB |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` | 6 | 1000 | climb, shared `neb_fmax=0.20`, 7 images, parallel NEB, `max_endpoint_mismatch=1.25` Å |
| `example_pt5_graphite.py` | `surface_cluster` | 6 | 1000 | no climb, shared `neb_fmax=0.20`, MIC + lattice rotation, parallel NEB |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` | 4 | 1000 | climb, shared `neb_fmax=0.20`, parallel NEB, no lattice rotation, `max_endpoint_mismatch=1.5` Å |
| `example_defected_graphite.py` | `surface` | 4 | 1000 | top-layer slab search on vacancy-defected graphite |
| `example_n_doped_graphite.py` | `surface_adsorbate` | 4 | 1000 | top-layer + OH on N-doped graphite |

GO-only:

| Script | System type | Notes |
|--------|-------------|-------|
| `example_pt5_orr_defected_graphite.py` | `surface_cluster` then `surface_cluster_adsorbate` | Four `run_go` searches: bare Pt5, then Pt5+O, Pt5+OH, Pt5+OOH on monovacancy graphite |

All graphite scripts build slabs via the HOPG 5×5 × 3-layer preset helpers
(`make_hopg_5x5_graphite_surface_config`,
`make_hopg_5x5_defected_graphite_surface_config`, or
`make_n_doped_graphite_surface_config` for N-doping). Slab geometry is not
hard-coded in the examples; only defect/dopant counts differ where noted
(`n_dopants=2` in the N-doped example).

```bash
pip install -e ".[mace]"
python examples/example_pt5_gas.py
python examples/example_pt5_oh_gas.py
python examples/example_pt5_graphite.py
python examples/example_pt5_2oh_graphite.py
python examples/example_defected_graphite.py
python examples/example_n_doped_graphite.py
python examples/example_pt5_orr_defected_graphite.py
```

Each run creates a new datetime `run_*` under `examples/results/{stem}_mace/`
(`{path_key}_searches/` and, for GO+TS, `{path_key}_ts_results/`; timing JSON
enabled). Path keys are component-aware, for example `Pt5`, `Pt5_OH`,
`Pt5_graphite`, `Pt5_OH_OH_graphite`, `defected_graphite`,
`OH_n_doped_graphite`. Reusing the same `output_stem` can seed GO from prior
DBs in that tree; use a fresh stem (or delete the old tree) for a clean
end-to-end check. Override the stem without editing the script via
`SCGO_EXAMPLE_OUTPUT_STEM=my_fresh_stem`. See
[`docs/source/output_layout.rst`](../docs/source/output_layout.rst)
(*On-disk layout*).
