Quick Start
===========

SCGO finds low-energy atomic structures with global optimization. This guide
shows how to use each supported workflow.

A **cluster** (also called the **core** when molecules are present) is the metal
nanoparticle whose shape you are searching. An **adsorbate** is a small molecule
or fragment (for example OH or CO) attached to the cluster or the surface. A
**slab** is a periodic surface the cluster or adsorbate sits on. You can also
search the top layers of the slab itself, with the bottom layers held still.

System Types
------------

You must specify one of six system types:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Type
     - What you search
   * - ``gas_cluster``
     - Cluster in vacuum
   * - ``surface_cluster``
     - Cluster on a slab
   * - ``gas_cluster_adsorbate``
     - Cluster plus adsorbates in vacuum
   * - ``surface_cluster_adsorbate``
     - Cluster plus adsorbates on a slab
   * - ``surface``
     - Top layers of a bare slab
   * - ``surface_adsorbate``
     - Top slab layers plus adsorbates, no cluster

For surface types, pass a ``surface_config``. For adsorbate types, pass
``adsorbates``. When the slab itself is the search target
(``surface`` / ``surface_adsorbate``), also set ``fix_all_slab_atoms=False``
and choose ``n_relax_top_slab_layers`` or ``n_fix_bottom_slab_layers``.

Gas Cluster
-----------

Optimize a simple cluster in vacuum.

**Fast test with EMT:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       ["Pt"] * 4,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

   for energy, atoms in results:
       print(f"Energy: {energy:.4f} eV, Formula: {atoms.get_chemical_formula()}")

For multi-element clusters, atom order follows the composition list you pass in
so genetic-algorithm crossover can pair structures safely. Heavier elements are
favoured during initialization while the population stays diverse. Details:
:doc:`/api/initialization`.

**Production run with MACE:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["calculator_kwargs"]["model_name"] = "mace_mp_small"

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster",
   )

On a Surface
------------

Use the built-in graphite surface or create your own slab.

**Using graphite preset:**

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Use :func:`~scgo.adsorption_energy` to compare adsorption energies on a slab.

**Using a custom slab:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go, make_surface_config
   from scgo.param_presets import get_default_params

   slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
   surface_config = make_surface_config(slab)

   results = run_go(
       "Pt4",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Defining Custom Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SurfaceSystemConfig`` or the simpler ``make_surface_config()`` helper.

**Slab motion options:**

- ``fix_all_slab_atoms=True``: entire slab stays frozen (default)
- ``n_relax_top_slab_layers=2``: allow top 2 layers to relax
- ``n_fix_bottom_slab_layers=1``: freeze bottom layer only
- Both layer counts = ``None``: entire slab can relax

Do not use ``n_relax_top_slab_layers`` together with ``n_fix_bottom_slab_layers``.

**Full example:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.surface import SurfaceSystemConfig

   slab = fcc111("Fe", size=(4, 4, 3), vacuum=12.0)

   surface_config = SurfaceSystemConfig(
       slab=slab,
       adsorption_height_min=1.2,
       adsorption_height_max=2.5,
       fix_all_slab_atoms=False,
       n_relax_top_slab_layers=2,
   )

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

With Adsorbates
---------------

Add adsorbate molecules (OH, CO, and so on) to your cluster.

**Gas phase with adsorbate:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       ["Pt"] * 5,
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

**Multiple adsorbates:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh1 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
   oh2 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=[oh1, oh2],
   )

Defining Custom Adsorbates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Any ASE ``Atoms`` object is a valid adsorbate fragment. The genetic algorithm
will:

- Place fragments on cluster surface sites
- Keep fragments rigid (bonds stay intact)
- Optionally reposition fragments during optimization

**Tuning placement:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.cluster_adsorbate import ClusterAdsorbateConfig

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   params = get_default_params()
   params["cluster_adsorbate_config"] = ClusterAdsorbateConfig(
       height_min=0.9,
       height_max=2.2,
       max_placement_attempts=200,
   )
   params["freeze_adsorbate_internal_geometry"] = True  # Keep fragment rigid

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

Use :func:`~scgo.is_true_minimum` or :func:`~scgo.perform_local_relaxation` to
check or re-relax candidates outside a full GO run.

Surface + Adsorbates
--------------------

Combine surface and adsorbates.

.. code-block:: python

   from ase import Atoms
   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)
   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster_adsorbate",
       adsorbates=oh,
   )

Slab as search target
---------------------

Use ``surface`` / ``surface_adsorbate`` when the search should move the **top
slab layers** rather than a deposited nanoparticle. Pass an empty composition
``[]`` for the nanoparticle core. Presets such as
:func:`~scgo.make_defected_graphite_surface_config` and
:func:`~scgo.make_n_doped_graphite_surface_config` set
``fix_all_slab_atoms=False``, ``n_relax_top_slab_layers=1``, and a filesystem
``name`` for path keys.

**Bare defected graphite:**

.. code-block:: python

   from scgo import run_go, make_defected_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_defected_graphite_surface_config(
       n_vacancies=1, seed=42
   )

   results = run_go(
       [],
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface",
   )

**OH on N-doped graphite:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go, make_n_doped_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_n_doped_graphite_surface_config(
       n_dopants=2, seed=42
   )
   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])

   results = run_go(
       [],
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_adsorbate",
       adsorbates=oh,
   )

Transition States
-----------------

Find transition states between optimized structures. SCGO builds a short list
of minima pairs, ranks them, and runs NEB only on the top candidates. The
budget is ``max_pairs``.

How candidate pairs are chosen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SCGO does not run NEB on every possible combination. It builds and ranks
candidates before spending any NEB force evaluations:

1. **Load and dedupe minima.** GO minima are loaded and filtered with the
   uniqueness rule (see :doc:`/uniqueness`). For TS, the distance gate is
   tighter (``0.1`` Å instead of ``0.7`` Å).
2. **Enumerate pairs.** Every remaining pair is considered.
3. **Hard gates — drop bad pairs.** A pair is skipped if its energy gap
   exceeds ``energy_gap_threshold`` (2.0 eV for bare systems, 0.75 eV with
   adsorbates), if its shape difference exceeds ``max_endpoint_mismatch``,
   or, for adsorbate plus metal core, if core RMS exceeds
   ``pair_core_rms_max``.
4. **Soft rank.** Survivors are scored with ``pair_score_*`` weights. The
   score favors a mid-range energy gap, similar cores, and some adsorbate
   site motion. Bare systems favor distinct fingerprints instead.
5. **Budget and oversampling.** Keep the top N pairs where N comes from the
   budget. Bare systems use ``max_pairs`` directly. Adsorbate systems with
   ``max_endpoint_mismatch`` set oversample to
   ``min(max_pairs * 10, max(max_pairs, 50))`` and re-rank by IDPP path
   quality before keeping ``max_pairs`` for NEB. When TorchSim or IDPP
   screening is not available, the oversampled list is still truncated to
   ``max_pairs`` before NEB.
6. **Run NEB.** Only those N pairs run NEB. Bare ``surface_cluster`` and
   ``surface`` use the same budget rule as bare gas (no oversampling even
   though ``max_endpoint_mismatch`` is set).

Hard gates always apply. Soft scores only matter when the pool is larger than
the budget. Pair selection and NEB endpoint setup share the same core overlay:
gas cores are fingerprint-matched then Kabsch-aligned (including 1-atom
translation); slab cores stay in the lab frame. Defaults by system type and
full knob tables: :doc:`/parameters` (Pair selection) and
:doc:`/validation_and_constraints`.

**TS from existing minima** (after a prior ``run_go`` or manual GO output):

.. code-block:: python

   from scgo import run_ts_search
   from scgo.param_presets import get_ts_search_params

   ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
   ts_params["max_pairs"] = 10

   # Campaign root: reads Pt5_searches/, writes Pt5_ts_results/ as sibling
   results = run_ts_search(
       "Pt5",
       ts_params=ts_params,
       seed=42,
       output_dir="results/pt5_gas_mace",
       system_type="gas_cluster",
   )

   # Or pass the searches directory directly (parent becomes campaign root)
   results = run_ts_search(
       "Pt5",
       ts_params=ts_params,
       seed=42,
       searches_dir="results/pt5_gas_mace/Pt5_searches",
       system_type="gas_cluster",
   )

**GO + TS combined:**

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   go_params = get_torchsim_ga_params(system_type="gas_cluster", seed=42)
   go_params["n_jobs"] = -2   # all but one CPU, across every parallel stage
   go_params["optimizer_params"]["ga"].update(
       niter=10,
       population_size=50,
   )


   ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
   ts_params["max_pairs"] = 15

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster",
   )

**On a surface:**

.. code-block:: python

   from scgo import run_go_ts, make_graphite_surface_config
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   go_params = get_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   go_params["n_jobs"] = -2   # all but one CPU, across every parallel stage


   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   ts_params["max_pairs"] = 10

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Campaigns
---------

Run multiple compositions in one call. Composition builders
(``build_one_element_compositions``, ``build_two_element_compositions``) live in
``scgo.runner_api``, not the top-level ``scgo`` package.

**Global optimization:**

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_one_element_compositions

   # Pt2, Pt3, Pt4, Pt5, Pt6
   compositions = build_one_element_compositions("Pt", min_atoms=2, max_atoms=6)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )
   # results is dict[path_key, list[(energy, Atoms)]]
   # (for gas_cluster, path_key matches the formula, e.g. "Pt5")

Failed compositions are logged, recorded as empty lists under their ``path_key``,
and skipped so the rest of the campaign continues. See :doc:`/api/initialization`
for multi-element atom ordering and placement.

**Binary compositions:**

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_two_element_compositions

   # All Au/Pt combinations with 2-4 total atoms
   compositions = build_two_element_compositions("Au", "Pt", min_atoms=2, max_atoms=4)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

**TS from existing minima** (each composition needs a prior ``{path_key}_searches/`` tree):

.. code-block:: python

   from scgo import run_ts_campaign
   from scgo.param_presets import get_ts_search_params
   from scgo.runner_api import build_one_element_compositions

   compositions = build_one_element_compositions("Pt", min_atoms=4, max_atoms=6)

   results = run_ts_campaign(
       compositions,
       ts_params=get_ts_search_params(system_type="gas_cluster", seed=42),
       seed=42,
       output_dir="benchmark/results",  # shared campaign root
       system_type="gas_cluster",
   )

**GO + TS for multiple compositions:**

.. code-block:: python

   from scgo import run_go_ts_campaign
   from scgo.param_presets import get_testing_params, get_ts_search_params
   from scgo.runner_api import build_one_element_compositions

   compositions = build_one_element_compositions("Pt", min_atoms=4, max_atoms=5)

   results = run_go_ts_campaign(
       compositions,
       go_params=get_testing_params(),
       ts_params=get_ts_search_params(system_type="gas_cluster", seed=42),
       seed=42,
       output_dir="benchmark/results",
       system_type="gas_cluster",
   )

See :doc:`/api/runner_api` for full signatures.

Output
------

See :doc:`/output_layout` for directory structure, path keys, run IDs, and file
formats. In brief: ``run_go`` writes ``{path_key}_searches/`` with ``run_*/``
subdirectories, and GO+TS creates a sibling ``{path_key}_ts_results/`` tree.

Parameters
----------

Quick parameter selection:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Preset
     - Use for
   * - ``get_testing_params()``
     - Fast tests (EMT calculator)
   * - ``get_default_params()``
     - Default production (MACE)
   * - ``get_torchsim_ga_params(...)``
     - GPU-accelerated with TorchSim
   * - ``get_ts_search_params(...)``
     - Transition state search

See :doc:`/parameters` for all options and :doc:`/api/param_presets` for details.

Examples
--------

Working examples live in the repository. See
`examples/README.md <https://github.com/rlaplaza-lab/scgo/blob/main/examples/README.md>`_
for the full list and usage notes. Each example enables ``write_timing_json``
so per-run ``timing.json`` and campaign ``go_ts_timing.json`` are written.
