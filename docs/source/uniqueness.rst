Uniqueness (de-duplication)
===========================

Global optimization will find the same isomer more than once. SCGO keeps a
structure only if it is **new in energy and in shape**. Both must match before
two results are treated as duplicates:

1. Energies differ by at most ``0.02`` eV (``energy_tolerance``).
2. The **moving** atoms have the same geometry: sorted interatomic distances
   agree closely (no single distance off by more than ``0.7`` Å; tighter gates
   apply to supported clusters — see below).

A high-energy copy of a known isomer is dropped. Two different shapes at almost
the same energy are both kept. Frozen atoms (the lower slab, for example) are
ignored.

What is compared
----------------

Only the atoms that the search is allowed to move go into the comparison:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - System type
     - Atoms compared
   * - ``gas_cluster`` / ``gas_cluster_adsorbate``
     - The whole cluster, including adsorbates
   * - ``surface_cluster`` / ``surface_cluster_adsorbate``
     - The deposited cluster (and adsorbates); the fixed slab is ignored
   * - ``surface`` / ``surface_adsorbate``
     - Mobile top slab layers (and adsorbates); fixed bottom layers are ignored

On a periodic slab, distances wrap through the cell so the same site on opposite
edges counts as the same geometry. That wrap is off for gas-phase clusters.

Role blocks and weighting
~~~~~~~~~~~~~~~~~~~~~~~~~

The moving atoms are further split into **role blocks** — ``mobile_slab``,
``deposit``, ``adsorbate`` — and each block's distances are normalized inside
the block before being combined. This fixes a real failure mode: a globally
optimized deposit sitting on a barely-moving same-element support used to have
its distance differences diluted by the support in one shared bucket.

Each block carries a weight (default ``1.0``). A weight of ``0`` excludes the
block completely. Distances *between* two blocks are also compared now; they
are what make an adsorbate's binding site (fcc vs hcp vs bridge) or a deposit's
contact with the support count, which pure intra-element fingerprints could
never see.

Type-aware defaults:

- ``surface`` / ``surface_adsorbate``: mobile top layers are the region of
  interest → weight ``1.0``.
- ``surface_cluster*`` with relaxed support layers: those layers join the
  comparison at weight ``0.2`` so lattice noise cannot drown the deposit;
  set ``{"mobile_slab": 0.0}`` to exclude them entirely.
- Gas-phase types: plain deposit/adsorbate blocks at weight ``1.0``.

Tighter gates for supported clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because block normalization keeps deposit/adsorbate differences undiluted,
``surface_cluster`` and ``surface_cluster_adsorbate`` use tighter default
gates: ``comparator_tol=0.010`` and ``comparator_pair_cor_max=0.45 Å`` instead
of ``0.015`` / ``0.7 Å``. Explicit non-default user values always win.

When it runs
------------

The same rule is applied:

- during a **genetic algorithm** search, so the population does not fill with
  copies (re-presented isomers also accrue a fitness penalty via a
  rediscovery counter)
- at the end of **basin hopping** (unless you set ``deduplicate=False``)
- at the end of every **campaign**, before connectivity and Hessian checks

The simple (1–2 atom) optimizer has no in-search filter; it relies on that
final campaign pass.
Unique structures are written under ``final_unique_minima/`` (see
:doc:`/output_layout`).

Throwing out identical **starting guesses** is a separate, simpler check. It
does not use the energy-and-shape rule above.

How to change it
----------------

Set the knobs on the optimizer you actually run
(``optimizer_params["ga"]``, ``"bh"``, or ``"simple"``). Defaults are shared;
the block-weighting knobs below exist on ``ga`` and ``bh`` (``simple`` only
runs trivial gas clusters and always uses the plain window):

.. list-table::
   :widths: 36 16 48
   :header-rows: 1

   * - Knob
     - Default
     - Meaning
   * - ``energy_tolerance``
     - ``0.02`` eV
     - How close two energies must be to even compare shapes
   * - ``comparator_pair_cor_max``
     - ``0.7`` Å (``0.45`` for supported clusters)
     - Largest allowed difference in any one interatomic distance
   * - ``comparator_tol``
     - ``0.015`` (``0.010`` for supported clusters)
     - How much overall mismatch is still "the same shape"
   * - ``comparator_n_top``
     - ``None``
     - Leave ``None`` (uses the moving atoms in the table above). Set it to
       force the legacy single-window comparison without role blocks
   * - ``comparator_component_weights``
     - ``None`` (type-aware)
     - Per-role weights, e.g. ``{"mobile_slab": 0.2}``; ``0`` excludes a block
   * - ``comparator_cross_weight``
     - ``1.0``
     - Weight of cross-block (binding-geometry) distance terms
   * - ``deduplicate`` (BH only)
     - ``True``
     - End-of-run BH filter. The campaign pass still runs if this is ``False``

Smaller values keep more structures as distinct. Larger values collapse more
near-copies::

   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["optimizer_params"]["ga"]["energy_tolerance"] = 0.01
   params["optimizer_params"]["ga"]["comparator_pair_cor_max"] = 0.4

On slabs, distance wrapping is on by default. Turn it off with
``comparator_use_mic=False`` when you build the surface.

.. caution::

   ``comparator_use_mic=False`` affects **GO uniqueness only**. Transition
   states always use minimum-image distances on surface types — TS dedupe,
   pair selection, and NEB interpolation force MIC via
   :func:`~scgo.system_types.resolve_neb_mic`, and the runner logs a warning
   when the knob disagrees.

Transition states
-----------------

TS search uses the **same energy-and-shape idea**, with a tighter distance
cutoff (``0.1`` Å instead of ``0.7`` Å):

- Minima loaded for pairing are first filtered with the **GO** rule
  (``minima_energy_tolerance`` plus the GO geometry cutoffs, including the
  role-block weighting and the tighter supported-cluster gates).
- Successful saddles are filtered into ``final_unique_ts/``
  (``ts_energy_tolerance`` plus ``similarity_tolerance`` and
  ``similarity_pair_cor_max``, compared over the same role blocks).

The same ``similarity_*`` knobs are reused when pair selection asks whether two
endpoints look alike. Choosing *which* pairs to connect (energy gap, overlay
checks, and so on) is a different step, documented under **Pair selection** in
:doc:`/parameters`.

The geometry test follows Vilhelmsen and Hammer, *Phys. Rev. Lett.* **108**,
126101 (2012). Knob tables for every optimizer live in :doc:`/parameters`.
