Uniqueness (de-duplication)
===========================

Global optimization will find the same isomer more than once. SCGO keeps a
structure only if it is **new in energy and in shape**. Both must match before
two results are treated as duplicates:

1. Energies differ by at most ``0.02`` eV (``energy_tolerance``).
2. The **moving** atoms have the same geometry: sorted interatomic distances
   agree closely (no single distance off by more than ``0.7`` Å).

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

When it runs
------------

The same rule is applied:

- during a **genetic algorithm** search, so the population does not fill with
  copies
- at the end of **basin hopping** (unless you set ``deduplicate=False``)
- at the end of every **campaign**, before connectivity and Hessian checks

**GA in-search check (performance note).** Acceptance is decided against the
current population (O(population size), not O(history)).  Each time an
isomer is re-presented to the population its incumbent's rediscovery count is
incremented; that count feeds the fitness penalty ``1/√(1 + L)`` used in
parent selection.  Geometry fingerprints are cached on each structure without
copying the ``Atoms`` object, so repeated comparisons reuse the fingerprint
computed on the first call.  The end-of-campaign ``filter_unique_minima`` pass
(BH and simple GO) is a separate full-history check and is unchanged.

The simple (1–2 atom) optimizer has no in-search filter; it relies on that
final campaign pass.
Unique structures are written under ``final_unique_minima/`` (see
:doc:`/output_layout`).

Throwing out identical **starting guesses** is a separate, simpler check. It
does not use the energy-and-shape rule above.

How to change it
----------------

Set the knobs on the optimizer you actually run
(``optimizer_params["ga"]``, ``"bh"``, or ``"simple"``). Defaults are shared:

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
     - ``0.7`` Å
     - Largest allowed difference in any one interatomic distance
   * - ``comparator_tol``
     - ``0.015``
     - How much overall mismatch is still "the same shape"
   * - ``comparator_n_top``
     - ``None``
     - Leave ``None`` (uses the moving atoms in the table above)
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
  (``minima_energy_tolerance`` plus the GO geometry cutoffs).
- Successful saddles are filtered into ``final_unique_ts/``
  (``ts_energy_tolerance`` plus ``similarity_tolerance`` and
  ``similarity_pair_cor_max``).

The same ``similarity_*`` knobs are reused when pair selection asks whether two
endpoints look alike. Choosing *which* pairs to connect (energy gap, overlay
checks, and so on) is a different step, documented under **Pair selection** in
:doc:`/parameters`.

The geometry test follows Vilhelmsen and Hammer, *Phys. Rev. Lett.* **108**,
126101 (2012). Knob tables for every optimizer live in :doc:`/parameters`.
