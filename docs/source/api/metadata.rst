Metadata
========

Structure tags, run-directory records, DB identity stamps, and output-JSON
provenance headers. These are **separate schemas** that share a package
namespace only.

Structure tags (ASE Atoms)
--------------------------

Per-structure annotations live in ``atoms.info['key_value_pairs']`` (ASE's
durable SQLite JSON column). Use :mod:`scgo.metadata.atoms` exclusively.

.. automodule:: scgo.metadata.atoms
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

Final-minima SQL tagging
------------------------

.. automodule:: scgo.metadata.persist
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

Output-JSON provenance header
-----------------------------

Shared header for ``metadata.json``, NEB/TS summaries, timing files, and
cluster-adsorbate relaxed-structure ``info`` dicts (``schema_version`` = 4).

.. automodule:: scgo.metadata.provenance
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

Run-directory records
---------------------

``run_*/metadata.json`` params/composition snapshots and run-id helpers.
:func:`~scgo.metadata.run_dir.resolve_run_id_from_db_path` returns ``None`` when
no canonical ``run_YYYYMMDD_HHMMSS_ffffff`` path segment is found (callers skip
that database rather than using the filename as a fake run id).

.. automodule:: scgo.metadata.run_dir
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

SQLite DB stamp
---------------

The ``scgo_metadata`` table marks SCGO-owned databases (``schema_version`` = 2).
This integer is distinct from the output-JSON header version.

.. automodule:: scgo.metadata.db_stamp
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*
