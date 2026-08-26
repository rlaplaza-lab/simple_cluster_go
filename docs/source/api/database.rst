Database access
===============

HPC-oriented SQLite helpers for reading and writing GO/TS optimizer databases.

Use :func:`~scgo.load_previous_run_results` or
:class:`~scgo.database.manager.SCGODatabaseManager` for the common case of
reloading minima from completed ``run_*`` searches. For low-level access,
:func:`~scgo.database.connection.get_connection` opens a scoped
:class:`~ase_ga.data.DataConnection` with SCGO PRAGMA settings applied.

:func:`~scgo.database.helpers.setup_database` writes at most one
``simulation_cell=True`` template row; reopening with ``remove_existing=False``
keeps that row and checks stored stoichiometry.

Structure tags, DB stamps, and run-dir JSON live in :doc:`metadata`.

.. automodule:: scgo.database
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

.. autoclass:: scgo.database.discovery.DatabaseDiscovery
   :members:
   :show-inheritance:
   :exclude-members: _*

.. automodule:: scgo.database.helpers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

.. automodule:: scgo.database.transactions
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

.. autofunction:: scgo.database.connection.get_connection

.. automodule:: scgo.database.registry
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*

.. automodule:: scgo.database.cache
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _*
