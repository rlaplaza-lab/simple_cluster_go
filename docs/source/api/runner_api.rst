Runner API
==========

High-level API entry points for global optimization and transition state searches.

.. automodule:: scgo.runner_api
   :members: run_go, run_go_campaign, run_ts_search, run_go_ts, run_go_ts_campaign
   :undoc-members:
   :show-inheritance:

run_go
------

Primary entry point for global optimization of atomic clusters.

.. autofunction:: scgo.runner_api.run_go

run_go_campaign
---------------

Run global optimization for multiple compositions sequentially.

.. autofunction:: scgo.runner_api.run_go_campaign

run_ts_search
-------------

Perform transition state search for a given composition.

.. autofunction:: scgo.runner_api.run_ts_search

run_go_ts
---------

Run global optimization followed by transition state search.

.. autofunction:: scgo.runner_api.run_go_ts

run_go_ts_campaign
------------------

Run GO+TS pipeline for multiple compositions sequentially.

.. autofunction:: scgo.runner_api.run_go_ts_campaign