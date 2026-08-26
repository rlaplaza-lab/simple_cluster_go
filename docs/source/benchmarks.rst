Benchmarks
==========

SCGO ships **long-running MLIP regression scripts** under ``benchmark/`` at the
repository root, separate from the fast unit tests of the benchmark helpers in
``tests/benchmark/``.

Dependencies
------------

- **MACE (default):** ``pip install -e ".[mace]"`` (TorchSim GA + MACE)
- **UMA:** separate env; ``--backend uma``
- **UPET:** separate env with ``scgo[upet]`` + pinned ``vesin==0.6.0``

Output layout
-------------

All scripts write under ``benchmark/results/``. See :doc:`/output_layout` for the
complete directory structure. TS runs add sibling ``{path_key}_ts_results/`` trees
with the same run-oriented layout (``run_*/``, summaries, deduplicated exports);
pair work lives under ``pair_*`` subdirs.

Benchmark GA presets enable ``write_timing_json`` and ``detailed_timing`` so CLI
profiling lines match ``{run_dir}/timing.json`` on disk.

Entry points
------------

.. code-block:: bash

   python -m benchmark.benchmark_Pt --help
   python -m benchmark.benchmark_Pt_surface_graphite --help
   python -m benchmark.benchmark_parallel_neb --help

Shared CLI flags (see ``benchmark.benchmark_common.add_common_benchmark_cli``):

- ``--backend {mace,uma}``: default from ``SCGO_BENCHMARK_BACKEND`` (``mace``)
- ``--model-name``, ``--seed``, ``--uma-task``, ``--clusters``, ``--niter``,
  ``--population-size``

Pytest
------

``pytest.ini`` sets ``testpaths = tests``, so ``benchmark/`` is out of scope
for the default run (a ``benchmark`` marker also exists for explicit runs). To
run MLIP regression hooks:

.. code-block:: bash

   pytest benchmark/ -m slow

See also `benchmark/README.md <https://github.com/rlaplaza-lab/scgo/blob/main/benchmark/README.md>`_
in the repository for maintainer notes.
