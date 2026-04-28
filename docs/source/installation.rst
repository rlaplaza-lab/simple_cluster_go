Installation
============

SCGO can be installed using either conda (recommended) or pip.

Prerequisites
-------------

- Python 3.12+
- SQLite with JSON1 extension
- CUDA (for GPU acceleration with MLIPs)

Conda Installation (Recommended)
--------------------------------

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/simple_cluster_go.git
   cd simple_cluster_go
   conda env create -f environment.yml
   conda activate scgo

The conda environment installs SCGO in editable mode with MACE/TorchSim support and development tools.

Pip Installation
----------------

Install with MACE support:

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/simple_cluster_go.git
   cd simple_cluster_go
   pip install -e ".[mace]"

Or with UMA support:

.. code-block:: bash

   pip install -e ".[uma]"

Development Installation
------------------------

For development with tests and linting:

.. code-block:: bash

   pip install -e ".[mace,dev]"  # or: pip install -e ".[uma,dev]"
   pre-commit install

Dependency Notes
----------------

- SCGO requires exactly one of the `[mace]` or `[uma]` extras for MLIP support
- The MACE and UMA extras use incompatible dependency stacks
- SQLite JSON1 extension is required (install `pysqlite3-binary` if needed)
- Sella is optional for advanced optimization features