# SCGO Documentation

This directory contains the Sphinx documentation source for SCGO (Simple Cluster Global Optimization), built with Sphinx.

## Building the Documentation

### Prerequisites

- Python 3.12+
- Documentation dependencies: `pip install -r requirements.txt`

### Building HTML

```bash
make html
```

The built documentation will be available in `build/html/index.html`.

### Building PDF

```bash
make latexpdf
```

The PDF will be available in `build/latex/scgo.pdf`.

## Documentation Structure

- `source/` — Sphinx source files (RST format)
  - `api/` — API reference documentation (auto-generated from docstrings)
  - `index.rst` — Main documentation index
  - `installation.rst` — Installation instructions
  - `quickstart.rst` — Quick start guide with working examples
- `conf.py` — Sphinx configuration
- `requirements.txt` — Documentation build requirements
- `Makefile` — Build automation

## Online Documentation

This documentation is automatically built and published on [ReadTheDocs](https://scgo.readthedocs.io/). The configuration is in `.readthedocs.yaml` in the project root.

## Writing Documentation

- Use reStructuredText (RST) format
- Follow Google-style docstrings in the Python code
- Use `.. autofunction::` and `.. automodule::` directives for API documentation
- Keep examples concise and practical

## Updating API Documentation

The API documentation is automatically generated from docstrings in the Python code. To update:

1. Add/improve docstrings in the source code
2. Run `make html` to rebuild
3. Commit both code and documentation changes

## Style Guide

- Use sentence case for headings
- Keep line length under 88 characters
- Use code blocks for examples
- Be consistent with existing documentation style