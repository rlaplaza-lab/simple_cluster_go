# SCGO Documentation

This directory contains the Sphinx documentation source for SCGO (Simple Cluster Global Optimization), built with Sphinx.

## Building the Documentation

```bash
pip install -e ".[mace]"
pip install -r docs/source/requirements.txt
cd docs && make html
```

The built documentation will be available in `docs/build/html/index.html`. For PDF: `make latexpdf` (output in `docs/build/latex/scgo.pdf`).

## Structure

- `source/` — Sphinx source files (RST format); `api/` holds the API reference auto-generated from docstrings
- `Makefile` — delegates to `source/Makefile`

## Online Documentation

This documentation is automatically built and published on [Read the Docs](https://scgo.readthedocs.io/). The configuration is in `.readthedocs.yaml` in the project root.

## Writing Documentation

- Use reStructuredText (RST) format and Google-style docstrings in Python code
- Use `.. autofunction::` / `.. automodule::` directives for API documentation
- Keep examples concise and practical; sentence case for headings

The API reference is generated from docstrings — improve them in source, then rebuild with `make html`.

## Releases (maintainers)

Publish via the GitHub Actions **Publish to PyPI** workflow (`workflow_dispatch`, `confirm=publish`). Configure trusted publishing for the `pypi` environment (and `testpypi` if used).
