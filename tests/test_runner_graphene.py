"""Smoke tests for the deprecated scgo.runner_graphene shim."""

from pathlib import Path

import pytest


def test_import_warns_deprecation() -> None:
    """Importing runner_graphene must emit a DeprecationWarning."""
    import importlib
    import sys

    # Remove cached module so the module-level warning fires again.
    sys.modules.pop("scgo.runner_graphene", None)
    with pytest.warns(DeprecationWarning, match="runner_surface"):
        importlib.import_module("scgo.runner_graphene")


def test_make_graphene_slab_periodic() -> None:
    with pytest.warns(DeprecationWarning):
        import importlib
        import sys

        sys.modules.pop("scgo.runner_graphene", None)
        mod = importlib.import_module("scgo.runner_graphene")
    slab = mod.make_graphene_slab()
    assert slab.pbc.any()
    assert "C" in slab.get_chemical_symbols()


def test_default_surface_config() -> None:
    with pytest.warns(DeprecationWarning):
        import importlib
        import sys

        sys.modules.pop("scgo.runner_graphene", None)
        mod = importlib.import_module("scgo.runner_graphene")
    cfg = mod.default_surface_config()
    assert cfg.fix_all_slab_atoms is True


def test_read_full_composition_from_first_xyz_missing_dir(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning):
        import importlib
        import sys

        sys.modules.pop("scgo.runner_graphene", None)
        mod = importlib.import_module("scgo.runner_graphene")
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="not found"):
        mod.read_full_composition_from_first_xyz(missing)


def test_read_full_composition_from_first_xyz_empty_dir(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning):
        import importlib
        import sys

        sys.modules.pop("scgo.runner_graphene", None)
        mod = importlib.import_module("scgo.runner_graphene")
    empty = tmp_path / "final_unique_minima"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No final_unique_minima"):
        mod.read_full_composition_from_first_xyz(empty)
