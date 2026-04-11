"""Smoke tests that pass without the MACE optional extra (e.g. UMA-only CI)."""

from __future__ import annotations


def test_import_scgo_without_eager_torchsim():
    import scgo

    assert scgo.__version__
    assert hasattr(scgo, "run_scgo_trials")


def test_lazy_ga_go_torchsim_import_error_message():
    import importlib.util

    if importlib.util.find_spec("mace") is not None:
        # MACE stack installed; lazy import should succeed.
        from scgo.algorithms import ga_go_torchsim

        assert ga_go_torchsim is not None
        return

    import pytest

    with pytest.raises(ImportError, match=r"scgo\[mace\]"):
        from scgo.algorithms import ga_go_torchsim  # noqa: F401
