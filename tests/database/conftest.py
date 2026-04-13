from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path

import pytest
from ase import Atoms

from scgo.database import close_data_connection, setup_database


@pytest.fixture
def temp_db_ctx():
    """
    Returns a context manager to set up a transient SQLite database.
    """

    @contextmanager
    def _setup(
        tmp_path: Path,
        filename: str,
        template: Atoms,
        *,
        initial_candidate: Atoms | None = None,
        **setup_kwargs,
    ):
        da = setup_database(
            tmp_path,
            filename,
            template,
            initial_candidate=initial_candidate,
            **setup_kwargs,
        )
        try:
            yield da, (Path(tmp_path) / filename)
        finally:
            close_data_connection(da)
            gc.collect()

    return _setup


@pytest.fixture(
    params=[
        {},
        {"wal_mode": True},
        {"busy_timeout": 60000},
    ]
)
def db_with_kwargs(request, tmp_path, pt3_atoms):
    """Parametrized DB fixture tested against different SQLite PRAGMAs."""
    db_file = tmp_path / "parametrized.db"
    da = setup_database(
        tmp_path,
        db_file.name,
        pt3_atoms,
        initial_candidate=pt3_atoms.copy(),
        **request.param,
    )
    yield da, db_file
    close_data_connection(da)
    gc.collect()
