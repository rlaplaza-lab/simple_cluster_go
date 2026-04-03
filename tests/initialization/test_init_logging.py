"""Test initialization logging to understand the duplicate messages."""

import logging

import numpy as np

from scgo.initialization import create_initial_cluster_batch


def test_create_initial_cluster_batch_logs_and_returns_population(caplog):
    composition = ["Pt"] * 4
    n_structures = 59
    rng = np.random.default_rng(42)
    caplog.set_level(logging.INFO)
    population = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=1,
    )

    assert isinstance(population, list)
    assert len(population) == n_structures
    # Ensure expected initialization log messages were emitted
    assert "Initialization for 4-atom clusters" in caplog.text
    assert "Strategy allocation" in caplog.text
