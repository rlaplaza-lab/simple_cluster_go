"""Simple optimization for 1-2 atom clusters.

This module provides a minimal optimization approach for very small clusters
(1-2 atoms) where basin hopping is unnecessary. For these cases, there's only
one meaningful structure to optimize, so we just perform a single local
optimization without iterations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer

from scgo.database import HPC_DATABASE_EXCEPTIONS, close_data_connection, setup_database
from scgo.database.metadata import persist_provenance
from scgo.database.sync import retry_with_backoff
from scgo.utils.helpers import (
    extract_minima_from_database,
    perform_local_relaxation,
)
from scgo.utils.logging import get_logger
from scgo.utils.validation import (
    validate_atoms,
    validate_calculator_attached,
    validate_positive,
)


def simple_go(
    atoms: Atoms,
    output_dir: str,
    rng: np.random.Generator,
    niter: int = 1,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = LBFGS,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    **kwargs: Any,
) -> list[tuple[float, Atoms]]:
    """Simple local optimization for 1-2 atom clusters.

    Performs a single local optimization without basin hopping iterations.

    Args:
        atoms: Initial Atoms object representing the cluster. Calculator must be attached.
        output_dir: Directory where ASE database will be stored.
        rng: Random number generator (numpy.random.Generator). Required.
        niter: Total number of iterations. Default 1.
        fmax: Maximum force criterion for convergence (eV/Å). Default 0.05.
        niter_local_relaxation: Maximum steps for each local relaxation. Default 250.
        optimizer: ASE optimizer class (e.g., BFGS) for local relaxations. Default LBFGS.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        run_id: Optional run id for database provenance (same as other optimizers).
        clean: If True, remove an existing database in the trial directory.
        **kwargs: Extra keys from shared ``global_optimizer_kwargs`` (ignored).

    Returns:
        List of (energy, Atoms) tuples for local minima found. Typically a single structure.

    Raises:
        TypeError: If atoms is not an Atoms object.
        ValueError: If calculator is not attached or atoms is not 1-2 atoms.
    """
    logger = get_logger(__name__)

    validate_atoms(atoms)
    calculator = validate_calculator_attached(atoms, "simple optimization")
    validate_positive("fmax", fmax, strict=True)

    n_atoms = len(atoms)
    if n_atoms < 1 or n_atoms > 2:
        raise ValueError(f"simple_go only supports 1-2 atoms, got {n_atoms} atoms")

    del kwargs  # unused keys from shared presets (e.g. GA-only options)
    # Detach calculator temporarily for DB setup to avoid pickling issues
    calc = atoms.calc
    atoms.calc = None
    da = setup_database(
        output_dir,
        "bh_go.db",
        atoms,
        initial_candidate=atoms,
        remove_existing=clean,
        run_id=run_id,
    )
    atoms.calc = calc

    logger.info(f"Performing simple optimization for {n_atoms}-atom cluster")

    _HPC_RETRY_EXCEPTIONS = HPC_DATABASE_EXCEPTIONS + (IOError,)

    try:
        a_optimized = retry_with_backoff(
            da.get_an_unrelaxed_candidate,
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
        )
        perform_local_relaxation(
            a_optimized,
            calculator,
            optimizer,
            fmax,
            niter_local_relaxation,
        )

        if run_id is not None and a_optimized is not None:
            persist_provenance(a_optimized, run_id=run_id)

        retry_with_backoff(
            lambda: da.add_relaxed_step(a_optimized),
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
        )

        all_candidates = retry_with_backoff(
            da.get_all_relaxed_candidates,
            max_retries=5,
            initial_delay=0.2,
            backoff_factor=2.0,
            exception_types=_HPC_RETRY_EXCEPTIONS,
        )
        all_minima = extract_minima_from_database(all_candidates)

        if not all_minima:
            return []

        return all_minima
    finally:
        close_data_connection(da, log_errors=False)
