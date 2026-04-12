"""VASP DFT input file generation utilities.

This module provides functions for generating VASP input files (INCAR, POSCAR, KPOINTS)
from optimized cluster structures for periodic DFT calculations.
"""

from __future__ import annotations

import os
from typing import Any

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import write

from scgo.utils.helpers import ensure_directory_exists, get_cluster_formula
from scgo.utils.logging import get_logger


def write_vasp_inputs(
    atoms: Atoms,
    output_dir: str,
    vasp_settings: dict[str, Any],
    vacuum: float = 10.0,
    *,
    center_structure: bool = True,
):
    """Writes VASP input files (INCAR, POSCAR, KPOINTS, POTCAR) for a given structure.

    This function prepares a directory for a VASP calculation by creating the
    necessary input files based on a template Atoms object and a dictionary of
    VASP settings.

    Args:
        atoms: The ASE Atoms object representing the structure.
        output_dir: The directory where the VASP input files will be written.
        vasp_settings: A dictionary of INCAR parameters (e.g., `{'encut': 400}`).
        vacuum: The amount of vacuum (in Angstrom) to add around the cluster
                in the simulation cell. Defaults to 10.0. Only used when
                ``center_structure`` is True.
        center_structure: If True (default), center the structure with ``vacuum``
            padding (gas-phase clusters). If False, keep positions (slabs /
            adsorbate-on-slab); set selective dynamics in INCAR if you freeze layers.
    """
    # Ensure the target directory exists before trying to write to it.
    ensure_directory_exists(output_dir)

    logger = get_logger(__name__)
    logger.info(f"Writing VASP inputs to {os.path.basename(output_dir)}")
    # Use a copy to avoid modifying the original atoms object
    atoms_for_vasp = atoms.copy()
    if center_structure:
        atoms_for_vasp.center(vacuum=vacuum)

    # Save the final, best structure for easy viewing
    cluster_formula = get_cluster_formula(atoms.get_chemical_symbols())

    # Remove tags array to ensure consistent XYZ format
    if "tags" in atoms_for_vasp.arrays:
        del atoms_for_vasp.arrays["tags"]

    write(os.path.join(output_dir, f"{cluster_formula}_best.xyz"), atoms_for_vasp)

    # Set up and write VASP files
    # NOTE: ASE needs the VASP_PP_PATH environment variable to be set
    vasp_calc = Vasp(
        directory=output_dir,
        xc="PBE",  # Exchange-correlation functional
        kpts=(1, 1, 1),  # Gamma-point only for a cluster
        gamma=True,
        **vasp_settings,
    )
    vasp_calc.write_input(atoms_for_vasp)


def prepare_vasp_calculations(
    unique_minima: list[tuple[float, Atoms]],
    base_dir: str,
    vasp_settings: dict[str, Any],
    vacuum: float,
    *,
    center_structure: bool = True,
):
    """Prepares VASP input files for a list of unique minima.

    This function iterates through a list of structures, creating a dedicated
    subdirectory for each one and writing the necessary VASP input files for
    further analysis (e.g., high-level DFT optimization or validation).

    Args:
        unique_minima: A list of (energy, Atoms) tuples representing the
                       unique structures to be calculated.
        base_dir: The base directory where subdirectories for each calculation
                  will be created.
        vasp_settings: A dictionary of VASP INCAR parameters that will be passed
                       to `write_vasp_inputs` for each structure.
        vacuum: The amount of vacuum to add around each cluster.
        center_structure: Forwarded to :func:`write_vasp_inputs`.
    """
    from functools import partial

    from scgo.utils.helpers import _prepare_calculator_calculations

    _prepare_calculator_calculations(
        unique_minima=unique_minima,
        base_dir=base_dir,
        calculator_name="VASP",
        write_function=partial(write_vasp_inputs, center_structure=center_structure),
        vasp_settings=vasp_settings,
        vacuum=vacuum,
    )
