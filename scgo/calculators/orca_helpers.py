"""ORCA quantum chemistry input file generation utilities.

This module provides functions for generating ORCA input files from optimized
cluster structures, enabling subsequent high-level quantum chemical calculations
and analysis.
"""

from __future__ import annotations

import os
from typing import Any

from ase import Atoms

from scgo.utils.helpers import ensure_directory_exists
from scgo.utils.logging import get_logger


def write_orca_inputs(
    atoms: Atoms,
    output_dir: str,
    orca_settings: dict[str, Any],
):
    """Writes a multi-job ORCA input file for a given structure.

    The first job is a geometry optimization and frequency calculation.
    The second job is a single point calculation reading orbitals from the first job.

    Args:
        atoms: The ASE Atoms object representing the structure.
        output_dir: The directory where the ORCA input file (`orca.inp`) will be
                    written.
        orca_settings: A dictionary of ORCA parameters. Expected keys include:
                       'charge', 'multiplicity', 'keywords', and 'blocks_str'.
                       If not provided, reasonable defaults are used.
    """
    ensure_directory_exists(output_dir)
    logger = get_logger(__name__)
    logger.info(f"Writing ORCA input to {os.path.basename(output_dir)}")

    # Defaults
    default_keywords_1 = "! RI PBE def2-tzvp def2/J Opt Freq VerySlowConv"
    default_keywords_2 = "! PBE0 def2-tzvp VerySlowConv MOread"
    default_blocks_str = """%pal
nprocs 24
end

%scf
MaxIter 1500
DIISMaxEq 15
directresetfreq 5
end"""
    charge = orca_settings.get("charge", 0)
    multiplicity = orca_settings.get("multiplicity", 1)

    # Job 1
    keywords1 = orca_settings.get("keywords", default_keywords_1)
    blocks1 = orca_settings.get("blocks_str", default_blocks_str)

    # Job 2
    keywords2 = orca_settings.get("keywords", default_keywords_2)
    # The second job should read the gbw file from the first job, which is named after the input file (orca.gbw)
    moinp_block = '%moinp "orca.gbw"'
    # Combine the moinp block with the other blocks for the second job
    blocks2 = f"{moinp_block}\n\n{blocks1}"

    # Build the input file content
    lines = []
    # Job 1
    lines.append(keywords1)
    lines.append("")
    lines.append(blocks1)
    lines.append("")
    lines.append(f"* xyz {charge} {multiplicity}")
    lines.extend(
        [
            f" {atom.symbol:2s} {atom.x:20.12f} {atom.y:20.12f} {atom.z:20.12f}"
            for atom in atoms
        ]
    )
    lines.append("*")
    lines.append("")

    # Job 2
    lines.append("$new_job")
    lines.append(keywords2)
    lines.append("")
    lines.append(blocks2)
    lines.append("")
    # The second job should read the optimized coordinates from the first job's output xyz file.
    lines.append("* xyzfile 0 1 orca.xyz")
    lines.append("")

    content = "\n".join(lines)

    input_path = os.path.join(output_dir, "orca.inp")
    with open(input_path, "w") as f:
        f.write(content)


def prepare_orca_calculations(
    unique_minima: list[tuple[float, Atoms]],
    base_dir: str,
    orca_settings: dict[str, Any],
):
    """Prepares ORCA input files for a list of unique minima.

    This function iterates through a list of structures, creating a dedicated
    subdirectory for each one and writing the necessary ORCA input files for
    further analysis (e.g., high-level DFT optimization or validation).

    Args:
        unique_minima: A list of (energy, Atoms) tuples representing the
                       unique structures to be calculated.
        base_dir: The base directory where subdirectories for each calculation
                  will be created.
        orca_settings: A dictionary of ORCA parameters that will be passed to
                       `write_orca_inputs` for each structure.
    """
    from scgo.utils.helpers import _prepare_calculator_calculations

    _prepare_calculator_calculations(
        unique_minima=unique_minima,
        base_dir=base_dir,
        calculator_name="ORCA",
        write_function=write_orca_inputs,
        orca_settings=orca_settings,
    )
