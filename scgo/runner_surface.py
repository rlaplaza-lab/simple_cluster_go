"""Generic slab-first surface runner helpers for SCGO example scripts.

Provides reusable utilities that work with *any* ASE slab ``Atoms`` object,
so example runner scripts only need to build or load their slab and call
these helpers — no surface-specific module required.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read

from scgo.surface.config import SurfaceSystemConfig


def make_surface_config(
    slab: Atoms,
    *,
    adsorption_height_min: float = 2.0,
    adsorption_height_max: float = 3.5,
    fix_all_slab_atoms: bool = True,
    comparator_use_mic: bool = True,
    max_placement_attempts: int = 500,
) -> SurfaceSystemConfig:
    """Build a ``SurfaceSystemConfig`` from an arbitrary ASE slab.

    Parameters
    ----------
    slab:
        Any periodic (or non-periodic) ASE ``Atoms`` representing the substrate.
    adsorption_height_min:
        Minimum adsorbate height above the slab surface (Angstrom).
    adsorption_height_max:
        Maximum adsorbate height above the slab surface (Angstrom).
    fix_all_slab_atoms:
        Whether to freeze every slab atom during local relaxation.
    comparator_use_mic:
        Use minimum image convention for duplicate detection (recommended
        when the slab has in-plane periodicity).
    max_placement_attempts:
        Maximum random placement retries per candidate structure.
    """
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=adsorption_height_min,
        adsorption_height_max=adsorption_height_max,
        fix_all_slab_atoms=fix_all_slab_atoms,
        comparator_use_mic=comparator_use_mic,
        max_placement_attempts=max_placement_attempts,
    )


def read_full_composition_from_first_xyz(
    final_unique_minima_dir: Path,
) -> list[str]:
    """Infer full system composition (slab + adsorbate) from the first final minimum XYZ.

    Parameters
    ----------
    final_unique_minima_dir:
        Path to the ``final_unique_minima/`` directory produced by a previous
        SCGO global-optimization run.

    Returns
    -------
    list[str]
        Chemical symbols of the full system (slab atoms followed by adsorbate atoms).

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no ``*.xyz`` files.
    """
    if not final_unique_minima_dir.exists():
        msg = f"Final unique minima directory not found: {final_unique_minima_dir}"
        raise FileNotFoundError(msg)

    first_xyz = next(final_unique_minima_dir.glob("*.xyz"), None)
    if first_xyz is None:
        msg = f"No final_unique_minima xyz files found in: {final_unique_minima_dir}"
        raise FileNotFoundError(msg)

    atoms = read(str(first_xyz))
    return list(atoms.get_chemical_symbols())
