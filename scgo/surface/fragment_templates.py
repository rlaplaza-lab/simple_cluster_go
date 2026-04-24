"""Default rigid-body templates for adsorbate fragments (hierarchical surface init)."""

from __future__ import annotations

import numpy as np
from ase import Atoms

# Default O–H length (Å) for built-in OH and paired-OH patterns (matches ClusterOHConfig)
DEFAULT_OH_BOND_LENGTH = 0.96


def build_default_fragment_template(
    symbols: list[str], *, oh_bond_length: float = DEFAULT_OH_BOND_LENGTH
) -> Atoms | None:
    """Return a gas-phase template for simple ``adsorbate_symbols`` lists, or ``None``.

    Supported patterns (exact symbol order):
        - ``["O", "H"]``: one OH
        - ``["O", "H", "O", "H"]``: two separated OH (for duplicated OH on a core)

    For other stoichiometries or orderings, pass an explicit
    :class:`ase.Atoms` template to ``run_go(..., adsorbate_fragment_template=...)``.
    """
    s = [str(x) for x in symbols]
    if s == ["O", "H"]:
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, oh_bond_length]], dtype=float)
        return Atoms(symbols=s, positions=pos)
    if s == ["O", "H", "O", "H"]:
        sep = 2.2
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, oh_bond_length],
                [sep, 0.0, 0.0],
                [sep, 0.0, oh_bond_length],
            ],
            dtype=float,
        )
        return Atoms(symbols=s, positions=pos)
    return None
