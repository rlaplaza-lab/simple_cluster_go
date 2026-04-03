"""ASE constraints for adsorbates on a cluster."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from ase.constraints import FixBondLength


def attach_fix_bond_lengths(
    atoms: Atoms,
    bond_pairs: Sequence[tuple[int, int]],
) -> None:
    """Append one :class:`~ase.constraints.FixBondLength` per pair (global indices).

    Args:
        atoms: Full system (core + adsorbate).
        bond_pairs: Pairs of atom indices in ``atoms`` whose distances to hold fixed.

    Raises:
        ValueError: Invalid indices or duplicate pair.
    """
    n = len(atoms)
    seen: set[tuple[int, int]] = set()
    new_constraints: list = list(atoms.constraints) if atoms.constraints else []
    for a, b in bond_pairs:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(
                f"Invalid bond pair ({a}, {b}) for structure with {n} atoms"
            )
        if a == b:
            raise ValueError(f"bond pair ({a}, {b}) must be two distinct atoms")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"duplicate bond pair {key}")
        seen.add(key)
        new_constraints.append(FixBondLength(int(a), int(b)))
    if new_constraints:
        atoms.set_constraint(new_constraints)


def attach_oh_bond_constraint(atoms: Atoms, o_index: int, h_index: int) -> None:
    """Append :class:`~ase.constraints.FixBondLength` between O and H.

    Args:
        atoms: Structure containing O and H at the given indices.
        o_index: Index of the oxygen atom.
        h_index: Index of the hydrogen atom.

    Raises:
        ValueError: Invalid indices or wrong chemical symbols.
    """
    n = len(atoms)
    if not (0 <= o_index < n and 0 <= h_index < n):
        raise ValueError(
            f"Invalid O/H indices o_index={o_index}, h_index={h_index} for n_atoms={n}"
        )
    if o_index == h_index:
        raise ValueError("o_index and h_index must differ")
    sym = atoms.get_chemical_symbols()
    if sym[o_index] != "O":
        raise ValueError(f"Atom at o_index={o_index} must be O, got {sym[o_index]!r}")
    if sym[h_index] != "H":
        raise ValueError(f"Atom at h_index={h_index} must be H, got {sym[h_index]!r}")

    attach_fix_bond_lengths(atoms, [(o_index, h_index)])
