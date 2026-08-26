"""TorchSim-side constraint implementations not provided by upstream torch-sim.

torch-sim ships ``FixAtoms``, ``FixCom`` and ``FixSymmetry`` but no bond-length
constraint. SCGO needs to honor ASE ``FixBondLengths`` during batched TorchSim
relaxation, so this module provides a TorchSim ``Constraint`` that restores
fixed bond lengths after every optimizer step, mirroring the ASE
``FixBondLengths`` semantics (project positions and forces so the constrained
relative distances are preserved).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixBondLengths as ASEFixBondLengths
from torch_sim.constraints import Constraint
from torch_sim.transforms import minimum_image_displacement

from scgo.exceptions import SCGOValidationError

__all__ = [
    "TorchSimFixBondLengths",
    "build_torchsim_fixbondlengths_from_ase_batch",
    "collect_ase_fixbondlengths",
]


def _mask_constraint_indices(idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renumber surviving indices into a dense ``[0, n)`` range after a keep-mask.

    Same algorithm as ``torch_sim.constraints._mask_constraint_indices``. Copied
    because torch-sim remaps ``atom_idx`` / ``system_idx`` after ``select_constraint``
    but never a ``pairs`` tensor.
    """
    dropped_before = torch.cumsum(~mask, dim=0)
    remapped = idx - dropped_before[idx]
    keep = torch.isin(idx, torch.where(mask)[0])
    return remapped[keep]


def _as_tensor(
    value: torch.Tensor | Sequence | np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert lists/arrays/tensors to ``dtype`` without forcing a host numpy copy.

    ``np.asarray(cuda_tensor)`` calls ``Tensor.numpy()`` and raises on CUDA. torch-sim
    moves constraints via ``Constraint.to(device)`` during ``initialize_state``, so
    constructors must accept already-on-device tensors.
    """
    if isinstance(value, torch.Tensor):
        out = value.detach().to(dtype=dtype)
    else:
        out = torch.as_tensor(value, dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


# Match ``normalize_slab_pbc`` / ASE degenerate-vector cutoff.
_LATTICE_ZERO = 1e-8


def _pbc_mask(pbc: torch.Tensor | bool, *, device: torch.device) -> torch.Tensor:
    """Broadcast a scalar or length-3 PBC flag to a ``(3,)`` bool tensor."""
    if isinstance(pbc, bool):
        return torch.tensor([pbc, pbc, pbc], dtype=torch.bool, device=device)
    mask = pbc.to(device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() == 1:
        return mask.expand(3)
    return mask[:3]


def _complete_cell_ase_rows(cell: torch.Tensor) -> torch.Tensor:
    """Torch port of ``ase.geometry.cell.complete_cell`` (rows = lattice vectors).

    ASE ``find_mic`` calls this before wrapping so mixed-PBC cells (zero vacuum
    vector) are a right-handed 3x3. One missing vector is replaced by the unit
    cross product of the other two; two missing vectors use an SVD completion;
    three missing vectors become the identity.
    """
    missing = torch.linalg.norm(cell, dim=1) <= _LATTICE_ZERO
    n_missing = int(missing.sum().item())
    if n_missing == 0:
        return cell
    if n_missing == 3:
        return torch.eye(3, dtype=cell.dtype, device=cell.device)
    out = cell.clone()
    if n_missing == 1:
        i = int(torch.nonzero(missing, as_tuple=False)[0, 0].item())
        dummy = torch.linalg.cross(out[i - 2], out[i - 1])
        out[i] = dummy / torch.linalg.norm(dummy)
        return out
    V, s, wt = torch.linalg.svd(out.T, full_matrices=True)
    scale = torch.diag(torch.stack((s[0], s.new_tensor(1.0), s.new_tensor(1.0))))
    completed = (V @ scale @ wt).T
    i0 = int(torch.nonzero(missing, as_tuple=False)[0, 0].item())
    if torch.linalg.det(completed) < 0:
        completed[i0] = -completed[i0]
    return completed


def _complete_cell(cell: torch.Tensor) -> torch.Tensor:
    """ASE ``complete_cell`` in TorchSim convention (columns = lattice vectors).

    Metatomic/vesin zeros non-periodic lattice vectors (see
    ``_prepare_atoms_for_metatomic_torchsim``). Those become zero *columns*
    here. Completing a copy for MIC leaves ``state.cell`` singular for the
    model, which is the metatomic requirement.
    """
    return _complete_cell_ase_rows(cell.T).T


def _minimum_image_displacement(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | bool,
) -> torch.Tensor:
    """Minimum-image wrap matching ASE ``find_mic`` on a metatomic-zeroed cell.

    ASE does ``pbc = cell.any(axis=1) & pbc`` then ``complete_cell`` then wrap.
    Cartesian dummy axes (e.g. ``e_z``) are not equivalent for skewed slabs.
    """
    pbc_mask = _pbc_mask(pbc, device=dr.device)
    present = torch.linalg.norm(cell, dim=0) > _LATTICE_ZERO
    return minimum_image_displacement(
        dr=dr,
        cell=_complete_cell(cell),
        pbc=pbc_mask & present,
    )


class TorchSimFixBondLengths(Constraint):
    """TorchSim constraint that keeps selected interatomic distances fixed.

    Unlike :class:`torch_sim.constraints.FixAtoms`, which pins fixed atoms to
    their current positions, this constraint restores each listed bond to its
    target length after every proposed position update. Both atoms of a bond are
    displaced by half the required correction along the bond axis, which keeps
    the center of mass of the diatomic pair stationary (no spurious translation
    is introduced). The corresponding force component along the bond axis is
    removed from both atoms so the optimizer does not keep fighting the
    constraint.
    """

    def __init__(
        self,
        pairs: torch.Tensor | list[list[int]],
        bond_lengths: torch.Tensor | list[float],
        system_idx: torch.Tensor | list[int] | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        pairs_t = _as_tensor(pairs, dtype=torch.long, device=device)
        if pairs_t.ndim != 2 or pairs_t.shape[1] != 2:
            raise SCGOValidationError(
                "FixBondLengths pairs must have shape (n_bonds, 2), "
                f"got {tuple(pairs_t.shape)}"
            )
        self.pairs = pairs_t
        self.bond_lengths = _as_tensor(bond_lengths, dtype=torch.float64, device=device)
        if self.bond_lengths.shape[0] != self.pairs.shape[0]:
            raise SCGOValidationError(
                "FixBondLengths bond_lengths must have one entry per pair "
                f"({self.pairs.shape[0]}), got {self.bond_lengths.shape[0]}"
            )
        if system_idx is None:
            self.system_idx = torch.zeros(
                self.pairs.shape[0], dtype=torch.long, device=self.pairs.device
            )
        else:
            self.system_idx = _as_tensor(system_idx, dtype=torch.long, device=device)

    def get_removed_dof(self, state: object) -> torch.Tensor:
        """One degree of freedom removed per constrained bond, per system."""
        counts = torch.zeros(state.n_systems, dtype=torch.long, device=state.device)
        if self.pairs.shape[0] == 0:
            return counts
        idx = self.system_idx.to(device=state.device)
        counts.index_add_(
            0,
            idx,
            torch.ones(idx.shape[0], dtype=counts.dtype, device=counts.device),
        )
        return counts

    def _bond_delta(
        self, positions: torch.Tensor, state: object, k: int
    ) -> tuple[int, int, torch.Tensor]:
        """MIC displacement from atom i to j for bond ``k``."""
        i = int(self.pairs[k, 0])
        j = int(self.pairs[k, 1])
        delta = _minimum_image_displacement(
            dr=(positions[j] - positions[i]).unsqueeze(0),
            cell=state.cell[int(self.system_idx[k])],
            pbc=state.pbc,
        ).squeeze(0)
        return i, j, delta

    def adjust_positions(self, state: object, new_positions: torch.Tensor) -> None:
        """Pull each constrained bond back to its target length."""
        for k in range(self.pairs.shape[0]):
            i, j, d = self._bond_delta(new_positions, state, k)
            dist = torch.linalg.norm(d)
            if dist <= 1e-12:
                continue
            correction = 0.5 * (dist - float(self.bond_lengths[k])) * (d / dist)
            new_positions[i] = new_positions[i] + correction
            new_positions[j] = new_positions[j] - correction

    def adjust_forces(self, state: object, forces: torch.Tensor) -> None:
        """Remove the bond-stretching component of the relative force."""
        for k in range(self.pairs.shape[0]):
            i, j, d = self._bond_delta(state.positions, state, k)
            dist = torch.linalg.norm(d)
            if dist <= 1e-12:
                continue
            direction = d / dist
            parallel = torch.dot(forces[j] - forces[i], direction) * direction
            forces[i] = forces[i] + 0.5 * parallel
            forces[j] = forces[j] - 0.5 * parallel

    def select_constraint(
        self,
        atom_mask: torch.Tensor,
        system_mask: torch.Tensor,
    ) -> Constraint | None:
        """Keep surviving bonds and pack pair/system indices into the filtered state."""
        if self.pairs.shape[0] == 0:
            return None
        if atom_mask.device != self.pairs.device:
            atom_mask = atom_mask.to(self.pairs.device)
        if system_mask.device != self.system_idx.device:
            system_mask = system_mask.to(self.system_idx.device)
        keep = atom_mask[self.pairs].all(dim=1) & system_mask[self.system_idx]
        if not keep.any():
            return None
        packed_pairs = _mask_constraint_indices(
            self.pairs[keep].reshape(-1), atom_mask
        ).reshape(-1, 2)
        packed_systems = _mask_constraint_indices(self.system_idx[keep], system_mask)
        return type(self)(
            packed_pairs,
            self.bond_lengths[keep],
            packed_systems,
            device=self.pairs.device,
        )

    def select_sub_constraint(
        self, atom_idx: torch.Tensor, sys_idx: int
    ) -> Constraint | None:
        """Keep bonds for ``sys_idx`` and renumber atoms to local indices."""
        if self.pairs.shape[0] == 0:
            return None
        if atom_idx.device != self.pairs.device:
            atom_idx = atom_idx.to(self.pairs.device)
        sys_mask = self.system_idx == sys_idx
        pairs = self.pairs[sys_mask]
        if pairs.shape[0] == 0:
            return None
        present = torch.isin(pairs, atom_idx).all(dim=1)
        pairs = pairs[present]
        if pairs.shape[0] == 0:
            return None
        local_index = {int(a): k for k, a in enumerate(atom_idx.tolist())}
        local_pairs = torch.tensor(
            [[local_index[int(a)], local_index[int(b)]] for a, b in pairs.tolist()],
            dtype=torch.long,
            device=self.pairs.device,
        )
        return type(self)(
            local_pairs,
            self.bond_lengths[sys_mask][present],
            torch.zeros(pairs.shape[0], dtype=torch.long),
            device=self.pairs.device,
        )

    def reindex(self, atom_offset: int, system_offset: int) -> TorchSimFixBondLengths:
        """Return a copy with global atom/system indices shifted."""
        return type(self)(
            self.pairs + atom_offset,
            self.bond_lengths,
            self.system_idx + system_offset,
            device=self.pairs.device,
        )

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> TorchSimFixBondLengths:
        """Merge already-reindexed bond constraints into one."""
        bond_constraints = [c for c in constraints if isinstance(c, cls)]
        if not bond_constraints:
            raise SCGOValidationError(
                f"{cls.__name__}.merge requires at least one {cls.__name__}."
            )
        device = bond_constraints[0].pairs.device
        pairs = torch.cat([c.pairs for c in bond_constraints])
        lengths = torch.cat([c.bond_lengths for c in bond_constraints])
        systems = torch.cat([c.system_idx for c in bond_constraints])
        return cls(pairs, lengths, systems, device=device)

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TorchSimFixBondLengths:
        """Return a copy with all internal tensors moved to *device*/*dtype*."""
        return type(self)(
            self.pairs.to(device=device),
            self.bond_lengths.to(device=device, dtype=dtype),
            self.system_idx.to(device=device),
            device=device,
        )


def collect_ase_fixbondlengths(
    atoms: Atoms, offset: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Extract ASE ``FixBondLengths`` pairs/lengths with global atom indices.

    Args:
        atoms: ASE structure (single system).
        offset: Atom-index offset for batching into a global index space.

    Returns:
        A list of ``(pairs, lengths, system_index)`` tuples, one per
        ``FixBondLengths`` constraint found on ``atoms``. ``pairs`` is a
        ``(n_bonds, 2)`` long tensor of global atom indices and ``lengths`` is a
        ``(n_bonds,)`` float64 tensor of target bond lengths (taken from the
        constraint's cached ``bondlengths`` when present, else measured from the
        current geometry).
    """
    out: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for sys_id, constraint in enumerate(atoms.constraints):
        if not isinstance(constraint, ASEFixBondLengths):
            continue
        pairs = np.asarray(constraint.pairs, dtype=np.int64) + offset
        if getattr(constraint, "bondlengths", None) is not None:
            lengths = np.asarray(constraint.bondlengths, dtype=np.float64)
        else:
            lengths = np.array(
                [atoms.get_distance(a, b, mic=True) for a, b in pairs - offset],
                dtype=np.float64,
            )
        out.append(
            (
                torch.as_tensor(pairs),
                torch.as_tensor(lengths),
                sys_id,
            )
        )
    return out


def build_torchsim_fixbondlengths_from_ase_batch(
    atoms_list: Sequence[Atoms],
    device: object,
) -> TorchSimFixBondLengths | None:
    """Map per-structure ASE ``FixBondLengths`` to one global TorchSim constraint.

    Mirrors :func:`build_torchsim_fixatoms_from_ase_batch`: torch-sim does not
    read ``atoms.constraints`` so SCGO builds the TorchSim constraint explicitly
    and attaches it to the ``SimState`` before calling ``ts.optimize``.

    Args:
        atoms_list: One or more ASE systems in batch order.
        device: ``torch.device`` for the index/length tensors.

    Returns:
        A :class:`TorchSimFixBondLengths` instance, or ``None`` if no
        ``FixBondLengths`` constraints are present.
    """
    pairs_all: list[torch.Tensor] = []
    lengths_all: list[torch.Tensor] = []
    systems_all: list[torch.Tensor] = []
    offset = 0
    for batch_idx, atoms in enumerate(atoms_list):
        for pairs, lengths, _sys_id in collect_ase_fixbondlengths(atoms, offset=offset):
            pairs_all.append(pairs)
            lengths_all.append(lengths)
            systems_all.append(
                torch.full((pairs.shape[0],), batch_idx, dtype=torch.long)
            )
        offset += len(atoms)
    if not pairs_all:
        return None
    return TorchSimFixBondLengths(
        torch.cat(pairs_all),
        torch.cat(lengths_all),
        torch.cat(systems_all),
        device=torch.device(device) if device is not None else None,
    )
