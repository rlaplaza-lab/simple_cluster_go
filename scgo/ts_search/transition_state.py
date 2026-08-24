"""Utilities for finding transition states with NEB and path interpolation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

from scgo.calculators import torchsim_helpers as _tsh
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_FMAX_THRESHOLD,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_TS_PAIR_COR_MAX,
)
from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.metadata.atoms import get_tag, set_tags
from scgo.metadata.provenance import is_cuda_oom_error, output_json_provenance
from scgo.system_types import SystemType, get_system_policy
from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_shared_mobile_atom_indices,
)
from scgo.utils.helpers import copy_atoms, extract_energy_from_atoms
from scgo.utils.logging import (
    get_logger,
    log_debug_v,
    log_warning_v,
)
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.timing_report import (
    build_timing_payload,
    log_timing_summary,
    write_timing_file,
)
from scgo.utils.torchsim_policy import (
    _require_torchsim,
    _require_torchsim_fairchem,
    _require_torchsim_upet,
    is_uma_like_calculator,
    is_upet_like_calculator,
)
from scgo.utils.ts_runner_kwargs import NebRunConfig
from scgo.utils.validation import validate_atoms, validate_calculator_attached

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

# Pre-NEB IDPP and finalize share this cap for discontinuous / unphysical barriers.
# It is a hard discontinuity guard for NEB profiles whose barrier is an artifact
# of image drift / endpoint mismatch rather than a real transition, NOT a physical
# barrier scale. It is tunable via ``neb_max_spurious_barrier`` and is only one of
# several TS-quality gates: interior-image clash, saddle prominence and
# endpoint-drift checks also reject unphysical NEB results.
MAX_SPURIOUS_NEB_BARRIER_EV: float = 8.0


def _detach_calc(atoms: Atoms | None) -> None:
    """Remove calculator from structure when present."""
    if atoms is None:
        return
    atoms.calc = None


def neb_max_atom_force(neb_forces: np.ndarray | list[float]) -> float:
    """ASE-compatible fmax: maximum per-atom Euclidean force norm."""
    forces = np.asarray(neb_forces, dtype=float).reshape(-1, 3)
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces, axis=1).max())


def attach_singlepoint_from_relax_output(
    atoms: Atoms,
    energy: float,
    relaxed_atoms: Atoms,
    *,
    require_forces: bool = True,
) -> None:
    """Attach ``SinglePointCalculator`` to ``atoms`` from one ``relax_batch`` result.

    Also stores ``potential_energy`` in atoms metadata so barrier finalize can
    still read energies if ASE invalidates the SinglePoint after a FIRE step.
    """
    energy_f = float(energy)
    forces = relaxed_atoms.arrays.get("forces")
    if forces is None and relaxed_atoms.calc is not None:
        with contextlib.suppress(AttributeError, NotImplementedError):
            forces = relaxed_atoms.get_forces()
    if forces is not None and getattr(forces, "size", 0) > 0:
        atoms.calc = SinglePointCalculator(atoms, energy=energy_f, forces=forces)
        set_tags(atoms, potential_energy=energy_f, raw_score=-energy_f)
        return
    if require_forces:
        raise SCGORuntimeError(
            "TorchSim did not return forces. Ensure the model is loaded with compute_forces=True."
        )
    atoms.calc = SinglePointCalculator(atoms, energy=energy_f)
    set_tags(atoms, potential_energy=energy_f, raw_score=-energy_f)


def _image_potential_energy(atoms: Atoms) -> float:
    """Return image energy from calculator or cached metadata.

    After an ASE optimizer ``step()``, ``SinglePointCalculator`` raises
    ``PropertyNotImplementedError`` because positions changed. Metadata written
    by :func:`attach_singlepoint_from_relax_output` remains valid for that
    pre-step geometry; callers should refresh PES after the final step when
    possible so positions and energies stay consistent.
    """
    with contextlib.suppress(AttributeError, NotImplementedError, RuntimeError):
        return float(atoms.get_potential_energy())
    stored = get_tag(atoms, "potential_energy", default=None)
    if stored is not None:
        return float(stored)
    extracted = extract_energy_from_atoms(atoms)
    if extracted is not None:
        return float(extracted)
    raise SCGORuntimeError(
        'The property "energy" is not available on NEB image '
        "(no calculator energy and no cached potential_energy metadata)."
    )


def _image_has_cached_forces(img: Atoms) -> bool:
    """True when ``img`` already carries PES forces (array or calculator cache)."""
    if img.arrays.get("forces") is not None:
        return True
    calc = img.calc
    if calc is None:
        return False
    with contextlib.suppress(AttributeError, NotImplementedError, RuntimeError):
        return calc.get_forces(img) is not None
    return False


def calculate_structure_similarity(
    atoms1: Atoms,
    atoms2: Atoms,
    tolerance: float = DEFAULT_COMPARATOR_TOL,
    pair_cor_max: float = DEFAULT_TS_PAIR_COR_MAX,
    *,
    ignore_fixed_atoms: bool = True,
    use_mic: bool = False,
    n_slab: int | None = None,
    comparator: PureInteratomicDistanceComparator | None = None,
) -> tuple[float, float, bool]:
    """Return ``(cum_diff, max_diff, are_similar)`` for two Atoms objects.

    Raises:
        SCGOValidationError: If the two structures have different atom counts.
    """
    if len(atoms1) != len(atoms2):
        raise SCGOValidationError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    if ignore_fixed_atoms:
        comparison_indices = get_shared_mobile_atom_indices(
            atoms1,
            atoms2,
            n_slab=n_slab,
        )
    else:
        comparison_indices = np.arange(len(atoms1), dtype=int)
    atoms1_cmp = atoms1[comparison_indices]
    atoms2_cmp = atoms2[comparison_indices]

    if comparator is None:
        comparator = PureInteratomicDistanceComparator(
            n_top=len(atoms1_cmp),
            tol=tolerance,
            pair_cor_max=pair_cor_max,
            mic=use_mic,
        )
    elif comparator.n_top != len(atoms1_cmp):
        # Mobile count changed; fall back to a matching comparator.
        comparator = PureInteratomicDistanceComparator(
            n_top=len(atoms1_cmp),
            tol=tolerance,
            pair_cor_max=pair_cor_max,
            mic=use_mic,
        )

    cum_diff, max_diff = comparator.get_differences(atoms1_cmp, atoms2_cmp)
    are_similar = comparator.looks_like(atoms1_cmp, atoms2_cmp)

    return cum_diff, max_diff, are_similar


class TorchSimNEB(NEB):
    """NEB that batches PES evaluations via TorchSim for GPU efficiency.

    Spring forces, climbing-image, and tangent method are ASE ``NEB`` physics.
    Only the per-image energy/force evaluation is replaced by a batched
    ``TorchSimBatchRelaxer.relax_batch(..., steps=0)`` single-point call
    (``torch_sim.static``), matching ASE calculator semantics at fixed positions.
    """

    def __init__(
        self,
        images: list[Atoms],
        relaxer: TorchSimBatchRelaxer,
        k: float | list[float] = 0.1,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        method: str = DEFAULT_NEB_TANGENT_METHOD,
    ):
        """Initialize NEB with images and a TorchSimBatchRelaxer."""
        super().__init__(
            images,
            k=k,
            climb=climb,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            method=method,
        )
        self.relaxer = relaxer
        self._force_calls = 0
        # Set by ``ParallelNEBBatch``: the batch runner counts one force call per
        # batched ``relax_batch`` the band participates in, so ``get_forces`` must
        # not double-count on top of that (see B2). The serial fallback leaves
        # this False and keeps owning the counter itself.
        self._force_calls_counted_externally = False

    def get_forces(self) -> np.ndarray:
        """Batch-evaluate PES forces with TorchSim and return NEB forces.

        When images already carry PES forces (for example because
        ``ParallelNEBBatch`` just evaluated them in a single batched call),
        reuse the cached arrays instead of re-invoking TorchSim.
        """
        if all(_image_has_cached_forces(img) for img in self.images):
            return super().get_forces()

        if not self._force_calls_counted_externally:
            self._force_calls += 1
        results = self.relaxer.relax_batch(self.images, steps=0)

        for atoms, (energy, relaxed_atoms) in zip(self.images, results, strict=True):
            attach_singlepoint_from_relax_output(
                atoms, energy, relaxed_atoms, require_forces=True
            )

        return super().get_forces()

    def get_force_calls(self) -> int:
        """Return the number of times forces have been evaluated."""
        return self._force_calls


def _local_distance_fingerprints(atoms: Atoms) -> np.ndarray:
    """Return per-atom sorted distance fingerprint (shape: n_atoms x (n_atoms-1)).

    The fingerprint is used only for robust endpoint atom matching and is
    intentionally simple and deterministic (no RNG).
    """
    pos = atoms.get_positions()
    n = len(atoms)
    fp = np.zeros((n, max(0, n - 1)), dtype=float)
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d = np.delete(d, i)
        d.sort()
        if d.size > 0:
            fp[i, : d.size] = d
    return fp


def _local_distance_fingerprints_mic(
    atoms: Atoms,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> np.ndarray:
    """MIC-aware distance fingerprints for periodic endpoint matching."""
    pos = atoms.get_positions()
    n = len(atoms)
    fp = np.zeros((n, max(0, n - 1)), dtype=float)
    for i in range(n):
        disp = pos - pos[i]
        disp_mic, _ = find_mic(disp, cell=cell, pbc=pbc)
        d = np.linalg.norm(disp_mic, axis=1)
        d = np.delete(d, i)
        d.sort()
        if d.size > 0:
            fp[i, : d.size] = d
    return fp


def _mic_matching_context(
    reactant: Atoms,
    *,
    n_slab: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (cell, pbc) for MIC-aware fingerprint matching, or (None, None)."""
    if not _requires_surface_pbc_alignment(reactant, n_slab=n_slab):
        return None, None
    return _cell_array(reactant.cell), _pbc_for_mic_alignment(reactant.pbc)


def _match_atoms_by_fingerprint(
    a1: Atoms,
    a2: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> list[int]:
    """Return mapping such that mapped_idx[i] is index in `a2` matching atom i in `a1`.

    Uses per-atom local-distance fingerprints and the Hungarian algorithm to
    obtain a permutation that is robust to rotations and permutations.
    When ``mic_cell`` and ``mic_pbc`` are set, fingerprints use minimum-image
    distances (required for slab endpoints near periodic boundaries).
    """
    if len(a1) != len(a2):
        raise SCGOValidationError("Atoms objects have different lengths")

    mapping = [-1] * len(a1)
    use_mic = mic_cell is not None and mic_pbc is not None
    if use_mic:
        fp1_all = _local_distance_fingerprints_mic(a1, mic_cell, mic_pbc)
        fp2_all = _local_distance_fingerprints_mic(a2, mic_cell, mic_pbc)
    else:
        fp1_all = _local_distance_fingerprints(a1)
        fp2_all = _local_distance_fingerprints(a2)
    # Match separately for each atomic number (handles mixed-species clusters)
    for z in set(a1.numbers):
        idx1 = [i for i, x in enumerate(a1.numbers) if x == z]
        idx2 = [i for i, x in enumerate(a2.numbers) if x == z]
        if len(idx1) != len(idx2):
            raise SCGOValidationError("Composition mismatch during endpoint matching")

        fp1 = fp1_all[idx1]
        fp2 = fp2_all[idx2]
        # Cost = L2 distance between fingerprints
        cost = np.linalg.norm(fp1[:, None, :] - fp2[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        for ri, ci in zip(r, c, strict=False):
            mapping[idx1[ri]] = idx2[ci]

    return mapping


def _core_block_match_method(n_slab: int) -> str:
    """Fingerprint gas cores (rotation-robust); spatial match slab cores (lab frame)."""
    return "spatial" if int(n_slab) > 0 else "fingerprint"


def _permute_atoms_block_to_match(
    a1_block: Atoms,
    a2_block: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
    method: str = "fingerprint",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, atomic_numbers) for a2_block permuted to match a1_block.

    ``method``:
    - ``"fingerprint"``: local-distance fingerprints (rotation-robust).
    - ``"spatial"``: Hungarian on Cartesian (MIC) distances — prefers minimal travel
      for NEB cores that are already roughly aligned.
    """
    if method == "spatial":
        return _permute_atoms_block_spatially(
            a1_block, a2_block, mic_cell=mic_cell, mic_pbc=mic_pbc
        )
    if method != "fingerprint":
        raise SCGOValidationError(f"Unknown block permute method: {method!r}")
    mapping = _match_atoms_by_fingerprint(
        a1_block, a2_block, mic_cell=mic_cell, mic_pbc=mic_pbc
    )
    pos2 = a2_block.get_positions()
    nums2 = a2_block.numbers
    return pos2[mapping], nums2[mapping]


def _permute_atoms_block_spatially(
    a1_block: Atoms,
    a2_block: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute a2 onto a1 by minimizing per-species Cartesian (MIC) travel."""
    if len(a1_block) != len(a2_block):
        raise SCGOValidationError("spatial block match: length mismatch")
    pos1 = a1_block.get_positions()
    pos2 = a2_block.get_positions()
    nums1 = a1_block.numbers
    nums2 = a2_block.numbers
    mapping = [-1] * len(a1_block)
    use_mic = mic_cell is not None and mic_pbc is not None and bool(np.any(mic_pbc))
    for z in set(nums1.tolist()):
        idx1 = [i for i, x in enumerate(nums1) if int(x) == int(z)]
        idx2 = [i for i, x in enumerate(nums2) if int(x) == int(z)]
        if len(idx1) != len(idx2):
            raise SCGOValidationError("spatial block match: composition mismatch")
        p1 = pos1[idx1]
        p2 = pos2[idx2]
        cost = np.zeros((len(idx1), len(idx2)), dtype=float)
        for i, r in enumerate(p1):
            dlt = p2 - r
            if use_mic:
                dlt, _ = find_mic(dlt, mic_cell, mic_pbc)
            cost[i, :] = np.linalg.norm(dlt, axis=1)
        rows, cols = linear_sum_assignment(cost)
        for ri, ci in zip(rows, cols, strict=True):
            mapping[idx1[ri]] = idx2[ci]
    order = np.asarray(mapping, dtype=int)
    return pos2[order], nums2[order]


def _match_adsorbate_fragments_by_com(
    a1_ads: Atoms,
    a2_ads: Atoms,
    fragment_lengths: list[int],
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute product adsorbate atoms so fragments match reactant by COM, then fingerprint."""
    n_ads = len(a1_ads)
    if len(a2_ads) != n_ads:
        raise SCGOValidationError("adsorbate fragment match: length mismatch")
    if sum(int(x) for x in fragment_lengths) != n_ads:
        raise SCGOValidationError(
            "adsorbate_fragment_lengths must sum to adsorbate atom count "
            f"(sum={sum(int(x) for x in fragment_lengths)}, n_ads={n_ads})"
        )
    if any(int(x) <= 0 for x in fragment_lengths):
        raise SCGOValidationError("adsorbate_fragment_lengths must be positive")

    # Single fragment: fall back to ordinary block matching.
    if len(fragment_lengths) <= 1:
        return _permute_atoms_block_to_match(
            a1_ads, a2_ads, mic_cell=mic_cell, mic_pbc=mic_pbc
        )

    # Build equal-length fragment groups only when all fragments share a length
    # (e.g. 2×OH). Mixed lengths are matched greedily by composition.
    r_pos = a1_ads.get_positions()
    p_pos = a2_ads.get_positions()
    r_num = a1_ads.numbers
    p_num = a2_ads.numbers

    frag_slices: list[slice] = []
    off = 0
    for fl in fragment_lengths:
        frag_slices.append(slice(off, off + int(fl)))
        off += int(fl)
    r_slices = p_slices = frag_slices

    r_coms = [r_pos[s].mean(axis=0) for s in r_slices]

    n_frag = len(fragment_lengths)
    cost = np.full((n_frag, n_frag), np.inf, dtype=float)
    for i, rs in enumerate(r_slices):
        for j, ps in enumerate(p_slices):
            if int(fragment_lengths[i]) != int(fragment_lengths[j]):
                continue
            if sorted(r_num[rs].tolist()) != sorted(p_num[ps].tolist()):
                continue
            dlt = p_pos[ps].mean(axis=0) - r_coms[i]
            if mic_cell is not None and mic_pbc is not None and bool(np.any(mic_pbc)):
                dlt, _ = find_mic(dlt.reshape(1, 3), mic_cell, mic_pbc)
                dlt = np.asarray(dlt, dtype=float).reshape(3)
            cost[i, j] = float(np.linalg.norm(dlt))

    if not np.isfinite(cost).any():
        # Composition/length mismatch across fragments: whole-block fallback.
        return _permute_atoms_block_to_match(
            a1_ads, a2_ads, mic_cell=mic_cell, mic_pbc=mic_pbc
        )

    rows, cols = linear_sum_assignment(cost)
    if not np.isfinite(cost[rows, cols]).all():
        return _permute_atoms_block_to_match(
            a1_ads, a2_ads, mic_cell=mic_cell, mic_pbc=mic_pbc
        )

    out_pos = np.empty_like(p_pos)
    out_num = np.empty_like(p_num)
    for i, j in zip(rows, cols, strict=True):
        rs, ps = r_slices[i], p_slices[j]
        p_blk, n_blk = _permute_atoms_block_to_match(
            a1_ads[rs], a2_ads[ps], mic_cell=mic_cell, mic_pbc=mic_pbc
        )
        out_pos[rs] = p_blk
        out_num[rs] = n_blk
    return out_pos, out_num


def _assign_atom_block(
    atoms: Atoms,
    start: int,
    stop: int,
    positions: np.ndarray,
    numbers: np.ndarray,
) -> None:
    """Write a contiguous position/number slice onto ``atoms`` in-place."""
    pos = atoms.get_positions().copy()
    nums = atoms.numbers.copy()
    pos[start:stop] = positions
    nums[start:stop] = numbers
    atoms.set_positions(pos, apply_constraint=False)
    atoms.numbers = nums


def _match_core_block(
    a1: Atoms,
    a2: Atoms,
    n_slab: int,
    n_core: int,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> None:
    """Permute product core atoms onto the reactant core (in-place)."""
    if n_core <= 0:
        return
    s1, s2 = int(n_slab), int(n_slab) + int(n_core)
    p_blk, n_blk = _permute_atoms_block_to_match(
        a1[s1:s2],
        a2[s1:s2],
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
        method=_core_block_match_method(n_slab),
    )
    _assign_atom_block(a2, s1, s2, p_blk, n_blk)


def _match_adsorbate_block(
    a1: Atoms,
    a2: Atoms,
    n_slab: int,
    n_core: int,
    n_ads: int,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
) -> None:
    """Permute product adsorbate atoms onto the reactant adsorbate (in-place)."""
    if n_ads <= 0:
        return
    t1 = int(n_slab) + int(n_core)
    t2 = t1 + int(n_ads)
    if adsorbate_fragment_lengths:
        p_blk, n_blk = _match_adsorbate_fragments_by_com(
            a1[t1:t2],
            a2[t1:t2],
            list(adsorbate_fragment_lengths),
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
        )
    else:
        p_blk, n_blk = _permute_atoms_block_to_match(
            a1[t1:t2], a2[t1:t2], mic_cell=mic_cell, mic_pbc=mic_pbc
        )
    _assign_atom_block(a2, t1, t2, p_blk, n_blk)


def _align_endpoints_blockwise(
    a1: Atoms,
    a2: Atoms,
    n_slab: int,
    n_core: int,
    n_ads: int,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    match_adsorbate: bool = True,
) -> None:
    """Match product to reactant per block (slab indices unchanged).

    Gas cores use fingerprints; slab cores use spatial matching. Adsorbate
    matching can be deferred (``match_adsorbate=False``) until after rigid
    overlay so fragment COMs are assigned in the aligned frame.
    """
    n = len(a1)
    if len(a2) != n:
        raise SCGOValidationError("align blockwise: endpoint length mismatch")
    if n_slab + n_core + n_ads != n:
        raise SCGOValidationError(
            f"align blockwise: n_slab+n_core+n_ads={n_slab + n_core + n_ads} != len={n}"
        )
    _match_core_block(a1, a2, n_slab, n_core, mic_cell=mic_cell, mic_pbc=mic_pbc)
    if match_adsorbate:
        _match_adsorbate_block(
            a1,
            a2,
            n_slab,
            n_core,
            n_ads,
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
            adsorbate_fragment_lengths=adsorbate_fragment_lengths,
        )


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return rotation matrix R that minimizes ||P - Q @ R|| (P and Q are centered)."""
    dim = int(P.shape[1])
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.ones(dim, dtype=float)
    d[-1] = float(np.sign(np.linalg.det(U @ Vt)) or 1.0)
    return U @ np.diag(d) @ Vt


def _kabsch_rotation_in_plane(
    P: np.ndarray, Q: np.ndarray, *, surface_normal_axis: int = 2
) -> np.ndarray:
    """Return 3x3 rotation that aligns Q to P using only in-plane degrees of freedom."""
    if surface_normal_axis not in (0, 1, 2):
        raise SCGOValidationError("surface_normal_axis must be 0, 1, or 2")
    plane_axes = [i for i in range(3) if i != surface_normal_axis]
    r2 = _kabsch_rotation(P[:, plane_axes], Q[:, plane_axes])
    rot = np.eye(3)
    for i, ia in enumerate(plane_axes):
        for j, ja in enumerate(plane_axes):
            rot[ia, ja] = r2[i, j]
    return rot


def _infer_surface_normal_axis(pbc: np.ndarray | list[bool]) -> int:
    """Guess vacuum/normal axis as the sole non-periodic direction, else z."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    open_axes = [i for i in range(3) if not pbc_arr[i]]
    if len(open_axes) == 1:
        return int(open_axes[0])
    return 2


def _fixed_atom_mask(atoms: Atoms) -> np.ndarray:
    """Return a boolean mask for atoms fixed by ``FixAtoms`` constraints."""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            mask[idx] = True
    return mask


def _anchor_mask(
    atoms: Atoms,
    *,
    n_slab: int,
    fixed_mask: np.ndarray,
) -> np.ndarray:
    """Mask of atoms used to anchor periodic endpoint alignment (slab frame)."""
    n = len(atoms)
    if np.any(fixed_mask):
        return fixed_mask
    if n_slab > 0:
        anchor = np.zeros(n, dtype=bool)
        anchor[: min(n_slab, n)] = True
        return anchor
    return np.zeros(n, dtype=bool)


def _mobile_alignment_mask(
    anchor_mask: np.ndarray,
    *,
    n_slab: int,
    n_atoms: int,
) -> np.ndarray:
    """Atoms that may receive rigid alignment (not slab prefix, not anchored)."""
    mobile = np.ones(n_atoms, dtype=bool)
    if n_slab > 0:
        mobile[: min(n_slab, n_atoms)] = False
    mobile &= ~anchor_mask
    return mobile


def _cell_array(cell: Any) -> np.ndarray:
    """Return a 3x3 cell matrix from ASE ``Cell`` or ndarray."""
    if hasattr(cell, "array"):
        return np.asarray(cell.array, dtype=float)
    return np.asarray(cell, dtype=float)


def _pbc_for_mic_alignment(pbc: np.ndarray | list[bool]) -> np.ndarray:
    """PBC mask for MIC: in-plane periodic, vacuum axis open (slab convention)."""
    pbc_arr = np.asarray(pbc, dtype=bool).copy()
    normal_axis = _infer_surface_normal_axis(pbc_arr)
    pbc_arr[normal_axis] = False
    return pbc_arr


def _inplane_periodic_axes(pbc: np.ndarray | list[bool]) -> tuple[int, int]:
    """Return the two in-plane periodic axis indices for a slab-like cell."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    periodic = [i for i in range(3) if pbc_arr[i]]
    if len(periodic) == 2:
        return int(periodic[0]), int(periodic[1])
    return 0, 1


def _validate_lattice_compatible_rotation(
    rot: np.ndarray,
    normal_axis: int,
    *,
    tol: float = 1e-6,
) -> None:
    """Fail-fast when a rotation would alter the vacuum axis or handedness."""
    if normal_axis not in (0, 1, 2):
        raise SCGOValidationError("normal_axis must be 0, 1, or 2")
    if abs(float(rot[normal_axis, normal_axis]) - 1.0) > tol:
        raise SCGOValidationError(
            "Rotation must preserve the surface normal axis (energy-equivalent)."
        )
    for i in range(3):
        if i != normal_axis and abs(float(rot[normal_axis, i])) > tol:
            raise SCGOValidationError(
                "Rotation must not mix the surface normal with in-plane axes."
            )
    if abs(float(np.linalg.det(rot)) - 1.0) > tol:
        raise SCGOValidationError(
            "Rotation determinant must be +1 for rigid alignment."
        )


def _lattice_translation_candidates(
    cell: np.ndarray,
    axis_a: int,
    axis_b: int,
    *,
    max_shift: int = 1,
) -> list[np.ndarray]:
    """Integer in-plane lattice translations (Cartesian vectors)."""
    if max_shift < 0:
        raise SCGOValidationError("max_shift must be non-negative")
    candidates: list[np.ndarray] = []
    for nx in range(-max_shift, max_shift + 1):
        for ny in range(-max_shift, max_shift + 1):
            delta = nx * cell[axis_a] + ny * cell[axis_b]
            candidates.append(np.asarray(delta, dtype=float))
    return candidates


def _snap_to_reactant_mic_frame(
    ref_pos: np.ndarray,
    pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    anchor_mask: np.ndarray,
) -> np.ndarray:
    """Express ``pos`` in the reactant periodic image (Cartesian, MIC-short)."""
    disp_mic, _ = find_mic(pos - ref_pos, cell=cell, pbc=pbc)
    if np.any(anchor_mask):
        disp_mic = disp_mic - np.mean(disp_mic[anchor_mask], axis=0)
    snapped = ref_pos + disp_mic
    if np.any(anchor_mask):
        snapped[anchor_mask] = ref_pos[anchor_mask]
    return snapped


def _score_mobile_endpoint_displacement(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    mobile_mask: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> tuple[float, float]:
    """Return (max, rms) mobile-atom displacement norms in the reactant MIC frame."""
    if not np.any(mobile_mask):
        return 0.0, 0.0
    disp_mic, _ = find_mic(prod_pos - ref_pos, cell=cell, pbc=pbc)
    norms = np.linalg.norm(disp_mic[mobile_mask], axis=1)
    return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))


def _core_alignment_mask(
    *,
    n_slab: int,
    n_core_mobile: int | None,
    n_atoms: int,
    mobile_mask: np.ndarray,
) -> np.ndarray:
    """Mask of core atoms used to drive rigid/PBC alignment when adsorbate blocks exist."""
    if n_core_mobile is None or int(n_core_mobile) <= 0:
        return mobile_mask
    core = np.zeros(n_atoms, dtype=bool)
    i0 = max(0, int(n_slab))
    i1 = min(n_atoms, i0 + int(n_core_mobile))
    core[i0:i1] = True
    core &= mobile_mask
    if not np.any(core):
        return mobile_mask
    return core


def _collective_mobile_lattice_snap(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    mobile_mask: np.ndarray,
    *,
    axis_a: int,
    axis_b: int,
    max_shift: int,
    score_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pick a uniform in-plane lattice image for mobile atoms before per-atom MIC snap."""
    if not np.any(mobile_mask):
        return prod_pos
    rank_mask = mobile_mask if score_mask is None else score_mask

    best_pos = prod_pos.copy()
    best_score, _ = _score_mobile_endpoint_displacement(
        ref_pos, best_pos, rank_mask, cell, pbc
    )
    for shift in _lattice_translation_candidates(
        cell, axis_a, axis_b, max_shift=max_shift
    ):
        shifted = prod_pos.copy()
        shifted[mobile_mask] += shift
        score, _ = _score_mobile_endpoint_displacement(
            ref_pos, shifted, rank_mask, cell, pbc
        )
        if score < best_score:
            best_score = score
            best_pos = shifted
    return best_pos


def _apply_global_inplane_kabsch(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    fit_mask: np.ndarray,
    *,
    normal_axis: int,
    anchor_mask: np.ndarray,
    apply_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one global in-plane rotation derived from ``fit_mask`` Kabsch.

    Rotation is fit on ``fit_mask`` atoms and applied to ``apply_mask`` (default:
    all atoms except anchors). Adsorbate cores can therefore drive alignment
    without ads hops dragging the frame.
    """
    idx = np.where(fit_mask)[0]
    if idx.size < 2:
        return prod_pos
    center = ref_pos[idx].mean(axis=0)
    p_ref_c = ref_pos[idx] - center
    p_prod_c = prod_pos[idx] - center
    rot = _kabsch_rotation_in_plane(p_ref_c, p_prod_c, surface_normal_axis=normal_axis)
    _validate_lattice_compatible_rotation(rot, normal_axis)
    out = prod_pos.copy()
    move = apply_mask if apply_mask is not None else ~anchor_mask
    if np.any(move):
        out[move] = (prod_pos[move] - center) @ rot.T + center
    if np.any(anchor_mask):
        out[anchor_mask] = ref_pos[anchor_mask]
    return out


def _align_product_surface_pbc(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    enable_cell_remap: bool = True,
    enable_lattice_rotation: bool = True,
    max_lattice_shift: int = 1,
    n_core_mobile: int | None = None,
) -> np.ndarray:
    """Align product to reactant using MIC, lattice shifts, and global in-plane rotation.

    **Single surface NEB alignment entry point.** Serial (:func:`find_transition_state`),
    parallel (:func:`run_parallel_neb_search`), and :func:`interpolate_path` all route
    slab/periodic endpoint prep through this helper (not mobile-only Kabsch).

    Only energy-equivalent transforms are considered:
    - collective uniform in-plane lattice image for mobile atoms,
    - per-atom minimum-image wrapping,
    - integer in-plane lattice translations up to ``max_lattice_shift`` cells,
    - global in-plane rigid rotation (same ``R`` for all atoms; evaluated jointly
      with each shift candidate; anchors reset to reactant afterward).

    When ``n_core_mobile`` is set, lattice-image / rotation scoring uses the core
    block while transforms still apply to all mobile atoms.

    Does **not** rotate mobile atoms independently of the lattice frame.
    """
    ref_pos = reactant.get_positions()
    cell = _cell_array(reactant.cell)
    pbc_mic = _pbc_for_mic_alignment(reactant.pbc)
    normal_axis = _infer_surface_normal_axis(reactant.pbc)
    axis_a, axis_b = _inplane_periodic_axes(pbc_mic)

    fixed_mask = _fixed_atom_mask(reactant)
    anchor_mask = _anchor_mask(reactant, n_slab=n_slab, fixed_mask=fixed_mask)
    mobile_mask = _mobile_alignment_mask(
        anchor_mask, n_slab=n_slab, n_atoms=len(reactant)
    )
    score_mask = _core_alignment_mask(
        n_slab=n_slab,
        n_core_mobile=n_core_mobile,
        n_atoms=len(reactant),
        mobile_mask=mobile_mask,
    )

    prod = np.asarray(product_positions, dtype=float).copy()
    if enable_cell_remap:
        prod = _collective_mobile_lattice_snap(
            ref_pos,
            prod,
            cell,
            pbc_mic,
            mobile_mask,
            axis_a=axis_a,
            axis_b=axis_b,
            max_shift=max_lattice_shift,
            score_mask=score_mask,
        )

    prod = _snap_to_reactant_mic_frame(ref_pos, prod, cell, pbc_mic, anchor_mask)

    best_pos = prod.copy()
    best_score, _ = _score_mobile_endpoint_displacement(
        ref_pos, best_pos, score_mask, cell, pbc_mic
    )

    shifts = _lattice_translation_candidates(
        cell, axis_a, axis_b, max_shift=max_lattice_shift
    )
    if not enable_cell_remap:
        shifts = [np.zeros(3, dtype=float)]

    for shift in shifts:
        prod_shifted = prod + shift
        prod_snapped = _snap_to_reactant_mic_frame(
            ref_pos, prod_shifted, cell, pbc_mic, anchor_mask
        )
        candidates: list[tuple[float, np.ndarray]] = []
        score, _ = _score_mobile_endpoint_displacement(
            ref_pos, prod_snapped, score_mask, cell, pbc_mic
        )
        candidates.append((score, prod_snapped))

        if enable_lattice_rotation:
            prod_rot = _apply_global_inplane_kabsch(
                ref_pos,
                prod_snapped,
                score_mask,
                normal_axis=normal_axis,
                anchor_mask=anchor_mask,
                apply_mask=mobile_mask,
            )
            prod_rot_snapped = _snap_to_reactant_mic_frame(
                ref_pos, prod_rot, cell, pbc_mic, anchor_mask
            )
            score_rot, _ = _score_mobile_endpoint_displacement(
                ref_pos, prod_rot_snapped, score_mask, cell, pbc_mic
            )
            candidates.append((score_rot, prod_rot_snapped))

        for score_c, pos_c in candidates:
            if score_c < best_score:
                best_score = score_c
                best_pos = pos_c

    return _snap_to_reactant_mic_frame(ref_pos, best_pos, cell, pbc_mic, anchor_mask)


def _requires_surface_pbc_alignment(reactant: Atoms, *, n_slab: int) -> bool:
    """True when endpoint alignment must use lattice-compatible surface PBC logic.

    Slab prefixes (``n_slab > 0``) and slab-like 2D PBC use MIC / in-plane
    lattice alignment. A gas cluster in a 3D vacuum box (``n_slab == 0``,
    three periodic axes or none) uses 3D Kabsch even if ``pbc`` is set.
    """
    if int(n_slab) > 0:
        return True
    pbc = np.asarray(reactant.pbc, dtype=bool)
    return int(np.count_nonzero(pbc)) == 2


def _align_product_for_neb(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    surface_cell_remap: bool = True,
    surface_lattice_rotation: bool = True,
    surface_max_lattice_shift: int = 1,
    n_core_mobile: int | None = None,
) -> np.ndarray:
    """Single NEB endpoint rigid-alignment entry point (gas Kabsch or surface PBC)."""
    if _requires_surface_pbc_alignment(reactant, n_slab=n_slab):
        return _align_product_surface_pbc(
            reactant,
            product_positions,
            n_slab=n_slab,
            enable_cell_remap=surface_cell_remap,
            enable_lattice_rotation=surface_lattice_rotation,
            max_lattice_shift=surface_max_lattice_shift,
            n_core_mobile=n_core_mobile,
        )
    return _align_product_kabsch_to_reactant(
        reactant,
        product_positions,
        n_slab=n_slab,
        in_plane_only=False,
        n_core_mobile=n_core_mobile,
    )


def _align_product_kabsch_to_reactant(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    in_plane_only: bool = False,
    n_core_mobile: int | None = None,
) -> np.ndarray:
    """Rigidly align product to reactant (gas-phase clusters without periodic endpoints).

    When ``n_core_mobile`` is set, Kabsch is derived from the core block only and
    applied to all mobile atoms so adsorbate hops do not drag the frame.
    """
    if n_slab > 0:
        raise SCGORuntimeError(
            "Slab NEB endpoints must use _align_product_surface_pbc, not Kabsch-only alignment."
        )
    ref_pos = reactant.get_positions()
    fixed_mask = _fixed_atom_mask(reactant)
    anchor_mask = _anchor_mask(reactant, n_slab=n_slab, fixed_mask=fixed_mask)
    mobile_mask = _mobile_alignment_mask(
        anchor_mask, n_slab=n_slab, n_atoms=len(reactant)
    )
    fit_mask = _core_alignment_mask(
        n_slab=n_slab,
        n_core_mobile=n_core_mobile,
        n_atoms=len(reactant),
        mobile_mask=mobile_mask,
    )

    if np.any(mobile_mask):
        out = product_positions.copy()
        p_ref = ref_pos[fit_mask]
        if p_ref.shape[0] < 1:
            return product_positions
        center_ref = p_ref.mean(axis=0)
        center_prod = product_positions[fit_mask].mean(axis=0)
        p_ref_c = p_ref - center_ref
        p_prod_c = product_positions[fit_mask] - center_prod
        if in_plane_only:
            rot = _kabsch_rotation_in_plane(
                p_ref_c,
                p_prod_c,
                surface_normal_axis=_infer_surface_normal_axis(reactant.pbc),
            )
        elif p_ref.shape[0] >= 2:
            rot = _kabsch_rotation(p_ref_c, p_prod_c)
        else:
            rot = np.eye(3)
        # Apply the core-derived transform to all mobile atoms.
        out[mobile_mask] = (
            product_positions[mobile_mask] - center_prod
        ) @ rot.T + center_ref
        if np.any(anchor_mask):
            out[anchor_mask] = ref_pos[anchor_mask]
        return out

    p_ref = ref_pos
    p_prod = product_positions
    center_ref = p_ref.mean(axis=0)
    center_prod = p_prod.mean(axis=0)
    p_ref_c = p_ref - center_ref
    p_prod_c = p_prod - center_prod
    if in_plane_only:
        rot = _kabsch_rotation_in_plane(
            p_ref_c,
            p_prod_c,
            surface_normal_axis=_infer_surface_normal_axis(reactant.pbc),
        )
    else:
        rot = _kabsch_rotation(p_ref_c, p_prod_c)
    return (p_prod_c @ rot.T) + center_ref


def _rematch_gas_core_and_kabsch(
    reactant: Atoms,
    pos_j: np.ndarray,
    nums_j: np.ndarray,
    *,
    n_core: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial-rematch a gas core in the overlaid frame and re-Kabsch.

    Fingerprints are chirality-blind, so Hungarian can assign a reflected
    labeling on near-symmetric cores. Proper Kabsch cannot overlay that;
    Cartesian rematch recovers a proper labeling.
    """
    n_core = max(0, int(n_core))
    pos_j = np.asarray(pos_j, dtype=float).copy()
    nums_j = np.asarray(nums_j, dtype=int).copy()
    if n_core <= 0 or n_core > len(pos_j) or n_core > len(reactant):
        return pos_j, nums_j
    core_i = Atoms(
        numbers=np.asarray(reactant.numbers[:n_core], dtype=int),
        positions=np.asarray(reactant.get_positions()[:n_core], dtype=float),
        cell=reactant.cell,
        pbc=reactant.pbc,
    )
    core_j = Atoms(
        numbers=nums_j[:n_core],
        positions=pos_j[:n_core],
        cell=reactant.cell,
        pbc=reactant.pbc,
    )
    refined_pos, refined_nums = _permute_atoms_block_to_match(
        core_i, core_j, method="spatial"
    )
    pos_j[:n_core] = refined_pos
    nums_j[:n_core] = refined_nums
    return (
        _align_product_kabsch_to_reactant(
            reactant, pos_j, n_slab=0, n_core_mobile=n_core
        ),
        nums_j,
    )


def _overlay_product_core(
    atoms_i: Atoms,
    pos_j: np.ndarray,
    nums_j: np.ndarray,
    *,
    n_slab: int,
    n_core: int,
    mic_cell: np.ndarray | None,
    mic_pbc: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute product core onto reactant; Kabsch-overlay gas mobile atoms.

    Shared by TS pairing and NEB endpoint prep. Layout is
    ``[slab | core | adsorbate]``; adsorbate atoms ride the core transform and
    are not permuted here.

    Gas (``n_slab == 0``): fingerprint correspondence, core-derived rigid
    motion (translation if ``n_core == 1``), spatial rematch, re-Kabsch.
    Slab: spatial match in the lab frame; no 3D Kabsch.
    """
    pos_j = np.asarray(pos_j, dtype=float).copy()
    nums_j = np.asarray(nums_j, dtype=int).copy()
    n_slab = max(0, int(n_slab))
    n_core = max(0, int(n_core))
    if n_core <= 0 or n_slab + n_core > len(pos_j):
        return pos_j, nums_j
    i0, i1 = n_slab, n_slab + n_core
    core_i = Atoms(
        numbers=np.asarray(atoms_i.numbers[i0:i1], dtype=int),
        positions=np.asarray(atoms_i.get_positions()[i0:i1], dtype=float),
        cell=atoms_i.cell,
        pbc=atoms_i.pbc,
    )
    core_j = Atoms(
        numbers=nums_j[i0:i1],
        positions=pos_j[i0:i1],
        cell=atoms_i.cell,
        pbc=atoms_i.pbc,
    )
    matched_core, matched_nums = _permute_atoms_block_to_match(
        core_i,
        core_j,
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
        method=_core_block_match_method(n_slab),
    )
    pos_j[i0:i1] = matched_core
    nums_j[i0:i1] = matched_nums
    if n_slab != 0:
        return pos_j, nums_j
    pos_j = _align_product_kabsch_to_reactant(
        atoms_i, pos_j, n_slab=0, n_core_mobile=n_core
    )
    return _rematch_gas_core_and_kabsch(atoms_i, pos_j, nums_j, n_core=n_core)


def _reorder_product_to_match_reactant(
    reactant: Atoms,
    product: Atoms,
    *,
    n_slab: int,
    n_core_mobile: int | None,
    n_adsorbate_mobile: int | None,
    adsorbate_fragment_lengths: list[int] | None = None,
    match_adsorbate: bool = True,
) -> np.ndarray:
    """Reorder product atoms (positions and species) to match reactant ordering.

    When block dimensions are provided, the core is matched first. Adsorbate
    matching is skipped when ``match_adsorbate`` is False so callers can
    rigid-align on the core and then match adsorbate in the overlaid frame.
    """
    n_atom = len(reactant)
    mic_cell, mic_pbc = _mic_matching_context(reactant, n_slab=n_slab)
    use_blocks = (
        n_core_mobile is not None
        and n_adsorbate_mobile is not None
        and n_slab + int(n_core_mobile) + int(n_adsorbate_mobile) == n_atom
    )
    if use_blocks:
        _align_endpoints_blockwise(
            reactant,
            product,
            n_slab,
            int(n_core_mobile),
            int(n_adsorbate_mobile),
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
            adsorbate_fragment_lengths=adsorbate_fragment_lengths,
            match_adsorbate=match_adsorbate,
        )
        return product.get_positions()
    if 0 < n_slab < n_atom:
        p_m, n_m = _permute_atoms_block_to_match(
            reactant[n_slab:],
            product[n_slab:],
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
        )
        pos = product.get_positions().copy()
        nums = product.numbers.copy()
        pos[n_slab:] = p_m
        nums[n_slab:] = n_m
        product.set_positions(pos, apply_constraint=False)
        product.numbers = nums
        return pos
    mapping = _match_atoms_by_fingerprint(
        reactant, product, mic_cell=mic_cell, mic_pbc=mic_pbc
    )
    product.set_positions(product.get_positions()[mapping], apply_constraint=False)
    product.numbers = product.numbers[mapping]
    return product.get_positions()


def _warn_if_interpolated_bonds_stretch(
    images: list[Atoms],
    *,
    tol: float,
    mic: bool,
    verbosity: int = 1,
) -> None:
    """Diagnostic: warn if interior images stretch any ``FixBondLengths`` pair.

    Compares each interpolated interior image's ``FixBondLengths`` pair distance
    against the endpoint pair distance (the larger of the two endpoints is the
    reference). This is a diagnostic only: it never raises, so a pathological
    IDPP interpolation does not abort a normal TS run.
    """
    if len(images) < 3:
        return
    bond_constraints = [
        c for c in images[0].constraints if isinstance(c, FixBondLengths)
    ]
    if not bond_constraints:
        return
    logger = get_logger(__name__)
    reactant = images[0]
    product = images[-1]
    for constraint in bond_constraints:
        for a, b in constraint.pairs:
            d_r = float(reactant.get_distance(int(a), int(b), mic=mic))
            d_p = float(product.get_distance(int(a), int(b), mic=mic))
            ref = max(d_r, d_p)
            for img in images[1:-1]:
                d = float(img.get_distance(int(a), int(b), mic=mic))
                if abs(d - ref) > tol:
                    log_warning_v(
                        logger,
                        "Post-interpolation FixBondLengths pair (%d, %d) stretched to "
                        "%.3f A (endpoint %.3f A, tol %.3f A) in an interior NEB image",
                        int(a),
                        int(b),
                        d,
                        ref,
                        tol,
                        verbosity=verbosity,
                    )
                    break  # one warning per violated pair is enough


def interpolate_path(
    atoms1: Atoms,
    atoms2: Atoms,
    n_images: int = 5,
    method: str = "idpp",
    mic: bool = False,
    *,
    align_endpoints: bool = True,
    perturb_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    system_type: SystemType | None = None,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    neb_surface_cell_remap: bool = True,
    neb_surface_lattice_rotation: bool = True,
    neb_surface_max_lattice_shift: int = 1,
    neb_interpolation_bond_tolerance_a: float | None = None,
    verbosity: int = 1,
) -> list[Atoms]:
    """Interpolate between two structures and return images including endpoints.

    ``align_endpoints`` (default True): reorder endpoint atoms to match reactant.
    For slab/surface workflows (``n_slab > 0`` or exactly two periodic axes),
    alignment uses ``_align_product_surface_pbc``: MIC-aware matching, collective
    mobile lattice-image selection, per-atom MIC snapping, optional integer
    in-plane lattice shifts (``neb_surface_max_lattice_shift``), and global
    in-plane rotation evaluated jointly with each shift, with anchors reset to
    the reactant slab frame (no independent mobile-only rotation). Gas-phase
    clusters (no slab prefix, including a 3D vacuum box with ``pbc=True``) use
    the same core overlay as pair selection (fingerprint + Kabsch + spatial
    rematch). Adsorbate blocks are matched in that overlaid frame.
    ``perturb_sigma``: optional Gaussian displacement (Å) on interior images only.
    ``rng``: optional NumPy Generator when ``perturb_sigma`` > 0.

    If ``n_slab`` + ``n_core_mobile`` + ``n_adsorbate_mobile`` equals
    ``len(atoms)``, match endpoints per slab / core / adsorbate block instead
    of one global permutation. When ``adsorbate_fragment_lengths`` is provided,
    adsorbate fragments are COM-matched before intra-fragment fingerprinting.

    For constrained slab systems we always interpolate with
    ``apply_constraint=False``; constraints remain attached and are enforced
    during subsequent NEB optimization.
    """
    validate_atoms(atoms1)
    validate_atoms(atoms2)

    a1_copy = copy_atoms(atoms1)
    a2_copy = copy_atoms(atoms2)

    surface_cell_remap = neb_surface_cell_remap
    surface_lattice_rotation = neb_surface_lattice_rotation
    if align_endpoints and system_type is not None:
        system_policy = get_system_policy(system_type)
        surface_cell_remap = (
            system_policy.neb_surface_cell_remap and neb_surface_cell_remap
        )
        surface_lattice_rotation = (
            system_policy.neb_surface_lattice_rotation and neb_surface_lattice_rotation
        )

    if align_endpoints:
        n_slab_i = int(n_slab)
        n_atom = len(a1_copy)
        use_blocks = (
            n_core_mobile is not None
            and n_adsorbate_mobile is not None
            and n_slab_i + int(n_core_mobile) + int(n_adsorbate_mobile) == n_atom
        )
        surface_pbc = _requires_surface_pbc_alignment(a1_copy, n_slab=n_slab_i)
        mic_cell, mic_pbc = _mic_matching_context(a1_copy, n_slab=n_slab_i)
        if use_blocks:
            pos_j, nums_j = _overlay_product_core(
                a1_copy,
                a2_copy.get_positions(),
                np.asarray(a2_copy.numbers, dtype=int),
                n_slab=n_slab_i,
                n_core=int(n_core_mobile),
                mic_cell=mic_cell,
                mic_pbc=mic_pbc,
            )
            if surface_pbc:
                pos_j = _align_product_for_neb(
                    a1_copy,
                    pos_j,
                    n_slab=n_slab_i,
                    surface_cell_remap=surface_cell_remap,
                    surface_lattice_rotation=surface_lattice_rotation,
                    surface_max_lattice_shift=neb_surface_max_lattice_shift,
                    n_core_mobile=n_core_mobile,
                )
            a2_copy.set_positions(pos_j, apply_constraint=False)
            a2_copy.numbers = nums_j
            _match_adsorbate_block(
                a1_copy,
                a2_copy,
                n_slab_i,
                int(n_core_mobile),
                int(n_adsorbate_mobile),
                mic_cell=mic_cell,
                mic_pbc=mic_pbc,
                adsorbate_fragment_lengths=adsorbate_fragment_lengths,
            )
        else:
            pos_j = _reorder_product_to_match_reactant(
                a1_copy,
                a2_copy,
                n_slab=n_slab_i,
                n_core_mobile=n_core_mobile,
                n_adsorbate_mobile=n_adsorbate_mobile,
                adsorbate_fragment_lengths=adsorbate_fragment_lengths,
                match_adsorbate=True,
            )
            pos_j = _align_product_for_neb(
                a1_copy,
                pos_j,
                n_slab=n_slab_i,
                surface_cell_remap=surface_cell_remap,
                surface_lattice_rotation=surface_lattice_rotation,
                surface_max_lattice_shift=neb_surface_max_lattice_shift,
                n_core_mobile=n_core_mobile,
            )
            if not surface_pbc:
                n_fit = (
                    int(n_core_mobile)
                    if n_core_mobile is not None and int(n_core_mobile) > 0
                    else n_atom
                )
                pos_j, nums_j = _rematch_gas_core_and_kabsch(
                    a1_copy,
                    pos_j,
                    np.asarray(a2_copy.numbers, dtype=int),
                    n_core=n_fit,
                )
                a2_copy.numbers = nums_j
            a2_copy.set_positions(pos_j, apply_constraint=False)
        # Keep species order consistent with reactant for downstream NEB.
        a2_copy.numbers = a1_copy.numbers.copy()
        if surface_pbc:
            a2_copy.set_cell(a1_copy.cell)
            a2_copy.pbc = a1_copy.pbc

    # Build the band from aligned endpoints; ASE interpolation only fills interiors.
    # ``a1_copy``/``a2_copy`` are already de-aliased via ``copy_atoms`` above, but
    # ``Atoms.copy()`` shallow-copies ``info`` so the interior images would share
    # ``a1_copy``'s nested ``key_value_pairs`` dict. ``set_tags`` (potential_energy /
    # raw_score) on one image would then overwrite every other image, so isolate
    # each interior copy with ``copy_atoms``.
    images = [a1_copy] + [copy_atoms(a1_copy) for _ in range(n_images)] + [a2_copy]
    neb = NEB(images, method=DEFAULT_NEB_TANGENT_METHOD)
    # Interpolate unconstrained positions first; endpoint/image constraints
    # (e.g., fixed slab atoms) are enforced during subsequent optimization.
    neb.interpolate(method=method, mic=mic, apply_constraint=False)
    images = neb.images

    # Diagnostic check (never raises): interior NEB images interpolated with
    # apply_constraint=False must not stretch any FixBondLengths pair far from
    # its endpoint length. A large stretch signals a pathological IDPP path.
    if neb_interpolation_bond_tolerance_a is not None and len(images) > 2:
        _warn_if_interpolated_bonds_stretch(
            images,
            tol=float(neb_interpolation_bond_tolerance_a),
            mic=mic,
            verbosity=verbosity,
        )

    if perturb_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        for img in images[1:-1]:
            disp = rng.normal(
                scale=float(perturb_sigma), size=img.get_positions().shape
            )
            img.set_positions(img.get_positions() + disp, apply_constraint=False)

    return images


def _mobile_min_pairwise_distance(
    atoms: Atoms,
    *,
    n_slab: int = 0,
    mic: bool = False,
) -> float:
    """Minimum pairwise distance among mobile atoms (slab prefix excluded)."""
    pos = atoms.get_positions()[max(0, int(n_slab)) :]
    n = len(pos)
    if n < 2:
        return float("inf")
    if mic and bool(np.any(atoms.pbc)):
        cell = _cell_array(atoms.cell)
        pbc = _pbc_for_mic_alignment(atoms.pbc)
        i_idx, j_idx = np.triu_indices(n, k=1)
        if i_idx.size == 0:
            return float("inf")
        dlt = pos[j_idx] - pos[i_idx]
        dlt_mic, _ = find_mic(dlt, cell, pbc)
        return float(np.linalg.norm(dlt_mic, axis=1).min())
    return float(np.min(pdist(pos)))


def _endpoint_mobile_max_displacement(
    reactant: Atoms,
    product: Atoms,
    *,
    n_slab: int = 0,
    mic: bool = False,
) -> float:
    """Max Cartesian displacement of mobile atoms between aligned endpoints."""
    i0 = max(0, int(n_slab))
    dlt = product.get_positions()[i0:] - reactant.get_positions()[i0:]
    if dlt.size == 0:
        return 0.0
    if mic and bool(np.any(reactant.pbc)):
        dlt, _ = find_mic(
            dlt, _cell_array(reactant.cell), _pbc_for_mic_alignment(reactant.pbc)
        )
    return float(np.linalg.norm(dlt, axis=1).max())


def neb_uses_two_stage_climb(
    climb: bool,
    neb_steps: int,
    *,
    initial_energies: list[float] | np.ndarray | None = None,
    allow_two_stage: bool = True,
    min_interior_barrier: float = 1.0,
) -> bool:
    """True when CI-NEB should relax without climb first, then enable climb.

    Two-stage helps when the IDPP band already has a *robust* interior maximum
    (climb can otherwise pin to a terminus on a messy barrier). It *hurts*:

    - endpoint-max / barrierless IDPP bands (no-climb collapses the MEP);
    - soft adsorbate hops with a shallow interior max (no-climb also flattens
      them; seen on graphite OH pairs with ~0.9 eV IDPP barriers).

    Soft interior maxima (barrier ``< min_interior_barrier``) climb from step 0.
    """
    if not (bool(climb) and int(neb_steps) >= 4 and bool(allow_two_stage)):
        return False
    if initial_energies is None:
        return True
    e = np.asarray(initial_energies, dtype=float)
    if e.size < 3 or not np.all(np.isfinite(e)):
        return True
    max_idx = int(np.argmax(e))
    if max_idx in (0, len(e) - 1):
        return False
    barrier = float(e[max_idx] - min(float(e[0]), float(e[-1])))
    return barrier >= float(min_interior_barrier)


def validate_initial_neb_path(
    images: list[Atoms],
    *,
    n_slab: int = 0,
    mic: bool = False,
    max_endpoint_mismatch: float | None = None,
    clash_distance: float = 0.7,
) -> None:
    """Reject discontinuous/clashing IDPP bands before NEB optimization.

    The interior-image clash check (min mobile pairwise distance vs
    ``clash_distance``) always runs. The aligned endpoint-displacement gate is
    only enabled when ``max_endpoint_mismatch`` is set (adsorbate/surface presets).

    Raises:
        SCGOValidationError: when the initial path is unsuitable for NEB.
    """
    if len(images) < 2:
        raise SCGOValidationError(
            "Initial NEB path rejected (clashing/discontinuous interpolation): "
            "fewer than 2 images"
        )
    if max_endpoint_mismatch is not None:
        cartesian_limit = max(6.0, 3.0 * float(max_endpoint_mismatch))
        max_disp = _endpoint_mobile_max_displacement(
            images[0], images[-1], n_slab=n_slab, mic=mic
        )
        if max_disp > cartesian_limit:
            raise SCGOValidationError(
                "Initial NEB path rejected (clashing/discontinuous interpolation): "
                f"aligned endpoint mobile max displacement {max_disp:.3f} Å exceeds "
                f"cartesian limit {cartesian_limit:.3f} Å"
            )
    interiors = images[1:-1] if len(images) > 2 else images
    for i, img in enumerate(interiors, start=1):
        min_d = _mobile_min_pairwise_distance(img, n_slab=n_slab, mic=mic)
        if min_d < float(clash_distance):
            raise SCGOValidationError(
                "Initial NEB path rejected (clashing/discontinuous interpolation): "
                f"image {i} min mobile distance {min_d:.3f} Å < {float(clash_distance):.3f} Å"
            )


def validate_initial_neb_energy_profile(
    energies: list[float] | np.ndarray,
    *,
    max_spurious_barrier: float = MAX_SPURIOUS_NEB_BARRIER_EV,
    reference_reactant_energy: float | None = None,
    reference_product_energy: float | None = None,
    max_endpoint_energy_drift: float = 0.5,
    min_saddle_prominence: float | None = 0.40,
) -> None:
    """Reject IDPP bands with absurdly high barriers (discontinuous paths).

    Endpoint-max IDPP profiles are allowed: climbing NEB can still locate an
    interior saddle after the band relaxes (observed for adsorbate OH hops).
    Huge barriers (tens of eV) usually indicate a discontinuous hop.

    When reference endpoint energies are supplied (canonical minima energies),
    also reject bands whose aligned endpoint single-points drifted by more than
    ``max_endpoint_energy_drift`` — a signature of registry-breaking alignment.

    When ``min_saddle_prominence`` is set, reject interior maxima that sit less
    than that above *both* endpoints (one-sided slides that CI-NEB collapses).
    """
    e = np.asarray(energies, dtype=float)
    if e.size < 3:
        return
    if not np.all(np.isfinite(e)):
        raise SCGOValidationError(
            "Initial NEB path rejected (energy profile): non-finite image energies"
        )
    barrier = float(e.max() - min(float(e[0]), float(e[-1])))
    if barrier > float(max_spurious_barrier):
        raise SCGOValidationError(
            "Initial NEB path rejected (energy profile): "
            f"IDPP barrier {barrier:.3f} eV exceeds "
            f"{float(max_spurious_barrier):.3f} eV (likely discontinuous)"
        )
    drift_limit = float(max_endpoint_energy_drift)
    if reference_reactant_energy is not None:
        drift_r = abs(float(e[0]) - float(reference_reactant_energy))
        if drift_r > drift_limit:
            raise SCGOValidationError(
                "Initial NEB path rejected (energy profile): "
                f"aligned reactant energy drifted by {drift_r:.3f} eV "
                f"(limit {drift_limit:.3f} eV)"
            )
    if reference_product_energy is not None:
        drift_p = abs(float(e[-1]) - float(reference_product_energy))
        if drift_p > drift_limit:
            raise SCGOValidationError(
                "Initial NEB path rejected (energy profile): "
                f"aligned product energy drifted by {drift_p:.3f} eV "
                f"(limit {drift_limit:.3f} eV)"
            )
    # Prominence gate only when validating against canonical minima (adsorbate
    # TorchSim path). Unit/mock bands without references keep the looser check.
    if (
        min_saddle_prominence is not None
        and reference_reactant_energy is not None
        and reference_product_energy is not None
    ):
        max_idx = int(np.argmax(e))
        if max_idx not in (0, len(e) - 1):
            prominence = float(e[max_idx] - max(float(e[0]), float(e[-1])))
            if prominence < float(min_saddle_prominence):
                raise SCGOValidationError(
                    "Initial NEB path rejected (energy profile): "
                    f"interior max prominence {prominence:.3f} eV is below "
                    f"{float(min_saddle_prominence):.3f} eV (one-sided slide)"
                )


def evaluate_neb_image_energies(images: list[Atoms], relaxer: Any) -> list[float]:
    """Single-point energies for a NEB band via TorchSim ``relax_batch(steps=0)``.

    Attaches energy/forces onto each live image for later NEB force reuse.
    """
    batch = relaxer.relax_batch(list(images), steps=0)
    energies: list[float] = []
    for atoms, (energy, relaxed_atoms) in zip(images, batch, strict=True):
        attach_singlepoint_from_relax_output(
            atoms, energy, relaxed_atoms, require_forces=True
        )
        energies.append(float(energy))
    return energies


def evaluate_neb_image_energies_ase(images: list[Atoms]) -> list[float]:
    """Single-point energies for a NEB band with ASE calculator-backed images."""
    return [float(img.get_potential_energy()) for img in images]


def idpp_band_optimization_priority(
    energies: list[float] | np.ndarray,
    *,
    min_saddle_prominence: float = 0.40,
) -> tuple[int, float, float]:
    """Sort key for adsorbate NEB attempt order (higher tuple sorts first).

    Prefers IDPP bands with a robust interior maximum (tier 2) over endpoint-max
    bands (tier 1). Soft interior maxima (prominence below the gate) get tier 0.
    Within a tier, larger prominence / barrier is preferred.
    """
    e = np.asarray(energies, dtype=float)
    if e.size < 3 or not np.all(np.isfinite(e)):
        return (0, 0.0, 0.0)
    max_idx = int(np.argmax(e))
    barrier = float(e[max_idx] - min(float(e[0]), float(e[-1])))
    if max_idx in (0, len(e) - 1):
        return (1, barrier, 0.0)
    prominence = float(e[max_idx] - max(float(e[0]), float(e[-1])))
    if prominence < float(min_saddle_prominence):
        return (0, prominence, barrier)
    return (2, prominence, barrier)


def make_ts_result(
    *,
    pair_id: str,
    n_images: int,
    spring_constant: float,
    use_torchsim: bool,
    fmax: float,
    neb_steps: int | str | None,
    interpolation_method: str,
    climb: bool,
    align_endpoints: bool,
    perturb_sigma: float,
    neb_interpolation_mic: bool,
    neb_tangent_method: str,
    use_parallel_neb: bool = False,
    reactant_energy: float | None = None,
    product_energy: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a normalized TS-result dict (failure shape, success-promoted later)."""
    return {
        "status": "failed",
        "pair_id": pair_id,
        "neb_converged": False,
        "n_images": n_images,
        "spring_constant": spring_constant,
        "reactant_energy": float(reactant_energy)
        if reactant_energy is not None
        else None,
        "product_energy": float(product_energy) if product_energy is not None else None,
        "ts_energy": None,
        "ts_image_index": None,
        "barrier_height": None,
        "barrier_forward": None,
        "barrier_reverse": None,
        "transition_state": None,
        "error": error,
        "use_torchsim": bool(use_torchsim),
        "use_parallel_neb": bool(use_parallel_neb),
        "fmax": float(fmax),
        "neb_steps": int(neb_steps)
        if isinstance(neb_steps, (int, np.integer))
        else neb_steps,
        "interpolation_method": interpolation_method,
        "climb": bool(climb),
        "align_endpoints": bool(align_endpoints),
        "perturb_sigma": float(perturb_sigma),
        "neb_interpolation_mic": bool(neb_interpolation_mic),
        "neb_tangent_method": neb_tangent_method,
        "final_fmax": None,
        "steps_taken": None,
    }


def minima_provenance_dict(minima: list, idx: int) -> dict[str, Any]:
    """Extract per-minimum GO provenance for JSON serialization."""
    if not minima or idx < 0 or idx >= len(minima):
        get_logger(__name__).warning(
            "Invalid minima index %s for %d minima; returning empty provenance",
            idx,
            len(minima) if minima else 0,
        )
        return {}

    energy, atoms = minima[idx]
    return {
        "run_id": get_tag(atoms, "run_id"),
        "source_db": get_tag(atoms, "source_db"),
        "source_db_relpath": get_tag(atoms, "source_db_relpath"),
        "systems_row_id": get_tag(atoms, "systems_row_id"),
        "confid": get_tag(atoms, "confid"),
        "gaid": get_tag(atoms, "gaid"),
        "unique_id": get_tag(atoms, "unique_id"),
        "final_id": get_tag(atoms, "final_id"),
        "energy": float(energy) if energy is not None else None,
    }


def attach_minima_traceability(
    result: dict[str, Any],
    minima: list[tuple[float, Any]],
    i: int,
    j: int,
) -> None:
    """Record minima list indices and endpoint provenance on one TS result."""
    result["minima_indices"] = [int(i), int(j)]
    result["minima_provenance"] = [
        minima_provenance_dict(minima, i),
        minima_provenance_dict(minima, j),
    ]


def _finalize_neb_result(
    result: dict[str, Any],
    images: list[Atoms],
    *,
    logger: Any | None = None,
    max_spurious_barrier: float = MAX_SPURIOUS_NEB_BARRIER_EV,
) -> None:
    """Populate ``result`` with TS / endpoint geometry, energies, and barriers.

    Mutates ``result`` in place. Assumes ``reactant_energy`` and
    ``product_energy`` are already set. Bands whose highest-energy image is an
    endpoint, and barriers above :data:`MAX_SPURIOUS_NEB_BARRIER_EV`, are marked
    failed.

    Raises:
        SCGORuntimeError: If an endpoint energy is missing, or if no image energy
            could be read.
    """
    pair_id = result.get("pair_id")

    react = images[0].copy()
    prod = images[-1].copy()
    _detach_calc(react)
    _detach_calc(prod)
    result["reactant_structure"] = react
    result["product_structure"] = prod

    max_energy_idx = 0
    max_energy = -np.inf
    ts_atoms: Atoms | None = None
    for idx, atoms in enumerate(images):
        energy = _image_potential_energy(atoms)
        if energy > max_energy:
            max_energy = energy
            max_energy_idx = idx
            ts_atoms = atoms

    if result.get("reactant_energy") is None or result.get("product_energy") is None:
        raise SCGORuntimeError(
            f"Missing endpoint energies after NEB for pair {pair_id}: "
            f"reactant={result.get('reactant_energy')}, product={result.get('product_energy')}"
        )
    if ts_atoms is None:
        raise SCGORuntimeError(f"No TS energy found after NEB for pair {pair_id}")

    reactant_energy = float(result["reactant_energy"])
    product_energy = float(result["product_energy"])
    ts_energy = float(max_energy)
    barrier_height = ts_energy - min(reactant_energy, product_energy)

    ts_copy = deepcopy(ts_atoms)
    _detach_calc(ts_copy)
    result["transition_state"] = ts_copy
    result["ts_energy"] = ts_energy
    result["ts_image_index"] = int(max_energy_idx)
    result["barrier_height"] = barrier_height
    result["barrier_forward"] = ts_energy - reactant_energy
    result["barrier_reverse"] = ts_energy - product_energy

    endpoint_ts = max_energy_idx == 0 or max_energy_idx == len(images) - 1
    # Match pre-NEB IDPP gate: absurd barriers are discontinuous / unphysical.
    max_final_barrier = max_spurious_barrier
    if endpoint_ts:
        result["status"] = "failed"
        result["neb_converged"] = False
        result["error"] = (
            f"Highest-energy image is an endpoint (image {max_energy_idx}); "
            "no interior saddle found"
        )
        if logger is not None:
            logger.debug(
                "NEB highest-energy image is an endpoint for pair %s (image %d) "
                "— marking as non-converged",
                pair_id,
                max_energy_idx,
            )
    elif barrier_height > max_final_barrier:
        result["status"] = "failed"
        result["neb_converged"] = False
        result["error"] = (
            f"NEB barrier {barrier_height:.3f} eV exceeds "
            f"{max_final_barrier:.3f} eV (likely discontinuous path)"
        )
        if logger is not None:
            logger.debug(
                "NEB barrier too high for pair %s (%.3f eV) — marking as failed",
                pair_id,
                barrier_height,
            )
    else:
        result["status"] = "success" if result.get("neb_converged") else "failed"


def find_transition_state(
    atoms1: Atoms,
    atoms2: Atoms,
    calculator: Calculator | None,
    output_dir: str,
    pair_id: str,
    rng: np.random.Generator | None = None,
    n_images: int = 3,
    spring_constant: float = 0.1,
    optimizer: type[Optimizer] = FIRE,
    fmax: float = DEFAULT_FMAX_THRESHOLD,
    neb_steps: int = 500,
    trajectory: str | None = None,
    verbosity: int = 1,
    use_torchsim: bool = False,
    torchsim_params: dict[str, Any] | None = None,
    climb: bool = False,
    interpolation_method: str = "idpp",
    align_endpoints: bool = True,
    perturb_sigma: float = 0.0,
    neb_interpolation_mic: bool = False,
    neb_tangent_method: str = DEFAULT_NEB_TANGENT_METHOD,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    max_endpoint_mismatch: float | None = None,
    neb_prescreen_clash_distance: float = 0.7,
    min_saddle_prominence: float = 0.40,
    neb_max_spurious_barrier: float = MAX_SPURIOUS_NEB_BARRIER_EV,
    neb_interpolation_bond_tolerance_a: float | None = None,
    neb_surface_cell_remap: bool = True,
    neb_surface_lattice_rotation: bool = True,
    neb_surface_max_lattice_shift: int = 1,
    relaxer: Any | None = None,
    neb_cfg: NebRunConfig | None = None,
) -> dict[str, Any]:
    """Run NEB to locate a transition state between two structures.

    Args:
        neb_interpolation_mic: Forwarded to :func:`interpolate_path` as ``mic``.
            Use ``True`` for periodic cells (e.g. slabs); default ``False`` for
            isolated clusters.
        neb_tangent_method: ASE NEB tangent method (``ase.mep.neb.NEB`` ``method``
            argument). Default ``improvedtangent`` matches ASE recommendations.
        n_slab: Blockwise alignment: slab length (default 0).
        n_core_mobile: Mobile core count (with ``n_adsorbate_mobile`` for blockwise NEB).
        n_adsorbate_mobile: Mobile adsorbate fragment count.
        adsorbate_fragment_lengths: Optional per-fragment lengths for adsorbate matching.
        max_endpoint_mismatch: Optional Å gate for post-alignment path quality.
        neb_surface_cell_remap: Enable in-plane lattice-image search (surface).
        neb_surface_lattice_rotation: Enable global in-plane rotation (surface).
        neb_surface_max_lattice_shift: Max integer cell index searched in-plane
            during remap (default ``1``).

    Returns:
        A summary dict with TS geometry, energies and convergence status.
    """
    logger = get_logger(__name__)

    # Resolve effective parameters. A NebRunConfig (used by both runners)
    # wins for the geometry/validation knobs; torchsim_params and system_type
    # are only taken from it when the explicit arguments are None (preserves
    # the explicit-kwargs call sites used by tests).
    if neb_cfg is not None:
        n_images = neb_cfg.neb_n_images
        spring_constant = neb_cfg.neb_spring_constant
        fmax = neb_cfg.neb_fmax
        neb_steps = neb_cfg.neb_steps
        climb = neb_cfg.neb_climb
        interpolation_method = neb_cfg.neb_interpolation_method
        align_endpoints = neb_cfg.neb_align_endpoints
        perturb_sigma = neb_cfg.neb_perturb_sigma
        neb_interpolation_mic = neb_cfg.neb_interpolation_mic
        neb_tangent_method = neb_cfg.neb_tangent_method
        n_slab = neb_cfg.n_slab
        n_core_mobile = neb_cfg.n_core_mobile
        n_adsorbate_mobile = neb_cfg.n_adsorbate_mobile
        adsorbate_fragment_lengths = neb_cfg.adsorbate_fragment_lengths
        max_endpoint_mismatch = neb_cfg.max_endpoint_mismatch
        neb_prescreen_clash_distance = neb_cfg.neb_prescreen_clash_distance
        min_saddle_prominence = neb_cfg.min_saddle_prominence
        neb_max_spurious_barrier = neb_cfg.neb_max_spurious_barrier
        neb_interpolation_bond_tolerance_a = neb_cfg.neb_interpolation_bond_tolerance_a
        neb_surface_cell_remap = neb_cfg.neb_surface_cell_remap
        neb_surface_lattice_rotation = neb_cfg.neb_surface_lattice_rotation
        neb_surface_max_lattice_shift = neb_cfg.neb_surface_max_lattice_shift
        if system_type is None:
            system_type = neb_cfg.system_type
        if torchsim_params is None:
            torchsim_params = neb_cfg.torchsim_params

    validate_atoms(atoms1)
    validate_atoms(atoms2)

    if use_torchsim:
        if is_uma_like_calculator(calculator):
            _require_torchsim_fairchem()
        elif is_upet_like_calculator(calculator):
            _require_torchsim_upet()
        else:
            _require_torchsim()
    else:
        validate_calculator_attached(atoms1, "NEB reactant")
        validate_calculator_attached(atoms2, "NEB product")

    if len(atoms1) != len(atoms2):
        raise SCGOValidationError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    if trajectory is None:
        trajectory = os.path.join(output_dir, f"neb_{pair_id}.traj")

    # Extract initial energies (safe for TorchSim where atoms have no calculator).
    reactant_energy = extract_energy_from_atoms(atoms1)
    product_energy = extract_energy_from_atoms(atoms2)

    # For ASE NEB we require explicit endpoint energies; for TorchSim the
    # relaxer computes them below.
    if not use_torchsim:
        if reactant_energy is None:
            raise SCGOValidationError(
                f"Cannot extract energy from reactant atoms for pair {pair_id}"
            )
        if product_energy is None:
            raise SCGOValidationError(
                f"Cannot extract energy from product atoms for pair {pair_id}"
            )

    log_debug_v(
        logger,
        "Finding transition state for pair %s",
        pair_id,
        verbosity=verbosity,
    )
    if reactant_energy is not None:
        log_debug_v(
            logger,
            "  Reactant energy: %.6f eV",
            reactant_energy,
            verbosity=verbosity,
        )
    if product_energy is not None:
        log_debug_v(
            logger,
            "  Product energy: %.6f eV",
            product_energy,
            verbosity=verbosity,
        )

    result = make_ts_result(
        pair_id=pair_id,
        n_images=n_images,
        spring_constant=spring_constant,
        use_torchsim=use_torchsim,
        fmax=fmax,
        neb_steps=neb_steps,
        interpolation_method=interpolation_method,
        climb=climb,
        align_endpoints=align_endpoints,
        perturb_sigma=perturb_sigma,
        neb_interpolation_mic=neb_interpolation_mic,
        neb_tangent_method=neb_tangent_method,
        reactant_energy=reactant_energy,
        product_energy=product_energy,
    )

    t_wall0: float | None = None
    neb_opt = 0.0
    try:
        t_wall0 = perf_counter()
        if np.allclose(atoms1.get_positions(), atoms2.get_positions(), atol=1e-8):
            raise SCGOValidationError(
                f"Endpoints are identical for pair {pair_id}; no interior TS"
            )

        log_debug_v(
            logger,
            "Generating initial path with %s interpolation",
            interpolation_method,
            verbosity=verbosity,
        )
        # Keep interpolation unconstrained; constraints are applied during NEB.
        images = interpolate_path(
            atoms1,
            atoms2,
            n_images=n_images,
            method=interpolation_method,
            mic=neb_interpolation_mic,
            align_endpoints=align_endpoints,
            perturb_sigma=perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=n_slab,
            n_core_mobile=n_core_mobile,
            n_adsorbate_mobile=n_adsorbate_mobile,
            adsorbate_fragment_lengths=adsorbate_fragment_lengths,
            neb_surface_cell_remap=neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_surface_max_lattice_shift,
            neb_interpolation_bond_tolerance_a=neb_interpolation_bond_tolerance_a,
            verbosity=verbosity,
        )
        validate_initial_neb_path(
            images,
            n_slab=n_slab,
            mic=neb_interpolation_mic,
            max_endpoint_mismatch=max_endpoint_mismatch,
            clash_distance=neb_prescreen_clash_distance,
        )

        if np.allclose(
            images[0].get_positions(), images[-1].get_positions(), atol=1e-8
        ):
            raise SCGOValidationError(
                f"Endpoints are identical for pair {pair_id}; no interior TS"
            )

        neb: NEB
        if use_torchsim:
            ts_relaxer = relaxer
            if ts_relaxer is None:
                ts_relaxer = _tsh.TorchSimBatchRelaxer(**(torchsim_params or {}))

            react_e = extract_energy_from_atoms(images[0])
            prod_e = extract_energy_from_atoms(images[-1])
            if react_e is not None and prod_e is not None:
                result["reactant_energy"] = float(react_e)
                result["product_energy"] = float(prod_e)
            else:
                ep_results = ts_relaxer.relax_batch([images[0], images[-1]], steps=0)
                for atoms, (energy, relaxed_atoms) in zip(
                    [images[0], images[-1]], ep_results, strict=True
                ):
                    attach_singlepoint_from_relax_output(
                        atoms, energy, relaxed_atoms, require_forces=True
                    )
                result["reactant_energy"] = float(ep_results[0][0])
                result["product_energy"] = float(ep_results[1][0])

            # Full-band SP only when the energy-profile gate is enabled (mirrors
            # parallel). Forces attach for step-0 reuse.
            band_energies: list[float] | None = None
            if max_endpoint_mismatch is not None:
                band_energies = evaluate_neb_image_energies(images, ts_relaxer)
                validate_initial_neb_energy_profile(
                    band_energies,
                    reference_reactant_energy=reactant_energy,
                    reference_product_energy=product_energy,
                    min_saddle_prominence=min_saddle_prominence,
                    max_spurious_barrier=neb_max_spurious_barrier,
                )
                result["reactant_energy"] = float(band_energies[0])
                result["product_energy"] = float(band_energies[-1])

            log_debug_v(
                logger,
                "Using TorchSim batched NEB (climb=%s)",
                climb,
                verbosity=verbosity,
            )

            steps_budget = int(neb_steps)
            use_two_stage = neb_uses_two_stage_climb(
                climb, steps_budget, initial_energies=band_energies
            )
            neb = TorchSimNEB(
                images,
                ts_relaxer,
                k=spring_constant,
                climb=bool(climb) and not use_two_stage,
                method=neb_tangent_method,
            )
            if band_energies is not None and all(
                _image_has_cached_forces(img) for img in images
            ):
                neb._force_calls += 1
        else:
            if calculator is None:
                raise SCGOValidationError("Calculator required when use_torchsim=False")
            # Each image should own its calculator (ASE NEB requires distinct
            # calculators). When a calculator cannot be deep-copied we fall back
            # to sharing the single instance and must tell ASE that is allowed;
            # otherwise ``NEB.get_forces`` raises ``ValueError`` for shared calcs.
            shared_calc = False
            for img in images:
                try:
                    img.calc = deepcopy(calculator)
                except (TypeError, AttributeError):
                    shared_calc = True
                    img.calc = calculator

            # Lightweight single-point energy pre-screen for the serial ASE path
            # (the TorchSim path computes band_energies above via relax_batch).
            band_energies = evaluate_neb_image_energies_ase(images)
            if max_endpoint_mismatch is not None:
                validate_initial_neb_energy_profile(
                    band_energies,
                    reference_reactant_energy=reactant_energy,
                    reference_product_energy=product_energy,
                    min_saddle_prominence=min_saddle_prominence,
                    max_spurious_barrier=neb_max_spurious_barrier,
                )

            steps_budget = int(neb_steps)
            use_two_stage = neb_uses_two_stage_climb(
                climb, steps_budget, initial_energies=band_energies
            )
            neb = NEB(
                images,
                k=spring_constant,
                climb=bool(climb) and not use_two_stage,
                method=neb_tangent_method,
                allow_shared_calculator=shared_calc,
            )

        opt_logfile = None if verbosity <= 1 else sys.stdout
        # Two-stage CI-NEB: relax without climb, then climb (see helper docstring).
        # Cap stage 1 at half the budget; stage 2 gets whatever remains after stage 1
        # actually used (so early stage-1 convergence does not starve climb).
        stage1_cap = steps_budget // 2 if use_two_stage else steps_budget

        log_debug_v(
            logger,
            "Starting NEB optimization with %s",
            optimizer.__name__,
            verbosity=verbosity,
        )

        t_neb0 = perf_counter()
        dyn: Optimizer = optimizer(neb, trajectory=trajectory, logfile=opt_logfile)  # type: ignore[arg-type]
        dyn.run(fmax=fmax, steps=stage1_cap)
        steps_taken = int(dyn.nsteps)
        if use_two_stage:
            neb.climb = True
            stage2_steps = max(1, steps_budget - steps_taken)
            log_debug_v(
                logger,
                "Enabling climbing image for second NEB stage (%d steps)",
                stage2_steps,
                verbosity=verbosity,
            )
            dyn = optimizer(neb, trajectory=trajectory, logfile=opt_logfile)  # type: ignore[arg-type]
            dyn.run(fmax=fmax, steps=stage2_steps)
            steps_taken += int(dyn.nsteps)
        neb_opt = perf_counter() - t_neb0

        try:
            neb_forces = neb.get_forces()
            final_fmax: float | None = neb_max_atom_force(neb_forces)
        except (AttributeError, RuntimeError, ValueError):
            final_fmax = None

        result["final_fmax"] = final_fmax
        result["neb_converged"] = final_fmax is not None and final_fmax < fmax
        result["steps_taken"] = steps_taken

        if not result["neb_converged"] and result.get("error") is None:
            result["error"] = (
                f"NEB did not converge (final_fmax={final_fmax}, fmax={fmax})"
            )

        fmax_str = f"{final_fmax:.6f}" if final_fmax is not None else "unknown"
        if result["neb_converged"]:
            log_debug_v(
                logger,
                "NEB converged in %d steps (final_fmax=%s < %.6f)",
                result["steps_taken"],
                fmax_str,
                fmax,
                verbosity=verbosity,
            )
        else:
            log_warning_v(
                logger,
                "NEB not converged after %d steps (final_fmax=%s, target_fmax=%.6f)",
                result["steps_taken"],
                fmax_str,
                fmax,
                verbosity=verbosity,
                min_verbosity=2,
            )

        # Last optimizer step can invalidate SinglePoint caches; refresh PES at
        # the final geometries before barrier finalize (TorchSim path only).
        if use_torchsim:
            neb.get_forces()

        _finalize_neb_result(
            result,
            neb.images,
            logger=logger,
            max_spurious_barrier=neb_max_spurious_barrier,
        )

        if use_torchsim and result["status"] == "success":
            result["force_calls"] = neb.get_force_calls()

        if result["status"] == "success":
            log_debug_v(
                logger,
                "TS found at image %d/%d",
                result["ts_image_index"],
                len(neb.images) - 1,
                verbosity=verbosity,
            )
            log_debug_v(
                logger,
                "  TS energy: %.6f eV",
                result["ts_energy"],
                verbosity=verbosity,
            )
            log_debug_v(
                logger,
                "  Barrier height: %.6f eV",
                result["barrier_height"],
                verbosity=verbosity,
            )
            if use_torchsim:
                log_debug_v(
                    logger,
                    "  GPU-batched force calls: %s",
                    result.get("force_calls"),
                    verbosity=verbosity,
                )

    except KeyboardInterrupt:
        raise
    except (ValueError, RuntimeError, OSError, SCGOValidationError) as e:
        result["error"] = str(e)
        if is_cuda_oom_error(e):
            cleanup_torch_cuda(logger=logger)
            log_warning_v(
                logger,
                "Detected CUDA out-of-memory during NEB for pair %s — attempted GPU cleanup",
                pair_id,
                verbosity=verbosity,
            )
        if isinstance(e, SCGOValidationError):
            result["status"] = "skipped"
        else:
            logger.error(
                "Failed to find TS for pair %s: %s: %s",
                pair_id,
                type(e).__name__,
                e,
                exc_info=(verbosity >= 2),
            )

    if t_wall0 is not None:
        total_s = perf_counter() - t_wall0
        ts_timings: dict[str, float] = {
            "kind": "neb",
            "total_wall_s": total_s,
            "neb_optimization_s": neb_opt,
            "cpu_non_relax_s": max(0.0, total_s - neb_opt),
        }
        result["timings_s"] = ts_timings
        neb_backend = "neb_torchsim" if use_torchsim else "neb_ase"
        if verbosity >= 2:
            log_timing_summary(logger, neb_backend, ts_timings, verbosity=verbosity)
        if write_timing_json:
            write_timing_file(
                output_dir,
                build_timing_payload(
                    backend=neb_backend,
                    timings_s=ts_timings,
                    extra={"pair_id": pair_id},
                ),
                filename=f"timing_{pair_id}.json",
            )

    return result


_PROVENANCE_KEYS = (
    "system_type",
    "use_torchsim",
    "use_parallel_neb",
    "climb",
    "align_endpoints",
    "perturb_sigma",
    "neb_interpolation_mic",
    "interpolation_method",
    "fmax",
    "neb_steps",
    "minima_indices",
    "minima_provenance",
)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON via a same-directory temp file then ``os.replace``."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_neb_",
        suffix=".json",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def load_completed_neb_result(
    output_dir: str | os.PathLike[str],
    pair_id: str,
) -> dict[str, Any] | None:
    """Return a prior NEB result if ``neb_{pair_id}_metadata.json`` is resume-ready.

    Resume requires ``status == "success"``, ``neb_converged`` true (missing
    treated as not converged), and a readable ``ts_{pair_id}.xyz``. Corrupt or
    incomplete artifacts return ``None`` so the pair is re-run.
    """
    metadata_path = os.path.join(str(output_dir), f"neb_{pair_id}_metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("status") != "success":
        return None
    if not bool(metadata.get("neb_converged", False)):
        return None

    ts_path = os.path.join(str(output_dir), f"ts_{pair_id}.xyz")
    if not os.path.isfile(ts_path):
        return None
    try:
        transition_state = read(ts_path)
    except (OSError, ValueError):
        return None

    use_torchsim = metadata.get("neb_backend") == "torchsim" or bool(
        metadata.get("use_torchsim", False)
    )
    result = make_ts_result(
        pair_id=str(metadata.get("pair_id", pair_id)),
        n_images=int(metadata.get("n_images") or 0),
        spring_constant=float(metadata.get("spring_constant") or 0.0),
        use_torchsim=use_torchsim,
        fmax=float(metadata.get("fmax") or 0.0),
        neb_steps=metadata.get("neb_steps"),
        interpolation_method=str(metadata.get("interpolation_method") or "idpp"),
        climb=bool(metadata.get("climb", False)),
        align_endpoints=bool(metadata.get("align_endpoints", True)),
        perturb_sigma=float(metadata.get("perturb_sigma") or 0.0),
        neb_interpolation_mic=bool(metadata.get("neb_interpolation_mic", False)),
        neb_tangent_method=str(
            metadata.get("neb_tangent_method") or DEFAULT_NEB_TANGENT_METHOD
        ),
        use_parallel_neb=bool(metadata.get("use_parallel_neb", False)),
        reactant_energy=metadata.get("reactant_energy"),
        product_energy=metadata.get("product_energy"),
        error=metadata.get("error"),
    )
    result["status"] = "success"
    result["neb_converged"] = True
    result["ts_energy"] = metadata.get("ts_energy")
    result["ts_image_index"] = metadata.get("ts_image_index")
    result["barrier_height"] = metadata.get("barrier_height")
    result["final_fmax"] = metadata.get("final_fmax")
    result["steps_taken"] = metadata.get("steps_taken")
    result["force_calls"] = metadata.get("force_calls")
    result["resumed"] = True
    result["transition_state"] = transition_state

    for label, key in (
        ("reactant", "reactant_structure"),
        ("product", "product_structure"),
    ):
        ep_path = os.path.join(str(output_dir), f"{label}_{pair_id}.xyz")
        if os.path.isfile(ep_path):
            with contextlib.suppress(OSError, ValueError):
                result[key] = read(ep_path)

    for key in _PROVENANCE_KEYS:
        if key in metadata:
            result[key] = metadata[key]
    return result


def save_neb_result(
    result: dict[str, Any],
    output_dir: str,
    pair_id: str,
    *,
    verbosity: int = 1,
) -> None:
    """Save NEB result: TS and endpoint XYZ (when present) plus metadata JSON.

    Writes:

    - ``ts_{pair_id}.xyz`` on success when a TS geometry is present
    - ``reactant_{pair_id}.xyz`` / ``product_{pair_id}.xyz`` when
      ``reactant_structure`` / ``product_structure`` are on the result dict
    - ``neb_{pair_id}_metadata.json`` (includes schema/version/time and NEB params)

    Metadata uses temp + ``os.replace``. Per-file paths log at verbosity >= 2.
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    if result["status"] == "success" and result["transition_state"] is not None:
        _detach_calc(result["transition_state"])
        ts_path = os.path.join(output_dir, f"ts_{pair_id}.xyz")
        write(ts_path, result["transition_state"])
        log_debug_v(
            logger,
            "Saved TS structure to %s",
            ts_path,
            verbosity=verbosity,
        )

    for label, key in (
        ("reactant", "reactant_structure"),
        ("product", "product_structure"),
    ):
        atoms = result.get(key)
        if atoms is not None:
            ep = atoms.copy()
            _detach_calc(ep)
            ep_path = os.path.join(output_dir, f"{label}_{pair_id}.xyz")
            write(ep_path, ep)
            log_debug_v(
                logger,
                "Saved %s endpoint structure to %s",
                label,
                ep_path,
                verbosity=verbosity,
            )

    extra = {key: result[key] for key in _PROVENANCE_KEYS if key in result}
    extra["neb_backend"] = (
        "torchsim" if result.get("use_torchsim") else result.get("neb_backend", "ase")
    )
    metadata = output_json_provenance(extra=extra)
    metadata.update(
        {
            "pair_id": result["pair_id"],
            "status": result["status"],
            "neb_converged": result["neb_converged"],
            "n_images": result["n_images"],
            "spring_constant": result["spring_constant"],
            "reactant_energy": result["reactant_energy"],
            "product_energy": result["product_energy"],
            "ts_energy": result["ts_energy"],
            "barrier_height": result["barrier_height"],
            "error": result["error"],
            "final_fmax": result.get("final_fmax"),
            "steps_taken": result.get("steps_taken"),
            "force_calls": result.get("force_calls"),
            "fmax": result.get("fmax"),
            "neb_steps": result.get("neb_steps"),
            "interpolation_method": result.get("interpolation_method"),
            "climb": result.get("climb"),
            "align_endpoints": result.get("align_endpoints"),
            "perturb_sigma": result.get("perturb_sigma"),
            "neb_interpolation_mic": result.get("neb_interpolation_mic"),
            "neb_tangent_method": result.get("neb_tangent_method"),
            "use_parallel_neb": result.get("use_parallel_neb"),
            "use_torchsim": result.get("use_torchsim"),
        }
    )

    if result["status"] == "success":
        metadata["ts_image_index"] = result.get("ts_image_index")

    metadata_path = os.path.join(output_dir, f"neb_{pair_id}_metadata.json")
    _atomic_write_json(metadata_path, metadata)

    log_debug_v(
        logger,
        "Saved NEB metadata to %s",
        metadata_path,
        verbosity=verbosity,
    )
