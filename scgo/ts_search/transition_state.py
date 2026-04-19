"""Utilities for finding transition states with NEB and path interpolation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import write
from ase.mep import NEB
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from scipy.optimize import linear_sum_assignment

from scgo.calculators import torchsim_helpers as _tsh
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database.metadata import get_metadata
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.torchsim_policy import (
    _require_torchsim,
    _require_torchsim_fairchem,
    is_uma_like_calculator,
)
from scgo.utils.ts_provenance import is_cuda_oom_error, ts_output_provenance
from scgo.utils.validation import validate_atoms, validate_calculator_attached

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _detach_calc(atoms: Atoms | None) -> None:
    """Best-effort removal of calculator from returned structures (frees GPU buffers)."""
    if atoms is None:
        return
    with contextlib.suppress(AttributeError, TypeError):
        atoms.calc = None


def attach_singlepoint_from_relax_output(
    atoms: Atoms,
    energy: float,
    relaxed_atoms: Atoms,
    *,
    require_forces: bool = True,
) -> None:
    """Attach ``SinglePointCalculator`` to ``atoms`` from one ``relax_batch`` result."""
    forces = relaxed_atoms.arrays.get("forces")
    if forces is None and relaxed_atoms.calc is not None:
        with contextlib.suppress(AttributeError, NotImplementedError):
            forces = relaxed_atoms.get_forces()
    if forces is not None and getattr(forces, "size", 0) > 0:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return
    if require_forces:
        raise RuntimeError(
            "TorchSim did not return forces. Ensure the model is loaded with compute_forces=True."
        )
    atoms.calc = SinglePointCalculator(atoms, energy=energy)


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
    pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
    *,
    ignore_fixed_atoms: bool = True,
) -> tuple[float, float, bool]:
    """Return (cum_diff, max_diff, are_similar) comparing two Atoms; raises ValueError if counts differ."""
    from scgo.utils.comparators import (
        PureInteratomicDistanceComparator,
        get_shared_mobile_atom_indices,
    )

    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    comparison_indices = (
        get_shared_mobile_atom_indices(atoms1, atoms2)
        if ignore_fixed_atoms
        else np.arange(len(atoms1), dtype=int)
    )
    atoms1_cmp = atoms1[comparison_indices]
    atoms2_cmp = atoms2[comparison_indices]

    comparator = PureInteratomicDistanceComparator(
        n_top=len(atoms1_cmp),
        tol=tolerance,
        pair_cor_max=pair_cor_max,
        mic=False,
    )

    cum_diff, max_diff = comparator.get_differences(atoms1_cmp, atoms2_cmp)
    are_similar = comparator.looks_like(atoms1_cmp, atoms2_cmp)

    return cum_diff, max_diff, are_similar


class TorchSimNEB(NEB):
    """NEB that batches PES evaluations via TorchSim for GPU efficiency."""

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

    def get_forces(self) -> np.ndarray:
        """Batch-evaluate PES forces with TorchSim and return NEB forces.

        When images already carry PES forces (for example because
        ``ParallelNEBBatch`` just evaluated them in a single batched call),
        reuse the cached arrays instead of re-invoking TorchSim.
        """
        if all(_image_has_cached_forces(img) for img in self.images):
            return super().get_forces()

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


def _match_atoms_by_fingerprint(a1: Atoms, a2: Atoms) -> list[int]:
    """Return mapping such that mapped_idx[i] is index in `a2` matching atom i in `a1`.

    Uses per-atom local-distance fingerprints and the Hungarian algorithm to
    obtain a permutation that is robust to rotations and permutations.
    """
    if len(a1) != len(a2):
        raise ValueError("Atoms objects have different lengths")

    mapping = [-1] * len(a1)
    # Match separately for each atomic number (handles mixed-species clusters)
    for z in set(a1.numbers):
        idx1 = [i for i, x in enumerate(a1.numbers) if x == z]
        idx2 = [i for i, x in enumerate(a2.numbers) if x == z]
        if len(idx1) != len(idx2):
            raise ValueError("Composition mismatch during endpoint matching")

        fp1 = _local_distance_fingerprints(a1)[idx1]
        fp2 = _local_distance_fingerprints(a2)[idx2]
        # Cost = L2 distance between fingerprints
        cost = np.linalg.norm(fp1[:, None, :] - fp2[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        for ri, ci in zip(r, c, strict=False):
            mapping[idx1[ri]] = idx2[ci]

    return mapping


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return rotation matrix R that minimizes ||P - Q @ R|| (P and Q are centered)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def _fixed_atom_mask(atoms: Atoms) -> np.ndarray:
    """Return a boolean mask for atoms fixed by ``FixAtoms`` constraints."""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            mask[idx] = True
    return mask


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
) -> list[Atoms]:
    """Interpolate between two Atoms and return images including endpoints.

    ``align_endpoints`` (default True): reorder endpoint atoms to match reactant.
    For periodic MIC workflows (``mic=True`` + periodic cell), alignment uses
    MIC displacements instead of Cartesian Kabsch rotation.
    ``perturb_sigma``: optional Gaussian displacement (Å) on interior images only.
    ``rng``: optional NumPy Generator when ``perturb_sigma`` > 0.

    **Fixed slab / supported-cluster NEB:** Endpoints often carry ``FixAtoms`` on the
    substrate. ASE's ``NEB.interpolate`` defaults to ``apply_constraint=True``, which
    projects each interpolation step onto constrained degrees of freedom and commonly
    yields broken or unphysical initial bands (or failures) for slab systems. This
    function always uses ``apply_constraint=False`` so the path is built in full
    coordinates first; ``FixAtoms`` still constrain each image during the subsequent
    NEB optimization because interior images are ``Atoms`` copies of the endpoints and
    retain the same constraints.
    """
    validate_atoms(atoms1)
    validate_atoms(atoms2)

    a1_copy = atoms1.copy()
    a2_copy = atoms2.copy()

    if align_endpoints:
        mapping = _match_atoms_by_fingerprint(a1_copy, a2_copy)
        new_pos = a2_copy.get_positions()[mapping]
        pbc_active = bool(np.any(a1_copy.pbc))
        if mic and pbc_active:
            # Preserve periodic geometry by pulling each product atom to the
            # nearest MIC image relative to the matched reactant atom.
            ref_pos = a1_copy.get_positions().copy()
            disp = new_pos - ref_pos
            disp_mic, _ = find_mic(disp, cell=a1_copy.cell, pbc=a1_copy.pbc)
            # For fixed slabs, anchor displacements to fixed atoms so alignment
            # does not spuriously shift the slab reference frame.
            fixed_mask = _fixed_atom_mask(a1_copy)
            if np.any(fixed_mask):
                fixed_shift = np.mean(disp_mic[fixed_mask], axis=0)
                disp_mic = disp_mic - fixed_shift
            a2_copy.set_positions(ref_pos + disp_mic)
        else:
            P = a1_copy.get_positions().copy()
            Q = new_pos.copy()
            Pc = P - P.mean(axis=0)
            Qc = Q - Q.mean(axis=0)
            R = _kabsch_rotation(Pc, Qc)
            Qrot = (Qc @ R.T) + P.mean(axis=0)
            a2_copy.set_positions(Qrot)

    images = [a1_copy] + [a1_copy.copy() for _ in range(n_images)] + [a2_copy]
    neb = NEB(images, method=DEFAULT_NEB_TANGENT_METHOD)
    # Interpolate unconstrained positions first; endpoint/image constraints
    # (e.g., fixed slab atoms) are enforced during subsequent optimization.
    neb.interpolate(method=method, mic=mic, apply_constraint=False)
    images = neb.images

    if perturb_sigma and perturb_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        for img in images[1:-1]:
            disp = rng.normal(
                scale=float(perturb_sigma), size=img.get_positions().shape
            )
            img.set_positions(img.get_positions() + disp)

    return images


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def _coerce_neb_steps(neb_steps: int | str | None) -> int | str | None:
    """Coerce numpy integer step counts to plain int (JSON-friendly)."""
    if isinstance(neb_steps, (int, np.integer)):
        return int(neb_steps)
    return neb_steps


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
        "neb_steps": _coerce_neb_steps(neb_steps),
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
        return {}

    energy, atoms = minima[idx]
    return {
        "run_id": get_metadata(atoms, "run_id"),
        "trial": get_metadata(atoms, "trial") or get_metadata(atoms, "trial_id"),
        "source_db": get_metadata(atoms, "source_db"),
        "source_db_relpath": get_metadata(atoms, "source_db_relpath"),
        "systems_row_id": get_metadata(atoms, "systems_row_id"),
        "confid": get_metadata(atoms, "confid"),
        "gaid": get_metadata(atoms, "gaid"),
        "unique_id": get_metadata(atoms, "unique_id"),
        "final_id": get_metadata(atoms, "final_id"),
        "energy": float(energy) if energy is not None else None,
    }


def _finalize_neb_result(
    result: dict[str, Any],
    images: list[Atoms],
    *,
    logger: Any | None = None,
) -> None:
    """Populate ``result`` with TS / endpoint geometry, energies, and barriers.

    Mutates ``result`` in place. Assumes ``reactant_energy`` and
    ``product_energy`` are already set; raises ``RuntimeError`` otherwise.
    Marks an endpoint-as-TS result as failed.
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
        energy = float(atoms.get_potential_energy())
        if energy > max_energy:
            max_energy = energy
            max_energy_idx = idx
            ts_atoms = atoms

    if result.get("reactant_energy") is None or result.get("product_energy") is None:
        raise RuntimeError(
            f"Missing endpoint energies after NEB for pair {pair_id}: "
            f"reactant={result.get('reactant_energy')}, product={result.get('product_energy')}"
        )
    if ts_atoms is None:
        raise RuntimeError(f"No TS energy found after NEB for pair {pair_id}")

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
    if endpoint_ts:
        result["status"] = "failed"
        result["neb_converged"] = False
        result["error"] = (
            f"NEB returned endpoint as TS (image {max_energy_idx}); "
            "no interior saddle located"
        )
        if logger is not None:
            logger.warning(
                "NEB reported endpoint as TS for pair %s (image %d) — marking as non-converged",
                pair_id,
                max_energy_idx,
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
    fmax: float = 0.05,
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
) -> dict[str, Any]:
    """Run NEB to locate a transition state between two structures.

    Args:
        neb_interpolation_mic: Forwarded to :func:`interpolate_path` as ``mic``.
            Use ``True`` for periodic cells (e.g. slabs); default ``False`` for
            isolated clusters.
        neb_tangent_method: ASE NEB tangent method (``ase.mep.neb.NEB`` ``method``
            argument). Default ``improvedtangent`` matches ASE recommendations.

    Returns:
        A summary dict with TS geometry, energies and convergence status.
    """
    logger = get_logger(__name__)

    validate_atoms(atoms1)
    validate_atoms(atoms2)

    if use_torchsim:
        if is_uma_like_calculator(calculator):
            _require_torchsim_fairchem()
        else:
            _require_torchsim()
    else:
        validate_calculator_attached(atoms1, "NEB reactant")
        validate_calculator_attached(atoms2, "NEB product")

    if len(atoms1) != len(atoms2):
        raise ValueError(
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
            raise ValueError(
                f"Cannot extract energy from reactant atoms for pair {pair_id}"
            )
        if product_energy is None:
            raise ValueError(
                f"Cannot extract energy from product atoms for pair {pair_id}"
            )

    if verbosity >= 1:
        logger.info(f"Finding transition state for pair {pair_id}")
        if reactant_energy is not None:
            logger.info(f"  Reactant energy: {reactant_energy:.6f} eV")
        if product_energy is not None:
            logger.info(f"  Product energy: {product_energy:.6f} eV")

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

    try:
        if verbosity >= 2:
            logger.info(
                f"Generating initial path with {interpolation_method} interpolation"
            )
        # Initial band: interpolate_path uses neb.interpolate(..., apply_constraint=False)
        # so FixAtoms on slab endpoints do not break path construction.
        images = interpolate_path(
            atoms1,
            atoms2,
            n_images=n_images,
            method=interpolation_method,
            mic=neb_interpolation_mic,
            align_endpoints=align_endpoints,
            perturb_sigma=perturb_sigma,
            rng=rng,
        )

        if np.allclose(
            images[0].get_positions(), images[-1].get_positions(), atol=1e-8
        ):
            raise ValueError(
                f"Endpoints are identical for pair {pair_id}; no interior TS"
            )

        neb: NEB
        if use_torchsim:
            relaxer = _tsh.TorchSimBatchRelaxer(**(torchsim_params or {}))

            ep_results = relaxer.relax_batch([images[0], images[-1]], steps=0)
            result["reactant_energy"] = float(ep_results[0][0])
            result["product_energy"] = float(ep_results[1][0])

            if verbosity >= 2:
                logger.info(f"Using TorchSim batched NEB (climb={climb})")

            neb = TorchSimNEB(
                images,
                relaxer,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )
        else:
            if calculator is None:
                raise ValueError("Calculator required when use_torchsim=False")
            for img in images:
                try:
                    img.calc = deepcopy(calculator)
                except (TypeError, AttributeError):
                    img.calc = calculator

            neb = NEB(
                images,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )

        opt_logfile = None if verbosity <= 1 else sys.stdout
        dyn: Optimizer = optimizer(neb, trajectory=trajectory, logfile=opt_logfile)  # type: ignore[arg-type]

        if verbosity >= 2:
            logger.info(f"Starting NEB optimization with {optimizer.__name__}")

        dyn.run(fmax=fmax, steps=neb_steps)

        try:
            neb_forces = neb.get_forces()
            final_fmax: float | None = float(np.max(np.abs(neb_forces)))
        except (AttributeError, RuntimeError, ValueError):
            final_fmax = None

        result["final_fmax"] = final_fmax
        result["neb_converged"] = final_fmax is not None and final_fmax < fmax
        result["steps_taken"] = int(dyn.nsteps)

        if not result["neb_converged"] and result.get("error") is None:
            result["error"] = (
                f"NEB did not converge (final_fmax={final_fmax}, fmax={fmax})"
            )

        if verbosity >= 1:
            fmax_str = f"{final_fmax:.6f}" if final_fmax is not None else "unknown"
            if result["neb_converged"]:
                logger.info(
                    "NEB converged in %d steps (final_fmax=%s < %.6f)",
                    result["steps_taken"],
                    fmax_str,
                    fmax,
                )
            else:
                logger.warning(
                    "NEB not converged after %d steps (final_fmax=%s, target_fmax=%.6f)",
                    result["steps_taken"],
                    fmax_str,
                    fmax,
                )

        _finalize_neb_result(result, neb.images, logger=logger)

        if use_torchsim and result["status"] == "success":
            result["force_calls"] = neb.get_force_calls()

        if verbosity >= 1 and result["status"] == "success":
            logger.info(
                f"TS found at image {result['ts_image_index']}/{len(neb.images) - 1}"
            )
            logger.info(f"  TS energy: {result['ts_energy']:.6f} eV")
            logger.info(f"  Barrier height: {result['barrier_height']:.6f} eV")
            if use_torchsim:
                logger.info(
                    "  GPU-batched force calls: %s",
                    result.get("force_calls"),
                )

    except KeyboardInterrupt:
        raise
    except (ValueError, RuntimeError, OSError) as e:
        result["error"] = str(e)
        if is_cuda_oom_error(e):
            cleanup_torch_cuda(logger=logger)
            if verbosity >= 1:
                logger.warning(
                    "Detected CUDA out-of-memory during NEB for pair %s — attempted GPU cleanup",
                    pair_id,
                )
        if verbosity >= 1:
            logger.error(
                f"Failed to find TS for pair {pair_id}: {type(e).__name__}: {e}"
            )

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_PROVENANCE_KEYS = (
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


def save_neb_result(
    result: dict[str, Any],
    output_dir: str,
    pair_id: str,
) -> None:
    """Save NEB result: TS and endpoint XYZ (when present) plus metadata JSON.

    Writes:
    - ``ts_{pair_id}.xyz`` on success when a TS geometry is present
    - ``reactant_{pair_id}.xyz`` / ``product_{pair_id}.xyz`` when
      ``reactant_structure`` / ``product_structure`` are on the result dict
    - ``neb_{pair_id}_metadata.json`` (includes schema/version/time and NEB params)
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    if result["status"] == "success" and result["transition_state"] is not None:
        _detach_calc(result["transition_state"])
        ts_path = os.path.join(output_dir, f"ts_{pair_id}.xyz")
        write(ts_path, result["transition_state"])
        logger.info(f"Saved TS structure to {ts_path}")

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
            logger.info(f"Saved {label} endpoint structure to {ep_path}")

    extra = {key: result[key] for key in _PROVENANCE_KEYS if key in result}
    extra["neb_backend"] = (
        "torchsim" if result.get("use_torchsim") else result.get("neb_backend", "ase")
    )
    metadata = ts_output_provenance(extra=extra)
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
        }
    )

    if result["status"] == "success":
        metadata["ts_image_index"] = result.get("ts_image_index")

    metadata_path = os.path.join(output_dir, f"neb_{pair_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved NEB metadata to {metadata_path}")
