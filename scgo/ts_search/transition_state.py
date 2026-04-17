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
from ase.vibrations import Vibrations
from scipy.optimize import linear_sum_assignment

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.torchsim_policy import coerce_find_transition_state_torchsim
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


def _make_torchsim_relaxer(**kwargs: Any) -> Any:
    """Build a TorchSim relaxer; resolve the class via module attribute for monkeypatching."""
    from scgo.calculators import torchsim_helpers as _tsh

    return _tsh.TorchSimBatchRelaxer(**kwargs)


def _attach_calculator_to_images(
    images: list[Atoms], calculator: Calculator | None
) -> None:
    """Attach ``calculator`` to each image (deepcopy when supported)."""
    if calculator is None:
        return
    for img in images:
        try:
            img.calc = deepcopy(calculator)
        except (TypeError, AttributeError):
            img.calc = calculator


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

        Optimization note: when images already carry PES forces (for example
        because a caller such as `ParallelNEBBatch` just evaluated them in a
        single batched call), avoid re-invoking the TorchSim relaxer. The
        relaxer and `_force_calls` are only updated when we actually perform a
        TorchSim evaluation here.
        """

        def _has_cached_forces(atoms: Atoms) -> bool:
            if "forces" in atoms.arrays and atoms.arrays["forces"] is not None:
                return True
            if atoms.calc is None:
                return False
            f = None
            with contextlib.suppress(
                AttributeError,
                NotImplementedError,
                RuntimeError,
                ValueError,
            ):
                f = atoms.get_forces()
            return f is not None and getattr(f, "size", 0) > 0

        if all(_has_cached_forces(img) for img in self.images):
            return super().get_forces()

        # Otherwise perform actual TorchSim batched evaluation
        self._force_calls += 1
        results = self.relaxer.relax_batch(self.images, steps=0)

        for atoms, (energy, relaxed_atoms) in zip(self.images, results, strict=True):
            attach_singlepoint_from_relax_output(
                atoms, energy, relaxed_atoms, require_forces=True
            )

        # Compute NEB forces (includes spring/tangent contributions)
        return super().get_forces()

    def get_force_calls(self) -> int:
        """Return the number of times forces have been evaluated."""
        return self._force_calls

    def increment_force_calls(self, n: int = 1) -> None:
        """Increment the internal force-evaluation call counter.

        This allows external batched runners (for example
        `ParallelNEBBatch`) to record that a relaxer evaluation covered this
        NEB's images so that `get_force_calls()` reflects both direct NEB
        evaluations and aggregated/batched evaluations.
        """
        self._force_calls += int(n)


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
    C = P.T @ Q
    U, S, Vt = np.linalg.svd(C)
    d = np.linalg.det(U @ Vt)
    # Guard against degenerate case where det is exactly zero (singular SVD);
    # treat as proper rotation (sign +1) to avoid a singular D matrix.
    sign_d = np.sign(d) if d != 0.0 else 1.0
    D = np.diag([1.0, 1.0, sign_d])
    R = U @ D @ Vt
    return R


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

    # Work on copies so originals are not mutated
    a1_copy = atoms1.copy()
    a2_copy = atoms2.copy()

    if align_endpoints:
        # Compute permutation mapping so atom identities are consistently ordered.
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
            # Non-periodic alignment: rigid Kabsch transform minimizes RMSD.
            P = a1_copy.get_positions().copy()
            Q = new_pos.copy()
            Pc = P - P.mean(axis=0)
            Qc = Q - Q.mean(axis=0)
            R = _kabsch_rotation(Pc, Qc)
            Qrot = (Qc @ R.T) + P.mean(axis=0)
            a2_copy.set_positions(Qrot)

    # Build images including endpoints (copies) and let ASE interpolate
    images = [a1_copy] + [a1_copy.copy() for _ in range(n_images)] + [a2_copy]
    neb = NEB(images, method=DEFAULT_NEB_TANGENT_METHOD)
    # Interpolate unconstrained positions first; endpoint/image constraints
    # (e.g., fixed slab atoms) are enforced during subsequent optimization.
    neb.interpolate(method=method, mic=mic, apply_constraint=False)
    images = neb.images

    # Optionally perturb interior images (endpoints unchanged)
    if perturb_sigma and perturb_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        for img in images[1:-1]:
            disp = rng.normal(
                scale=float(perturb_sigma), size=img.get_positions().shape
            )
            img.set_positions(img.get_positions() + disp)

    return images


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
    validate_ts_by_frequency: bool = False,
    imag_freq_threshold: float = 50.0,
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

    use_torchsim = coerce_find_transition_state_torchsim(
        use_torchsim=use_torchsim,
        calculator=calculator,
        pair_id=pair_id,
        logger=logger,
    )

    if not use_torchsim:
        validate_calculator_attached(atoms1, "NEB reactant")
        validate_calculator_attached(atoms2, "NEB product")

    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    if trajectory is None:
        trajectory = os.path.join(output_dir, f"neb_{pair_id}.traj")

    # Extract initial energies using extract_energy_from_atoms (safe for TorchSim where atoms have no calculator)
    reactant_energy_tmp = extract_energy_from_atoms(atoms1)
    product_energy_tmp = extract_energy_from_atoms(atoms2)

    # For standard calculator-driven NEB we require explicit endpoint energies.
    # For TorchSim workflows endpoints may not carry energies on the Atoms and
    # will instead be computed by the relaxer; allow None in that case and
    # defer strict validation to after the relaxer call.
    if not use_torchsim:
        if reactant_energy_tmp is None:
            raise ValueError(
                f"Cannot extract energy from reactant atoms for pair {pair_id}"
            )
        if product_energy_tmp is None:
            raise ValueError(
                f"Cannot extract energy from product atoms for pair {pair_id}"
            )

    reactant_energy = (
        float(reactant_energy_tmp) if reactant_energy_tmp is not None else None
    )
    product_energy = (
        float(product_energy_tmp) if product_energy_tmp is not None else None
    )

    if verbosity >= 1:
        logger.info(f"Finding transition state for pair {pair_id}")
        if reactant_energy is not None:
            logger.info(f"  Reactant energy: {reactant_energy:.6f} eV")
        if product_energy is not None:
            logger.info(f"  Product energy: {product_energy:.6f} eV")

    result = {
        "status": "failed",
        "barrier_height": None,
        "barrier_forward": None,
        "barrier_reverse": None,
        "transition_state": None,
        "ts_energy": None,
        "ts_image_index": None,
        "reactant_energy": reactant_energy,
        "product_energy": product_energy,
        "neb_converged": False,
        "error": None,
        "pair_id": pair_id,
        "n_images": n_images,
        "spring_constant": spring_constant,
        "use_torchsim": use_torchsim,
        "climb": climb,
        "align_endpoints": bool(align_endpoints),
        "perturb_sigma": float(perturb_sigma),
        "neb_interpolation_mic": bool(neb_interpolation_mic),
        "neb_tangent_method": neb_tangent_method,
        "fmax": float(fmax),
        "neb_steps": int(neb_steps)
        if isinstance(neb_steps, (int, np.integer))
        else neb_steps,
        "interpolation_method": interpolation_method,
    }

    try:
        # Generate initial path using specified interpolation method
        if verbosity >= 2:
            logger.info(
                f"Generating initial path with {interpolation_method} interpolation"
            )
        # Initial band: interpolate_path uses neb.interpolate(..., apply_constraint=False)
        # so FixAtoms on slab endpoints do not break path construction (see docstring).
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

        pos0 = images[0].get_positions()
        posN = images[-1].get_positions()
        if np.allclose(pos0, posN, atol=1e-8):
            if verbosity >= 1:
                logger.warning(
                    "Endpoints are identical for pair %s; skipping NEB and "
                    "marking as endpoint failure",
                    pair_id,
                )

            result["neb_converged"] = False
            result["status"] = "failed"
            result["error"] = "endpoints identical; no interior TS (endpoint)"
            result["transition_state"] = deepcopy(images[0])
            _detach_calc(result["transition_state"])
            ep_e = extract_energy_from_atoms(images[0])
            result["ts_energy"] = float(ep_e) if ep_e is not None else None
            result["ts_image_index"] = 0
            result["barrier_height"] = 0.0
            result["barrier_forward"] = 0.0
            result["barrier_reverse"] = 0.0
            result["reactant_structure"] = deepcopy(images[0])
            result["product_structure"] = deepcopy(images[0])
            _detach_calc(result["reactant_structure"])
            _detach_calc(result["product_structure"])
            return result

        neb: NEB
        if use_torchsim:
            ts_params = torchsim_params or {}
            relaxer = _make_torchsim_relaxer(**ts_params)

            # Compute endpoint energies with TorchSim (single-point calculation)
            endpoints = [images[0], images[-1]]
            ep_results = relaxer.relax_batch(endpoints, steps=0)
            result["reactant_energy"] = ep_results[0][0]
            result["product_energy"] = ep_results[1][0]

            if verbosity >= 2:
                logger.info(f"Using TorchSim batched NEB (climb={climb})")

            # Create TorchSimNEB with GPU-efficient batched force evaluation
            neb = TorchSimNEB(
                images,
                relaxer,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )
        else:
            # Standard ASE NEB
            if calculator is None:
                raise ValueError("Calculator required when use_torchsim=False")

            _attach_calculator_to_images(images, calculator)

            neb = NEB(
                images,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )

        # Run optimization (silence ASE optimizer output by default to match GA/BH)
        opt_logfile = None if verbosity <= 1 else sys.stdout
        dyn: Optimizer = optimizer(neb, trajectory=trajectory, logfile=opt_logfile)  # type: ignore[arg-type]

        if verbosity >= 2:
            logger.info(f"Starting NEB optimization with {optimizer.__name__}")

        dyn.run(fmax=fmax, steps=neb_steps)

        # Derive final NEB forces as a proxy for convergence instead of assuming success.
        try:
            neb_forces = neb.get_forces()
            final_fmax = float(np.max(np.abs(neb_forces)))
        except (AttributeError, RuntimeError, ValueError):
            # Missing/invalid NEB forces — treat as non-converged
            final_fmax = None

        result["final_fmax"] = final_fmax
        result["neb_converged"] = final_fmax is not None and final_fmax < fmax

        # Record optimizer step count when available so callers can see how far
        # the NEB progressed (ASE Optimizer implementations commonly expose
        # `.nsteps` or `.get_number_of_steps()`). This is useful because the
        # ASE optimizer output is usually silenced in library runs.
        steps_taken = None
        try:
            if hasattr(dyn, "get_number_of_steps") and callable(
                dyn.get_number_of_steps
            ):
                steps_taken = int(dyn.get_number_of_steps())
            elif hasattr(dyn, "nsteps"):
                steps_taken = int(dyn.nsteps)
            else:
                steps_taken = None
        except (TypeError, ValueError):
            steps_taken = None

        result["steps_taken"] = steps_taken

        # If NEB did not converge, mark the run as failed so callers don't
        # mistakenly treat a non-converged NEB as a successful TS search.
        if not result["neb_converged"] and result.get("error") is None:
            result["error"] = (
                f"NEB did not converge (final_fmax={final_fmax}, fmax={fmax})"
            )

        # Log a concise convergence summary so users can quickly see progress
        # even when ASE optimizer logging is suppressed.
        if verbosity >= 1:
            if result["neb_converged"]:
                logger.info(
                    "NEB converged in %s steps (final_fmax=%.6f < %.6f)",
                    steps_taken if steps_taken is not None else "unknown",
                    final_fmax if final_fmax is not None else float("nan"),
                    fmax,
                )
            else:
                logger.warning(
                    "NEB not converged after %s steps (final_fmax=%s, target_fmax=%.6f)",
                    steps_taken if steps_taken is not None else neb_steps,
                    f"{final_fmax:.6f}" if final_fmax is not None else "unknown",
                    fmax,
                )
        # Find TS as the highest-energy image
        images = neb.images
        result["reactant_structure"] = deepcopy(images[0])
        result["product_structure"] = deepcopy(images[-1])
        _detach_calc(result["reactant_structure"])
        _detach_calc(result["product_structure"])
        max_energy_idx = 0
        max_energy = -np.inf

        ts_energy = None
        ts_atoms = None

        for idx, atoms in enumerate(images):
            energy = atoms.get_potential_energy()
            if energy > max_energy:
                max_energy = energy
                max_energy_idx = idx
                ts_energy = energy
                ts_atoms = atoms

        # Ensure endpoint energies and TS energy are available for numeric operations
        if (
            result.get("reactant_energy") is None
            or result.get("product_energy") is None
        ):
            raise RuntimeError(
                f"Missing endpoint energies after NEB for pair {pair_id}: "
                f"reactant={result.get('reactant_energy')}, product={result.get('product_energy')}"
            )
        if ts_energy is None:
            raise RuntimeError(f"No TS energy found after NEB for pair {pair_id}")

        reactant_energy_result = float(result["reactant_energy"])
        product_energy_result = float(result["product_energy"])
        min_endpoint_energy = min(reactant_energy_result, product_energy_result)

        barrier_height = ts_energy - min_endpoint_energy

        if max_energy_idx == 0 or max_energy_idx == len(images) - 1:
            result["neb_converged"] = False
            result["status"] = "failed"
            result["error"] = (
                f"NEB returned endpoint as TS (image {max_energy_idx}); "
                "no interior saddle located"
            )
            result["transition_state"] = deepcopy(ts_atoms)
            _detach_calc(result["transition_state"])
            result["ts_energy"] = float(ts_energy)
            result["ts_image_index"] = max_energy_idx
            result["barrier_height"] = float(barrier_height)
            result["barrier_forward"] = float(ts_energy - reactant_energy_result)
            result["barrier_reverse"] = float(ts_energy - product_energy_result)

            if verbosity >= 1:
                logger.warning(
                    "NEB reported endpoint as TS for pair %s (image %d) — marking as non-converged",
                    pair_id,
                    max_energy_idx,
                )
        else:
            result["status"] = "success" if result["neb_converged"] else "failed"
            result["barrier_height"] = float(barrier_height)
            result["barrier_forward"] = float(ts_energy - reactant_energy_result)
            result["barrier_reverse"] = float(ts_energy - product_energy_result)
            result["transition_state"] = deepcopy(ts_atoms)
            _detach_calc(result["transition_state"])
            result["ts_energy"] = float(ts_energy)
            result["ts_image_index"] = max_energy_idx

            if use_torchsim:
                result["force_calls"] = neb.get_force_calls()

            if verbosity >= 1:
                logger.info(f"TS found at image {max_energy_idx}/{len(images) - 1}")
                logger.info(f"  TS energy: {ts_energy:.6f} eV")
                logger.info(f"  Barrier height: {barrier_height:.6f} eV")
                if use_torchsim:
                    logger.info(
                        "  GPU-batched force calls: %s",
                        result["force_calls"],
                    )

        if (
            result["status"] == "success"
            and validate_ts_by_frequency
            and ts_atoms is not None
            and calculator is not None
        ):
            try:
                ts_atoms_check = ts_atoms.copy()
                ts_atoms_check.calc = calculator
                vib = Vibrations(ts_atoms_check, name=f"ts_vib_{pair_id}")
                vib.run()
                frequencies = vib.get_frequencies()
                vib.clean()

                # Count significant imaginary frequencies (more negative than -threshold)
                # Small imaginary frequencies between -threshold and 0 are treated as numerical noise
                imag_mask_significant = frequencies < -imag_freq_threshold
                num_significant_imag = int(np.sum(imag_mask_significant))

                # Count all imaginary frequencies (including small numerical noise)
                imag_mask_all = frequencies < 0.0
                num_all_imag = int(np.sum(imag_mask_all))

                if verbosity >= 1:
                    logger.info(
                        f"TS vibrational analysis for pair {pair_id}: "
                        f"{num_significant_imag} significant imaginary freq (< -{imag_freq_threshold} cm-1), "
                        f"{num_all_imag} total imaginary (< 0 cm-1)"
                    )
                    if num_all_imag > num_significant_imag:
                        small_imag = frequencies[
                            (frequencies < 0.0) & (frequencies >= -imag_freq_threshold)
                        ]
                        logger.debug(
                            f"  {len(small_imag)} small imaginary modes (noise, threshold {imag_freq_threshold} cm-1): "
                            f"{np.round(small_imag, 1)}"
                        )

                # TS must have exactly 1 significant imaginary frequency
                if num_significant_imag != 1:
                    result["status"] = "failed"
                    result["error"] = (
                        f"TS validation failed: expected exactly 1 significant imaginary frequency "
                        f"(threshold -{imag_freq_threshold} cm-1), found {num_significant_imag}. "
                        f"Total imaginary modes: {num_all_imag}."
                    )
                    result["ts_vib_frequencies"] = frequencies.tolist()
                    result["ts_vib_num_imag"] = num_all_imag
                    result["ts_vib_num_significant_imag"] = num_significant_imag
                    result["ts_vib_validated"] = False
                    if verbosity >= 1:
                        logger.info("TS rejected: %s", result["error"])
                else:
                    result["ts_vib_frequencies"] = frequencies.tolist()
                    result["ts_vib_num_imag"] = num_all_imag
                    result["ts_vib_num_significant_imag"] = num_significant_imag
                    result["ts_vib_validated"] = True
                    if verbosity >= 1:
                        logger.info(
                            "TS validated: exactly one significant imaginary frequency"
                        )

            except (OSError, RuntimeError, ValueError) as e:
                result["status"] = "failed"
                result["error"] = f"TS vibrational analysis failed: {e}"
                result["ts_vib_validated"] = False
                if verbosity >= 1:
                    logger.error(
                        f"TS vibrational analysis failed for pair {pair_id}: {e}"
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


def _neb_provenance_extra(result: dict[str, Any]) -> dict[str, Any]:
    """Subset of result fields useful for JSON provenance."""
    extra: dict[str, Any] = {}
    for key in (
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
    ):
        if key in result:
            extra[key] = result[key]
    extra["neb_backend"] = (
        "torchsim" if result.get("use_torchsim") else result.get("neb_backend", "ase")
    )
    return extra


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

    metadata = ts_output_provenance(extra=_neb_provenance_extra(result))
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
