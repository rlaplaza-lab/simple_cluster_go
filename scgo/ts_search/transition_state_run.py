"""High-level API for transition state finding via NEB.

Use :func:`run_transition_state_search` from this package, for example::

    from scgo.ts_search import run_transition_state_search
    from scgo.param_presets import get_default_params

    results = run_transition_state_search(
        ["Pt", "Pt", "Pt"],
        base_dir="Pt3_searches",
        params=get_default_params(),
        seed=42,
    )
"""

from __future__ import annotations

import contextlib
import glob
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
)
from scgo.database.metadata import add_metadata, get_metadata
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import (
    attach_slab_constraints_from_surface_config,
    surface_slab_constraint_summary,
)
from scgo.utils.helpers import (
    auto_niter_ts,
    filter_unique_minima,
    get_cluster_formula,
    validate_pair_id,
)
from scgo.utils.logging import configure_logging, get_logger
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import cleanup_torch_cuda, get_calculator_class
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags
from scgo.utils.ts_provenance import is_cuda_oom_error
from scgo.utils.validation import validate_composition

from .parallel_neb import ParallelNEBBatch
from .transition_state import (
    TorchSimNEB,
    _detach_calc,
    _make_torchsim_relaxer,
    find_transition_state,
    interpolate_path,
    save_neb_result,
)
from .transition_state_io import (
    _minima_provenance_dict,
    load_minima_by_composition,
    save_transition_state_results,
    select_structure_pairs,
    write_final_unique_ts,
)
from .ts_network import (
    add_ts_to_database,
    save_ts_network_metadata,
)


def _try_cleanup_cuda(logger: Any) -> None:
    """Release CUDA caches between pairs; ``cleanup_torch_cuda`` does not raise."""
    cleanup_torch_cuda(logger=logger)


def _neb_endpoint_copies(
    atoms_i: Atoms,
    atoms_j: Atoms,
    surface_config: SurfaceSystemConfig | None,
) -> tuple[Atoms, Atoms]:
    """Copy minima endpoints and, when requested, match GO slab ``FixAtoms`` policy."""
    react = atoms_i.copy()
    prod = atoms_j.copy()
    if surface_config is not None:
        attach_slab_constraints_from_surface_config(react, surface_config)
        attach_slab_constraints_from_surface_config(prod, surface_config)
    return react, prod


def _attach_minima_traceability(
    result: dict[str, Any],
    minima: list[tuple[float, Any]],
    i: int,
    j: int,
) -> None:
    """Record list indices and per-endpoint GO provenance on a TS result dict."""
    result["minima_indices"] = [int(i), int(j)]
    result["minima_provenance"] = [
        _minima_provenance_dict(minima, i),
        _minima_provenance_dict(minima, j),
    ]


def _build_failed_ts_result(
    *,
    pair_id: str,
    error: Exception,
    energy_i: float | None,
    energy_j: float | None,
    neb_n_images: int,
    neb_spring_constant: float,
    use_torchsim: bool,
    neb_fmax: float,
    neb_steps: int | str,
    neb_interpolation_method: str,
    neb_climb: bool,
    neb_align_endpoints: bool,
    neb_perturb_sigma: float,
    neb_interpolation_mic: bool,
    neb_tangent_method: str = DEFAULT_NEB_TANGENT_METHOD,
) -> dict[str, Any]:
    """Build a normalized failed-result payload for TS search."""
    return {
        "status": "failed",
        "pair_id": pair_id,
        "neb_converged": False,
        "n_images": neb_n_images,
        "spring_constant": neb_spring_constant,
        "reactant_energy": float(energy_i) if energy_i is not None else None,
        "product_energy": float(energy_j) if energy_j is not None else None,
        "ts_energy": None,
        "barrier_height": None,
        "barrier_forward": None,
        "barrier_reverse": None,
        "transition_state": None,
        "ts_image_index": None,
        "error": str(error),
        "use_torchsim": use_torchsim,
        "use_parallel_neb": False,
        "fmax": neb_fmax,
        "neb_steps": int(neb_steps)
        if isinstance(neb_steps, (int, np.integer))
        else neb_steps,
        "interpolation_method": neb_interpolation_method,
        "climb": neb_climb,
        "align_endpoints": neb_align_endpoints,
        "perturb_sigma": neb_perturb_sigma,
        "neb_interpolation_mic": neb_interpolation_mic,
        "neb_tangent_method": neb_tangent_method,
        "retry_attempted": False,
        "retry_success": False,
        "retry_history": [],
        "final_fmax": None,
        "steps_taken": None,
    }


def run_transition_state_search(
    composition: list[str],
    base_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    max_pairs: int | None = None,
    energy_gap_threshold: float | None = 1.0,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = 0.1,
    neb_n_images: int = 3,
    neb_spring_constant: float = 0.1,
    neb_fmax: float = 0.05,
    neb_steps: int | str = "auto",
    neb_climb: bool = False,
    neb_interpolation_method: str = "idpp",
    neb_align_endpoints: bool = True,
    neb_perturb_sigma: float = 0.0,
    neb_interpolation_mic: bool = False,
    neb_tangent_method: str = DEFAULT_NEB_TANGENT_METHOD,
    use_torchsim: bool = False,
    use_parallel_neb: bool = False,
    torchsim_params: dict | None = None,
    validate_ts_by_frequency: bool = False,
    imag_freq_threshold: float = 50.0,
    # Post-processing controls
    dedupe_minima: bool = True,
    minima_energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    dedupe_ts: bool = True,
    tag_ts_in_db: bool = True,
    ts_energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    surface_config: SurfaceSystemConfig | None = None,
) -> list[dict[str, Any]]:
    """Run transition state search for clusters of given composition.

    Loads minima from previous global optimization searches, pairs nearby structures,
    and finds transition states connecting them using nudged elastic band (NEB) with
    geodesic interpolation for initial path generation.

    Args:
        composition: List of atomic symbols specifying the cluster composition,
            e.g., ["Pt", "Pt", "Pt"] for Pt₃.
        base_dir: Base directory containing run_*/ subdirectories with
            previous optimization results. If None, uses ``{formula}_searches``.
        params: Dictionary of run parameters including:
            - "calculator": Calculator name (e.g., "MACE", "EMT"). Required.
            - "calculator_kwargs": Optional kwargs for calculator initialization.
            Other fields are ignored.
        seed: Integer seed for random number generation. Default None.
        verbosity: Logging verbosity (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        max_pairs: Maximum number of structure pairs to evaluate. If None, evaluates all pairs.
            Default None.
        energy_gap_threshold: Only pair structures with energy gap below this threshold (eV).
            If None, pairs all structures. Default 1.0.
        similarity_tolerance: Cumulative difference tolerance for structure comparison.
            Structures with cumulative difference below this are considered too similar to pair.
            Default `DEFAULT_COMPARATOR_TOL`.
        similarity_pair_cor_max: Maximum single distance difference tolerance for similarity.
            Default 0.1 Å.
        neb_n_images: Number of intermediate NEB images. Default 3 (recommended).
        neb_spring_constant: Spring constant for NEB band (eV/Ų). Default 0.1 (MACE gas sweep).
        neb_fmax: Maximum force convergence for NEB (eV/Å). Default 0.05.
        neb_steps: Maximum NEB optimization steps. Default 'auto' (resolved with auto_niter_ts).
        neb_climb: Use climbing image NEB for better TS convergence. Default False.
        neb_interpolation_method: Path interpolation method ('idpp' or 'linear'). Default 'idpp'.
        neb_interpolation_mic: If True, use minimum-image convention for NEB path
            interpolation (ASE ``NEB.interpolate(mic=True)``). Use for periodic cells
            (e.g. slabs). Default False (gas-phase clusters).
        neb_tangent_method: ASE NEB tangent method (``ase.mep.neb.NEB`` ``method``).
            Default ``improvedtangent``.
        use_torchsim: Use TorchSim for GPU-efficient batched force evaluation (MACE only).
            For ``calculator="UMA"`` (or without ``scgo[mace]``), this is coerced to
            ``False`` and ASE NEB with the selected calculator is used instead.
        torchsim_params: Optional parameters for TorchSimBatchRelaxer when use_torchsim=True. If
            `torchsim_params['max_steps']` is 'auto', it will be resolved the same way as `neb_steps`.
        surface_config: When set, the same :class:`scgo.surface.config.SurfaceSystemConfig`
            used for GA (``optimizer_params["ga"]["surface_config"]``). Endpoint structures
            are copied per pair and :func:`attach_slab_constraints_from_surface_config`
            is applied so NEB slab fixing matches global optimization (fully frozen slab,
            bottom-N layers fixed, or fully mobile slab per config). Default None leaves
            constraints on loaded minima unchanged.

    Returns:
        List of result dictionaries from find_transition_state(). Each contains:
        - "status": "success" or "failed"
        - "pair_id": Structure pair identifier
        - "barrier_height": Activation energy (eV)
        - "transition_state": Atoms object of TS (if successful).  The returned
            Atoms will have any attached calculator removed to avoid holding GPU
            memory across calls.
        - "reactant_energy" / "product_energy": Endpoint energies (eV)
        - "neb_converged": Whether NEB converged
        - Other metadata fields

    Raises:
        ValueError: If composition is empty or invalid, or calculator is unavailable.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    _try_cleanup_cuda(logger)

    validate_composition(composition, allow_empty=False)
    rng = ensure_rng(seed)

    if use_parallel_neb and not use_torchsim:
        raise ValueError("use_parallel_neb requires use_torchsim=True")

    formula = get_cluster_formula(composition)
    if base_dir is not None:
        ts_output_dir = str(Path(base_dir))
    else:
        ts_output_dir = str(Path(f"{formula}_searches"))

    # Default params if not provided
    if params is None:
        params = {"calculator": "EMT", "calculator_kwargs": {}}

    # Initialize calculator class once; actual instances will be created per
    # pair so that any internal caches (weights, layers, autograd graphs) are
    # freed promptly when the object is deleted.  This avoids slow memory
    # growth when a single calculator is reused across many NEBs.
    calculator_name = params.get("calculator", "EMT")
    calculator_kwargs = params.get("calculator_kwargs", {})

    use_torchsim, use_parallel_neb = resolve_ts_torchsim_flags(
        calculator_name,
        use_torchsim,
        use_parallel_neb,
        logger=logger,
    )

    try:
        calculator_class = get_calculator_class(calculator_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to locate calculator class {calculator_name}: {e}")
        raise ValueError(f"Cannot initialize calculator: {e}") from e

    # Load minima from database
    if verbosity >= 1:
        logger.info(f"Loading minima for composition {formula}")

    minima_by_formula = load_minima_by_composition(
        ts_output_dir, composition, prefer_final_unique=True
    )

    # Resolve any 'auto' step-counts to concrete integers using the
    # TS-specific heuristic (`auto_niter_ts`) which provides a larger NEB budget.
    if neb_steps in ("auto", None):
        neb_steps = auto_niter_ts(composition)

    # Ensure we don't mutate caller dict; prefer a concise ternary as suggested by ruff
    torchsim_params = {} if torchsim_params is None else dict(torchsim_params)

    if torchsim_params.get("max_steps") in ("auto", None):
        torchsim_params["max_steps"] = auto_niter_ts(composition)

    run_context: dict[str, Any] = {
        "calculator_name": calculator_name,
        "neb_fmax": neb_fmax,
        "neb_steps_resolved": int(neb_steps)
        if isinstance(neb_steps, int)
        else neb_steps,
        "neb_backend": "torchsim" if use_torchsim else "ase",
        "use_parallel_neb": use_parallel_neb,
        "neb_climb": neb_climb,
        "neb_interpolation_method": neb_interpolation_method,
        "neb_n_images": neb_n_images,
        "neb_spring_constant": neb_spring_constant,
        "neb_align_endpoints": neb_align_endpoints,
        "neb_perturb_sigma": neb_perturb_sigma,
        "neb_interpolation_mic": neb_interpolation_mic,
        "neb_tangent_method": neb_tangent_method,
    }
    if surface_config is not None:
        run_context["surface_slab_constraints"] = surface_slab_constraint_summary(
            surface_config
        )

    if not minima_by_formula:
        logger.error(f"No minima found in {ts_output_dir}")
        _try_cleanup_cuda(logger)
        return []

    minima = minima_by_formula.get(formula, [])

    # Deduplicate minima coming from multiple run_* databases to avoid
    # quadratic blow-up during pairwise similarity filtering. This mirrors the
    # final deduplication performed in `run_minima.py` for global optimization results.
    if dedupe_minima:
        original_count = len(minima)
        minima = filter_unique_minima(minima, minima_energy_tolerance)
        if verbosity >= 1 and len(minima) != original_count:
            logger.info(
                f"Deduplicated minima for {formula}: {original_count} -> {len(minima)} unique entries"
            )

    if len(minima) < 2:
        logger.error(f"Only {len(minima)} minima found, need at least 2 to find TS")
        _try_cleanup_cuda(logger)
        return []

    if verbosity >= 1:
        logger.info(f"Found {len(minima)} minima for {formula}")

    # Select structure pairs
    pairs = select_structure_pairs(
        minima,
        max_pairs=max_pairs,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
        surface_aware=bool(neb_interpolation_mic),
    )

    if not pairs:
        logger.error("No suitable pairs found for TS search")
        _try_cleanup_cuda(logger)
        return []

    if verbosity >= 1:
        logger.info(f"Selected {len(pairs)} structure pairs for TS search")

    # Find transition states
    ts_results = []
    result_dir = Path(ts_output_dir) / f"ts_results_{formula}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # If requested, run multiple TorchSim NEBs together using ParallelNEBBatch.
    if use_parallel_neb:
        if not use_torchsim:
            raise ValueError("use_parallel_neb requires use_torchsim=True")

        ts_params = torchsim_params or {}
        relaxer = _make_torchsim_relaxer(**ts_params)

        # Prepare endpoint batch (collect reactant/product images for all pairs)
        endpoints = []
        pair_end_idx = []  # (start_idx, end_idx) per pair in endpoints list
        for i, j in pairs:
            ri, rj = _neb_endpoint_copies(minima[i][1], minima[j][1], surface_config)
            endpoints.append(ri)
            endpoints.append(rj)
            pair_end_idx.append((len(endpoints) - 2, len(endpoints)))

        # Compute endpoint single-point energies with the shared relaxer
        ep_results = relaxer.relax_batch(endpoints, steps=0)

        # Map endpoint energies back to pairs
        pair_endpoint_energies: list[tuple[float, float]] = []
        for start, end in pair_end_idx:
            react_e = ep_results[start][0]
            prod_e = ep_results[end - 1][0]
            pair_endpoint_energies.append((float(react_e), float(prod_e)))

        # Build TorchSimNEB instances for all pairs (images copied so callers' atoms not mutated)
        neb_instances: list[TorchSimNEB] = []
        pair_ids: list[str] = []
        pair_meta: list[dict] = []
        for idx_pair, (i, j) in enumerate(pairs):
            energy_i, atoms_i = minima[i]
            energy_j, atoms_j = minima[j]
            pair_id = f"{i}_{j}"
            react_ep, prod_ep = _neb_endpoint_copies(atoms_i, atoms_j, surface_config)
            images = interpolate_path(
                react_ep,
                prod_ep,
                n_images=neb_n_images,
                method=neb_interpolation_method,
                mic=neb_interpolation_mic,
                align_endpoints=neb_align_endpoints,
                perturb_sigma=neb_perturb_sigma,
                rng=rng,
            )

            neb = TorchSimNEB(
                images,
                relaxer,
                k=neb_spring_constant,
                climb=neb_climb,
                method=neb_tangent_method,
            )
            neb_instances.append(neb)
            pair_ids.append(pair_id)

            # Store reactant/product energy (from batch) for later result construction
            react_e, prod_e = pair_endpoint_energies[idx_pair]
            pair_meta.append({"reactant_energy": react_e, "product_energy": prod_e})

        # Run parallel NEB optimization (global step cap scaled by number of NEBs)
        neb_steps_i = int(neb_steps)
        max_total = neb_steps_i * max(1, len(neb_instances))
        batch = ParallelNEBBatch(neb_instances, relaxer, max_total_steps=max_total)
        batch_results = batch.run_optimization(fmax=neb_fmax, max_steps=neb_steps)

        # Convert ParallelNEBBatch summaries to find_transition_state-like results
        for neb_idx, neb in enumerate(neb_instances):
            summary = batch_results[neb_idx]
            result: dict[str, Any] = {
                "status": "failed",
                "barrier_height": None,
                "barrier_forward": None,
                "barrier_reverse": None,
                "transition_state": None,
                "ts_energy": None,
                "ts_image_index": None,
                "reactant_energy": pair_meta[neb_idx]["reactant_energy"],
                "product_energy": pair_meta[neb_idx]["product_energy"],
                "neb_converged": bool(summary.get("converged", False)),
                "error": summary.get("error"),
                "pair_id": pair_ids[neb_idx],
                "n_images": neb_n_images,
                "spring_constant": neb_spring_constant,
                "use_torchsim": True,
                "climb": neb_climb,
                "neb_interpolation_mic": bool(neb_interpolation_mic),
                "neb_tangent_method": neb_tangent_method,
            }

            # Extract TS as the highest-energy image (energies should be available via SinglePointCalculator)
            images = neb.images
            max_energy = -float("inf")
            max_idx = None
            ts_atoms = None
            for idx_img, atoms in enumerate(images):
                try:
                    e = float(atoms.get_potential_energy())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    e = None
                if e is not None and e > max_energy:
                    max_energy = e
                    max_idx = idx_img
                    ts_atoms = atoms

            if max_idx is not None:
                result["ts_energy"] = float(max_energy)
                result["ts_image_index"] = int(max_idx)
                result["transition_state"] = ts_atoms.copy()

                # Compute barriers relative to lower endpoint
                min_endpoint = min(result["reactant_energy"], result["product_energy"])
                result["barrier_height"] = float(result["ts_energy"] - min_endpoint)
                result["barrier_forward"] = (
                    float(result["ts_energy"] - result["reactant_energy"])
                    if result.get("reactant_energy") is not None
                    else None
                )
                result["barrier_reverse"] = (
                    float(result["ts_energy"] - result["product_energy"])
                    if result.get("product_energy") is not None
                    else None
                )

            # Endpoint-as-TS is considered non-converged/failed for consistency
            # with the serial find_transition_state() path.
            endpoint_ts = (
                max_idx in (0, len(images) - 1) if max_idx is not None else False
            )
            if endpoint_ts:
                result["neb_converged"] = False
                result["error"] = (
                    f"NEB returned endpoint as TS (image {max_idx}); "
                    "no interior saddle located"
                )

            # Final status: only treat as success if NEB converged, TS energy exists,
            # and the TS is interior to the band.
            if (
                result["neb_converged"]
                and result.get("ts_energy") is not None
                and not endpoint_ts
            ):
                result["status"] = "success"
            else:
                # If NEB reports converged but no TS energy was extracted, mark failed
                if result["neb_converged"] and result.get("ts_energy") is None:
                    logger.warning(
                        "Parallel NEB converged but no TS energy for pair %s; marking as failed",
                        result.get("pair_id"),
                    )

            # Add optimizer/force call metadata when available
            result["final_fmax"] = summary.get("final_fmax")
            result["force_calls"] = summary.get("force_calls")
            result["steps_taken"] = summary.get("steps_taken")
            result["fmax"] = neb_fmax
            result["neb_steps"] = (
                int(neb_steps) if isinstance(neb_steps, int) else neb_steps
            )
            result["interpolation_method"] = neb_interpolation_method
            result["use_parallel_neb"] = True
            result["align_endpoints"] = neb_align_endpoints
            result["perturb_sigma"] = neb_perturb_sigma
            react_atoms = neb.images[0].copy()
            prod_atoms = neb.images[-1].copy()
            _detach_calc(react_atoms)
            _detach_calc(prod_atoms)
            result["reactant_structure"] = react_atoms
            result["product_structure"] = prod_atoms

            # drop any calculator from the copied TS returned by ParallelNEBBatch
            if result.get("transition_state") is not None:
                _detach_calc(result["transition_state"])

            pi, pj = pairs[neb_idx]
            _attach_minima_traceability(result, minima, pi, pj)

            ts_results.append(result)

            # Save result (same behavior as non-parallel path)
            save_neb_result(result, str(result_dir), result["pair_id"])

            _try_cleanup_cuda(logger)

    else:
        for idx, (i, j) in enumerate(pairs, 1):
            _try_cleanup_cuda(logger)

            energy_i, atoms_i = minima[i]
            energy_j, atoms_j = minima[j]

            pair_id = f"{i}_{j}"

            if verbosity >= 1:
                logger.info("[%d/%d] Finding TS for pair %s", idx, len(pairs), pair_id)
                logger.info(
                    "  Structure %d: %s eV",
                    i,
                    f"{energy_i:.6f}" if energy_i is not None else "None",
                )
                logger.info(
                    "  Structure %d: %s eV",
                    j,
                    f"{energy_j:.6f}" if energy_j is not None else "None",
                )

            result: dict[str, Any] | None = None
            calculator: Any = None
            for attempt in range(2):
                react_ep, prod_ep = _neb_endpoint_copies(
                    atoms_i, atoms_j, surface_config
                )
                if not use_torchsim:
                    calculator = calculator_class(**calculator_kwargs)
                    react_ep.calc = calculator
                    prod_ep.calc = calculator
                try:
                    result = find_transition_state(
                        react_ep,
                        prod_ep,
                        calculator if not use_torchsim else None,
                        output_dir=str(result_dir),
                        pair_id=pair_id,
                        rng=rng,
                        n_images=neb_n_images,
                        spring_constant=neb_spring_constant,
                        fmax=neb_fmax,
                        neb_steps=neb_steps,
                        climb=neb_climb,
                        interpolation_method=neb_interpolation_method,
                        align_endpoints=neb_align_endpoints,
                        perturb_sigma=neb_perturb_sigma,
                        neb_interpolation_mic=neb_interpolation_mic,
                        neb_tangent_method=neb_tangent_method,
                        use_torchsim=use_torchsim,
                        torchsim_params=torchsim_params,
                        verbosity=verbosity,
                        validate_ts_by_frequency=validate_ts_by_frequency,
                        imag_freq_threshold=imag_freq_threshold,
                    )
                    break
                except KeyboardInterrupt:
                    raise
                except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    AttributeError,
                    TypeError,
                ) as e:
                    oom = is_cuda_oom_error(e)
                    if oom and attempt == 0:
                        logger.warning(
                            "OOM during TS search for pair %s (attempt %d): %s; clearing cache and retrying",
                            pair_id,
                            attempt + 1,
                            e,
                        )
                        _try_cleanup_cuda(logger)
                        continue

                    logger.exception(
                        "Unexpected error while finding TS for pair %s: %s", pair_id, e
                    )
                    if oom:
                        _try_cleanup_cuda(logger)
                        logger.warning(
                            "Detected CUDA OOM for pair %s; freed cached GPU memory",
                            pair_id,
                        )

                    result = _build_failed_ts_result(
                        pair_id=pair_id,
                        error=e,
                        energy_i=energy_i,
                        energy_j=energy_j,
                        neb_n_images=neb_n_images,
                        neb_spring_constant=neb_spring_constant,
                        use_torchsim=use_torchsim,
                        neb_fmax=neb_fmax,
                        neb_steps=neb_steps,
                        neb_interpolation_method=neb_interpolation_method,
                        neb_climb=neb_climb,
                        neb_align_endpoints=neb_align_endpoints,
                        neb_perturb_sigma=neb_perturb_sigma,
                        neb_interpolation_mic=neb_interpolation_mic,
                        neb_tangent_method=neb_tangent_method,
                    )
                    break
            assert result is not None

            if result.get("transition_state") is not None:
                _detach_calc(result["transition_state"])

            _attach_minima_traceability(result, minima, i, j)

            ts_results.append(result)

            # Save result
            save_neb_result(result, str(result_dir), pair_id)

            if not use_torchsim and calculator is not None:
                del calculator

            _try_cleanup_cuda(logger)
    # Save summary
    save_transition_state_results(
        ts_results,
        str(result_dir),
        composition,
        run_context=run_context,
    )

    # Also save TS network metadata (used by downstream analysis and by tests)
    # `minima` is the list of minima loaded earlier in this function.
    save_ts_network_metadata(
        ts_results,
        str(result_dir),
        composition,
        minima_count=len(minima),
        minima=minima,
        minima_base_dir=ts_output_dir,
        run_context=run_context,
    )

    # --- Post-processing: deduplicate unique TSs and optionally tag DBs ---
    if dedupe_ts or tag_ts_in_db:
        unique_ts = write_final_unique_ts(
            ts_results,
            str(result_dir),
            composition,
            energy_tolerance=ts_energy_tolerance,
            minima=minima,
            minima_base_dir=ts_output_dir,
            run_context=run_context,
        )

        if tag_ts_in_db and unique_ts:
            db_files = glob.glob(
                os.path.join(ts_output_dir, "run_*", "**", "*.db"), recursive=True
            )
            basename_to_path = {os.path.basename(p): p for p in db_files}
            logger.debug(
                "Tagging: discovered DB basenames for %s: %s",
                ts_output_dir,
                list(basename_to_path.keys()),
            )

            for item in unique_ts:
                ts_energy = item.get("ts_energy")
                atoms_obj = item.get("_atoms_obj")

                edge_list: list[dict[str, Any]] = list(
                    item.get("connected_edges") or []
                )
                if not edge_list:
                    logger.warning(
                        "Skipping unique TS without connected_edges while tagging DB: %s",
                        item.get("filename"),
                    )
                    continue

                for edge in edge_list:
                    pair_id = edge.get("pair_id")
                    mi = edge.get("minima_indices")
                    if (
                        pair_id is None
                        or not isinstance(mi, (list, tuple))
                        or len(mi) != 2
                    ):
                        continue
                    i, j = int(mi[0]), int(mi[1])
                    barrier = edge.get("barrier_height")
                    neb_conv = edge.get("neb_converged")
                    endpoint_prov = edge.get("minima_provenance")

                    db_candidate = None
                    src_db_i = None
                    src_db_j = None
                    if 0 <= i < len(minima):
                        src_db_i = get_metadata(minima[i][1], "source_db")
                    if 0 <= j < len(minima):
                        src_db_j = get_metadata(minima[j][1], "source_db")

                    if src_db_i and src_db_i in basename_to_path:
                        db_candidate = basename_to_path[src_db_i]
                    elif src_db_j and src_db_j in basename_to_path:
                        db_candidate = basename_to_path[src_db_j]

                    if db_candidate is None:
                        logger.warning(
                            "No minima DB found to tag TS %s (searched run_*/*.db under %s); "
                            "available_db_basenames=%s src_db_i=%s src_db_j=%s",
                            pair_id,
                            ts_output_dir,
                            list(basename_to_path.keys()),
                            src_db_i,
                            src_db_j,
                        )
                        continue

                    try:
                        atoms_for_db = (
                            atoms_obj.copy() if atoms_obj is not None else None
                        )

                        if atoms_for_db is not None:

                            def _get_min_id(idx: int, key: str):
                                if not (0 <= idx < len(minima)):
                                    return None
                                return get_metadata(minima[idx][1], key)

                            # Persist GO provenance on TS so add_ts_to_database can store it
                            run_id_src = _get_min_id(i, "run_id") or _get_min_id(
                                j, "run_id"
                            )
                            trial_src = (
                                _get_min_id(i, "trial")
                                or _get_min_id(i, "trial_id")
                                or _get_min_id(j, "trial")
                                or _get_min_id(j, "trial_id")
                            )
                            if run_id_src is not None or trial_src is not None:
                                add_metadata(
                                    atoms_for_db,
                                    run_id=run_id_src,
                                    trial_id=trial_src,
                                )

                            add_metadata(
                                atoms_for_db,
                                connects=[i, j],
                                minima_source_db=[src_db_i, src_db_j],
                                minima_confids=[
                                    _get_min_id(i, "confid"),
                                    _get_min_id(j, "confid"),
                                ],
                                minima_unique_ids=[
                                    _get_min_id(i, "unique_id"),
                                    _get_min_id(j, "unique_id"),
                                ],
                                ts_connects_minima=f"{i}_{j}",
                            )

                        success = add_ts_to_database(
                            ts_structure=atoms_for_db,
                            ts_energy=float(ts_energy),
                            minima_idx_1=int(i),
                            minima_idx_2=int(j),
                            db_file=db_candidate,
                            pair_id=str(pair_id),
                            barrier_height=float(barrier)
                            if barrier is not None
                            else 0.0,
                            endpoint_provenance=endpoint_prov,
                            canonical_ts=True,
                            neb_converged=bool(neb_conv),
                        )
                        if not success:
                            logger.warning(
                                "add_ts_to_database returned False for %s -> %s",
                                pair_id,
                                db_candidate,
                            )
                    except (
                        sqlite3.DatabaseError,
                        sqlite3.OperationalError,
                        OSError,
                        ValueError,
                    ) as e:
                        logger.exception(
                            "Failed to add TS %s to DB %s (%s)",
                            pair_id,
                            db_candidate,
                            type(e).__name__,
                        )

    # Final summary for TS search
    if verbosity >= 1:
        num_success = sum(1 for r in ts_results if r.get("status") == "success")
        logger.info(
            f"TS search complete for {formula}: {len(ts_results)} result(s) ({num_success} successful)."
        )
        logger.info(f"Results written to: {result_dir}")

    _try_cleanup_cuda(logger)

    return ts_results


def integrate_ts_to_database(
    ts_results: list[dict[str, Any]],
    minima_database_file: str,
    verbosity: int = 1,
) -> int:
    """Add found transition states to the minima database.

    Iterates over ``ts_results`` and calls the module-level ``add_ts_to_database`` for
    each successful TS. Returns the number of TS entries successfully added.

    Row-level failures are logged and skipped; systemic filesystem/DB errors
    cause an early return (or are re-raised by underlying helpers).
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    if not os.path.exists(minima_database_file):
        logger.warning(f"Minima database not found: {minima_database_file}")
        return 0

    added = 0
    for result in ts_results:
        if result.get("status") != "success":
            continue

        try:
            pair_id = result.get("pair_id")
            ts_structure = result.get("transition_state")
            ts_energy = result.get("ts_energy")
            barrier = result.get("barrier_height")

            minima_idx_1, minima_idx_2 = -1, -1
            mid = result.get("minima_indices")
            if isinstance(mid, (list, tuple)) and len(mid) == 2:
                minima_idx_1, minima_idx_2 = int(mid[0]), int(mid[1])
            elif pair_id is not None:
                with contextlib.suppress(ValueError, TypeError):
                    minima_idx_1, minima_idx_2 = validate_pair_id(str(pair_id))

            endpoint_provenance = result.get("minima_provenance")

            success = add_ts_to_database(
                ts_structure=ts_structure,
                ts_energy=float(ts_energy) if ts_energy is not None else 0.0,
                minima_idx_1=int(minima_idx_1),
                minima_idx_2=int(minima_idx_2),
                db_file=minima_database_file,
                pair_id=pair_id,
                barrier_height=float(barrier) if barrier is not None else 0.0,
                endpoint_provenance=endpoint_provenance,
                canonical_ts=False,
                neb_converged=bool(result.get("neb_converged", False)),
            )

            if success:
                added += 1
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            ValueError,
        ) as e:
            logger.exception(
                "Failed to add TS %s to DB %s: %s",
                result.get("pair_id"),
                minima_database_file,
                type(e).__name__,
            )
            continue

    return added


def run_transition_state_campaign(
    compositions: list[list[str]],
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    neb_params: dict | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run transition state search across multiple compositions.

    Convenience function to run TS searches for multiple cluster compositions
    in sequence.

    Args:
        compositions: List of compositions, each a list of atomic symbols.
            E.g., [["Pt"], ["Pt", "Pt"], ["Pt", "Au"]]
        output_dir: Base output directory. Minima are read from
            ``{output_dir}/{formula}_searches``. If None, uses ``{formula}_searches``
            per composition under the current working directory.
        params: Global optimization parameters with "calculator" field.
        seed: Random seed. Default None.
        verbosity: Logging verbosity (0, 1, or 2). Default 1.
        neb_params: Dictionary of NEB-specific parameters forwarded to
            :func:`run_transition_state_search` (same keys as that function's kwargs),
            including ``surface_config`` to align slab ``FixAtoms`` with GA.

            The TS results for each composition will have calculators detached
            before being stored in the returned mapping to prevent memory leaks
            when running many compositions in sequence.
            - "max_pairs": Maximum structure pairs to evaluate
            - "energy_gap_threshold": Energy cutoff for pairing
            - "neb_n_images": Number of intermediate images
            - "neb_spring_constant": Spring constant
            - "neb_fmax": Force convergence criterion
            - "neb_steps": Maximum optimization steps
            - "neb_tangent_method": ASE NEB tangent method (e.g. ``improvedtangent``)
            - "surface_config": Optional :class:`scgo.surface.config.SurfaceSystemConfig`

    Returns:
        Dictionary mapping formula strings to lists of TS result dictionaries.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    neb_params = neb_params or {}

    campaign_results = {}

    for composition in compositions:
        formula = get_cluster_formula(composition)

        if verbosity >= 1:
            logger.info("Running TS search campaign for %s", formula)

        if output_dir is not None:
            comp_output_dir = str(Path(output_dir) / f"{formula}_searches")
        else:
            comp_output_dir = f"{formula}_searches"

        try:
            logger.debug(
                "Using minima base directory for %s: %s", formula, comp_output_dir
            )
            results = run_transition_state_search(
                composition,
                base_dir=comp_output_dir,
                params=params,
                seed=seed,
                verbosity=verbosity,
                **neb_params,
            )
            for r in results:
                if r.get("transition_state") is not None:
                    _detach_calc(r["transition_state"])
            campaign_results[formula] = results
        except KeyboardInterrupt:
            raise
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error(
                f"Failed to run TS search for {formula}: {type(e).__name__}: {e}"
            )
            campaign_results[formula] = []

    return campaign_results
