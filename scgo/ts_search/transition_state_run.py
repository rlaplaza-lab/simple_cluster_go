"""Transition state finding via NEB (implementation module).

Most users should call :func:`~scgo.runner_api.run_transition_state_search` from
``scgo`` (composition string / :class:`~ase.Atoms` / symbol list, plus
``ts_kwargs`` for NEB options). This module exposes the same operations with a
flat keyword interface (symbol list composition), for example::

    from scgo.ts_search import run_transition_state_search
    from scgo.param_presets import get_default_params

    results = run_transition_state_search(
        ["Pt", "Pt", "Pt"],
        output_dir="Pt3_searches",
        params=get_default_params(),
        seed=42,
    )
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Any

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
)
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

from .parallel_neb import run_parallel_neb_search
from .transition_state import (
    _detach_calc,
    attach_minima_traceability,
    find_transition_state,
    make_ts_result,
    save_neb_result,
)
from .transition_state_io import (
    load_minima_by_composition,
    save_transition_state_results,
    select_structure_pairs,
    write_final_unique_ts,
)
from .ts_network import (
    add_ts_to_database,
    save_ts_network_metadata,
    tag_unique_ts_in_databases,
)

__all__ = [
    "run_transition_state_search",
    "run_transition_state_campaign",
    "integrate_ts_to_database",
]


def _run_serial_neb_search(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Any]],
    *,
    result_dir: Path,
    calculator_class: Any,
    calculator_kwargs: dict[str, Any],
    surface_config: SurfaceSystemConfig | None,
    rng: Any,
    use_torchsim: bool,
    torchsim_params: dict[str, Any],
    neb_n_images: int,
    neb_spring_constant: float,
    neb_fmax: float,
    neb_steps: int,
    neb_climb: bool,
    neb_interpolation_method: str,
    neb_align_endpoints: bool,
    neb_perturb_sigma: float,
    neb_interpolation_mic: bool,
    neb_tangent_method: str,
    verbosity: int,
) -> list[dict[str, Any]]:
    """Run NEBs sequentially via :func:`find_transition_state` (one calc per pair)."""
    logger = get_logger(__name__)
    ts_results: list[dict[str, Any]] = []

    for idx, (i, j) in enumerate(pairs, 1):
        cleanup_torch_cuda(logger=logger)

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

        react_ep = atoms_i.copy()
        prod_ep = atoms_j.copy()
        if surface_config is not None:
            attach_slab_constraints_from_surface_config(react_ep, surface_config)
            attach_slab_constraints_from_surface_config(prod_ep, surface_config)

        calculator: Any = None
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
            )
        except (RuntimeError, ValueError) as e:
            logger.error(
                "Unexpected error while finding TS for pair %s: %s: %s",
                pair_id,
                type(e).__name__,
                e,
            )
            if is_cuda_oom_error(e):
                cleanup_torch_cuda(logger=logger)
                logger.warning(
                    "Detected CUDA OOM for pair %s; freed cached GPU memory",
                    pair_id,
                )
            result = make_ts_result(
                pair_id=pair_id,
                n_images=neb_n_images,
                spring_constant=neb_spring_constant,
                use_torchsim=use_torchsim,
                fmax=neb_fmax,
                neb_steps=neb_steps,
                interpolation_method=neb_interpolation_method,
                climb=neb_climb,
                align_endpoints=neb_align_endpoints,
                perturb_sigma=neb_perturb_sigma,
                neb_interpolation_mic=neb_interpolation_mic,
                neb_tangent_method=neb_tangent_method,
                reactant_energy=energy_i,
                product_energy=energy_j,
                error=str(e),
            )

        if result.get("transition_state") is not None:
            _detach_calc(result["transition_state"])

        attach_minima_traceability(result, minima, i, j)
        ts_results.append(result)
        save_neb_result(result, str(result_dir), pair_id)

        if not use_torchsim and calculator is not None:
            del calculator

        cleanup_torch_cuda(logger=logger)

    return ts_results


def run_transition_state_search(
    composition: list[str],
    output_dir: str | Path | None = None,
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
        output_dir: Base directory containing run_*/ subdirectories with
            previous optimization results. If None, uses ``{formula}_searches``.
        params: Dictionary of run parameters including:
            - "calculator": Calculator name (e.g., "MACE", "EMT"). Required.
            - "calculator_kwargs": Optional kwargs for calculator initialization.
            Other fields are ignored.
        seed: Integer seed for random number generation. Default None.
        verbosity: Logging verbosity (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        max_pairs: Maximum number of structure pairs to evaluate. If None, evaluates all pairs.
        energy_gap_threshold: Only pair structures with energy gap below this threshold (eV).
        similarity_tolerance: Cumulative difference tolerance for structure comparison.
        similarity_pair_cor_max: Maximum single distance difference tolerance for similarity.
        neb_n_images: Number of intermediate NEB images. Default 3 (recommended).
        neb_spring_constant: Spring constant for NEB band (eV/Ų). Default 0.1.
        neb_fmax: Maximum force convergence for NEB (eV/Å). Default 0.05.
        neb_steps: Maximum NEB optimization steps. Default 'auto' (resolved with auto_niter_ts).
        neb_climb: Use climbing image NEB for better TS convergence. Default False.
        neb_interpolation_method: Path interpolation method ('idpp' or 'linear'). Default 'idpp'.
        neb_interpolation_mic: If True, use minimum-image convention for NEB path
            interpolation. Use for periodic cells (e.g. slabs). Default False.
        neb_tangent_method: ASE NEB tangent method.
        use_torchsim: Use TorchSim for GPU-efficient batched force evaluation (MACE/UMA only).
        torchsim_params: Optional parameters for TorchSimBatchRelaxer when use_torchsim=True.
        surface_config: When set, the same :class:`scgo.surface.config.SurfaceSystemConfig`
            used for GA. Endpoint structures are copied per pair and slab
            constraints are applied to match GO behavior.

    Returns:
        List of result dictionaries from :func:`find_transition_state`.

    Raises:
        ValueError: If composition is empty or invalid, or calculator is unavailable.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)
    cleanup_torch_cuda(logger=logger)

    validate_composition(composition, allow_empty=False)
    rng = ensure_rng(seed)

    if use_parallel_neb and not use_torchsim:
        raise ValueError("use_parallel_neb requires use_torchsim=True")

    formula = get_cluster_formula(composition)
    ts_output_dir = (
        str(Path(output_dir))
        if output_dir is not None
        else str(Path(f"{formula}_searches"))
    )

    if params is None:
        params = {"calculator": "EMT", "calculator_kwargs": {}}
    calculator_name = params.get("calculator", "EMT")
    calculator_kwargs = params.get("calculator_kwargs", {})

    use_torchsim, use_parallel_neb = resolve_ts_torchsim_flags(
        calculator_name,
        use_torchsim,
        use_parallel_neb,
    )

    try:
        calculator_class = get_calculator_class(calculator_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to locate calculator class {calculator_name}: {e}")
        raise ValueError(f"Cannot initialize calculator: {e}") from e

    if verbosity >= 1:
        logger.info(f"Loading minima for composition {formula}")

    minima_by_formula = load_minima_by_composition(
        ts_output_dir, composition, prefer_final_unique=True
    )

    if neb_steps in ("auto", None):
        neb_steps = auto_niter_ts(composition)

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
        cleanup_torch_cuda(logger=logger)
        return []

    minima = minima_by_formula.get(formula, [])

    if dedupe_minima:
        original_count = len(minima)
        minima = filter_unique_minima(minima, minima_energy_tolerance)
        if verbosity >= 1 and len(minima) != original_count:
            logger.info(
                f"Deduplicated minima for {formula}: {original_count} -> {len(minima)} unique entries"
            )

    if len(minima) < 2:
        logger.error(f"Only {len(minima)} minima found, need at least 2 to find TS")
        cleanup_torch_cuda(logger=logger)
        return []

    if verbosity >= 1:
        logger.info(f"Found {len(minima)} minima for {formula}")

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
        cleanup_torch_cuda(logger=logger)
        return []

    if verbosity >= 1:
        logger.info(f"Selected {len(pairs)} structure pairs for TS search")

    result_dir = Path(ts_output_dir) / f"ts_results_{formula}"
    result_dir.mkdir(parents=True, exist_ok=True)

    if use_parallel_neb:
        ts_results = run_parallel_neb_search(
            pairs,
            minima,
            result_dir=result_dir,
            surface_config=surface_config,
            rng=rng,
            neb_n_images=neb_n_images,
            neb_spring_constant=neb_spring_constant,
            neb_fmax=neb_fmax,
            neb_steps=int(neb_steps),
            neb_climb=neb_climb,
            neb_interpolation_method=neb_interpolation_method,
            neb_align_endpoints=neb_align_endpoints,
            neb_perturb_sigma=neb_perturb_sigma,
            neb_interpolation_mic=neb_interpolation_mic,
            neb_tangent_method=neb_tangent_method,
            torchsim_params=torchsim_params,
        )
        cleanup_torch_cuda(logger=logger)
    else:
        ts_results = _run_serial_neb_search(
            pairs,
            minima,
            result_dir=result_dir,
            calculator_class=calculator_class,
            calculator_kwargs=calculator_kwargs,
            surface_config=surface_config,
            rng=rng,
            use_torchsim=use_torchsim,
            torchsim_params=torchsim_params,
            neb_n_images=neb_n_images,
            neb_spring_constant=neb_spring_constant,
            neb_fmax=neb_fmax,
            neb_steps=int(neb_steps) if isinstance(neb_steps, int) else neb_steps,
            neb_climb=neb_climb,
            neb_interpolation_method=neb_interpolation_method,
            neb_align_endpoints=neb_align_endpoints,
            neb_perturb_sigma=neb_perturb_sigma,
            neb_interpolation_mic=neb_interpolation_mic,
            neb_tangent_method=neb_tangent_method,
            verbosity=verbosity,
        )

    save_transition_state_results(
        ts_results,
        str(result_dir),
        composition,
        run_context=run_context,
    )

    save_ts_network_metadata(
        ts_results,
        str(result_dir),
        composition,
        minima_count=len(minima),
        minima=minima,
        minima_base_dir=ts_output_dir,
        run_context=run_context,
    )

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
            tag_unique_ts_in_databases(unique_ts, minima, ts_output_dir)

    if verbosity >= 1:
        num_success = sum(1 for r in ts_results if r.get("status") == "success")
        logger.info(
            f"TS search complete for {formula}: {len(ts_results)} result(s) ({num_success} successful)."
        )
        logger.info(f"Results written to: {result_dir}")

    cleanup_torch_cuda(logger=logger)

    return ts_results


def integrate_ts_to_database(
    ts_results: list[dict[str, Any]],
    minima_database_file: str,
    verbosity: int = 1,
) -> int:
    """Add found transition states to the minima database.

    Iterates over ``ts_results`` and calls the module-level :func:`add_ts_to_database`
    for each successful TS. Returns the number of TS entries successfully added.

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
            if ts_structure is None or ts_energy is None or barrier is None:
                logger.warning(
                    "Skipping TS %s DB integration due to missing fields: structure=%s ts_energy=%s barrier=%s",
                    pair_id,
                    ts_structure is not None,
                    ts_energy,
                    barrier,
                )
                continue

            minima_idx_1, minima_idx_2 = -1, -1
            mid = result.get("minima_indices")
            if isinstance(mid, (list, tuple)) and len(mid) == 2:
                minima_idx_1, minima_idx_2 = int(mid[0]), int(mid[1])
            elif pair_id is not None:
                with contextlib.suppress(ValueError, TypeError):
                    minima_idx_1, minima_idx_2 = validate_pair_id(str(pair_id))

            success = add_ts_to_database(
                ts_structure=ts_structure,
                ts_energy=float(ts_energy),
                minima_idx_1=int(minima_idx_1),
                minima_idx_2=int(minima_idx_2),
                db_file=minima_database_file,
                pair_id=pair_id,
                barrier_height=float(barrier),
                endpoint_provenance=result.get("minima_provenance"),
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
            logger.error(
                "Failed to add TS %s to DB %s: %s: %s",
                result.get("pair_id"),
                minima_database_file,
                type(e).__name__,
                e,
            )
            continue

    return added


def run_transition_state_campaign(
    compositions: list[list[str]],
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    ts_kwargs: dict | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run :func:`run_transition_state_search` for multiple compositions in sequence.

    Minima are read from ``{output_dir}/{formula}_searches`` (or just
    ``{formula}_searches`` when ``output_dir`` is None). Extra search/NEB
    arguments are forwarded via ``ts_kwargs``. Failures for one composition never
    abort the whole campaign — they are logged and that formula gets an empty
    result list.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    ts_kwargs = ts_kwargs or {}
    campaign_results: dict[str, list[dict[str, Any]]] = {}

    for composition in compositions:
        formula = get_cluster_formula(composition)
        comp_output_dir = (
            str(Path(output_dir) / f"{formula}_searches")
            if output_dir is not None
            else f"{formula}_searches"
        )
        if verbosity >= 1:
            logger.info("Running TS search campaign for %s", formula)

        try:
            results = run_transition_state_search(
                composition,
                output_dir=comp_output_dir,
                params=params,
                seed=seed,
                verbosity=verbosity,
                **ts_kwargs,
            )
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error("Failed to run TS search for %s: %s", formula, e)
            campaign_results[formula] = []
            continue

        for r in results:
            if r.get("transition_state") is not None:
                _detach_calc(r["transition_state"])
        campaign_results[formula] = results

    return campaign_results
