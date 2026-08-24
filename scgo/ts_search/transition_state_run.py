"""Transition-state orchestration for NEB-based searches.

This module is the implementation layer behind ``scgo.runner_api`` TS helpers.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from scgo.algorithms.ga_common import resolve_neb_mobile_dims
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_TS_PAIR_COR_MAX,
)
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.exceptions import SCGOValidationError
from scgo.metadata.provenance import is_cuda_oom_error
from scgo.metadata.run_dir import ensure_run_id, save_run_dir_record
from scgo.param_presets import default_energy_gap_threshold, get_ts_defaults
from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import (
    surface_slab_constraint_summary,
)
from scgo.system_types import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    SystemType,
    as_adsorbate_definition,
    get_system_policy,
    resolve_connectivity_factor,
    resolve_neb_mic,
    validate_minimum_structure,
    validate_system_type_settings,
)
from scgo.utils.comparators import get_shared_mobile_atom_indices
from scgo.utils.helpers import (
    auto_niter_ts,
    copy_atoms,
    filter_unique_minima,
    get_cluster_formula,
)
from scgo.utils.logging import (
    configure_logging,
    get_logger,
    log_debug_v,
    log_info_v,
)
from scgo.utils.output_paths import resolve_ts_campaign_paths
from scgo.utils.path_keys import resolve_run_path_key
from scgo.utils.phase_logging import log_neb_search_summaries
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import cleanup_torch_cuda, get_calculator_class
from scgo.utils.timing_report import (
    build_timing_payload,
    log_timing_summary,
    sum_neb_seconds_from_ts_results,
    write_timing_file,
)
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags
from scgo.utils.ts_runner_kwargs import NebRunConfig
from scgo.utils.validation import validate_composition

from .neb_endpoints import prepare_neb_endpoints
from .parallel_neb import _evaluate_bands_in_chunks, run_parallel_neb_search
from .transition_state import (
    _detach_calc,
    attach_minima_traceability,
    find_transition_state,
    idpp_band_optimization_priority,
    interpolate_path,
    load_completed_neb_result,
    make_ts_result,
    save_neb_result,
    validate_initial_neb_energy_profile,
    validate_initial_neb_path,
)
from .transition_state_io import (
    load_minima_by_composition,
    resolve_ts_pair_select_cap,
    save_transition_state_results,
    select_structure_pairs,
    write_final_unique_ts,
)
from .ts_network import (
    save_ts_network_metadata,
    tag_unique_ts_in_databases,
)

__all__ = [
    "run_transition_state_search",
    "run_transition_state_campaign",
]


def _prioritize_adsorbate_pairs_by_idpp(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Any]],
    *,
    max_pairs: int,
    relaxer: Any,
    neb_n_images: int,
    neb_interpolation_method: str,
    neb_interpolation_mic: bool,
    neb_align_endpoints: bool,
    neb_perturb_sigma: float,
    rng: Any,
    system_type: SystemType | None,
    n_slab: int,
    n_core_mobile: int | None,
    n_adsorbate_mobile: int | None,
    adsorbate_fragment_lengths: list[int] | None,
    neb_surface_cell_remap: bool,
    neb_surface_lattice_rotation: bool,
    neb_surface_max_lattice_shift: int,
    max_endpoint_mismatch: float,
    neb_prescreen_clash_distance: float,
    min_saddle_prominence: float,
    neb_max_spurious_barrier: float,
    neb_interpolation_bond_tolerance_a: float | None,
    parallel_neb_max_batch_atoms: int | None,
    parallel_neb_max_bands: int | None,
    logger: Any,
    verbosity: int = 1,
) -> list[tuple[int, int]]:
    """Keep up to ``max_pairs`` adsorbate bands, preferring robust IDPP interiors.

    Endpoint-max IDPP paths are retained only when the oversampled pool holds no
    robust-interior candidate at all (CI-NEB can still salvage some).

    The per-pair image construction and geometry-only path validation run on the
    CPU (no GPU). All energy evaluations are then fused into per-band batches and
    evaluated through :func:`_evaluate_bands_in_chunks`, so the oversampled
    screen respects the same atom budget / band cap and CUDA-OOM retry used by
    the main pre-screen (a single unbounded ``relax_batch`` over every candidate
    image OOM'd the 16 GB T4). ``relaxer`` must be a
    :class:`TorchSimBatchRelaxer`.
    """
    # CPU-only stage: build images and validate geometry. Pairs that fail CPU
    # validation are dropped here so the batched energy eval never sees them.
    valid_pairs: list[tuple[int, int, list[Any]]] = []
    for i, j in pairs:
        try:
            images = interpolate_path(
                copy_atoms(minima[i][1]),
                copy_atoms(minima[j][1]),
                n_images=neb_n_images,
                method=neb_interpolation_method,
                mic=neb_interpolation_mic,
                align_endpoints=neb_align_endpoints,
                perturb_sigma=neb_perturb_sigma,
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
        except (SCGOValidationError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Adsorbate pair %s_%s dropped during IDPP geometry screen: %s",
                i,
                j,
                exc,
            )
            continue
        valid_pairs.append((i, j, images))

    if not valid_pairs:
        logger.info(
            "Adsorbate IDPP priority screen: 0/%d pairs passed geometry screening",
            len(pairs),
        )
        return []

    # Per-band energy evaluation through the atom-budgeted, OOM-retrying chunker
    # (no single unbounded fused launch, which OOM'd the 16 GB T4). Energy lists
    # are returned per band, in input order.
    bands = [images for _i, _j, images in valid_pairs]
    band_cap = (
        int(parallel_neb_max_bands)
        if parallel_neb_max_bands is not None and int(parallel_neb_max_bands) > 0
        else None
    )
    band_energy_lists = _evaluate_bands_in_chunks(
        bands,
        relaxer,
        atom_budget=parallel_neb_max_batch_atoms,
        band_cap=band_cap,
    )

    # Re-associate per-band energies with each pair.
    ranked: list[tuple[tuple[int, float, float], int, int]] = []
    for (i, j, _images), energies in zip(valid_pairs, band_energy_lists, strict=True):
        try:
            validate_initial_neb_energy_profile(
                energies,
                reference_reactant_energy=float(minima[i][0]),
                reference_product_energy=float(minima[j][0]),
                min_saddle_prominence=min_saddle_prominence,
                max_spurious_barrier=neb_max_spurious_barrier,
            )
        except (SCGOValidationError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Adsorbate pair %s_%s dropped during IDPP energy screen: %s",
                i,
                j,
                exc,
            )
            continue
        priority = idpp_band_optimization_priority(
            energies, min_saddle_prominence=min_saddle_prominence
        )
        if priority[0] <= 0:
            continue
        ranked.append((priority, i, j))

    ranked.sort(
        key=lambda item: (-item[0][0], -item[0][1], -item[0][2], item[1], item[2])
    )
    robust = [item for item in ranked if item[0][0] >= 2]
    # When the oversampled pool has activated IDPP bands, do not spend the
    # NEB budget on endpoint-max slides (CI-NEB often climbs into junk).
    chosen = robust if robust else ranked
    kept = [(i, j) for _priority, i, j in chosen[: int(max_pairs)]]
    logger.info(
        "Adsorbate IDPP priority screen: %d/%d pairs kept "
        "(%d robust-interior candidates in pool)",
        len(kept),
        len(pairs),
        len(robust),
    )
    return kept


def _run_serial_neb_search(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Any]],
    *,
    neb_cfg: NebRunConfig,
    run_dir: Path,
    calculator_class: Any,
    calculator_kwargs: dict[str, Any],
    rng: Any,
    use_torchsim: bool,
    verbosity: int,
    write_timing_json: bool = False,
) -> list[dict[str, Any]]:
    """Run NEBs sequentially via :func:`find_transition_state`.

    Without TorchSim a fresh ASE calculator is built for each pair; the TorchSim
    path instead shares one ``TorchSimBatchRelaxer`` across all pairs.
    """
    logger = get_logger(__name__)
    ts_results: list[dict[str, Any]] = []
    torchsim_params = neb_cfg.torchsim_params or {}
    system_type = neb_cfg.system_type
    neb_steps = (
        int(neb_cfg.neb_steps)
        if isinstance(neb_cfg.neb_steps, int)
        else neb_cfg.neb_steps
    )

    shared_relaxer = None
    if use_torchsim:
        from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

        shared_relaxer = TorchSimBatchRelaxer(**torchsim_params)

    for idx, (i, j) in enumerate(pairs, 1):
        if not use_torchsim:
            cleanup_torch_cuda(logger=logger)

        energy_i, atoms_i = minima[i]
        energy_j, atoms_j = minima[j]
        pair_id = f"{i}_{j}"
        pair_dir = run_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        resumed = load_completed_neb_result(pair_dir, pair_id)
        if resumed is not None:
            log_info_v(
                logger,
                "[%d/%d] Skipping pair %s (resumed success)",
                idx,
                len(pairs),
                pair_id,
                verbosity=verbosity,
            )
            resumed["system_type"] = system_type
            if "minima_indices" not in resumed:
                attach_minima_traceability(resumed, minima, i, j)
            ts_results.append(resumed)
            continue

        log_debug_v(
            logger,
            "[%d/%d] Finding TS for pair %s",
            idx,
            len(pairs),
            pair_id,
            verbosity=verbosity,
        )

        try:
            react_ep, prod_ep = prepare_neb_endpoints(atoms_i, atoms_j, neb_cfg)
        except (ValueError, SCGOValidationError) as e:
            skipped = make_ts_result(
                pair_id=pair_id,
                n_images=neb_cfg.neb_n_images,
                spring_constant=neb_cfg.neb_spring_constant,
                use_torchsim=use_torchsim,
                fmax=neb_cfg.neb_fmax,
                neb_steps=neb_steps,
                interpolation_method=neb_cfg.neb_interpolation_method,
                climb=neb_cfg.neb_climb,
                align_endpoints=neb_cfg.neb_align_endpoints,
                perturb_sigma=neb_cfg.neb_perturb_sigma,
                neb_interpolation_mic=neb_cfg.neb_interpolation_mic,
                neb_tangent_method=neb_cfg.neb_tangent_method,
                reactant_energy=energy_i,
                product_energy=energy_j,
                error=str(e),
            )
            skipped["status"] = "skipped"
            attach_minima_traceability(skipped, minima, i, j)
            ts_results.append(skipped)
            continue

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
                output_dir=str(pair_dir),
                pair_id=pair_id,
                rng=rng,
                use_torchsim=use_torchsim,
                relaxer=shared_relaxer,
                verbosity=verbosity,
                write_timing_json=write_timing_json,
                neb_cfg=neb_cfg,
            )
        except (RuntimeError, ValueError, SCGOValidationError) as e:
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
                n_images=neb_cfg.neb_n_images,
                spring_constant=neb_cfg.neb_spring_constant,
                use_torchsim=use_torchsim,
                fmax=neb_cfg.neb_fmax,
                neb_steps=neb_steps,
                interpolation_method=neb_cfg.neb_interpolation_method,
                climb=neb_cfg.neb_climb,
                align_endpoints=neb_cfg.neb_align_endpoints,
                perturb_sigma=neb_cfg.neb_perturb_sigma,
                neb_interpolation_mic=neb_cfg.neb_interpolation_mic,
                neb_tangent_method=neb_cfg.neb_tangent_method,
                reactant_energy=energy_i,
                product_energy=energy_j,
                error=str(e),
            )

        if result.get("transition_state") is not None:
            _detach_calc(result["transition_state"])
        result["system_type"] = system_type

        attach_minima_traceability(result, minima, i, j)
        ts_results.append(result)
        save_neb_result(result, str(pair_dir), pair_id, verbosity=verbosity)

        if not use_torchsim and calculator is not None:
            del calculator

    if use_torchsim:
        cleanup_torch_cuda(logger=logger)

    log_neb_search_summaries(
        logger,
        ts_results,
        verbosity=verbosity,
        run_dir=str(run_dir),
    )
    return ts_results


def _warn_on_surface_mobile_indices(
    minima: list[tuple[float, Any]],
    *,
    system_type: SystemType,
    n_slab: int = 0,
) -> None:
    r"""Log diagnostics when surface minima lack a usable mobile partition.

    When ``n_slab > 0`` (from ``surface_config``), pair comparison uses the slab
    prefix from the live surface template, not stored ``n_slab_atoms`` metadata.
    Warnings about ``all atoms mobile`` apply only when that partition is
    unavailable.
    """
    logger = get_logger(__name__)
    policy = get_system_policy(system_type)
    if not policy.uses_surface:
        return

    slab_n = int(n_slab) if n_slab > 0 else None
    sample_count = min(3, len(minima))
    for i in range(sample_count):
        for j in range(i + 1, sample_count):
            ai = minima[i][1]
            aj = minima[j][1]
            try:
                shared = get_shared_mobile_atom_indices(ai, aj, n_slab=slab_n)
            except (ValueError, SCGOValidationError):
                logger.warning(
                    "Surface TS pair (%d,%d) has no shared mobile atoms for comparison; "
                    "pair similarity may be skipped",
                    i,
                    j,
                )
                continue
            if slab_n is None and shared.size >= len(ai):
                logger.warning(
                    "Surface TS pair (%d,%d) compares all atoms as mobile; pass "
                    "surface_config so TS can use len(slab) as n_slab, or ensure "
                    "minima carry FixAtoms / n_slab_atoms metadata",
                    i,
                    j,
                )


def _apply_surface_ts_geometry_gate(
    ts_results: list[dict[str, Any]],
    *,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType,
    adsorbate_definition: Any | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: Any | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    binding_penetration_tolerance_a: float = 0.1,
    layer_cluster_threshold_ang: float = 0.4,
    n_slab_deposit: int | None = None,
    run_dir: Path | None = None,
    verbosity: int = 1,
) -> None:
    """Reject successful TS results that violate supported-deposit geometry.

    When a result is demoted, rewrite ``pair_*/neb_*_metadata.json`` under
    ``run_dir`` so resume cannot reload a stale success.
    """
    if surface_config is None:
        return
    policy = get_system_policy(system_type)
    if not (policy.uses_surface and policy.needs_supported_deposit_validation):
        return

    n_slab = len(surface_config.slab)
    cf = resolve_connectivity_factor(
        connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        surface_config=surface_config,
    )
    checks = (
        ("reactant", "reactant_structure"),
        ("product", "product_structure"),
        ("transition_state", "transition_state"),
    )
    for result in ts_results:
        if result.get("status") != "success":
            continue
        demoted = False
        for label, key in checks:
            atoms = result.get(key)
            if atoms is None:
                result["status"] = "failed"
                result["neb_converged"] = False
                result["error"] = f"Missing {label} structure for surface TS validation"
                demoted = True
                break
            try:
                validate_minimum_structure(
                    atoms,
                    system_type=system_type,
                    surface_config=surface_config,
                    n_slab=n_slab,
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=cf,
                    cluster_adsorbate_config=cluster_adsorbate_config,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    binding_penetration_tolerance_a=binding_penetration_tolerance_a,
                    n_slab_deposit=n_slab_deposit,
                )
            except SCGOValidationError as exc:
                result["status"] = "failed"
                result["neb_converged"] = False
                result["error"] = f"{label} failed surface geometry validation: {exc}"
                demoted = True
                break
        if demoted and run_dir is not None:
            pair_id = str(result.get("pair_id") or "")
            if pair_id:
                pair_dir = run_dir / f"pair_{pair_id}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                save_neb_result(result, str(pair_dir), pair_id, verbosity=verbosity)


def run_transition_state_search(
    composition: list[str],
    system_type: SystemType,
    output_dir: str | Path | None = None,
    searches_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    max_pairs: int | None = None,
    energy_gap_threshold: float | None = None,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = DEFAULT_TS_PAIR_COR_MAX,
    pair_core_rms_max: float | None = None,
    pair_score_gap_center: float | None = None,
    pair_score_gap_width: float | None = None,
    pair_score_cum_scale: float | None = None,
    pair_score_mismatch_scale: float | None = None,
    pair_score_core_rms_scale: float | None = None,
    pair_score_w_gap: float | None = None,
    pair_score_w_distinct: float | None = None,
    pair_score_w_mismatch: float | None = None,
    pair_score_w_core: float | None = None,
    neb_n_images: int | None = None,
    neb_spring_constant: float | None = None,
    neb_fmax: float | None = None,
    neb_steps: int | str | None = None,
    neb_climb: bool | None = None,
    neb_interpolation_method: str | None = None,
    neb_align_endpoints: bool | None = None,
    neb_perturb_sigma: float | None = None,
    neb_interpolation_mic: bool | None = None,
    neb_surface_cell_remap: bool | None = None,
    neb_surface_lattice_rotation: bool | None = None,
    neb_surface_max_lattice_shift: int | None = None,
    neb_tangent_method: str = DEFAULT_NEB_TANGENT_METHOD,
    max_endpoint_mismatch: float | None = None,
    neb_prescreen_clash_distance: float | None = None,
    min_saddle_prominence: float | None = None,
    neb_max_spurious_barrier: float | None = None,
    binding_penetration_tolerance_a: float | None = None,
    layer_cluster_threshold_ang: float | None = None,
    neb_interpolation_bond_tolerance_a: float | None = None,
    use_torchsim: bool = False,
    use_parallel_neb: bool | None = None,
    parallel_neb_max_bands: int | None = None,
    parallel_neb_max_batch_atoms: int | None = None,
    torchsim_params: dict | None = None,
    # Post-processing controls
    dedupe_minima: bool = True,
    minima_energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    dedupe_ts: bool = True,
    tag_ts_in_db: bool = True,
    ts_energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    surface_config: SurfaceSystemConfig | None = None,
    write_timing_json: bool = False,
    adsorbate_definition: Any | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run transition state search for clusters of given composition.

    Loads minima from previous global optimization searches, pairs nearby structures,
    and finds transition states connecting them using nudged elastic band (NEB) with
    IDPP (default) or linear interpolation for initial path generation.

    Prefer :func:`scgo.param_presets.get_ts_search_params` (or ``run_ts_search`` /
    ``run_go_ts``) for production defaults: shared ``neb_fmax=0.20`` and parallel
    TorchSim NEB for every system type; bare gas uses 5 images; adsorbates use
    7 images, climb, and ``energy_gap_threshold=0.75``. NEB knobs left as
    ``None`` (spring constant, fmax, steps, interpolation method, endpoint
    alignment, MIC, surface remap/rotation/shift, images, climb) resolve from
    the same per-system presets via :func:`~scgo.param_presets.get_ts_defaults`
    — e.g. adsorbate types get spring ``0.5`` / steps ``4000``, bare surfaces
    steps ``2000``. ``use_parallel_neb`` defaults to ``True`` whenever
    TorchSim is enabled (``None`` → on with TorchSim, off without). When
    ``neb_n_images``, ``neb_climb``, or ``energy_gap_threshold`` are omitted
    (``None``), values are taken from :func:`~scgo.param_presets.get_ts_defaults`
    and the same adsorbate-aware gap rule as :func:`~scgo.get_ts_search_params`.

    Args:
        composition: List of atomic symbols for the mobile region. For high-level
            ``run_go_ts`` / ``run_ts_search`` with ``*_adsorbate`` types, pass
            core-only symbols plus ``adsorbates=``; the runner supplies the full
            mobile composition here. For surface types without explicit adsorbate
            blocks, this is the supported cluster on the slab.
        output_dir: Campaign root directory. TS results are written to
            ``{path_key}_ts_results/`` as a sibling of ``{path_key}_searches/``.
            If None, uses the current working directory.
        searches_dir: Optional explicit path to the GO searches directory
            (``{path_key}_searches/``). When set, minima are loaded from here
            instead of ``{output_dir}/{path_key}_searches``.
        params: Dictionary of run parameters including:
            - "calculator": Calculator name (e.g., "MACE", "EMT"). Required.
            - "calculator_kwargs": Optional kwargs for calculator initialization.
            Other fields are ignored.
        seed: Integer seed for random number generation. Default None.
        verbosity: Logging verbosity (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
        max_pairs: Maximum number of structure pairs to evaluate. If None, evaluates all pairs.
        energy_gap_threshold: Only pair structures with energy gap below this threshold (eV).
            ``None`` selects the system-aware preset (``2.0`` bare / ``0.75`` adsorbate).
            Pass ``float("inf")`` to disable the gap filter.
        similarity_tolerance: Cumulative difference tolerance for structure comparison.
        similarity_pair_cor_max: Maximum single distance difference tolerance for similarity.
        pair_core_rms_max: Hard max core RMS (Å) for adsorbate+core pair gating.
        pair_score_gap_center: Preferred endpoint energy gap (eV) for ranking.
        pair_score_gap_width: Gaussian width (eV) around the preferred gap.
        pair_score_cum_scale: Scale (Å) for distinctness / adsorbate-hop ranking.
        pair_score_mismatch_scale: Scale (Å) for fingerprint ``max_diff`` ranking.
        pair_score_core_rms_scale: Scale (Å) for soft core-RMS ranking.
        pair_score_w_gap: Ranking weight for the energy-gap term.
        pair_score_w_distinct: Ranking weight for distinctness / adsorbate hop.
        pair_score_w_mismatch: Ranking weight for fingerprint mismatch.
        pair_score_w_core: Ranking weight for core RMS (adsorbate+core only).
        neb_n_images: Number of intermediate NEB images. ``None`` selects the
            system-aware preset (``5`` bare / ``7`` adsorbate).
        neb_spring_constant: Spring constant for NEB band (eV/Å²). ``None``
            selects the system-aware preset (``0.1`` bare / ``0.5`` adsorbate).
        neb_fmax: Maximum force convergence for NEB (eV/Å). Default 0.20
            (shared across system types; same as presets).
        neb_steps: Maximum NEB optimization steps. ``None`` selects the
            system-aware preset (``"auto"`` bare gas, resolved with
            auto_niter_ts / ``2000`` surfaces / ``4000`` adsorbate).
        neb_climb: Use climbing image NEB for better TS convergence. ``None``
            selects the system-aware preset (``False`` bare / ``True`` adsorbate).
        neb_interpolation_method: Path interpolation method ('idpp' or 'linear'). Default 'idpp'.
        neb_interpolation_mic: If True, use minimum-image convention for NEB path
            interpolation. Use for periodic cells (e.g. slabs). Default False.
        neb_tangent_method: ASE NEB tangent method.
        max_endpoint_mismatch: Optional Å geometric gate on comparator ``max_diff``;
            when set (adsorbate presets), also enables pre-NEB path/energy checks.
        use_torchsim: Use TorchSim for GPU-efficient batched force evaluation
            (MACE/UMA/UPET only).
            Low-level default ``False``; presets set ``True``.
        use_parallel_neb: Batch multiple NEB bands (requires TorchSim). Default
            ``None`` resolves to ``True`` when ``use_torchsim=True`` (same as
            presets) and ``False`` otherwise. Explicit ``True`` without TorchSim
            raises.
        parallel_neb_max_bands: Cap concurrent bands inside parallel NEB
            (``None`` = no band cap). Applied together with
            ``parallel_neb_max_batch_atoms``. Surface presets use ``4`` to avoid
            GPU OOM on large slab cells while keeping the parallel NEB path.
        parallel_neb_max_batch_atoms: Atom budget (sum of ``n_images * n_atoms``)
            for one fused parallel-NEB force batch. Applied together with
            ``parallel_neb_max_bands``; ``None`` disables the atom budget.
            Presets use ``6000`` (gas) / ``4000`` (surface).
        torchsim_params: Optional parameters for TorchSimBatchRelaxer when use_torchsim=True.
        surface_config: When set, the same :class:`scgo.surface.config.SurfaceSystemConfig`
            used for GA. Endpoint structures are copied per pair and slab
            constraints are applied to match GO behavior.
        adsorbate_definition: Optional; two-block mobile runs use blockwise NEB alignment
            when ``neb_align_endpoints`` is True.
        run_id: Optional existing TS run directory name under
            ``{path_key}_ts_results/``. When set, resumes that run and skips pairs
            whose ``neb_{pair_id}_metadata.json`` already has ``status="success"``.
            When ``None``, a new run id is generated.

    Returns:
        List of per-pair TS result dictionaries (one entry per selected pair,
        including skipped and failed pairs).

    Raises:
        SCGOValidationError: If the composition is invalid, ``params`` is missing
            or lacks ``"calculator"``, the calculator class cannot be located, or
            ``use_parallel_neb=True`` is requested without TorchSim.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)
    cleanup_torch_cuda(logger=logger)

    connectivity_factor = resolve_connectivity_factor(
        connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        surface_config=surface_config,
    )

    system_policy = get_system_policy(system_type)
    ts_defaults = get_ts_defaults(system_type)
    # Direct callers omitting NEB knobs resolve them from the same per-system
    # presets that ``get_ts_search_params`` / ``coerce_ts_params_to_runner_kwargs``
    # use, so a bare ``run_transition_state_search(...)`` call no longer runs
    # with hardcoded signature defaults (k=0.1, steps="auto", ...).
    if neb_n_images is None:
        neb_n_images = int(ts_defaults["neb_n_images"])
    if neb_spring_constant is None:
        neb_spring_constant = float(ts_defaults["neb_spring_constant"])
    if neb_fmax is None:
        neb_fmax = float(ts_defaults["neb_fmax"])
    if neb_steps is None:
        neb_steps = ts_defaults["neb_steps"]
    if neb_climb is None:
        neb_climb = bool(ts_defaults["neb_climb"])
    if neb_interpolation_method is None:
        neb_interpolation_method = str(ts_defaults["neb_interpolation_method"])
    if neb_align_endpoints is None:
        neb_align_endpoints = bool(ts_defaults["neb_align_endpoints"])
    if neb_perturb_sigma is None:
        neb_perturb_sigma = float(ts_defaults["neb_perturb_sigma"])
    if neb_interpolation_mic is None:
        neb_interpolation_mic = bool(ts_defaults["neb_interpolation_mic"])
    if neb_surface_cell_remap is None:
        neb_surface_cell_remap = bool(ts_defaults["neb_surface_cell_remap"])
    if neb_surface_lattice_rotation is None:
        neb_surface_lattice_rotation = bool(ts_defaults["neb_surface_lattice_rotation"])
    if neb_surface_max_lattice_shift is None:
        neb_surface_max_lattice_shift = int(
            ts_defaults["neb_surface_max_lattice_shift"]
        )
    if neb_prescreen_clash_distance is None:
        neb_prescreen_clash_distance = float(
            ts_defaults["neb_prescreen_clash_distance"]
        )
    if min_saddle_prominence is None:
        min_saddle_prominence = float(ts_defaults["min_saddle_prominence"])
    if neb_max_spurious_barrier is None:
        neb_max_spurious_barrier = float(ts_defaults["neb_max_spurious_barrier"])
    if binding_penetration_tolerance_a is None:
        binding_penetration_tolerance_a = float(
            ts_defaults["binding_penetration_tolerance_a"]
        )
    if layer_cluster_threshold_ang is None:
        layer_cluster_threshold_ang = float(ts_defaults["layer_cluster_threshold_ang"])
    if neb_interpolation_bond_tolerance_a is None:
        neb_interpolation_bond_tolerance_a = float(
            ts_defaults["neb_interpolation_bond_tolerance_a"]
        )
    if energy_gap_threshold is None:
        energy_gap_threshold = default_energy_gap_threshold(system_policy.has_adsorbate)
    validate_composition(
        composition,
        allow_empty=system_policy.slab_is_search_target
        and not system_policy.has_adsorbate,
    )
    validate_system_type_settings(
        system_type=system_type,
        surface_config=surface_config,
    )
    adsorbate_composition = list(composition)
    if system_policy.uses_surface:
        composition = full_adsorbate_slab_composition(
            adsorbate_composition, surface_config
        )
    if system_policy.neb_force_mic:
        neb_interpolation_mic = True
    if (
        system_policy.uses_surface
        and surface_config is not None
        and not surface_config.comparator_use_mic
    ):
        logger.warning(
            "comparator_use_mic=False affects GO comparators only; TS "
            "dedupe/pairing/NEB force MIC for surface types (resolve_neb_mic)."
        )
    if not system_policy.uses_surface:
        neb_surface_cell_remap = False
        neb_surface_lattice_rotation = False
    else:
        neb_surface_cell_remap = (
            system_policy.neb_surface_cell_remap and neb_surface_cell_remap
        )
        neb_surface_lattice_rotation = (
            system_policy.neb_surface_lattice_rotation and neb_surface_lattice_rotation
        )
    neb_n_slab = (
        len(surface_config.slab)
        if surface_config is not None and system_policy.uses_surface
        else 0
    )
    slab_search_partition = None
    if (
        system_policy.slab_is_search_target
        and surface_config is not None
        and system_policy.uses_surface
    ):
        from scgo.surface.partition import prepare_slab_search_surface_config

        surface_config, slab_search_partition = prepare_slab_search_surface_config(
            surface_config
        )
        # Match GA: fixed bottom layers are the NEB ``n_slab`` prefix; top
        # layers (+ adsorbates) remain mobile.
        neb_n_slab = int(slab_search_partition.n_fixed)
    neb_n_core_m, neb_n_ads_m, neb_ads_frag_lengths = resolve_neb_mobile_dims(
        system_type,
        adsorbate_composition,
        adsorbate_definition,
        neb_align_endpoints=neb_align_endpoints,
    )
    rng = ensure_rng(seed)

    if use_parallel_neb is True and not use_torchsim:
        raise SCGOValidationError("use_parallel_neb requires use_torchsim=True")

    path_key_formula = resolve_run_path_key(
        adsorbate_composition if system_policy.uses_surface else composition,
        system_type=system_type,
        adsorbate_definition=adsorbate_definition,
        surface_config=surface_config if system_policy.uses_surface else None,
    )
    formula = get_cluster_formula(composition)
    _campaign_root, minima_dir, ts_results_root = resolve_ts_campaign_paths(
        output_dir,
        path_key_formula,
        searches_dir=searches_dir,
    )

    if params is None:
        raise SCGOValidationError(
            "params is required for transition-state search. Build with "
            "get_ts_search_params(system_type=...) or initialize_ts_params()."
        )
    if "calculator" not in params:
        raise SCGOValidationError(
            "params must include 'calculator'. Build with get_ts_search_params()."
        )
    calculator_name = str(params["calculator"])
    calculator_kwargs = params.get("calculator_kwargs") or {}

    use_torchsim, use_parallel_neb = resolve_ts_torchsim_flags(
        calculator_name,
        use_torchsim,
        use_parallel_neb,
    )

    try:
        calculator_class = get_calculator_class(calculator_name)
    except ValueError as e:
        logger.error("Failed to locate calculator class %s: %s", calculator_name, e)
        raise SCGOValidationError(f"Cannot initialize calculator: {e}") from e

    log_info_v(
        logger, "Loading minima for composition %s", formula, verbosity=verbosity
    )

    minima_by_formula = load_minima_by_composition(
        str(minima_dir), composition, prefer_final_unique=True
    )

    if neb_steps in ("auto", None):
        neb_steps = auto_niter_ts(composition)

    torchsim_params = {} if torchsim_params is None else dict(torchsim_params)
    if torchsim_params.get("max_steps") in ("auto", None):
        torchsim_params["max_steps"] = auto_niter_ts(composition)
    # Preset default: TorchSim float32 for the TS path (much faster FP32/TF32
    # GPU kernels). Callers may override torchsim_params["dtype"] before this
    # point; only set it when unset to avoid clobbering an explicit choice.
    if use_torchsim and "dtype" not in torchsim_params:
        import torch

        torchsim_params["dtype"] = torch.float32

    run_context: dict[str, Any] = {
        "system_type": system_type,
        "calculator_name": calculator_name,
        "neb_fmax": neb_fmax,
        "neb_steps_resolved": int(neb_steps)
        if isinstance(neb_steps, int)
        else neb_steps,
        "neb_backend": "torchsim" if use_torchsim else "ase",
        "use_parallel_neb": use_parallel_neb,
        "parallel_neb_max_bands": parallel_neb_max_bands,
        "parallel_neb_max_batch_atoms": parallel_neb_max_batch_atoms,
        "neb_climb": neb_climb,
        "neb_interpolation_method": neb_interpolation_method,
        "neb_n_images": neb_n_images,
        "neb_spring_constant": neb_spring_constant,
        "neb_align_endpoints": neb_align_endpoints,
        "neb_perturb_sigma": neb_perturb_sigma,
        "neb_interpolation_mic": neb_interpolation_mic,
        "neb_surface_cell_remap": neb_surface_cell_remap,
        "neb_surface_lattice_rotation": neb_surface_lattice_rotation,
        "neb_surface_max_lattice_shift": neb_surface_max_lattice_shift,
        "neb_tangent_method": neb_tangent_method,
    }
    if surface_config is not None:
        run_context["surface_slab_constraints"] = surface_slab_constraint_summary(
            surface_config
        )

    if not minima_by_formula:
        n_dbs = len(
            list_discovered_db_paths_with_run(
                str(minima_dir), composition=composition, use_cache=False
            )
        )
        xyz_dir = Path(minima_dir) / "final_unique_minima"
        n_xyz = len(list(xyz_dir.glob("*.xyz"))) if xyz_dir.is_dir() else 0
        logger.error(
            "No minima found in %s (discovered_dbs=%d, "
            "final_unique_minima_xyz=%d). TS loads only DB rows tagged "
            "final_unique_minimum; XYZ under final_unique_minima is export-only.",
            minima_dir,
            n_dbs,
            n_xyz,
        )
        cleanup_torch_cuda(logger=logger)
        return []

    minima = minima_by_formula.get(formula, [])

    if dedupe_minima:
        original_count = len(minima)
        # Match NEB MIC geometry (resolve_neb_mic), not GA comparator knob.
        ts_dedupe_mic = resolve_neb_mic(system_type)
        if slab_search_partition is not None and minima:
            # Slab-target types fingerprint only the [fixed | MOBILE] tail
            # (top layers + adsorbates), matching GO-phase semantics
            # (run_trials ``search_mobile_count`` / core.py:844-847): frozen
            # bottom layers cannot move, so their geometry must not gate
            # pre-pair uniqueness nor dilute the cumulative difference.
            dedupe_n_top = max(
                1, len(minima[0][1]) - int(slab_search_partition.n_fixed)
            )
        else:
            # Match GA ``n_to_optimize``: trailing mobile atoms (core +
            # adsorbates).
            dedupe_n_top = len(adsorbate_composition)
            if neb_n_core_m is not None and neb_n_ads_m is not None:
                dedupe_n_top = int(neb_n_core_m) + int(neb_n_ads_m)
        minima = filter_unique_minima(
            minima,
            minima_energy_tolerance,
            n_top=dedupe_n_top,
            mic=ts_dedupe_mic,
        )
        if verbosity >= 1 and len(minima) != original_count:
            log_info_v(
                logger,
                "Deduplicated minima for %s: %d -> %d unique entries",
                formula,
                original_count,
                len(minima),
                verbosity=verbosity,
            )

    if len(minima) < 2:
        logger.error("Only %d minima found, need at least 2 to find TS", len(minima))
        cleanup_torch_cuda(logger=logger)
        return []

    log_info_v(
        logger,
        "Found %d minima for %s",
        len(minima),
        formula,
        verbosity=verbosity,
    )
    # Surface adsorbate on a searchable top layer: run metadata / resolve dims
    # only know the adsorbate fragment (e.g. OH → n_ads=2, n_core=0). Infer the
    # middle mobile-slab block as the NEB "core" so layout is
    # [fixed | top_layer | adsorbate] for pairing, hop gates, and FixBondLengths.
    if (
        slab_search_partition is not None
        and neb_n_ads_m is not None
        and int(neb_n_ads_m) > 0
        and minima
        and (neb_n_core_m is None or int(neb_n_core_m) == 0)
    ):
        inferred_core = len(minima[0][1]) - int(neb_n_slab) - int(neb_n_ads_m)
        if inferred_core > 0:
            neb_n_core_m = inferred_core
            log_info_v(
                logger,
                "Inferred n_core_mobile=%d for slab-search adsorbate "
                "(n_slab=%d n_ads=%d n_atoms=%d)",
                neb_n_core_m,
                neb_n_slab,
                neb_n_ads_m,
                len(minima[0][1]),
                verbosity=verbosity,
            )
    if neb_align_endpoints:
        log_debug_v(
            logger,
            "NEB endpoint alignment enabled (align=%s, mic=%s, cell_remap=%s, "
            "lattice_rotation=%s)",
            neb_align_endpoints,
            neb_interpolation_mic,
            neb_surface_cell_remap,
            neb_surface_lattice_rotation,
            verbosity=verbosity,
        )
    _warn_on_surface_mobile_indices(minima, system_type=system_type, n_slab=neb_n_slab)

    # Adsorbate: oversample for IDPP re-rank. Bare: select cap is the NEB budget.
    pair_select_cap = resolve_ts_pair_select_cap(
        max_pairs,
        has_adsorbate=bool(system_policy.has_adsorbate),
        max_endpoint_mismatch=max_endpoint_mismatch,
    )
    pairs = select_structure_pairs(
        minima,
        max_pairs=pair_select_cap,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
        surface_aware=bool(system_policy.uses_surface),
        use_mic=resolve_neb_mic(system_type),
        n_slab=neb_n_slab if neb_n_slab > 0 else None,
        max_endpoint_mismatch=max_endpoint_mismatch,
        adsorbate_aware=bool(system_policy.has_adsorbate),
        n_core_mobile=neb_n_core_m,
        n_adsorbate_mobile=neb_n_ads_m,
        pair_core_rms_max=pair_core_rms_max,
        pair_score_gap_center=pair_score_gap_center,
        pair_score_gap_width=pair_score_gap_width,
        pair_score_cum_scale=pair_score_cum_scale,
        pair_score_mismatch_scale=pair_score_mismatch_scale,
        pair_score_core_rms_scale=pair_score_core_rms_scale,
        pair_score_w_gap=pair_score_w_gap,
        pair_score_w_distinct=pair_score_w_distinct,
        pair_score_w_mismatch=pair_score_w_mismatch,
        pair_score_w_core=pair_score_w_core,
    )

    if not pairs:
        logger.error("No suitable pairs found for TS search")
        cleanup_torch_cuda(logger=logger)
        return []

    # The adsorbate IDPP priority screen needs a TorchSim relaxer, and the
    # parallel NEB runner can reuse the same one. Build it at most once, and only
    # when something will actually use it: the serial path builds and owns its own
    # relaxer inside _run_serial_neb_search.
    needs_idpp_screen = (
        bool(system_policy.has_adsorbate)
        and use_torchsim
        and max_endpoint_mismatch is not None
        and max_pairs is not None
        and int(max_pairs) > 0
        and len(pairs) > int(max_pairs)
    )
    shared_relaxer = None
    if use_torchsim and (use_parallel_neb or needs_idpp_screen):
        from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

        # Size the relaxer for the largest fused NEB force batch (mirrors the GO
        # expected_max_atoms pattern) so the native autobatcher probe stays capped
        # to the actual workload. coerce_ts_params_to_runner_kwargs normally injects
        # these already; this covers direct callers. Native torch-sim estimation
        # runs on the real batches, so no synthetic probe geometry is needed.
        relaxer_params = dict(torchsim_params or {})
        if (
            parallel_neb_max_batch_atoms is not None
            and int(parallel_neb_max_batch_atoms) > 0
        ):
            relaxer_params.setdefault(
                "expected_max_atoms", int(parallel_neb_max_batch_atoms)
            )
            relaxer_params.setdefault(
                "max_atoms_to_try", int(parallel_neb_max_batch_atoms)
            )
        shared_relaxer = TorchSimBatchRelaxer(**relaxer_params)

    if needs_idpp_screen:
        pairs = _prioritize_adsorbate_pairs_by_idpp(
            pairs,
            minima,
            max_pairs=int(max_pairs),
            relaxer=shared_relaxer,
            neb_n_images=neb_n_images,
            neb_interpolation_method=neb_interpolation_method,
            neb_interpolation_mic=neb_interpolation_mic,
            neb_align_endpoints=neb_align_endpoints,
            neb_perturb_sigma=neb_perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=neb_n_slab,
            n_core_mobile=neb_n_core_m,
            n_adsorbate_mobile=neb_n_ads_m,
            adsorbate_fragment_lengths=neb_ads_frag_lengths,
            neb_surface_cell_remap=neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_surface_max_lattice_shift,
            max_endpoint_mismatch=float(max_endpoint_mismatch),
            neb_prescreen_clash_distance=neb_prescreen_clash_distance,
            min_saddle_prominence=float(min_saddle_prominence),
            neb_max_spurious_barrier=float(neb_max_spurious_barrier),
            neb_interpolation_bond_tolerance_a=neb_interpolation_bond_tolerance_a,
            parallel_neb_max_batch_atoms=parallel_neb_max_batch_atoms,
            parallel_neb_max_bands=parallel_neb_max_bands,
            logger=logger,
            verbosity=verbosity,
        )

        if not pairs:
            logger.error(
                "No adsorbate pairs survived IDPP priority screening for TS search"
            )
            return []
    if max_pairs is not None and int(max_pairs) > 0:
        pairs = pairs[: int(max_pairs)]

    log_info_v(
        logger,
        "Selected %d structure pairs for TS search",
        len(pairs),
        verbosity=verbosity,
    )

    ts_results_root.mkdir(parents=True, exist_ok=True)
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)
    run_dir = ts_results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_dir_record(
        str(run_dir),
        run_id,
        record={
            "path_key": path_key_formula,
            "composition": list(composition),
            "formula": formula or path_key_formula,
            "params": run_context,
        },
    )

    t_ts0 = perf_counter()
    parallel_meta: dict[str, float] = {}
    neb_cfg = NebRunConfig(
        neb_n_images=neb_n_images,
        neb_spring_constant=neb_spring_constant,
        neb_fmax=neb_fmax,
        neb_steps=neb_steps,
        neb_climb=neb_climb,
        neb_interpolation_method=neb_interpolation_method,
        neb_align_endpoints=neb_align_endpoints,
        neb_perturb_sigma=neb_perturb_sigma,
        neb_interpolation_mic=neb_interpolation_mic,
        neb_tangent_method=neb_tangent_method,
        neb_surface_cell_remap=neb_surface_cell_remap,
        neb_surface_lattice_rotation=neb_surface_lattice_rotation,
        neb_surface_max_lattice_shift=neb_surface_max_lattice_shift,
        n_slab=neb_n_slab,
        n_core_mobile=neb_n_core_m,
        n_adsorbate_mobile=neb_n_ads_m,
        adsorbate_fragment_lengths=neb_ads_frag_lengths,
        max_endpoint_mismatch=max_endpoint_mismatch,
        neb_prescreen_clash_distance=neb_prescreen_clash_distance,
        min_saddle_prominence=min_saddle_prominence,
        neb_max_spurious_barrier=neb_max_spurious_barrier,
        layer_cluster_threshold_ang=layer_cluster_threshold_ang,
        neb_interpolation_bond_tolerance_a=neb_interpolation_bond_tolerance_a,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
        system_type=system_type,
        surface_config=surface_config,
        torchsim_params=torchsim_params,
        cluster_adsorbate_config=cluster_adsorbate_config,
        parallel_neb_max_batch_atoms=parallel_neb_max_batch_atoms,
    )
    if use_parallel_neb:
        # Always use the parallel runner when requested; surface presets pass
        # parallel_neb_max_bands=4 so bands are chunked (OOM-safe on slabs).
        ts_results, parallel_meta = run_parallel_neb_search(
            pairs,
            minima,
            neb_cfg=neb_cfg,
            run_dir=run_dir,
            rng=rng,
            parallel_neb_max_bands=parallel_neb_max_bands,
            relaxer=shared_relaxer,
            verbosity=verbosity,
        )
        cleanup_torch_cuda(logger=logger)
    else:
        ts_results = _run_serial_neb_search(
            pairs,
            minima,
            neb_cfg=neb_cfg,
            run_dir=run_dir,
            calculator_class=calculator_class,
            calculator_kwargs=calculator_kwargs,
            rng=rng,
            use_torchsim=use_torchsim,
            verbosity=verbosity,
            write_timing_json=write_timing_json,
        )

    ts_phase_wall = perf_counter() - t_ts0
    neb_sum = float(parallel_meta.get("neb_batch_optimization_s", 0.0))
    if neb_sum <= 0.0:
        logger.debug(
            "Timing key neb_batch_optimization_s unavailable; "
            "summing per-pair NEB timings"
        )
        neb_sum = sum_neb_seconds_from_ts_results(ts_results)
    ts_rollup: dict[str, float] = {
        "kind": "neb",
        "total_wall_s": ts_phase_wall,
        "neb_optimization_s": neb_sum,
        "cpu_non_relax_s": max(0.0, ts_phase_wall - neb_sum),
    }
    log_timing_summary(
        logger,
        "ts_search",
        ts_rollup,
        verbosity=verbosity,
    )
    if write_timing_json:
        write_timing_file(
            str(run_dir),
            build_timing_payload(
                backend="ts_search",
                timings_s=ts_rollup,
                run_id=run_id,
                extra={
                    "counters": {
                        "n_results": len(ts_results),
                        "n_success": sum(
                            1 for r in ts_results if r.get("status") == "success"
                        ),
                    },
                    "parallel_batch_s": parallel_meta,
                },
            ),
        )

    _apply_surface_ts_geometry_gate(
        ts_results,
        surface_config=surface_config,
        system_type=system_type,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
        binding_penetration_tolerance_a=binding_penetration_tolerance_a,
        n_slab_deposit=(
            int(slab_search_partition.n_fixed)
            if slab_search_partition is not None
            else None
        ),
        run_dir=run_dir,
        verbosity=verbosity,
    )

    save_transition_state_results(
        ts_results,
        str(ts_results_root),
        composition,
        run_context=run_context,
        run_id=run_id,
    )

    save_ts_network_metadata(
        ts_results,
        str(ts_results_root),
        composition,
        minima_count=len(minima),
        minima=minima,
        minima_base_dir=str(minima_dir),
        run_context=run_context,
        path_key=path_key_formula,
    )

    if dedupe_ts or tag_ts_in_db:
        unique_ts = write_final_unique_ts(
            ts_results,
            str(ts_results_root),
            composition,
            energy_tolerance=ts_energy_tolerance,
            similarity_tolerance=similarity_tolerance,
            similarity_pair_cor_max=similarity_pair_cor_max,
            minima=minima,
            minima_base_dir=str(minima_dir),
            run_context=run_context,
            surface_aware=system_policy.uses_surface,
            n_slab=neb_n_slab if neb_n_slab > 0 else None,
            path_key=path_key_formula,
        )

        if tag_ts_in_db and unique_ts:
            tag_unique_ts_in_databases(unique_ts, minima, str(minima_dir))

    num_success = sum(1 for r in ts_results if r.get("status") == "success")
    log_info_v(
        logger,
        "TS search complete for %s: %d result(s) (%d successful)",
        formula,
        len(ts_results),
        num_success,
        verbosity=verbosity,
    )
    log_info_v(logger, "Results written to: %s", ts_results_root, verbosity=verbosity)

    cleanup_torch_cuda(logger=logger)

    return ts_results


def run_transition_state_campaign(
    compositions: list[list[str]],
    system_type: SystemType,
    output_dir: str | Path | None = None,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    ts_kwargs: dict | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run :func:`~scgo.ts_search.run_transition_state_search` for multiple compositions in sequence.

    ``output_dir`` is the campaign root. Minima are read from
    ``{output_dir}/{path_key}_searches`` (or ``{path_key}_searches`` under the
    current working directory when ``output_dir`` is None). TS results are
    written to sibling ``{path_key}_ts_results/`` directories. Extra search/NEB
    arguments are forwarded via ``ts_kwargs``. Failures for one composition never
    abort the whole campaign — they are logged and that path key gets an empty
    result list.
    """
    configure_logging(verbosity)
    logger = get_logger(__name__)

    ts_kwargs = dict(ts_kwargs or {})
    # ``params`` and ``system_type`` are named arguments of this function and are
    # forwarded explicitly below; duplicates inside ``ts_kwargs`` (e.g. from
    # ``coerce_ts_params_to_runner_kwargs``) would raise ``TypeError`` at the
    # ``**ts_kwargs`` expansion. The explicit arguments win.
    ts_kwargs.pop("system_type", None)
    ts_kwargs_params = ts_kwargs.pop("params", None)
    if params is None:
        params = ts_kwargs_params
    campaign_results: dict[str, list[dict[str, Any]]] = {}
    ads_def = as_adsorbate_definition(ts_kwargs.get("adsorbate_definition"))
    if ads_def is not None:
        ts_kwargs["adsorbate_definition"] = ads_def
    surface_cfg = ts_kwargs.get("surface_config")
    if not isinstance(surface_cfg, SurfaceSystemConfig):
        surface_cfg = None

    for composition in compositions:
        path_key = resolve_run_path_key(
            composition,
            system_type=system_type,
            adsorbate_definition=ads_def,
            surface_config=surface_cfg,
        )
        campaign_root = (
            str(Path(output_dir).expanduser().resolve())
            if output_dir is not None
            else None
        )
        log_info_v(
            logger,
            "Running TS search campaign for %s",
            path_key,
            verbosity=verbosity,
        )

        results = run_transition_state_search(
            composition,
            output_dir=campaign_root,
            params=params,
            seed=seed,
            verbosity=verbosity,
            system_type=system_type,
            **ts_kwargs,
        )

        for r in results:
            if r.get("transition_state") is not None:
                _detach_calc(r["transition_state"])
        campaign_results[path_key] = results

    return campaign_results
