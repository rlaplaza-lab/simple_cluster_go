"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from scgo.calculators import torchsim_helpers as _tsh
from scgo.constants import DEFAULT_FMAX_THRESHOLD
from scgo.exceptions import SCGOValidationError
from scgo.metadata.provenance import is_cuda_oom_error
from scgo.utils.logging import get_logger
from scgo.utils.phase_logging import log_neb_search_summaries
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.ts_runner_kwargs import NebRunConfig

from .neb_endpoints import prepare_neb_endpoints
from .transition_state import (
    TorchSimNEB,
    _detach_calc,
    _finalize_neb_result,
    _image_has_cached_forces,
    attach_minima_traceability,
    attach_singlepoint_from_relax_output,
    evaluate_neb_image_energies,
    interpolate_path,
    load_completed_neb_result,
    make_ts_result,
    neb_max_atom_force,
    neb_uses_two_stage_climb,
    save_neb_result,
    validate_initial_neb_energy_profile,
    validate_initial_neb_path,
)

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

logger = get_logger(__name__)


def _neb_image_dedup_key(atoms: Atoms) -> tuple:
    """Hashable key for deduplicating NEB images across bands.

    Positions alone are not enough: surface bands enable ``neb_surface_cell_remap``
    and ``neb_surface_lattice_rotation``, which legitimately produce identical
    Cartesian positions in *different* cells. Cell and PBC are part of the key so
    such images never collide and receive a neighbor's energy/forces.
    """
    return (
        tuple(atoms.get_chemical_symbols()),
        tuple(np.round(atoms.get_positions().ravel(), 6)),
        tuple(np.round(np.asarray(atoms.get_cell()).ravel(), 6)),
        tuple(bool(p) for p in atoms.get_pbc()),
    )


def _band_atom_cost(neb: TorchSimNEB) -> int:
    """Atoms in one fused force batch for this band (``n_images * n_atoms``)."""
    images = neb.images
    if not images:
        return 0
    return len(images) * len(images[0])


def chunk_band_indices_by_atom_budget(
    indices: list[int],
    costs: list[int],
    max_batch_atoms: int | None,
) -> list[list[int]]:
    """Greedily bin ``indices`` so each chunk's summed atom cost fits the budget.

    ``costs`` is indexed by band index (parallel to ``neb_instances``). Input
    order is preserved, so chunking is deterministic. A single band exceeding the
    budget still gets its own chunk (never dropped). ``max_batch_atoms`` of
    ``None`` or ``<= 0`` disables budgeting and returns one chunk.
    """
    if not indices:
        return []
    if max_batch_atoms is None or int(max_batch_atoms) <= 0:
        return [list(indices)]
    budget = int(max_batch_atoms)
    chunks: list[list[int]] = []
    current: list[int] = []
    current_atoms = 0
    for idx in indices:
        cost = int(costs[idx])
        if current and current_atoms + cost > budget:
            chunks.append(current)
            current = []
            current_atoms = 0
        current.append(idx)
        current_atoms += cost
    if current:
        chunks.append(current)
    return chunks


def _apply_band_cap(chunks: list[list[int]], band_cap: int | None) -> list[list[int]]:
    """Further split atom-budget chunks so none exceeds *band_cap* bands."""
    if band_cap is None:
        return chunks
    capped: list[list[int]] = []
    for chunk in chunks:
        capped.extend(chunk[k : k + band_cap] for k in range(0, len(chunk), band_cap))
    return capped


# The soft sentinel "NEB did not converge after N steps" is the only error text
# that stays climb-eligible: a band that merely exhausted stage-1's half budget
# is the normal interior-max IDPP case and MUST still proceed to the climb pass.
# Every other non-empty error (non-finite forces, "not processed", CUDA OOM, or
# any arbitrary exception message) marks a hard failure.
_STAGE1_SOFT_NONCONVERGENCE = "did not converge"


def _stage1_band_climb_eligible(summary: dict[str, Any]) -> bool:
    """Return True when a stage-1 band should proceed to the CI-NEB climb pass.

    A band is climb-eligible when it actually took at least one optimizer step
    and did not *hard*-fail. The previous ``not summary.get("error")`` test
    filtered out every band that merely exhausted stage-1's half budget (which
    is stamped ``error="NEB did not converge after N steps"``), so in the normal
    case no band ever climbed.

    Preference order:

    * the explicit ``summary["failed"]`` boolean set by
      :meth:`ParallelNEBBatch.run_optimization` (``neb_idx in self.failed_nebs``);
    * a string sniff of ``summary["error"]`` as a fallback for summaries produced
      outside ``run_optimization`` (e.g. the OOM-retry stubs), where only an
      empty error or the soft "did not converge" sentinel stays climb-eligible
      and any other error text is treated as a hard failure.
    """
    if int(summary.get("steps_taken") or 0) <= 0:
        return False
    if "failed" in summary:
        return not bool(summary["failed"])
    error_text = str(summary.get("error") or "").lower()
    if not error_text:
        return True
    return _STAGE1_SOFT_NONCONVERGENCE in error_text


def _evaluate_bands_in_chunks(
    bands: list[list[Atoms]],
    relaxer: Any,
    *,
    atom_budget: int | None,
    band_cap: int | None,
) -> list[list[float]]:
    """Batched single-point energies per band, chunked to fit GPU memory.

    ``bands`` is a list of image lists (one per band). Returns per-band energy
    lists in the same order as the input. Each fused
    :func:`evaluate_neb_image_energies` call is bounded by ``atom_budget`` (the
    summed ``n_images * n_atoms`` cost) and ``band_cap`` (max bands per batch),
    mirroring the chunking used for the optimization pass. On CUDA OOM a chunk is
    re-binned at half its atom cost and retried once (``cleanup_torch_cuda`` runs
    between attempts).

    This replaces a single ``evaluate_neb_image_energies(all_images)`` over every
    image of every band, which bypassed the atom budget and OOM recovery and
    could silently OOM the whole pre-screen for large ``setup_pairs`` lists.
    """
    if not bands:
        return []
    band_costs = [len(imgs) * len(imgs[0]) if imgs else 0 for imgs in bands]

    def _chunk(indices: list[int], budget: int | None) -> list[list[int]]:
        chunks = chunk_band_indices_by_atom_budget(indices, band_costs, budget)
        return _apply_band_cap(chunks, band_cap)

    results: list[list[float] | None] = [None] * len(bands)

    def _run_chunk(chunk: list[int]) -> None:
        chunk_images = [img for band_i in chunk for img in bands[band_i]]
        energies = evaluate_neb_image_energies(chunk_images, relaxer)
        offset = 0
        for band_i in chunk:
            n = len(bands[band_i])
            results[band_i] = [float(e) for e in energies[offset : offset + n]]
            offset += n

    for chunk in _chunk(list(range(len(bands))), atom_budget):
        try:
            _run_chunk(chunk)
        except (RuntimeError, MemoryError) as exc:
            if not is_cuda_oom_error(exc):
                raise
            logger.warning(
                "Pre-screen energy eval hit CUDA OOM for %d band(s) (%s); "
                "retrying once at half the chunk atom cost",
                len(chunk),
                exc,
            )
            cleanup_torch_cuda(logger=logger)
            chunk_atoms = sum(band_costs[i] for i in chunk)
            retry_budget = max(1, chunk_atoms // 2)
            for sub_chunk in chunk_band_indices_by_atom_budget(
                chunk, band_costs, retry_budget
            ):
                _run_chunk(sub_chunk)

    if not all(r is not None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise SCGOValidationError(
            f"Parallel NEB returned incomplete results; missing bands: {missing}"
        )
    return results  # type: ignore[return-value]


class ParallelNEBBatch:
    """Coordinate multiple TorchSimNEB instances and run batched evaluations."""

    def __init__(
        self,
        neb_instances: list[TorchSimNEB],
        relaxer: TorchSimBatchRelaxer,
        max_total_steps: int = 1000,
        optimizer: type = FIRE,
    ):
        """Initialize with NEBs, relaxer, max steps, and ASE optimizer (default FIRE)."""
        self.neb_instances = neb_instances
        self.relaxer = relaxer
        self.max_total_steps = max_total_steps
        self.optimizer_cls = optimizer

        self.active_nebs = list(range(len(neb_instances)))
        self.converged_nebs: dict[int, bool] = {}
        self.failed_nebs: dict[int, str] = {}
        self.step_count = 0

        # The batch runner owns ``force_calls`` for these bands (B2): it counts
        # one call per batched relax_batch a band participates in, so the bands
        # must not also self-count inside ``TorchSimNEB.get_forces``.
        for neb in neb_instances:
            if hasattr(neb, "_force_calls_counted_externally"):
                neb._force_calls_counted_externally = True

        # Per-NEB optimizer instances (created lazily). Uses ASE optimizers
        # (default: FIRE) so stepping respects NEB forces / spring terms.
        self._optimizers: dict[int, object] = {}

    def _step_optimizer(self, neb_idx: int) -> None:
        """Advance one band's optimizer using the just-computed batched forces.

        ``optimizer.step()`` is called with no arguments: the ASE
        ``Dynamics.step()`` API takes none, and ``FIRE.step`` keeps ``f=None``
        only for backwards compatibility. Called that way ``FIRE.step()`` falls
        back to ``optimizable.get_gradient().reshape(-1, 3)``, and
        ``NEBOptimizable.get_gradient`` is ``neb.get_forces().ravel()`` (NEB
        forces are already the descent direction, no sign flip), so the value is
        identical to the forces this batch just computed (checked against ASE
        3.26).

        Crucially this does *not* cost an extra TorchSim call: every image still
        carries the SinglePoint results attached moments ago, so
        ``TorchSimNEB.get_forces`` takes its cached-forces fast path instead of
        dispatching an unbatched per-band ``relax_batch``.
        """
        optimizer = self._optimizers.get(neb_idx)
        if optimizer is None:
            optimizer = self.optimizer_cls(
                self.neb_instances[neb_idx], logfile=None, trajectory=None
            )
            self._optimizers[neb_idx] = optimizer
        optimizer.step()

    def run_optimization(
        self,
        fmax: float = DEFAULT_FMAX_THRESHOLD,
        max_steps: int = 500,
    ) -> list[dict[str, Any]]:
        """Optimize NEBs using batched evaluations; return per-NEB summaries.

        Raises:
            RuntimeError: (or ``torch.cuda.OutOfMemoryError``) when a batched
                force evaluation runs out of GPU memory. CUDA OOM is *not*
                converted into per-band error dicts: the caller
                (``run_parallel_neb_search``) re-bins the chunk at half its
                own atom cost and retries. Non-OOM ``RuntimeError``/``ValueError``
                still mark the active bands failed and end the loop, because
                those indicate bad input rather than GPU pressure.
        """
        if not self.neb_instances:
            logger.error("No NEB instances provided to run_optimization")
            return []

        results = [
            {
                "converged": False,
                "steps_taken": 0,
                "final_fmax": None,
                "error": None,
                "force_calls": None,
                "failed": False,
            }
            for _ in self.neb_instances
        ]

        step_cap = min(self.max_total_steps, int(max_steps))
        while self.active_nebs and self.step_count < step_cap:
            unique_images: list[Atoms] = []
            unique_index: dict[tuple, int] = {}
            neb_image_map: list[tuple[int, int, int]] = []
            batch_participants: list[int] = []
            # After step 0, endpoints keep cached SinglePoint energy/forces.
            evaluate_endpoints = self.step_count == 0

            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                n_img = len(neb.images)
                participates = False
                for img_idx, atoms in enumerate(neb.images):
                    is_endpoint = img_idx == 0 or img_idx == n_img - 1
                    if is_endpoint and not evaluate_endpoints:
                        continue
                    key = _neb_image_dedup_key(atoms)
                    if key not in unique_index:
                        unique_index[key] = len(unique_images)
                        unique_images.append(atoms)
                    unique_slot = unique_index[key]
                    neb_image_map.append((neb_idx, img_idx, unique_slot))
                    participates = True
                if participates:
                    batch_participants.append(neb_idx)

            if not unique_images:
                break

            logger.debug(
                f"Step {self.step_count}: Evaluating {len(unique_images)} unique images "
                f"({len(neb_image_map)} total slots) from {len(self.active_nebs)} active NEBs"
                f"{'' if evaluate_endpoints else ' (interiors only)'}"
            )

            # Reuse energy-screen forces at step 0 when present.
            reuse_cached = evaluate_endpoints and all(
                _image_has_cached_forces(img) for img in unique_images
            )

            if not reuse_cached:
                try:
                    unique_results = self.relaxer.relax_batch(unique_images, steps=0)
                except (RuntimeError, ValueError) as e:
                    if is_cuda_oom_error(e):
                        # Propagate GPU pressure so ``_run_chunk_with_oom_retry`` can
                        # re-bin this chunk at half its atom cost. Swallowing it here
                        # (the historical behavior) made that safety net unreachable
                        # and silently produced zero transition states.
                        logger.warning(
                            "Batched force evaluation hit CUDA OOM at step %d "
                            "(%d image(s), %d band(s)); propagating for re-binning: %s",
                            self.step_count,
                            len(unique_images),
                            len(self.active_nebs),
                            e,
                        )
                        raise
                    kind = (
                        "Invalid input"
                        if isinstance(e, ValueError)
                        else "Batched force evaluation"
                    )
                    logger.error("%s failed: %s", kind, e)
                    for neb_idx in self.active_nebs:
                        self.failed_nebs[neb_idx] = str(e)
                        results[neb_idx]["error"] = str(e)
                    break

                for neb_idx, img_idx, unique_slot in neb_image_map:
                    energy, relaxed_atoms = unique_results[unique_slot]
                    atoms = self.neb_instances[neb_idx].images[img_idx]
                    attach_singlepoint_from_relax_output(
                        atoms, energy, relaxed_atoms, require_forces=True
                    )

            # Count one PES eval per participating band (includes screen reuse).
            for neb_idx in batch_participants:
                self.neb_instances[neb_idx]._force_calls += 1

            still_active: list[int] = []
            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                try:
                    neb_forces = neb.get_forces()
                    max_force = neb_max_atom_force(neb_forces)

                    results[neb_idx]["final_fmax"] = max_force
                    results[neb_idx]["steps_taken"] = self.step_count + 1

                    if not np.isfinite(max_force):
                        msg = (
                            "NEB forces are non-finite "
                            f"(fmax={max_force!r}); refusing optimizer step"
                        )
                        results[neb_idx]["converged"] = False
                        self.failed_nebs[neb_idx] = msg
                        results[neb_idx]["error"] = msg
                        logger.debug("NEB %d step failed: %s", neb_idx, msg)
                        continue
                    if max_force < fmax:
                        results[neb_idx]["converged"] = True
                        self.converged_nebs[neb_idx] = True
                        logger.debug(
                            f"NEB {neb_idx} finished: converged, fmax={max_force:.6f}"
                        )
                    else:
                        self._step_optimizer(neb_idx)
                        still_active.append(neb_idx)
                except (RuntimeError, ValueError) as e:
                    if is_cuda_oom_error(e):
                        # Same contract as the batched eval above: OOM belongs to
                        # the chunk-level retry, not to per-band bookkeeping.
                        raise
                    logger.debug("NEB %d step failed: %s", neb_idx, e)
                    self.failed_nebs[neb_idx] = str(e)
                    results[neb_idx]["error"] = str(e)

            self.active_nebs = still_active
            self.step_count += 1

            if not self.active_nebs:
                break

        for neb_idx in range(len(self.neb_instances)):
            # Record the hard-failure flag so ``_stage1_band_climb_eligible`` can
            # key off it directly instead of sniffing the error string.
            results[neb_idx]["failed"] = neb_idx in self.failed_nebs
            if neb_idx not in self.converged_nebs and neb_idx not in self.failed_nebs:
                steps = results[neb_idx]["steps_taken"] or 0
                results[neb_idx]["error"] = (
                    f"NEB did not converge after {steps} steps"
                    if steps
                    else "NEB not processed"
                )

        # Final FIRE.step() invalidates SinglePoint caches on moved images.
        # Refresh PES at the final geometries so barrier finalize can read energies.
        self._refresh_pes_after_optimization()

        logger.debug(
            "Parallel NEB batch complete: %d steps, %d converged, %d failed",
            self.step_count,
            len(self.converged_nebs),
            len(self.failed_nebs),
        )

        return results

    def _refresh_pes_after_optimization(self) -> None:
        """Re-evaluate images of every non-failed NEB at their final positions.

        No optimizer step is taken; failed bands are skipped.
        """
        unique_images: list[Atoms] = []
        unique_index: dict[tuple, int] = {}
        neb_image_map: list[tuple[int, int, int]] = []

        for neb_idx, neb in enumerate(self.neb_instances):
            if neb_idx in self.failed_nebs:
                continue
            for img_idx, atoms in enumerate(neb.images):
                key = _neb_image_dedup_key(atoms)
                if key not in unique_index:
                    unique_index[key] = len(unique_images)
                    unique_images.append(atoms)
                neb_image_map.append((neb_idx, img_idx, unique_index[key]))

        if not unique_images:
            return

        try:
            unique_results = self.relaxer.relax_batch(unique_images, steps=0)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Final NEB PES refresh failed (%s); finalize will use cached energies if present",
                e,
            )
            return

        for neb_idx, img_idx, unique_slot in neb_image_map:
            energy, relaxed_atoms = unique_results[unique_slot]
            atoms = self.neb_instances[neb_idx].images[img_idx]
            attach_singlepoint_from_relax_output(
                atoms, energy, relaxed_atoms, require_forces=True
            )

    def get_summary(self) -> dict[str, int]:
        """Return counts of total, converged and failed NEBs."""
        return {
            "total_nebs": len(self.neb_instances),
            "converged": len(self.converged_nebs),
            "failed": len(self.failed_nebs),
            "total_steps": self.step_count,
        }


def run_parallel_neb_search(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Atoms]],
    *,
    neb_cfg: NebRunConfig,
    run_dir: Path,
    rng: np.random.Generator | None,
    parallel_neb_max_bands: int | None = None,
    relaxer: Any | None = None,
    verbosity: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run all pairs through ParallelNEBBatch. Returns (results, timing meta).

    Bands are chunked so each fused force batch fits GPU memory
    (``cleanup_torch_cuda`` runs between chunks). Both bounds always apply:

    * bands are greedily binned so the summed ``n_images * n_atoms`` per batch
      stays within ``neb_cfg.parallel_neb_max_batch_atoms`` (``None`` = no atom
      budget);
    * ``parallel_neb_max_bands`` (>0) additionally caps how many bands share a
      batch. Surface presets pass ``4``.

    A chunk that hits CUDA OOM is re-binned at half its own atom cost and retried
    once (after ``cleanup_torch_cuda``); only the sub-chunks that still OOM have
    their bands marked failed.

    ``relaxer`` lets the caller reuse a single :class:`TorchSimBatchRelaxer`
    (e.g. the one built for the IDPP screen) instead of constructing a fresh
    model load. When ``None``, a relaxer is built from ``neb_cfg.torchsim_params``.
    """
    t_parallel0 = perf_counter()
    torchsim_params = neb_cfg.torchsim_params or {}
    # Reuse the caller-provided relaxer (e.g. the shared IDPP-screen relaxer)
    # instead of constructing a second model load.
    if relaxer is None:
        relaxer = _tsh.TorchSimBatchRelaxer(**torchsim_params)
    neb_steps_i = int(neb_cfg.neb_steps)
    system_type = neb_cfg.system_type

    neb_instances: list[TorchSimNEB] = []
    # Parallel to neb_instances: (pair_index_in_results, i, j)
    neb_meta: list[tuple[int, int, int]] = []
    # Per-band two-stage flag (endpoint-max IDPP → climb from step 0).
    neb_two_stage: list[bool] = []
    pair_results: list[dict[str, Any] | None] = [None] * len(pairs)

    def _make_pair_ts_result(
        pair_id: str,
        *,
        react_e: float,
        prod_e: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return make_ts_result(
            pair_id=pair_id,
            n_images=neb_cfg.neb_n_images,
            spring_constant=neb_cfg.neb_spring_constant,
            use_torchsim=True,
            fmax=neb_cfg.neb_fmax,
            neb_steps=neb_cfg.neb_steps,
            interpolation_method=neb_cfg.neb_interpolation_method,
            climb=neb_cfg.neb_climb,
            align_endpoints=neb_cfg.neb_align_endpoints,
            perturb_sigma=neb_cfg.neb_perturb_sigma,
            neb_interpolation_mic=neb_cfg.neb_interpolation_mic,
            neb_tangent_method=neb_cfg.neb_tangent_method,
            use_parallel_neb=True,
            reactant_energy=react_e,
            product_energy=prod_e,
            error=error,
        )

    def _record_skipped_pair(
        pair_ord: int,
        pair_id: str,
        i: int,
        j: int,
        react_e: float,
        prod_e: float,
        error: str,
    ) -> None:
        skipped = _make_pair_ts_result(
            pair_id, react_e=react_e, prod_e=prod_e, error=error
        )
        skipped["status"] = "skipped"
        skipped["system_type"] = system_type
        attach_minima_traceability(skipped, minima, i, j)
        pair_dir = run_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_neb_result(skipped, str(pair_dir), pair_id, verbosity=verbosity)
        pair_results[pair_ord] = skipped

    # Build NEB instances for every valid pair first; defer the per-band single-point
    # energy eval so all bands can be fused into one relax_batch(steps=0) call below.
    setup_pairs: list[tuple[int, str, int, int, float, float, list[Any]]] = []
    for pair_ord, (i, j) in enumerate(pairs):
        pair_id = f"{i}_{j}"
        pair_dir = run_dir / f"pair_{pair_id}"
        resumed = load_completed_neb_result(pair_dir, pair_id)
        if resumed is not None:
            logger.info("Skipping pair %s (resumed success)", pair_id)
            resumed["system_type"] = system_type
            if "minima_indices" not in resumed:
                attach_minima_traceability(resumed, minima, i, j)
            pair_results[pair_ord] = resumed
            continue

        react_e = float(minima[i][0])
        prod_e = float(minima[j][0])
        try:
            react_ep, prod_ep = prepare_neb_endpoints(
                minima[i][1],
                minima[j][1],
                neb_cfg,
            )
        except (ValueError, SCGOValidationError) as e:
            _record_skipped_pair(pair_ord, pair_id, i, j, react_e, prod_e, str(e))
            continue

        images = interpolate_path(
            react_ep,
            prod_ep,
            n_images=neb_cfg.neb_n_images,
            method=neb_cfg.neb_interpolation_method,
            mic=neb_cfg.neb_interpolation_mic,
            align_endpoints=neb_cfg.neb_align_endpoints,
            perturb_sigma=neb_cfg.neb_perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=neb_cfg.n_slab,
            n_core_mobile=neb_cfg.n_core_mobile,
            n_adsorbate_mobile=neb_cfg.n_adsorbate_mobile,
            adsorbate_fragment_lengths=neb_cfg.adsorbate_fragment_lengths,
            neb_surface_cell_remap=neb_cfg.neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_cfg.neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_cfg.neb_surface_max_lattice_shift,
            neb_interpolation_bond_tolerance_a=(
                neb_cfg.neb_interpolation_bond_tolerance_a
            ),
            verbosity=verbosity,
        )
        try:
            validate_initial_neb_path(
                images,
                n_slab=neb_cfg.n_slab,
                mic=neb_cfg.neb_interpolation_mic,
                max_endpoint_mismatch=neb_cfg.max_endpoint_mismatch,
                clash_distance=neb_cfg.neb_prescreen_clash_distance,
            )
        except SCGOValidationError as e:
            _record_skipped_pair(pair_ord, pair_id, i, j, react_e, prod_e, str(e))
            continue
        setup_pairs.append((pair_ord, pair_id, i, j, react_e, prod_e, images))

    # Batched single-point energy pre-screen across all valid bands, chunked to
    # fit GPU memory (per-band, input order preserved). Only needed when the
    # ``max_endpoint_mismatch`` energy-profile gate is enabled. Chunking mirrors
    # the optimization pass so a large ``setup_pairs`` list cannot OOM the screen
    # without recovery.
    if setup_pairs and neb_cfg.max_endpoint_mismatch is not None:
        prescreen_band_cap = (
            int(parallel_neb_max_bands)
            if parallel_neb_max_bands is not None and int(parallel_neb_max_bands) > 0
            else None
        )
        band_energy_lists = _evaluate_bands_in_chunks(
            [imgs for _ord, _pid, _i, _j, _re, _pe, imgs in setup_pairs],
            relaxer,
            atom_budget=neb_cfg.parallel_neb_max_batch_atoms,
            band_cap=prescreen_band_cap,
        )
    else:
        band_energy_lists = []

    for setup_i, (pair_ord, pair_id, i, j, react_e, prod_e, images) in enumerate(
        setup_pairs
    ):
        band_energies: list[float] | None = None
        if band_energy_lists:
            band_energies = band_energy_lists[setup_i]
            try:
                validate_initial_neb_energy_profile(
                    band_energies,
                    reference_reactant_energy=react_e,
                    reference_product_energy=prod_e,
                    min_saddle_prominence=neb_cfg.min_saddle_prominence,
                    max_spurious_barrier=neb_cfg.neb_max_spurious_barrier,
                )
            except SCGOValidationError as e:
                _record_skipped_pair(pair_ord, pair_id, i, j, react_e, prod_e, str(e))
                continue
        pair_two_stage = neb_uses_two_stage_climb(
            neb_cfg.neb_climb, neb_steps_i, initial_energies=band_energies
        )
        neb_instances.append(
            TorchSimNEB(
                images,
                relaxer,
                k=neb_cfg.neb_spring_constant,
                climb=bool(neb_cfg.neb_climb) and not pair_two_stage,
                method=neb_cfg.neb_tangent_method,
            )
        )
        neb_two_stage.append(pair_two_stage)
        if band_energies is not None:
            react_e = float(band_energies[0])
            prod_e = float(band_energies[-1])
        result = _make_pair_ts_result(pair_id, react_e=react_e, prod_e=prod_e)
        result["system_type"] = system_type
        pair_results[pair_ord] = result
        neb_meta.append((pair_ord, i, j))

    def _finalize_and_persist_band(neb_idx: int) -> None:
        """Finalize one optimized band and write pair artifacts immediately."""
        pair_ord, i, j = neb_meta[neb_idx]
        neb = neb_instances[neb_idx]
        summary = batch_results[neb_idx]
        result = pair_results[pair_ord]
        if result is None:
            raise SCGOValidationError("Parallel NEB produced a missing pair result")
        result["neb_converged"] = bool(summary.get("converged", False))
        result["error"] = summary.get("error")
        result["final_fmax"] = summary.get("final_fmax")
        result["force_calls"] = neb.get_force_calls()
        result["steps_taken"] = summary.get("steps_taken")

        # Batch failures (e.g. CUDA OOM) leave only GO endpoint energies on the
        # band; finalize would overwrite the real error with endpoint-as-TS.
        error_text = str(result.get("error") or "")
        batch_never_ran = bool(
            error_text
            and (result.get("force_calls") or 0) == 0
            and not result.get("steps_taken")
        )
        # Non-finite NEB forces (nan/inf fmax) mean the band's geometry and
        # energies are meaningless even though steps were taken, so finalize must
        # not turn them into a reported saddle either.
        forces_non_finite = "non-finite" in error_text.lower()
        band_unusable = batch_never_ran or forces_non_finite
        if band_unusable:
            result["status"] = "failed"
            result["neb_converged"] = False
            logger.warning(
                "Parallel NEB band unusable for pair %s (%s): %s",
                result.get("pair_id"),
                "non-finite forces" if forces_non_finite else "no steps taken",
                error_text,
            )
        else:
            try:
                _finalize_neb_result(
                    result,
                    neb.images,
                    logger=logger,
                    max_spurious_barrier=neb_cfg.neb_max_spurious_barrier,
                )
            except (RuntimeError, SCGOValidationError) as e:
                result["status"] = "failed"
                result["neb_converged"] = False
                result["error"] = str(e)
                _detach_calc(result.get("transition_state"))

        if result["neb_converged"] and result.get("status") != "success":
            result["neb_converged"] = False
            logger.warning(
                "Parallel NEB converged but no usable TS for pair %s; marking failed",
                result.get("pair_id"),
            )

        attach_minima_traceability(result, minima, i, j)
        pair_id = str(result["pair_id"])
        pair_dir = run_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_neb_result(result, str(pair_dir), pair_id, verbosity=verbosity)

    if neb_instances:
        t_batch0 = perf_counter()
        batch_results = [
            {
                "converged": False,
                "final_fmax": None,
                "steps_taken": 0,
                "error": None,
            }
            for _ in neb_instances
        ]
        band_cap = (
            int(parallel_neb_max_bands)
            if parallel_neb_max_bands is not None and int(parallel_neb_max_bands) > 0
            else None
        )
        band_costs = [_band_atom_cost(neb) for neb in neb_instances]
        atom_budget = neb_cfg.parallel_neb_max_batch_atoms
        if band_cap is not None and band_cap < len(neb_instances):
            logger.info(
                "Parallel NEB concurrency capped at %d band(s) "
                "(%d total; explicit parallel_neb_max_bands override)",
                band_cap,
                len(neb_instances),
            )
        if atom_budget is not None and int(atom_budget) > 0:
            logger.info(
                "Parallel NEB chunking by atom budget: %d atoms/force-batch "
                "(%d band(s), costs %s)",
                int(atom_budget),
                len(neb_instances),
                band_costs,
            )

        def _chunk_indices(
            indices: list[int], *, budget: int | None = None
        ) -> list[list[int]]:
            """Chunk band indices honoring *both* the atom budget and band cap.

            ``parallel_neb_max_bands`` used to override the atom budget entirely,
            so a band cap that still exceeded GPU capacity produced one oversized
            fused force batch. Chunking at ``min(band_cap, atom_budget)`` keeps the
            scgo-side bound honest even when the memory scaler is unavailable.
            """
            if not indices:
                return []
            chunks = chunk_band_indices_by_atom_budget(
                indices, band_costs, budget if budget is not None else atom_budget
            )
            return _apply_band_cap(chunks, band_cap)

        def _run_chunk_with_oom_retry(
            chunk: list[int],
            *,
            max_total_steps: int,
            max_steps: int,
        ) -> list[dict[str, Any]]:
            """Run one chunk; on CUDA OOM re-bin at half its atom cost and retry.

            Returns per-band summaries in ``chunk`` order. Bands that still fail
            after the retry carry the OOM error text with ``steps_taken=0`` so the
            caller's ``batch_never_ran`` guard can skip :func:`_finalize_neb_result`
            for them downstream.
            """
            try:
                return _run_chunk(chunk, max_total_steps, max_steps)
            except (RuntimeError, MemoryError) as exc:
                if not is_cuda_oom_error(exc):
                    raise
                logger.warning(
                    "Parallel NEB chunk of %d band(s) hit CUDA OOM (%s); "
                    "retrying once at half the chunk atom cost",
                    len(chunk),
                    exc,
                )
            cleanup_torch_cuda(logger=logger)
            chunk_atoms = sum(band_costs[i] for i in chunk)
            retry_budget = max(1, chunk_atoms // 2)
            summaries: dict[int, dict[str, Any]] = {}
            for sub_chunk in chunk_band_indices_by_atom_budget(
                chunk, band_costs, retry_budget
            ):
                try:
                    sub_results = _run_chunk(sub_chunk, max_total_steps, max_steps)
                except (RuntimeError, MemoryError) as exc:
                    if not is_cuda_oom_error(exc):
                        raise
                    logger.error(
                        "Parallel NEB retry still OOM for %d band(s): %s",
                        len(sub_chunk),
                        exc,
                    )
                    cleanup_torch_cuda(logger=logger)
                    sub_results = [
                        {
                            "converged": False,
                            "final_fmax": None,
                            "steps_taken": 0,
                            "error": str(exc),
                        }
                        for _ in sub_chunk
                    ]
                for local_i, band_i in enumerate(sub_chunk):
                    summaries[band_i] = sub_results[local_i]
            return [summaries[i] for i in chunk]

        def _run_chunk(
            chunk: list[int], max_total_steps: int, max_steps: int
        ) -> list[dict[str, Any]]:
            chunk_nebs = [neb_instances[i] for i in chunk]
            batch = ParallelNEBBatch(
                chunk_nebs, relaxer, max_total_steps=max_total_steps
            )
            try:
                return batch.run_optimization(
                    fmax=neb_cfg.neb_fmax, max_steps=max_steps
                )
            finally:
                del batch
                cleanup_torch_cuda(logger=logger)

        # Single-stage climb bands (typical endpoint-max IDPP adsorbate paths).
        single_idx = [i for i, ts in enumerate(neb_two_stage) if not ts]
        two_idx = [i for i, ts in enumerate(neb_two_stage) if ts]
        # Chunk each stage list independently so two-stage bands (which need a
        # separate climb pass) never share a force batch with single-stage bands.
        for chunk in _chunk_indices(single_idx):
            chunk_results = _run_chunk_with_oom_retry(
                chunk, max_total_steps=neb_steps_i, max_steps=neb_steps_i
            )
            for local_i, neb_i in enumerate(chunk):
                batch_results[neb_i] = chunk_results[local_i]
            for neb_i in chunk:
                _finalize_and_persist_band(neb_i)
        for chunk in _chunk_indices(two_idx):
            # Interior-max IDPP: relax without climb, then climb (always).
            chunk_nebs = [neb_instances[i] for i in chunk]
            stage1_cap = neb_steps_i // 2
            stage1_results = _run_chunk_with_oom_retry(
                chunk, max_total_steps=stage1_cap, max_steps=stage1_cap
            )
            for neb in chunk_nebs:
                neb.climb = True
            climb_local = [
                i
                for i, summary in enumerate(stage1_results)
                if _stage1_band_climb_eligible(summary)
            ]
            if climb_local:
                steps1_vals = [
                    int(stage1_results[i].get("steps_taken") or 0) for i in climb_local
                ]
                # Stage 2 shares one step budget across the whole chunk, so the
                # slowest stage-1 band (max steps taken) sets it for everyone.
                # This is intentionally conservative: it guarantees no band in the
                # chunk exceeds the overall neb_steps budget, at the cost of
                # shrinking the climb budget for bands that converged quickly.
                # The neb_steps_i // 2 floor keeps a usable climb pass regardless.
                stage2_steps = max(
                    neb_steps_i // 2,
                    neb_steps_i - max(steps1_vals),
                    1,
                )
                stage2_chunk = [chunk[i] for i in climb_local]
                stage2_results = _run_chunk_with_oom_retry(
                    stage2_chunk,
                    max_total_steps=stage2_steps,
                    max_steps=stage2_steps,
                )
                for local_i, s1_i in enumerate(climb_local):
                    s2 = stage2_results[local_i]
                    s1 = stage1_results[s1_i]
                    steps1 = int(s1.get("steps_taken") or 0)
                    steps2 = int(s2.get("steps_taken") or 0)
                    stage1_results[s1_i] = {
                        "converged": bool(s2.get("converged", False)),
                        "final_fmax": s2.get("final_fmax", s1.get("final_fmax")),
                        "steps_taken": steps1 + steps2,
                        "error": s2.get("error") or s1.get("error"),
                    }
            for local_i, neb_i in enumerate(chunk):
                batch_results[neb_i] = stage1_results[local_i]
            for neb_i in chunk:
                _finalize_and_persist_band(neb_i)
        neb_batch_s = perf_counter() - t_batch0
    else:
        batch_results = []
        neb_batch_s = 0.0

    wall_total = perf_counter() - t_parallel0
    n_active = max(1, len(neb_instances))
    neb_each = neb_batch_s / n_active
    wall_each = wall_total / max(1, len(pairs))

    # Timings are averages; pair metadata was already written per chunk.
    for pair_ord, _i, _j in neb_meta:
        result = pair_results[pair_ord]
        if result is None:
            raise SCGOValidationError("Parallel NEB produced a missing pair result")
        result["timings_s"] = {
            "kind": "neb",
            "total_wall_avg_s": wall_each,
            "neb_optimization_avg_s": neb_each,
            "cpu_non_relax_avg_s": max(0.0, wall_each - neb_each),
        }

    meta = {
        "neb_batch_optimization_s": neb_batch_s,
        "parallel_wall_s": wall_total,
    }
    if not all(r is not None for r in pair_results):
        raise SCGOValidationError("Parallel NEB produced a missing pair result")
    results: list[dict[str, Any]] = [r for r in pair_results if r is not None]
    log_neb_search_summaries(logger, results, verbosity=verbosity, run_dir=str(run_dir))
    return results, meta
