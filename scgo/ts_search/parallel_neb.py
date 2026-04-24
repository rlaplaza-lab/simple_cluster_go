"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from scgo.calculators import torchsim_helpers as _tsh
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    SystemType,
    validate_structure_for_system_type,
)
from scgo.utils.logging import get_logger

from .transition_state import (
    TorchSimNEB,
    _detach_calc,
    _finalize_neb_result,
    attach_minima_traceability,
    attach_singlepoint_from_relax_output,
    interpolate_path,
    make_ts_result,
    save_neb_result,
)

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

logger = get_logger(__name__)


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

        # Per-NEB optimizer instances (created lazily). Uses ASE optimizers
        # (default: FIRE) so stepping respects NEB forces / spring terms.
        self._optimizers: dict[int, object] = {}

    def run_optimization(
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
    ) -> list[dict[str, Any]]:
        """Optimize NEBs using batched evaluations; return per-NEB summaries."""
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
            }
            for _ in self.neb_instances
        ]

        step_cap = min(self.max_total_steps, int(max_steps))
        while self.active_nebs and self.step_count < step_cap:
            all_images: list[Atoms] = []
            neb_image_map: list[tuple[int, int, int]] = []

            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                start_idx = len(all_images)
                all_images.extend(neb.images)
                neb_image_map.append((neb_idx, start_idx, len(all_images)))

            if not all_images:
                break

            logger.debug(
                f"Step {self.step_count}: Evaluating {len(all_images)} images "
                f"from {len(self.active_nebs)} active NEBs"
            )

            try:
                batch_results = self.relaxer.relax_batch(all_images, steps=0)
            except (RuntimeError, ValueError) as e:
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

            for neb_idx, _, _ in neb_image_map:
                self.neb_instances[neb_idx]._force_calls += 1

            for neb_idx, start_idx, end_idx in neb_image_map:
                neb = self.neb_instances[neb_idx]
                if len(neb.images) != end_idx - start_idx:
                    raise RuntimeError(
                        f"NEB {neb_idx}: image count mismatch. "
                        f"NEB has {len(neb.images)} images, "
                        f"batch returned {end_idx - start_idx} results"
                    )
                for atoms, (energy, relaxed_atoms) in zip(
                    neb.images,
                    batch_results[start_idx:end_idx],
                    strict=True,
                ):
                    attach_singlepoint_from_relax_output(
                        atoms, energy, relaxed_atoms, require_forces=False
                    )

            still_active: list[int] = []
            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                try:
                    neb_forces = neb.get_forces()
                    max_force = float(np.max(np.abs(neb_forces)))

                    results[neb_idx]["final_fmax"] = max_force
                    results[neb_idx]["steps_taken"] = self.step_count + 1

                    if max_force < fmax:
                        results[neb_idx]["converged"] = True
                        self.converged_nebs[neb_idx] = True
                        logger.debug(
                            f"NEB {neb_idx} finished: converged, fmax={max_force:.6f}"
                        )
                    else:
                        if neb_idx not in self._optimizers:
                            self._optimizers[neb_idx] = self.optimizer_cls(
                                neb, logfile=None, trajectory=None
                            )
                        self._optimizers[neb_idx].step()
                        still_active.append(neb_idx)
                except (RuntimeError, ValueError) as e:
                    logger.debug(f"NEB {neb_idx} step failed: {e}")
                    self.failed_nebs[neb_idx] = str(e)
                    results[neb_idx]["error"] = str(e)

            self.active_nebs = still_active
            self.step_count += 1

            if not self.active_nebs:
                break

        for neb_idx in range(len(self.neb_instances)):
            if neb_idx not in self.converged_nebs and neb_idx not in self.failed_nebs:
                steps = results[neb_idx]["steps_taken"] or 0
                results[neb_idx]["error"] = (
                    f"NEB did not converge after {steps} steps"
                    if steps
                    else "NEB not processed"
                )

        logger.info(
            f"Parallel NEB batch complete: {self.step_count} steps, "
            f"{len(self.converged_nebs)} converged, {len(self.failed_nebs)} failed"
        )

        return results

    def get_summary(self) -> dict[str, int]:
        """Return counts of total, converged and failed NEBs."""
        return {
            "total_nebs": len(self.neb_instances),
            "converged": len(self.converged_nebs),
            "failed": len(self.failed_nebs),
            "total_steps": self.step_count,
        }


def _neb_endpoint_copies(
    atoms_i: Atoms,
    atoms_j: Atoms,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType,
    n_slab: int = 0,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> tuple[Atoms, Atoms]:
    """Copy minima endpoints, optionally re-attaching surface FixAtoms constraints."""
    from scgo.surface.constraints import attach_slab_constraints_from_surface_config

    react = atoms_i.copy()
    prod = atoms_j.copy()
    if surface_config is not None:
        attach_slab_constraints_from_surface_config(react, surface_config)
        attach_slab_constraints_from_surface_config(prod, surface_config)
    validate_structure_for_system_type(
        react,
        system_type=system_type,
        surface_config=surface_config,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
    )
    validate_structure_for_system_type(
        prod,
        system_type=system_type,
        surface_config=surface_config,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
    )
    return react, prod


def run_parallel_neb_search(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Atoms]],
    *,
    result_dir: Path,
    surface_config: SurfaceSystemConfig | None,
    rng: np.random.Generator | None,
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
    torchsim_params: dict[str, Any],
    system_type: SystemType,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run all pairs through ParallelNEBBatch. Returns (results, timing meta)."""
    t_parallel0 = perf_counter()
    relaxer = _tsh.TorchSimBatchRelaxer(**(torchsim_params or {}))

    # Endpoint single-point energies in one batched call.
    endpoints: list[Atoms] = []
    pair_endpoint_index: list[tuple[int, int]] = []
    for i, j in pairs:
        ri, rj = _neb_endpoint_copies(
            minima[i][1],
            minima[j][1],
            surface_config,
            system_type,
            n_slab=n_slab,
            adsorbate_definition=adsorbate_definition,
        )
        endpoints.append(ri)
        endpoints.append(rj)
        pair_endpoint_index.append((len(endpoints) - 2, len(endpoints) - 1))

    ep_results = relaxer.relax_batch(endpoints, steps=0)
    pair_endpoint_energies = [
        (float(ep_results[a][0]), float(ep_results[b][0]))
        for a, b in pair_endpoint_index
    ]

    neb_instances: list[TorchSimNEB] = []
    pair_results: list[dict[str, Any]] = []
    for (i, j), (react_e, prod_e) in zip(pairs, pair_endpoint_energies, strict=True):
        pair_id = f"{i}_{j}"
        react_ep, prod_ep = _neb_endpoint_copies(
            minima[i][1],
            minima[j][1],
            surface_config,
            system_type,
            n_slab=n_slab,
            adsorbate_definition=adsorbate_definition,
        )
        images = interpolate_path(
            react_ep,
            prod_ep,
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
        )
        neb_instances.append(
            TorchSimNEB(
                images,
                relaxer,
                k=neb_spring_constant,
                climb=neb_climb,
                method=neb_tangent_method,
            )
        )
        pair_results.append(
            make_ts_result(
                pair_id=pair_id,
                n_images=neb_n_images,
                spring_constant=neb_spring_constant,
                use_torchsim=True,
                fmax=neb_fmax,
                neb_steps=neb_steps,
                interpolation_method=neb_interpolation_method,
                climb=neb_climb,
                align_endpoints=neb_align_endpoints,
                perturb_sigma=neb_perturb_sigma,
                neb_interpolation_mic=neb_interpolation_mic,
                neb_tangent_method=neb_tangent_method,
                use_parallel_neb=True,
                reactant_energy=react_e,
                product_energy=prod_e,
            )
        )
        pair_results[-1]["system_type"] = system_type

    neb_steps_i = int(neb_steps)
    batch = ParallelNEBBatch(neb_instances, relaxer, max_total_steps=neb_steps_i)
    t_batch0 = perf_counter()
    batch_results = batch.run_optimization(fmax=neb_fmax, max_steps=neb_steps_i)
    neb_batch_s = perf_counter() - t_batch0
    wall_total = perf_counter() - t_parallel0
    n_p = max(1, len(pair_results))
    neb_each = neb_batch_s / n_p
    wall_each = wall_total / n_p

    for idx, neb in enumerate(neb_instances):
        summary = batch_results[idx]
        result = pair_results[idx]
        result["neb_converged"] = bool(summary.get("converged", False))
        result["error"] = summary.get("error")
        result["final_fmax"] = summary.get("final_fmax")
        result["force_calls"] = neb.get_force_calls()
        result["steps_taken"] = summary.get("steps_taken")

        try:
            _finalize_neb_result(result, neb.images, logger=logger)
        except RuntimeError as e:
            result["status"] = "failed"
            result["error"] = str(e)
            _detach_calc(result.get("transition_state"))

        if result["neb_converged"] and result.get("status") != "success":
            logger.warning(
                "Parallel NEB converged but no usable TS for pair %s; marking failed",
                result.get("pair_id"),
            )

        i, j = pairs[idx]
        attach_minima_traceability(result, minima, i, j)
        save_neb_result(result, str(result_dir), result["pair_id"])
        result["timings_s"] = {
            "total_wall_s": wall_each,
            "neb_optimization_s": neb_each,
            "cpu_non_relax_s": max(0.0, wall_each - neb_each),
        }

    meta = {
        "neb_batch_optimization_s": neb_batch_s,
        "parallel_wall_s": wall_total,
    }
    return pair_results, meta
