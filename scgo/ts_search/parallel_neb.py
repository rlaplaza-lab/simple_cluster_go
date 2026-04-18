"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.optimize import FIRE

from scgo.ts_search.transition_state import (
    TorchSimNEB,
    attach_singlepoint_from_relax_output,
)
from scgo.utils.logging import get_logger

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

        # Track optimization state
        self.active_nebs = list(range(len(neb_instances)))  # Indices of active NEBs
        self.converged_nebs: dict[int, bool] = {}  # neb_idx -> converged
        self.failed_nebs: dict[int, str] = {}  # neb_idx -> error_msg
        self.step_count = 0

        # Per-NEB optimizer instances (created lazily). Uses ASE optimizers
        # (default: FIRE) so stepping respects NEB forces / spring terms.
        self._optimizers: dict[int, object] = {}  # neb_idx -> Optimizer instance

    def run_optimization(
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
    ) -> list[dict[str, Any]]:
        """Optimize NEBs using batched evaluations; return per-NEB summaries."""

        # Validate input
        if not self.neb_instances:
            logger.error("No NEB instances provided to run_optimization")
            return []

        results = [
            {
                "converged": False,
                "steps_taken": 0,
                "final_fmax": None,
                "error": None,
                "force_calls": 0,
            }
            for _ in self.neb_instances
        ]

        step_cap = min(self.max_total_steps, int(max_steps))
        try:
            while self.active_nebs and self.step_count < step_cap:
                # Collect images from all active NEBs
                all_images = []
                neb_image_map = []

                for neb_idx in self.active_nebs:
                    neb = self.neb_instances[neb_idx]
                    start_idx = len(all_images)
                    all_images.extend(neb.images)
                    end_idx = len(all_images)
                    neb_image_map.append((neb_idx, start_idx, end_idx))

                if not all_images:
                    break

                # Single batched GPU call for all images
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

                # Redistribute forces/energies back to individual NEBs
                for neb_idx, start_idx, end_idx in neb_image_map:
                    neb = self.neb_instances[neb_idx]

                    # Verify batch results match NEB images
                    neb_image_count = len(neb.images)
                    batch_image_count = end_idx - start_idx
                    if neb_image_count != batch_image_count:
                        raise RuntimeError(
                            f"NEB {neb_idx}: image count mismatch. "
                            f"NEB has {neb_image_count} images, "
                            f"batch returned {batch_image_count} results"
                        )

                    # Update atoms with energies and forces from relaxed_atoms
                    for atoms, (energy, relaxed_atoms) in zip(
                        neb.images,
                        batch_results[start_idx:end_idx],
                        strict=True,
                    ):
                        attach_singlepoint_from_relax_output(
                            atoms, energy, relaxed_atoms, require_forces=False
                        )

                # Now let each NEB compute its forces and step
                still_active = []
                for neb_idx in self.active_nebs:
                    neb = self.neb_instances[neb_idx]

                    try:
                        # Use ASE NEB's get_forces to compute NEB forces from PES forces
                        neb_result = neb.get_forces()
                        max_force = np.max(np.abs(neb_result))

                        results[neb_idx]["final_fmax"] = float(max_force)
                        results[neb_idx]["force_calls"] += 1
                        results[neb_idx]["steps_taken"] = self.step_count + 1

                        # Use an ASE optimizer per-NEB to perform a single
                        # adaptive optimization step. This ensures spring/tangent
                        # contributions from `neb.get_forces()` are respected and
                        # avoids the crude fixed-step update.
                        if max_force < fmax:
                            results[neb_idx]["converged"] = max_force < fmax
                            self.converged_nebs[neb_idx] = max_force < fmax
                            logger.debug(
                                f"NEB {neb_idx} finished: "
                                f"converged={max_force < fmax}, fmax={max_force:.6f}"
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

                # Prepare for the next iteration: keep only still-active NEBs and
                # advance the global step counter so the max_steps cutoff can take effect.
                self.active_nebs = still_active
                self.step_count += 1

                # If nothing remains active, exit early
                if not self.active_nebs:
                    break

        except (RuntimeError, OSError) as e:
            logger.error(
                f"Fatal error in parallel NEB optimization: {type(e).__name__}: {e}"
            )
            for neb_idx in self.active_nebs:
                results[neb_idx]["error"] = str(e)

        # Finalize results
        for neb_idx in range(len(self.neb_instances)):
            if neb_idx not in self.converged_nebs and neb_idx not in self.failed_nebs:
                if (
                    results[neb_idx]["steps_taken"]
                    and results[neb_idx]["steps_taken"] > 0
                ):
                    results[neb_idx]["error"] = (
                        f"NEB did not converge after {results[neb_idx]['steps_taken']} steps"
                    )
                else:
                    results[neb_idx]["error"] = "NEB not processed"

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
