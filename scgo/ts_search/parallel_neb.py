"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE

from scgo.ts_search.transition_state import TorchSimNEB
from scgo.utils.logging import get_logger

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

logger = get_logger(__name__)


def _pes_forces_from_relaxed(relaxed_atoms: Any) -> np.ndarray | None:
    """Return PES forces from TorchSim-relaxed atoms, or None if unavailable."""
    forces = relaxed_atoms.arrays.get("forces")
    if forces is not None and getattr(forces, "size", 0) > 0:
        return np.asarray(forces)
    return None


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

        try:
            while self.active_nebs and self.step_count < self.max_total_steps:
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

                # Record batched relaxer invocation on per-NEB counters so
                # `TorchSimNEB.get_force_calls()` reflects batched evaluations
                # in addition to direct NEB-invoked evaluations.
                for neb_idx, _, _ in neb_image_map:
                    self.neb_instances[neb_idx].increment_force_calls()

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
                        forces = _pes_forces_from_relaxed(relaxed_atoms)
                        if forces is not None:
                            atoms.calc = SinglePointCalculator(
                                atoms, energy=energy, forces=forces
                            )
                        else:
                            atoms.calc = SinglePointCalculator(atoms, energy=energy)

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

                            try:
                                # NEB forces are already cached in SinglePointCalculator on
                                # each image; step() will call neb.get_forces() internally.
                                self._optimizers[neb_idx].step()
                            except (
                                RuntimeError,
                                ValueError,
                                TypeError,
                                AttributeError,
                            ) as e:
                                logger.error(
                                    f"NEB {neb_idx} optimizer step failed: {type(e).__name__}: {e}"
                                )
                                results[neb_idx]["error"] = str(e)
                                self.failed_nebs[neb_idx] = str(e)
                                continue

                            # Do NOT recompute NEB forces per-NEB here: that
                            # triggers an individual TorchSim evaluation per NEB,
                            # defeating the batched GPU strategy.  The next outer
                            # loop iteration will batch-evaluate all active NEBs
                            # together.  For the final iteration the pre-step fmax
                            # is a conservative (over-)estimate of residual forces.

                            if self.step_count < max_steps - 1:
                                still_active.append(neb_idx)
                            else:
                                # Last allowed iteration: use pre-step fmax as a
                                # conservative convergence proxy (true post-step
                                # fmax would require an extra evaluation).
                                results[neb_idx]["converged"] = (
                                    results[neb_idx]["final_fmax"] < fmax
                                )
                                if results[neb_idx]["converged"]:
                                    self.converged_nebs[neb_idx] = True

                    except RuntimeError as e:
                        logger.debug(f"NEB {neb_idx} force computation error: {e}")
                        self.failed_nebs[neb_idx] = str(e)
                        results[neb_idx]["error"] = str(e)
                    except ValueError as e:
                        logger.debug(f"NEB {neb_idx} invalid force calculation: {e}")
                        self.failed_nebs[neb_idx] = str(e)
                        results[neb_idx]["error"] = str(e)
                    except KeyboardInterrupt:
                        raise
                    except (AttributeError, TypeError) as e:
                        # Treat unexpected attribute/type errors as NEB-specific failures
                        logger.error(
                            f"Unexpected NEB {neb_idx} step error ({type(e).__name__}): {e}"
                        )
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
