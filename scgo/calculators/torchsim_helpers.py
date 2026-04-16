"""Utilities for integrating TorchSim batched relaxations with SCGO.

This module wraps the TorchSim high-level optimization API so SCGO can relax
multiple candidate structures in a single batched call.

Important:
- Imports for optional stacks (TorchSim, MACE, FairChem) are **lazy** so SCGO can
  be imported in minimal environments without pulling MLIP dependencies.
- TorchSim can run with multiple model families. SCGO supports MACE and
  FairChem/UMA via TorchSim model wrappers.
"""

from __future__ import annotations

import functools
import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms as ASEFixAtoms

from scgo.database.metadata import update_metadata
from scgo.utils.helpers import ensure_float64_forces
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "MemoryScalerCache",
    "TorchSimBatchRelaxer",
    "build_torchsim_fixatoms_from_ase_batch",
    "collect_ase_fixatoms_indices",
    "get_global_memory_scaler_cache",
]


class MemoryScalerCache:
    """Disk-backed cache for TorchSim ``max_memory_scaler`` (GPU probing takes ~70s).

    Essential for performance: without caching, each first run in a cluster size
    forces expensive memory estimation via forward passes. Saves ~70s per campaign.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_file: str = "memory_scaler_cache.json",
    ):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "scgo" / "torchsim"
        self._cache_dir = Path(cache_dir)
        self._cache_path = self._cache_dir / cache_file
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk if it exists."""
        if not self._cache_path.exists():
            return {}
        try:
            with open(self._cache_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load memory scaler cache: %s", exc)
            return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save memory scaler cache: %s", exc)

    def _make_key(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
    ) -> str:
        """Create a cache key from parameters (n_atoms binned to nearest 5)."""
        atom_bin = ((n_atoms + 4) // 5) * 5
        return f"{model_name}|{memory_scales_with}|{device}|atoms_{atom_bin}"

    def get(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
    ) -> float | None:
        """Get cached max_memory_scaler if available."""
        key = self._make_key(n_atoms, model_name, memory_scales_with, device)
        return self._cache.get(key)

    def set(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
        value: float,
    ) -> None:
        """Cache a max_memory_scaler value to disk."""
        key = self._make_key(n_atoms, model_name, memory_scales_with, device)
        self._cache[key] = value
        self._save_cache()

    def clear(self) -> None:
        """Clear the cache."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()


# Global cache instance shared across all TorchSimBatchRelaxer instances
_GLOBAL_MEMORY_SCALER_CACHE = MemoryScalerCache()


def collect_ase_fixatoms_indices(atoms: Atoms) -> list[int]:
    """Return sorted unique indices constrained by ASE :class:`ase.constraints.FixAtoms`.

    Other ASE constraint types are ignored (not represented in TorchSim today).
    """
    out: list[int] = []
    for c in atoms.constraints:
        if isinstance(c, ASEFixAtoms):
            out.extend(int(i) for i in c.index)
    return sorted(set(out))


def _patch_torchsim_fixatoms_subconstraint_device() -> None:
    """Align ``atom_idx`` device with constraint indices before ``torch.isin``.

    ``torch_sim.state._split_state`` passes a CPU ``torch.arange`` into
    ``FixAtoms.select_sub_constraint`` while batched constraints live on CUDA,
    which makes ``torch.isin`` fail. This is applied once per process.
    """
    # Lazy import: only patch when TorchSim is actually available/used.
    try:
        from torch_sim.constraints import FixAtoms as TSFixAtoms  # type: ignore
    except Exception:
        return

    if getattr(TSFixAtoms.select_sub_constraint, "_scgo_patched", False):
        return

    _orig = TSFixAtoms.select_sub_constraint

    def select_sub_constraint(self, atom_idx: torch.Tensor, sys_idx: int):
        if atom_idx.device != self.atom_idx.device:
            atom_idx = atom_idx.to(self.atom_idx.device)
        return _orig(self, atom_idx, sys_idx)

    select_sub_constraint._scgo_patched = True  # type: ignore[attr-defined]
    TSFixAtoms.select_sub_constraint = select_sub_constraint  # type: ignore[assignment]


def build_torchsim_fixatoms_from_ase_batch(
    atoms_list: Sequence[Atoms],
    device: object,
) -> object | None:
    """Map per-structure ASE ``FixAtoms`` to one TorchSim ``FixAtoms`` (global indices).

    TorchSim's :func:`torch_sim.io.atoms_to_state` does **not** read
    ``atoms.constraints``; this builds the constraint the optimizer expects.

    Args:
        atoms_list: One or more ASE systems in batch order (same order as
            :func:`torch_sim.io.atoms_to_state`).
        device: ``torch.device`` for the index tensor (match compute device).

    Returns:
        A ``torch_sim.constraints.FixAtoms`` instance, or ``None`` if nothing to fix.
    """
    # Lazy import: do not require TorchSim until needed.
    from torch_sim.constraints import FixAtoms as TSFixAtoms  # type: ignore

    merged: list[int] = []
    offset = 0
    for atoms in atoms_list:
        merged.extend(offset + idx for idx in collect_ase_fixatoms_indices(atoms))
        offset += len(atoms)
    if not merged:
        return None
    idx_t = torch.tensor(merged, device=device, dtype=torch.long)
    return TSFixAtoms(atom_idx=idx_t)


_patch_torchsim_fixatoms_subconstraint_device()


def _load_default_mace_model(
    *,
    device,
    dtype,
    mace_model_name: str = "mace_matpes_0",
    compute_forces: bool = True,
    compute_stress: bool = False,
):
    """Create a TorchSim MACE model given a canonical model identifier."""
    # Lazy imports: only required for the MACE TorchSim path.
    from mace.calculators.foundations_models import mace_mp  # type: ignore
    from torch_sim.models.mace import MaceModel  # type: ignore

    from scgo.calculators.mace_helpers import MaceUrls

    model_selector = getattr(MaceUrls, mace_model_name, mace_model_name)
    raw_model = mace_mp(
        model=model_selector,
        return_raw_model=True,
        default_dtype=str(dtype).removeprefix("torch."),
        device=device,
    )
    return MaceModel(
        model=raw_model,
        device=device,
        dtype=dtype,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )


def _load_default_fairchem_model(
    *,
    device,
    dtype,
    fairchem_model_name: str,
    fairchem_task_name: str | None,
    compute_stress: bool = False,
):
    """Create a TorchSim FairChem model for UMA checkpoints."""
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    return FairChemModel(
        model=fairchem_model_name,
        task_name=fairchem_task_name,
        device=device,
        dtype=dtype,
        compute_stress=compute_stress,
    )


@dataclass(eq=False)
class TorchSimBatchRelaxer:
    """Batched relaxer that offloads geometry optimization to TorchSim.

    ASE :class:`ase.constraints.FixAtoms` on input structures are translated to
    TorchSim's internal ``FixAtoms`` before optimization (TorchSim's ASE reader
    does not import ``atoms.constraints``).

    Parameters
    ----------
    device:
        Optional torch device. Defaults to CUDA when available, otherwise CPU.
    dtype:
        Torch dtype. Defaults to float32, matching TorchSim defaults for MLIPs.
    model:
        Optional TorchSim model implementing ``ModelInterface``. If omitted, a
        MACE foundation model specified by ``mace_model_name`` is loaded.
    mace_model_name:
        Name of the TorchSim ``MaceUrls`` member to load when ``model`` is not
        provided (default: ``"mace_matpes_0"``).
    optimizer_name:
        Name of TorchSim optimizer (e.g., "fire"), resolved to ``ts.Optimizer.*``.
    force_tol:
        Force convergence threshold (eV/Å). ``None`` uses TorchSim defaults.
    expected_max_atoms:
        Optional atom count for upfront memory probing (e.g., ``cluster_size * population_size``).
        If provided, probes GPU memory at initialization time to cache the ``max_memory_scaler``,
        avoiding expensive re-probing (~1s) on first ``relax_batch()`` call. Recommended for
        GA/BH campaigns with known population sizes.
    runner_kwargs:
        Extra keyword arguments forwarded to ``torch_sim.optimize``.

    """

    device: object | None = None
    dtype: object | None = None
    model: object | None = None
    model_kind: str = "mace"  # "mace" or "fairchem"
    mace_model_name: str = "mace_matpes_0"
    fairchem_model_name: str | None = None
    fairchem_task_name: str | None = None
    optimizer_name: str = "fire"
    force_tol: float | None = 0.05
    max_steps: int | None = 100
    # Autobatching controls; use "binning" to let TorchSim pick optimal batch sizes
    autobatch_strategy: str | None = "binning"  # options: "binning", "inflight", None
    memory_scales_with: str = "n_atoms_x_density"
    max_memory_scaler: float | None = None
    expected_max_atoms: int | None = (
        None  # Probe memory upfront with this atom count (cluster_size * pop_size)
    )
    init_kwargs: dict | None = None
    step_kwargs: dict | None = None
    runner_kwargs: dict | None = None
    seed: int | None = None
    compile_model: bool = False  # torch.compile can speed up MACE but adds startup cost

    def __post_init__(self) -> None:
        self._torch = torch
        # Lazy import: only require TorchSim when actually instantiating the relaxer.
        import torch_sim as ts  # type: ignore

        self._ts = ts
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype is None:
            # Match ASE MACE wrapper default of float64 for parity
            self.dtype = torch.float64

        # Optional seeding (do not force deterministic algorithms to avoid CuBLAS constraints)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if isinstance(self.optimizer_name, str):
            try:
                self.optimizer = getattr(ts.Optimizer, self.optimizer_name.lower())
            except AttributeError as exc:
                available = [x for x in dir(ts.Optimizer) if not x.startswith("_")]
                raise ValueError(
                    f"Unknown TorchSim optimizer '{self.optimizer_name}'. "
                    f"Available: {available}",
                ) from exc
        else:
            self.optimizer = self.optimizer_name

        if self.model is None:
            mk = str(self.model_kind or "mace").strip().lower()
            if mk == "mace":
                self.model = _load_default_mace_model(
                    device=self.device,
                    dtype=self.dtype,
                    mace_model_name=self.mace_model_name,
                )
            elif mk in ("fairchem", "uma"):
                if not self.fairchem_model_name:
                    raise ValueError(
                        "TorchSimBatchRelaxer(model_kind='fairchem') requires fairchem_model_name"
                    )
                self.model = _load_default_fairchem_model(
                    device=self.device,
                    dtype=self.dtype,
                    fairchem_model_name=str(self.fairchem_model_name),
                    fairchem_task_name=self.fairchem_task_name,
                )
            else:
                raise ValueError(
                    f"Unknown model_kind {self.model_kind!r}; expected 'mace' or 'fairchem'"
                )
        self._patch_model_for_cuda()

        if self.compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Store device string for cache key (e.g., "cuda" or "cpu")
        self._device_str = str(self.device).split(":")[0]

        self._runner_kwargs = dict(self.runner_kwargs or {})
        if self.autobatch_strategy and "autobatcher" not in self._runner_kwargs:
            if self.autobatch_strategy in ("binning", "inflight"):
                if str(self.device).split(":")[0] == "cpu":
                    autobatcher = None
                else:
                    autobatcher = self._ts.InFlightAutoBatcher(
                        model=self.model,
                        memory_scales_with=self.memory_scales_with,
                        max_memory_scaler=self.max_memory_scaler,
                    )
            else:
                autobatcher = None
            if autobatcher is not None:
                self._runner_kwargs["autobatcher"] = autobatcher
        if self.init_kwargs and "init_kwargs" not in self._runner_kwargs:
            self._runner_kwargs["init_kwargs"] = dict(self.init_kwargs)
        if self.step_kwargs and "step_kwargs" not in self._runner_kwargs:
            self._runner_kwargs["step_kwargs"] = dict(self.step_kwargs)
        if self.force_tol is not None and "convergence_fn" not in self._runner_kwargs:
            self._runner_kwargs["convergence_fn"] = ts.generate_force_convergence_fn(
                force_tol=self.force_tol,
                include_cell_forces=False,
            )
        # Cap iterations; default 100 matches ASE GA niter_local_relaxation default
        if "max_steps" not in self._runner_kwargs and self.max_steps is not None:
            self._runner_kwargs["max_steps"] = self.max_steps

        # Probe memory upfront if expected_max_atoms provided (avoids runtime probing cost)
        if self.expected_max_atoms is not None and self.max_memory_scaler is None:
            self._probe_memory_upfront(self.expected_max_atoms)

    def _probe_memory_upfront(self, n_atoms: int) -> None:
        """Probe GPU memory with a dummy structure at init time to cache memory scaler.

        This avoids expensive probing (~1s) on first relax_batch() call. The probing
        only happens once; subsequent calls use the cached value.

        Args:
            n_atoms: Atom count to probe with (typically: cluster_size * population_size).
        """
        if (
            self.max_memory_scaler is not None
            or "autobatcher" not in self._runner_kwargs
        ):
            return  # Skip if already set or autobatcher not in use

        try:
            # Check if already cached from previous runs
            cached_scaler = _GLOBAL_MEMORY_SCALER_CACHE.get(
                n_atoms=n_atoms,
                model_name=self._cache_model_name(),
                memory_scales_with=self.memory_scales_with,
                device=self._device_str,
            )
            if cached_scaler is not None:
                self._runner_kwargs["autobatcher"].max_memory_scaler = cached_scaler
                logger.info("Used cached memory scaler from disk (avoided probing)")
                return

            # Create a dummy structure with n_atoms for probing
            dummy = bulk(
                "Cu", "fcc", a=3.61, cubic=True
            )  # Use Cu for generic structure
            while len(dummy) < n_atoms:
                dummy = dummy.repeat((2, 2, 2))
            dummy = dummy[:n_atoms]  # Trim to exact size
            dummy.center(vacuum=3.0)

            # Run a single-step optimization to trigger memory estimation
            logger.info(
                f"Probing GPU memory with {n_atoms} atoms (cluster_size * population)..."
            )
            initial_time = time.time()
            _ = self._ts.optimize(
                system=[dummy],
                model=self.model,
                optimizer=self.optimizer,
                max_steps=1,
                **{k: v for k, v in self._runner_kwargs.items() if k != "max_steps"},
            )
            probe_time = time.time() - initial_time

            # Extract and cache the computed memory scaler
            autobatcher = self._runner_kwargs.get("autobatcher")
            if (
                autobatcher
                and hasattr(autobatcher, "max_memory_scaler")
                and autobatcher.max_memory_scaler
            ):
                _GLOBAL_MEMORY_SCALER_CACHE.set(
                    n_atoms=n_atoms,
                    model_name=self._cache_model_name(),
                    memory_scales_with=self.memory_scales_with,
                    device=self._device_str,
                    value=float(autobatcher.max_memory_scaler),
                )
                logger.info(
                    f"Memory probing complete ({probe_time:.2f}s). "
                    f"Scaler cached for {n_atoms} atoms."
                )
        except Exception as e:
            logger.warning(
                f"Memory probing failed (non-fatal): {e}. Will retry on first relax_batch()."
            )

    def relax_batch(
        self, atoms_list: Sequence[Atoms], steps: int | None = None
    ) -> list[tuple[float, Atoms]]:
        """Relax a batch of ASE ``Atoms`` objects using TorchSim.

        Args:
            atoms_list: List of Atoms objects to relax.
            steps: Optional override for max_steps. Set to 0 for single-point calculation.

        Returns:
            A list of ``(energy, atoms)`` with matching order to the input
        list. Energies are converted to Python floats in eV.
        """
        if not atoms_list:
            return []

        # Use max atom count in batch for smarter memory caching (worst-case estimate)
        max_atoms_in_batch = max(len(atoms) for atoms in atoms_list)

        # Try to apply cached memory scaler to avoid expensive re-probing (~70s per new cluster size)
        if self.max_memory_scaler is None and "autobatcher" in self._runner_kwargs:
            cached_scaler = _GLOBAL_MEMORY_SCALER_CACHE.get(
                n_atoms=max_atoms_in_batch,
                model_name=self._cache_model_name(),
                memory_scales_with=self.memory_scales_with,
                device=self._device_str,
            )
            if cached_scaler is not None:
                self._runner_kwargs["autobatcher"].max_memory_scaler = cached_scaler

        runner_kwargs = self._runner_kwargs.copy()
        if steps is not None:
            runner_kwargs["max_steps"] = steps

        atoms_seq = list(atoms_list)

        # torch_sim.io.atoms_to_state ignores ASE constraints; map FixAtoms -> TorchSim.
        ts_fix = build_torchsim_fixatoms_from_ase_batch(atoms_seq, self.device)
        if ts_fix is not None:
            system_in = self._ts.initialize_state(
                atoms_seq,
                self.device,
                self.dtype,
            )
            system_in.constraints = ts_fix
        else:
            system_in = atoms_seq

        # `steps=0` is intentionally used as a single-point mode in TS/NEB paths
        # (endpoint energies and batched force evaluations). Some torch-sim versions
        # emit a generic warning about max steps reached in this mode; suppress only
        # that specific, non-actionable warning.
        max_steps_now = runner_kwargs.get("max_steps", self.max_steps)
        if max_steps_now == 0:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"All systems have reached the maximum number of steps: 0\.",
                    category=UserWarning,
                )
                state = self._ts.optimize(  # type: ignore[call-arg]
                    system=system_in,
                    model=self.model,
                    optimizer=self.optimizer,
                    **runner_kwargs,
                )
        else:
            state = self._ts.optimize(  # type: ignore[call-arg]
                system=system_in,
                model=self.model,
                optimizer=self.optimizer,
                **runner_kwargs,
            )

        # Cache the memory scaler if we computed a new estimate (avoid ~70s re-probing)
        if self.max_memory_scaler is None and "autobatcher" in self._runner_kwargs:
            autobatcher = self._runner_kwargs["autobatcher"]
            if (
                hasattr(autobatcher, "max_memory_scaler")
                and autobatcher.max_memory_scaler
            ):
                _GLOBAL_MEMORY_SCALER_CACHE.set(
                    n_atoms=max_atoms_in_batch,
                    model_name=self._cache_model_name(),
                    memory_scales_with=self.memory_scales_with,
                    device=self._device_str,
                    value=float(autobatcher.max_memory_scaler),
                )

        energies_tensor = getattr(state, "energy", None)
        if energies_tensor is None:
            raise RuntimeError("TorchSim optimize did not return energy information")

        energies = [float(val) for val in energies_tensor.detach().cpu().tolist()]

        forces_tensor = getattr(state, "forces", None)
        forces_list = None
        if forces_tensor is not None:
            forces_np = forces_tensor.detach().cpu().numpy()  # Shape: (total_atoms, 3)

        relaxed_atoms = state.to_atoms()
        if len(relaxed_atoms) != len(energies):
            raise RuntimeError(
                "TorchSim returned mismatched counts for atoms and energies",
            )

        # Split forces by number of atoms per structure
        if forces_tensor is not None:
            forces_list = []
            offset = 0
            for atoms in relaxed_atoms:
                n_atoms = len(atoms)
                struct_forces = forces_np[
                    offset : offset + n_atoms
                ]  # Shape: (n_atoms, 3)
                forces_list.append(struct_forces)
                offset += n_atoms

            if offset != forces_np.shape[0]:
                raise RuntimeError(
                    f"Forces shape mismatch: expected {offset} total atoms, "
                    f"got {forces_np.shape[0]} forces"
                )

        results: list[tuple[float, Atoms]] = []
        for idx, (energy, relaxed) in enumerate(
            zip(energies, relaxed_atoms, strict=True)
        ):
            if forces_list is not None:
                relaxed.arrays["forces"] = np.asarray(
                    forces_list[idx], dtype=np.float64
                )
            elif "forces" in relaxed.arrays or relaxed.calc is not None:
                ensure_float64_forces(relaxed)

            relaxed.info.setdefault("key_value_pairs", {})

            update_metadata(
                relaxed,
                potential_energy=energy,
                raw_score=-energy,
            )
            results.append((energy, relaxed))
        return results

    def __deepcopy__(self, memo):  # pragma: no cover - deepcopy helper
        memo[id(self)] = self
        return self

    def _cache_model_name(self) -> str:
        mk = str(self.model_kind or "mace").strip().lower()
        if mk == "mace":
            return str(self.mace_model_name)
        if mk in ("fairchem", "uma"):
            return str(self.fairchem_model_name or "fairchem")
        return str(mk)

    def _patch_model_for_cuda(self) -> None:
        """Ensure TorchSim models handle CUDA atomic numbers safely."""
        setup_fn = getattr(self.model, "setup_from_system_idx", None)
        if setup_fn is None:
            return

        if getattr(setup_fn, "__wrapped_scgo__", False):  # already patched
            return

        @functools.wraps(setup_fn)
        def patched_setup(atomic_numbers, system_idx):
            original_device = None
            if hasattr(atomic_numbers, "is_cuda") and atomic_numbers.is_cuda:
                original_device = atomic_numbers.device
                atomic_numbers = atomic_numbers.cpu()
            result = setup_fn(atomic_numbers, system_idx)
            if original_device is not None and hasattr(self.model, "atomic_numbers"):
                # Restore tensors expected on the model device after CPU work.
                self.model.atomic_numbers = self.model.atomic_numbers.to(
                    original_device,
                )
            return result

        patched_setup.__wrapped_scgo__ = True  # type: ignore[attr-defined]
        self.model.setup_from_system_idx = patched_setup  # type: ignore[assignment]


def get_global_memory_scaler_cache() -> MemoryScalerCache:
    """Return the process-wide :class:`MemoryScalerCache` used by default."""
    return _GLOBAL_MEMORY_SCALER_CACHE
