#!/usr/bin/env python3
"""Microbenchmark: measure TorchSim memory probing overhead.

Tests whether InFlightAutoBatcher memory estimation is fast enough
to justify removing the disk-backed cache.
"""

import time

import torch
import torch_sim as ts
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel


def setup_model(device):
    """Load MACE model once."""
    print(f"Loading MACE model on {device}...", flush=True)
    raw_model = mace_mp(
        model="small",
        return_raw_model=True,
        default_dtype="float32",
        device=device,
    )
    model = MaceModel(
        model=raw_model,
        device=device,
        dtype=torch.float32,
        compute_forces=True,
        compute_stress=False,
    )
    return model


def create_pt8_batch(n_copies=1):
    """Create a batch of Pt8 clusters."""
    pt8 = bulk("Pt", "fcc", a=3.92, cubic=True)
    pt8 = pt8.repeat((2, 2, 2))[:8]  # Get exactly 8 atoms
    pt8.center(vacuum=3.0)
    return [pt8.copy() for _ in range(n_copies)]


def scenario_a_no_autobatcher(model, atoms_list):
    """Baseline: optimize without autobatcher."""
    print("\n=== Scenario A: No autobatcher (baseline) ===")

    t0 = time.time()
    ts.optimize(
        system=atoms_list,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=5,  # Just 5 steps
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    return elapsed


def scenario_b_autobatcher_none(model, atoms_list):
    """Test: autobatcher with max_memory_scaler=None (forces probing)."""
    print("\n=== Scenario B: Autobatcher with max_memory_scaler=None ===")

    # Create autobatcher with no memory limit → forces estimation
    ab = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with="n_atoms_x_density",
        max_memory_scaler=None,
    )

    t0 = time.time()
    ts.optimize(
        system=atoms_list,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=5,
        autobatcher=ab,
    )
    elapsed = time.time() - t0

    # Extract computed memory scaler
    max_scaler = getattr(ab, "max_memory_scaler", None)
    print(f"Time: {elapsed:.3f}s")
    print(f"Computed max_memory_scaler: {max_scaler}")
    return elapsed, max_scaler


def scenario_c_autobatcher_preset(model, atoms_list, max_scaler):
    """Test: autobatcher with pre-set max_memory_scaler (from cache)."""
    print("\n=== Scenario C: Autobatcher with max_memory_scaler preset ===")

    # Create autobatcher with pre-computed memory limit
    ab = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with="n_atoms_x_density",
        max_memory_scaler=max_scaler,
    )

    t0 = time.time()
    ts.optimize(
        system=atoms_list,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=5,
        autobatcher=ab,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    return elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TorchSim version: {ts.__version__}")

    if str(device) == "cpu":
        print("\n⚠️  GPU NOT AVAILABLE. Running on CPU (autobatcher disabled anyway).")
        print("Results will not be representative. Need GPU for valid test.")
        return

    # Setup
    model = setup_model(device)
    atoms_list = create_pt8_batch(n_copies=1)
    print(f"System: Pt8 ({len(atoms_list[0])} atoms)")
    print("Running 3 scenarios with 5 optimization steps each\n")

    # Run scenarios
    time_a = scenario_a_no_autobatcher(model, atoms_list)
    time_b, max_scaler = scenario_b_autobatcher_none(model, atoms_list)
    time_c = scenario_c_autobatcher_preset(model, atoms_list, max_scaler)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Scenario A (no autobatcher):           {time_a:.3f}s")
    print(f"Scenario B (autobatcher, None):        {time_b:.3f}s")
    print(f"Scenario C (autobatcher, preset):      {time_c:.3f}s")
    print(f"\nMemory probing overhead (B - C):       {time_b - time_c:.3f}s")
    print(f"Autobatcher overhead vs baseline (B - A): {time_b - time_a:.3f}s")

    # Decision
    overhead_ms = (time_b - time_c) * 1000
    print(f"\n{'=' * 60}")
    if overhead_ms < 50:
        print("✓ OVERHEAD IS SMALL (<50ms): Remove cache")
        print("  Memory probing is fast enough in TorchSim 0.5.2+")
        return_code = 0
    elif overhead_ms < 200:
        print("⚠ OVERHEAD IS MODERATE (50-200ms): Keep in-memory cache only")
        print("  Simplify by removing disk I/O, keep session cache")
        return_code = 1
    else:
        print("✗ OVERHEAD IS LARGE (>200ms): Keep full disk-backed cache")
        print("  Memory probing is expensive, caching is justified")
        return_code = 2
    print(f"{'=' * 60}")

    return return_code


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
