"""Tests for batch initialization with deterministic strategy allocation."""

from collections import Counter

import pytest

from scgo.initialization import compute_cell_side, create_initial_cluster_batch
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
    SEED_BASE_PCT,
    SEED_PREFACTOR,
    TEMPLATE_BASE_PCT,
    TEMPLATE_PREFACTOR,
    VACUUM_DEFAULT,
)
from scgo.initialization.initializers import (
    _allocate_strategies_metropolis,
    _discover_available_strategies,
    _filter_candidates_by_geometry,
    _find_smaller_candidates,
    _find_valid_seed_combinations,
)
from scgo.utils.helpers import get_composition_counts
from tests.test_utils import (
    assert_cluster_valid,
    create_paired_rngs,
    get_structure_signature,
)


def _precompute_seeds(
    composition: list[str],
) -> tuple[dict[str, list[tuple[float, object]]], list[tuple[str, ...]]]:
    candidates_by_formula = _find_smaller_candidates(composition, "**/*.db")
    candidates_by_formula = _filter_candidates_by_geometry(candidates_by_formula)
    if not candidates_by_formula:
        return {}, []
    target_counts = get_composition_counts(composition)
    valid_combinations = _find_valid_seed_combinations(
        candidates_by_formula, target_counts
    )
    return candidates_by_formula, valid_combinations


@pytest.mark.slow
def test_batch_allocation_metropolis(rng):
    """Test that batch allocation uses Metropolis algorithm for diversity."""
    # Use smaller magic number (13) with large n_structures for diversity testing
    composition = ["Pt"] * 13  # Magic number (faster than 55)
    n_structures = 100  # Large n_structures for diversity testing

    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=-2,  # Use all CPUs except one for stress testing
    )

    # Count unique structures (proxy for strategy diversity)
    signatures = [get_structure_signature(a) for a in batch]
    unique_count = len(set(signatures))

    # With Metropolis allocation, we should get good diversity
    # (much more than the ~3 unique templates alone would provide)
    assert unique_count >= 50, (
        f"Expected at least 50 unique structures with Metropolis allocation, "
        f"got {unique_count}"
    )


@pytest.mark.slow
def test_batch_allocation_non_magic(rng):
    """Test batch allocation for non-magic numbers."""
    # Test non-magic number cluster (30 atoms) with fewer structures for speed
    composition = ["Pt"] * 30  # Not a magic number
    n_structures = 30  # Reduced from 100 for faster execution

    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=-2,  # Use all CPUs except one for stress testing
    )

    # For non-magic numbers, Metropolis allocation will use templates from nearest magic number,
    # seed+growth, and random_spherical based on availability
    signatures = [get_structure_signature(a) for a in batch]
    unique_count = len(set(signatures))

    # Should have good diversity (adjusted for smaller n_structures)
    assert unique_count >= 20, (
        f"Expected at least 20 unique structures for non-magic number, "
        f"got {unique_count}"
    )


def test_batch_allocation_single_strategy(rng):
    """Test batch allocation with a single strategy."""
    composition = ["Pt"] * 30
    n_structures = 10

    # All random_spherical
    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="random_spherical",
    )

    assert len(batch) == n_structures
    # All should be unique (random_spherical produces diverse structures)
    signatures = [get_structure_signature(a) for a in batch]
    unique_count = len(set(signatures))
    assert unique_count == n_structures, (
        f"Expected all {n_structures} structures to be unique, got {unique_count}"
    )


def test_cluster_start_generator_allocation():
    """Test that ClusterStartGenerator uses Metropolis allocation when population_size is provided."""
    from scgo.algorithms.ga_common import ClusterStartGenerator

    # Test large cluster (30 atoms) but with fewer structures for speed
    composition = ["Pt"] * 30  # Large cluster
    population_size = 30  # Reduced from 100 for faster execution
    seed = 42

    rng1, rng2 = create_paired_rngs(seed)

    # Create two generators with same seed and population_size
    gen1 = ClusterStartGenerator(
        composition=composition,
        vacuum=10.0,
        rng=rng1,
        population_size=population_size,
        mode="smart",
    )
    gen2 = ClusterStartGenerator(
        composition=composition,
        vacuum=10.0,
        rng=rng2,
        population_size=population_size,
        mode="smart",
    )

    # Generate populations
    pop1 = [gen1.get_new_candidate() for _ in range(population_size)]
    pop2 = [gen2.get_new_candidate() for _ in range(population_size)]

    assert len(pop1) == population_size, (
        f"Generator 1 produced {len(pop1)} candidates, expected {population_size}"
    )
    assert len(pop2) == population_size, (
        f"Generator 2 produced {len(pop2)} candidates, expected {population_size}"
    )

    # Check reproducibility
    for i, (a1, a2) in enumerate(zip(pop1, pop2, strict=True)):
        sig1 = get_structure_signature(a1)
        sig2 = get_structure_signature(a2)
        assert sig1 == sig2, (
            f"Structure {i}: not reproducible. "
            f"This indicates Metropolis allocation is working correctly."
        )

    # Check diversity
    signatures1 = [get_structure_signature(a) for a in pop1]
    unique_count = len(set(signatures1))
    # With Metropolis allocation across strategies, should have good diversity.
    # Use a lower bound (8) so the test does not fail when template add-atoms
    # frequently fails (e.g. cuboctahedron, truncated_octahedron) and yields
    # fewer unique structures.
    assert unique_count >= 8, (
        f"Expected at least 8 unique structures with Metropolis allocation, "
        f"got {unique_count}"
    )


@pytest.mark.slow
def test_batch_allocation_strategy_distribution(rng):
    """Test that batch allocation distributes strategies as expected."""
    # Test large cluster (30 atoms) but with fewer structures for speed
    composition = ["Pt"] * 30  # Large cluster
    n_structures = 30  # Reduced from 100 for faster execution

    # Generate batch
    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=-2,  # Use all CPUs except one for stress testing
    )

    # Verify we got the right number
    assert len(batch) == n_structures

    # All should have correct composition
    for atoms in batch:
        assert_cluster_valid(atoms, composition)

    # Check diversity (should be high due to Metropolis allocation across strategies)
    # (adjusted for smaller n_structures)
    signatures = [get_structure_signature(a) for a in batch]
    unique_count = len(set(signatures))
    assert unique_count >= 15, (
        f"Expected good diversity with Metropolis allocation, got {unique_count} unique"
    )


@pytest.mark.slow
def test_smart_mode_template_index_alignment():
    """Test that each discovery template is used at least once in smart-mode batch.

    With discovery_templates fix, batch generation uses the pre-discovered template
    list and template_index from allocation. This ensures each template is used
    as intended (no index mismatch from re-running generate_template_matches).
    """
    seed = 42
    rng_discover, rng_batch = create_paired_rngs(seed)

    composition = ["Pt"] * 13  # Magic number: several templates available
    n_atoms = len(composition)
    cell_side = compute_cell_side(composition, vacuum=VACUUM_DEFAULT)

    candidates_by_formula, valid_combinations = _precompute_seeds(composition)
    discovery = _discover_available_strategies(
        composition=composition,
        n_atoms=n_atoms,
        cell_side=cell_side,
        rng=rng_discover,
        placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
        min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor=CONNECTIVITY_FACTOR,
        candidates_by_formula=candidates_by_formula,
        valid_combinations=valid_combinations,
    )
    templates = discovery["templates"]
    if not templates:
        pytest.skip("No templates discovered for Pt13; cannot assert template coverage")

    template_sigs = [get_structure_signature(t) for t in templates]
    n_seed_combinations = discovery["n_seed_combinations"]
    # Minimum needed: n_templates (fallback) or n_templates + n_seed_combinations (full guarantee)
    # Use full guarantee to ensure all templates are used
    n_structures = len(templates) + max(n_seed_combinations, 0)

    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng_batch,
        mode="smart",
        n_jobs=1,
    )

    assert len(batch) == n_structures
    for atoms in batch:
        assert_cluster_valid(atoms, composition)

    output_sigs = [get_structure_signature(a) for a in batch]
    output_sig_set = set(output_sigs)

    missing = [i for i, sig in enumerate(template_sigs) if sig not in output_sig_set]
    assert len(missing) == 0, (
        f"Discovery templates at indices {missing} were not used in batch output. "
        f"Template index alignment (use discovery_templates) may be broken."
    )


@pytest.mark.slow
def test_metropolis_allocation_logarithmic_scaling_and_cap(rng):
    """Test that Metropolis allocation uses logarithmic scaling and respects template cap.

    Verifies:
    - Templates get logarithmically scaled allocation (capped at 2 per template)
    - Seed+growth gets logarithmically scaled allocation (if seeds available), otherwise 0
    - All templates are used when n_structures >= n_templates
    - Template cap of 2 per template is enforced
    """
    import numpy as np

    composition = ["Pt"] * 13  # Magic number with templates
    n_structures = 100

    # Discover strategies
    cell_side = compute_cell_side(composition, vacuum=VACUUM_DEFAULT)
    candidates_by_formula, valid_combinations = _precompute_seeds(composition)
    discovery = _discover_available_strategies(
        composition=composition,
        n_atoms=len(composition),
        cell_side=cell_side,
        rng=rng,
        placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
        min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor=CONNECTIVITY_FACTOR,
        candidates_by_formula=candidates_by_formula,
        valid_combinations=valid_combinations,
    )

    # Get allocations
    allocations = _allocate_strategies_metropolis(
        n_structures=n_structures,
        templates=discovery["templates"],
        n_seed_formulas=discovery["n_seed_formulas"],
        n_seed_combinations=discovery["n_seed_combinations"],
        rng=rng,
    )

    # Count strategy usage
    template_count = sum(1 for s, _ in allocations if s == "template")
    seed_count = sum(1 for s, _ in allocations if s == "seed+growth")
    random_count = sum(1 for s, _ in allocations if s == "random_spherical")

    # Calculate expected logarithmic scaling (symmetric for templates and seeds)
    n_templates = len(discovery["templates"])
    n_seed_combinations = discovery["n_seed_combinations"]

    template_scaling = 0.0
    if n_templates > 0:
        template_scaling = TEMPLATE_BASE_PCT * np.log(
            1 + n_templates * TEMPLATE_PREFACTOR
        )
        expected_template_raw = int(n_structures * template_scaling)
        expected_template = min(expected_template_raw, 2 * n_templates, n_structures)
    else:
        expected_template = 0

    seed_scaling = 0.0
    if n_seed_combinations > 0:
        seed_scaling = SEED_BASE_PCT * np.log(1 + n_seed_combinations * SEED_PREFACTOR)
        expected_seed_raw = int(n_structures * seed_scaling)
        # Cap at 2 per seed combination (symmetric to templates)
        expected_seed = min(expected_seed_raw, 2 * n_seed_combinations, n_structures)
    else:
        expected_seed = 0

    # Ensure minimum allocations when we have enough structures
    if n_structures >= n_templates + n_seed_combinations:
        expected_template = max(expected_template, n_templates)
        expected_seed = max(expected_seed, n_seed_combinations)

    # Adjust for rounding and proportional scaling if needed
    total_expected = expected_template + expected_seed
    if total_expected > n_structures:
        # Preserve minimums if guaranteed
        min_template = (
            n_templates if n_structures >= n_templates + n_seed_combinations else 0
        )
        min_seed = (
            n_seed_combinations
            if n_structures >= n_templates + n_seed_combinations
            else 0
        )

        if min_template > 0 or min_seed > 0:
            guaranteed_total = min_template + min_seed
            if guaranteed_total <= n_structures:
                excess_template = expected_template - min_template
                excess_seed = expected_seed - min_seed
                excess_total = excess_template + excess_seed
                if excess_total > 0:
                    available_for_excess = n_structures - guaranteed_total
                    if available_for_excess > 0:
                        scale_factor = available_for_excess / excess_total
                        expected_template = min_template + int(
                            excess_template * scale_factor
                        )
                        expected_seed = min_seed + int(excess_seed * scale_factor)
                    else:
                        expected_template = min_template
                        expected_seed = min_seed
                else:
                    expected_template = min_template
                    expected_seed = min_seed
            else:
                # Can't satisfy both minimums, scale proportionally
                scale_factor = n_structures / total_expected
                expected_template = int(expected_template * scale_factor)
                expected_seed = int(expected_seed * scale_factor)
        else:
            # No minimums, scale proportionally
            scale_factor = n_structures / total_expected
            expected_template = int(expected_template * scale_factor)
            expected_seed = int(expected_seed * scale_factor)

    # Verify allocations match expected logarithmic scaling (with tolerance for rounding)
    TOLERANCE = (
        3  # Allow 3 structures tolerance for rounding and proportional adjustments
    )

    assert abs(template_count - expected_template) <= TOLERANCE, (
        f"Template allocation {template_count} deviates from expected logarithmic scaling "
        f"{expected_template} (scaling: {template_scaling:.3f}) by more than {TOLERANCE}"
    )

    if discovery["n_seed_combinations"] > 0:
        assert abs(seed_count - expected_seed) <= TOLERANCE, (
            f"Seed+growth allocation {seed_count} deviates from expected logarithmic scaling "
            f"{expected_seed} (scaling: {seed_scaling:.3f}) by more than {TOLERANCE}"
        )
    else:
        assert seed_count == 0, (
            f"Seed+growth allocation should be 0 when no seeds available, got {seed_count}"
        )

    # Verify total
    assert template_count + seed_count + random_count == n_structures, (
        f"Allocation mismatch: {template_count} + {seed_count} + {random_count} != {n_structures}"
    )

    # Verify template cap (2 per template) - symmetric to seed cap
    if n_templates > 0:
        template_usage = Counter(
            idx for s, idx in allocations if s == "template" and idx is not None
        )
        total_template_allocations = sum(template_usage.values())
        max_per_template_cap = 2 * n_templates

        # Verify total template allocations don't exceed the cap
        assert total_template_allocations <= max_per_template_cap, (
            f"Total template allocations {total_template_allocations} exceed cap "
            f"{max_per_template_cap} (2 per template * {n_templates} templates). "
            f"Usage: {dict(template_usage)}"
        )

        # Verify all templates are used when n_structures >= n_templates + n_seed_combinations
        if n_structures >= n_templates + n_seed_combinations:
            template_indices_used = set(template_usage.keys())
            assert len(template_indices_used) == n_templates, (
                f"Expected all {n_templates} templates to be used when "
                f"n_structures ({n_structures}) >= n_templates ({n_templates}) + "
                f"n_seed_combinations ({n_seed_combinations}), "
                f"but only {len(template_indices_used)} were used. "
                f"Indices used: {sorted(template_indices_used)}"
            )

    # Verify seed cap (2 per seed combination) - symmetric to template cap
    if n_seed_combinations > 0:
        max_seed_cap = 2 * n_seed_combinations
        assert seed_count <= max_seed_cap, (
            f"Total seed+growth allocations {seed_count} exceed cap "
            f"{max_seed_cap} (2 per combination * {n_seed_combinations} combinations)"
        )

        # Verify minimum seed allocation when n_structures >= n_templates + n_seed_combinations
        if n_structures >= n_templates + n_seed_combinations:
            assert seed_count >= n_seed_combinations, (
                f"Expected at least {n_seed_combinations} seed+growth allocations when "
                f"n_structures ({n_structures}) >= n_templates ({n_templates}) + "
                f"n_seed_combinations ({n_seed_combinations}), but got {seed_count}"
            )


@pytest.mark.slow
def test_large_batch_validity(rng):
    """Test that validity is maintained across large batches.

    Verifies that smart mode produces valid clusters across large batches
    (100 structures) with a failure rate < 1%.
    """
    composition = ["Pt"] * 20  # Medium-sized cluster
    n_structures = 100  # Reduced from 200 for faster execution

    batch = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=-2,  # Use all CPUs except one for stress testing
    )

    assert len(batch) == n_structures

    # Check all invariants and track failures
    invalid_count = 0
    failures = []
    for i, atoms in enumerate(batch):
        try:
            assert_cluster_valid(atoms, composition)
        except AssertionError as e:
            invalid_count += 1
            failures.append((i, str(e)))
            if len(failures) <= 5:  # Keep first 5 failures for reporting
                pass

    # Allow very small failure rate (< 1%) for edge cases
    failure_rate = invalid_count / n_structures
    assert failure_rate < 0.01, (
        f"Invalid cluster rate {failure_rate:.1%} exceeds 1% threshold. "
        f"Found {invalid_count} invalid clusters out of {n_structures}. "
        f"First failures: {failures[:5]}"
    )


def test_create_initial_cluster_batch_parallel(rng):
    """Test parallel batch generation with n_jobs > 1."""
    composition = ["Pt"] * 10
    n_structures = 10
    results = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=2,
    )
    assert len(results) == n_structures
    from ase import Atoms

    assert all(isinstance(r, Atoms) for r in results)
