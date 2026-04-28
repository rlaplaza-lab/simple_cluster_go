"""Strategy allocation logic for cluster initialization.

This module determines how to distribute requested structure counts across
different initialization strategies (templates, seeds, random placement).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.utils.logging import get_logger

from .initialization_config import (
    SEED_BASE_PCT,
    SEED_PREFACTOR,
    TEMPLATE_BASE_PCT,
    TEMPLATE_PREFACTOR,
)

logger = get_logger(__name__)


def _calculate_target_allocations(
    n_templates: int, n_seed_combinations: int, n_structures: int
) -> dict[str, int]:
    """Calculate target counts for each strategy based on logarithmic scaling."""
    targets = {"template": 0, "seed": 0}

    if n_templates > 0:
        template_scaling = TEMPLATE_BASE_PCT * np.log(
            1 + n_templates * TEMPLATE_PREFACTOR
        )
        target_template_raw = int(n_structures * template_scaling)
        targets["template"] = min(
            target_template_raw,
            2 * n_templates,  # Cap at 2 per template
            n_structures,
        )

    if n_seed_combinations > 0:
        seed_scaling = SEED_BASE_PCT * np.log(1 + n_seed_combinations * SEED_PREFACTOR)
        target_seed_raw = int(n_structures * seed_scaling)
        targets["seed"] = min(
            target_seed_raw,
            2 * n_seed_combinations,  # Cap at 2 per combination
            n_structures,
        )

    return targets


def _distribute_remaining(
    targets: dict[str, int],
    remaining: int,
    n_templates: int,
    n_seed_combinations: int,
) -> dict[str, int]:
    """Distribute remaining slots to templates and seeds up to caps."""
    if remaining <= 0:
        return targets

    # Prefer filling up to caps
    if n_templates > 0:
        template_cap = 2 * n_templates
        if targets["template"] < template_cap:
            add = min(remaining, template_cap - targets["template"])
            targets["template"] += add
            remaining -= add

    if remaining > 0 and n_seed_combinations > 0:
        seed_cap = 2 * n_seed_combinations
        if targets["seed"] < seed_cap:
            add = min(remaining, seed_cap - targets["seed"])
            targets["seed"] += add
            remaining -= add

    return targets


def _apply_guarantees(
    targets: dict[str, int],
    n_templates: int,
    n_seed_combinations: int,
    n_structures: int,
) -> dict[str, int]:
    """Apply minimum guarantees when structures >= options."""
    min_template = 0
    min_seed = 0

    if n_structures >= n_templates + n_seed_combinations:
        min_template = n_templates
        min_seed = n_seed_combinations
        targets["template"] = max(targets["template"], min_template)
        targets["seed"] = max(targets["seed"], min_seed)

    total_requested = targets["template"] + targets["seed"]

    if total_requested > n_structures:
        # Scale down if requested more than available
        if min_template > 0 or min_seed > 0:
            guaranteed = min_template + min_seed
            if guaranteed <= n_structures:
                excess_template = targets["template"] - min_template
                excess_seed = targets["seed"] - min_seed
                excess_total = excess_template + excess_seed

                if excess_total > 0:
                    available = n_structures - guaranteed
                    scale = available / excess_total
                    targets["template"] = min_template + int(excess_template * scale)
                    targets["seed"] = min_seed + int(excess_seed * scale)
                else:
                    targets["template"] = min_template
                    targets["seed"] = min_seed
            else:
                scale = n_structures / total_requested
                targets["template"] = int(targets["template"] * scale)
                targets["seed"] = int(targets["seed"] * scale)
        else:
            scale = n_structures / total_requested
            targets["template"] = int(targets["template"] * scale)
            targets["seed"] = int(targets["seed"] * scale)

        # If we have space left due to rounding, distribute it
        current_total = targets["template"] + targets["seed"]
        if current_total < n_structures:
            targets = _distribute_remaining(
                targets, n_structures - current_total, n_templates, n_seed_combinations
            )

    return targets


def _generate_allocations_list(
    targets: dict[str, int],
    n_structures: int,
    templates: list[Atoms],
    n_seed_combinations: int,
    rng: np.random.Generator,
) -> list[tuple[str, int | None]]:
    """Generate the list of allocation tuples from target counts."""
    allocations: list[tuple[str, int | None]] = []
    n_templates = len(templates)
    template_usage_count = [0] * n_templates

    # 1. Template allocations
    if n_templates > 0:
        if n_structures >= n_templates:
            indices = list(range(n_templates))
            rng.shuffle(indices)
            for idx in indices:
                allocations.append(("template", idx))
                template_usage_count[idx] += 1

        current_count = len(allocations)
        needed = targets["template"] - current_count

        if needed > 0:
            for _ in range(needed):
                weights = [1.0 / (1 + c) for c in template_usage_count]
                probs = np.array(weights) / sum(weights)
                idx = rng.choice(n_templates, p=probs)
                allocations.append(("template", idx))
                template_usage_count[idx] += 1

    # 2. Seed allocations
    remaining = n_structures - len(allocations)
    seed_count = min(targets["seed"], remaining)

    if n_seed_combinations > 0 and n_structures >= n_templates + n_seed_combinations:
        seed_count = max(seed_count, n_seed_combinations)
        seed_count = min(seed_count, remaining)

    allocations.extend([("seed+growth", None)] * seed_count)

    # 3. Random allocations
    remaining = n_structures - len(allocations)
    allocations.extend([("random_spherical", None)] * remaining)

    return allocations


def _allocate_strategies_metropolis(
    n_structures: int,
    templates: list[Atoms],
    n_seed_formulas: int,
    n_seed_combinations: int,
    rng: np.random.Generator,
    n_atoms: int = 0,
) -> list[tuple[str, int | None]]:
    """Allocate structures across strategies using logarithmic scaling with caps."""
    n_templates = len(templates)

    # 1. Calculate initial targets
    targets = _calculate_target_allocations(
        n_templates, n_seed_combinations, n_structures
    )

    # 2. Apply guarantees and scaling
    targets = _apply_guarantees(targets, n_templates, n_seed_combinations, n_structures)

    # 3. Generate actual allocations list
    allocations = _generate_allocations_list(
        targets, n_structures, templates, n_seed_combinations, rng
    )

    # Logging
    template_count = sum(1 for s, _ in allocations if s == "template")
    seed_count = sum(1 for s, _ in allocations if s == "seed+growth")
    random_count = sum(1 for s, _ in allocations if s == "random_spherical")

    if n_structures > 1:
        logger.info(
            f"Initialization for {n_atoms}-atom clusters: "
            f"{n_templates} template(s), {n_seed_formulas} seed formula(s), "
            f"{n_seed_combinations} seed combination(s) available"
        )
        logger.info(
            f"Strategy allocation ({len(allocations)} structures): "
            f"{template_count} template, {seed_count} seed+growth, {random_count} random"
        )

    return allocations
