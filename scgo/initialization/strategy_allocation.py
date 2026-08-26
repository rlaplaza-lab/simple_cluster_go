"""Strategy allocation logic for cluster initialization.

This module determines how to distribute requested structure counts across
different initialization strategies (templates, seeds, random placement).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.utils.logging import get_logger

from .initialization_config import (
    EXACT_BASE_PCT,
    EXACT_PREFACTOR,
    SEED_BASE_PCT,
    SEED_PREFACTOR,
    TEMPLATE_BASE_PCT,
    TEMPLATE_PREFACTOR,
)

logger = get_logger(__name__)


def _calculate_target_allocations(
    n_templates: int,
    n_seed_combinations: int,
    n_structures: int,
    n_exact: int = 0,
) -> dict[str, int]:
    """Calculate target counts for each strategy based on logarithmic scaling."""
    targets = {"template": 0, "seed": 0, "exact": 0}

    if n_templates > 0:
        template_scaling = TEMPLATE_BASE_PCT * np.log(
            1 + n_templates * TEMPLATE_PREFACTOR
        )
        target_template_raw = int(n_structures * template_scaling)
        targets["template"] = min(
            target_template_raw,
            2 * n_templates,  # Cap at 2 per template
            n_structures,
            n_templates,  # Never ask for more template slots than templates exist
        )

    if n_seed_combinations > 0:
        seed_scaling = SEED_BASE_PCT * np.log(1 + n_seed_combinations * SEED_PREFACTOR)
        target_seed_raw = int(n_structures * seed_scaling)
        targets["seed"] = min(
            target_seed_raw,
            2 * n_seed_combinations,  # Cap at 2 per combination
            n_structures,
        )

    if n_exact > 0:
        exact_scaling = EXACT_BASE_PCT * np.log(1 + n_exact * EXACT_PREFACTOR)
        target_exact_raw = int(n_structures * exact_scaling)
        targets["exact"] = min(
            target_exact_raw,
            2 * n_exact,  # Cap at 2 per exact candidate
            n_structures,
        )

    return targets


def _distribute_remaining(
    targets: dict[str, int],
    remaining: int,
    n_templates: int,
    n_seed_combinations: int,
    n_exact: int = 0,
) -> dict[str, int]:
    """Distribute remaining slots to templates, seeds and exact up to caps."""
    if remaining <= 0:
        return targets

    # Prefer filling up to caps (templates first, then seeds, then exact)
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

    if remaining > 0 and n_exact > 0:
        exact_cap = 2 * n_exact
        if targets["exact"] < exact_cap:
            add = min(remaining, exact_cap - targets["exact"])
            targets["exact"] += add
            remaining -= add

    return targets


def _apply_guarantees(
    targets: dict[str, int],
    n_templates: int,
    n_seed_combinations: int,
    n_structures: int,
    n_exact: int = 0,
) -> dict[str, int]:
    """Apply minimum guarantees when structures >= options."""
    min_template = 0
    min_seed = 0
    min_exact = 0

    if n_structures >= n_templates + n_seed_combinations + n_exact:
        min_template = n_templates
        min_seed = n_seed_combinations
        min_exact = n_exact
        targets["template"] = max(targets["template"], min_template)
        targets["seed"] = max(targets["seed"], min_seed)
        targets["exact"] = max(targets["exact"], min_exact)

    total_requested = targets["template"] + targets["seed"] + targets["exact"]

    if total_requested > n_structures:
        guaranteed = min_template + min_seed + min_exact
        if guaranteed > 0 and guaranteed <= n_structures:
            excess_template = targets["template"] - min_template
            excess_seed = targets["seed"] - min_seed
            excess_exact = targets["exact"] - min_exact
            excess_total = excess_template + excess_seed + excess_exact
            if excess_total > 0:
                available = n_structures - guaranteed
                scale = available / excess_total
                targets["template"] = min_template + int(excess_template * scale)
                targets["seed"] = min_seed + int(excess_seed * scale)
                targets["exact"] = min_exact + int(excess_exact * scale)

        # Fallback scale-down when there were no (or too many) guaranteed slots.
        if targets["template"] + targets["seed"] + targets["exact"] > n_structures:
            scale = n_structures / total_requested
            targets["template"] = int(targets["template"] * scale)
            targets["seed"] = int(targets["seed"] * scale)
            targets["exact"] = int(targets["exact"] * scale)

        # If we have space left due to rounding, distribute it
        current_total = targets["template"] + targets["seed"] + targets["exact"]
        if current_total < n_structures:
            targets = _distribute_remaining(
                targets,
                n_structures - current_total,
                n_templates,
                n_seed_combinations,
                n_exact,
            )

    return targets


def _generate_allocations_list(
    targets: dict[str, int],
    n_structures: int,
    templates: list[Atoms],
    n_seed_combinations: int,
    rng: np.random.Generator,
    n_exact: int = 0,
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

    # 3. Exact-match allocations (bounded enrichment tier)
    remaining = n_structures - len(allocations)
    exact_count = min(targets["exact"], remaining)
    allocations.extend([("exact", None)] * exact_count)

    # 4. Random allocations
    remaining = n_structures - len(allocations)
    allocations.extend([("random_spherical", None)] * remaining)

    return allocations


def _allocate_initialization_strategies(
    n_structures: int,
    templates: list[Atoms],
    n_seed_formulas: int,
    n_seed_combinations: int,
    rng: np.random.Generator,
    n_atoms: int = 0,
    n_exact: int = 0,
) -> list[tuple[str, int | None]]:
    """Allocate structures across strategies using logarithmic scaling with caps."""
    n_templates = len(templates)

    # 1. Calculate initial targets
    targets = _calculate_target_allocations(
        n_templates, n_seed_combinations, n_structures, n_exact
    )

    # 2. Apply guarantees and scaling
    targets = _apply_guarantees(
        targets, n_templates, n_seed_combinations, n_structures, n_exact
    )

    # 3. Generate actual allocations list
    allocations = _generate_allocations_list(
        targets, n_structures, templates, n_seed_combinations, rng, n_exact
    )

    # Logging (include single-structure runs for operational visibility)
    template_count = sum(1 for s, _ in allocations if s == "template")
    seed_count = sum(1 for s, _ in allocations if s == "seed+growth")
    exact_count = sum(1 for s, _ in allocations if s == "exact")
    random_count = sum(1 for s, _ in allocations if s == "random_spherical")

    logger.info(
        "Initialization for %d-atom clusters: %d template(s), %d seed formula(s), "
        "%d seed combination(s), %d exact match(es) available",
        n_atoms,
        n_templates,
        n_seed_formulas,
        n_seed_combinations,
        n_exact,
    )
    logger.info(
        "Strategy allocation (%d structure(s)): %d template, %d seed+growth, "
        "%d exact, %d random",
        len(allocations),
        template_count,
        seed_count,
        exact_count,
        random_count,
    )

    return allocations
