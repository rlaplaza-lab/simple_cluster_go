from __future__ import annotations

from typing import Any


def parse_energy_from_xyz_comment(comment: dict[str, Any]) -> float | None:
    """Parse energy value from an ASE Atoms.info dictionary.

    Extracts the energy value from the dictionary, which is conventionally stored
    as the last value.

    Args:
        comment: The atoms.info dictionary from an ASE Atoms object.

    Returns:
        The parsed energy as a float, or None if parsing fails.
    """
    try:
        return float(list(comment.values())[-1])
    except (ValueError, IndexError, TypeError, AttributeError):
        return None
