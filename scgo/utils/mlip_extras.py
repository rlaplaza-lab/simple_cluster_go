"""Optional MLIP install extras (MACE vs UMA) — detect conflicts."""

from __future__ import annotations

import importlib.util

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def installed_mace_and_uma() -> tuple[bool, bool]:
    """Return (mace_stack_present, fairchem_present) using importlib only."""
    mace = importlib.util.find_spec("mace") is not None
    uma = importlib.util.find_spec("fairchem") is not None
    return mace, uma


def ensure_mace_uma_not_both_installed() -> None:
    """Fail if both stacks are importable (unsupported mixed environment)."""
    mace, uma = installed_mace_and_uma()
    if mace and uma:
        msg = (
            "Both the MACE stack and fairchem-core are importable. "
            "Prefer a single extra: pip install 'scgo[mace]' or pip install 'scgo[uma]' "
            "in separate environments to avoid dependency conflicts."
        )
        logger.warning(msg)
        raise RuntimeError(msg)
