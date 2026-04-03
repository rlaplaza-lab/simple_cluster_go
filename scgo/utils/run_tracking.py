"""Run tracking and metadata management for SCGO campaigns.

This module provides functions for generating run IDs, saving and loading
run metadata, and managing run-specific directory structures.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from scgo.utils.helpers import ensure_directory_exists, get_cluster_formula
from scgo.utils.logging import get_logger
from scgo.utils.ts_provenance import ts_output_provenance


class RunMetadataJSONEncoder(json.JSONEncoder):
    """JSON encoder: ``type`` objects become their ``__name__`` (for params snapshots)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)


@dataclass
class RunMetadata:
    """Metadata for a single run."""

    run_id: str
    timestamp: str
    composition: list[str] | None = None
    formula: str | None = None  # Chemical formula for quick filtering
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        """Create metadata from dictionary (ignores TS-parity provenance keys)."""
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            composition=data.get("composition"),
            formula=data.get("formula"),
            params=data.get("params"),
        )


def generate_run_id() -> str:
    """Generate timestamp-based run ID with microsecond granularity.

    Returns:
        Run ID in format: run_YYYYMMDD_HHMMSS_ffffff
        Example: "run_20250124_143022_123456"
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.microsecond
    return f"run_{timestamp}_{microseconds:06d}"


def ensure_run_id(run_id: str | None, verbosity: int = 0, logger=None) -> str:
    """Ensure a run_id exists, generating one if needed and logging if appropriate.

    This helper consolidates the common pattern of generating a run_id if None
    and logging it when verbosity is sufficient.

    Args:
        run_id: Existing run_id or None to generate a new one.
        verbosity: Logging verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        logger: Optional logger instance. If None, will create one if needed.

    Returns:
        Run ID (existing or newly generated).
    """
    if run_id is None:
        run_id = generate_run_id()
        if verbosity >= 1:
            if logger is None:
                logger = get_logger(__name__)
            logger.info(f"Generated run ID: {run_id}")
    return run_id


def save_run_metadata(
    run_dir: str,
    run_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save run metadata to metadata.json file.

    Args:
        run_dir: Directory where metadata file will be saved.
        run_id: Run ID for this run.
        metadata: Optional dictionary of additional metadata to store.
            Common keys: composition, params, etc.
    """
    ensure_directory_exists(run_dir)

    # Extract composition and compute formula if available
    composition = metadata.get("composition") if metadata else None
    formula = metadata.get("formula") if metadata else None
    if composition and not formula:
        formula = get_cluster_formula(composition)

    metadata_obj = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        composition=composition,
        formula=formula,
        params=metadata.get("params") if metadata else None,
    )

    payload = {**ts_output_provenance(), **metadata_obj.to_dict()}

    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(payload, f, indent=2, cls=RunMetadataJSONEncoder)


def load_run_metadata(run_dir: str) -> RunMetadata | None:
    """Load run metadata from metadata.json file.

    Args:
        run_dir: Directory containing metadata.json file.

    Returns:
        RunMetadata object if metadata file exists and is valid, None otherwise.
    """
    metadata_file = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file) as f:
            data = json.load(f)
        return RunMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        return None


def get_run_directories(base_output_dir: str) -> list[str]:
    """Get list of all run directories in base output directory.

    Args:
        base_output_dir: Base directory to search for run_* subdirectories.

    Returns:
        List of run directory paths (full paths), sorted by name.
    """
    if not os.path.exists(base_output_dir):
        return []

    run_dirs = [
        os.path.join(base_output_dir, item)
        for item in os.listdir(base_output_dir)
        if (
            item.startswith("run_")
            and os.path.isdir(os.path.join(base_output_dir, item))
            and get_run_id_from_dir(item) is not None
        )
    ]

    return sorted(run_dirs)


def get_run_id_from_dir(run_dir: str) -> str | None:
    """Extract run ID from directory name.

    Args:
        run_dir: Directory path (may be full path or just name).

    Returns:
        Run ID if directory name matches pattern, None otherwise.
    """
    dir_name = os.path.basename(run_dir)
    # Format: run_YYYYMMDD_HHMMSS_ffffff (26 characters: 4 + 8 + 1 + 6 + 1 + 6)
    if dir_name.startswith("run_") and len(dir_name) == 26:
        # Additional validation: should match pattern run_YYYYMMDD_HHMMSS_ffffff
        parts = dir_name.split("_")
        if (
            len(parts) == 4
            and len(parts[1]) == 8
            and len(parts[2]) == 6
            and len(parts[3]) == 6
        ):
            return dir_name
    return None
