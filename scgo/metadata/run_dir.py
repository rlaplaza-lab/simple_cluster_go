"""Run-directory record (``metadata.json``) and run-id helpers."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scgo.exceptions import SCGOFileError
from scgo.metadata.provenance import output_json_provenance
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger, log_info_v

logger = get_logger(__name__)


class RunDirJSONEncoder(json.JSONEncoder):
    """JSON encoder: ``type`` objects become their ``__name__`` (for params snapshots).

    NumPy scalars (``np.int64``, ``np.float64``, ...) are converted to their
    native Python equivalents so params snapshots stay serializable. Dataclass
    instances (e.g. :class:`~scgo.system_types.AdsorbateDefinition`) are expanded
    to plain dicts so they survive params archival.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, type):
            return obj.__name__
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


@dataclass
class RunDirRecord:
    """Per-run params/composition snapshot written to ``metadata.json``.

    ``path_key`` is the component-aware directory identity (see
    :func:`scgo.utils.path_keys.resolve_run_path_key`). The single run timestamp
    is the provenance header's ``created_at``; this record carries no timestamp.
    """

    run_id: str
    path_key: str | None = None
    composition: list[str] | None = None
    formula: str | None = None
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDirRecord:
        """Create record from a ``metadata.json`` payload.

        Raises:
            KeyError: when ``run_id`` or ``path_key`` is missing.
            TypeError: when ``data`` is not a mapping.
        """
        return cls(
            run_id=data["run_id"],
            path_key=data["path_key"],
            composition=data.get("composition"),
            formula=data.get("formula"),
            params=data.get("params"),
        )


def generate_run_id() -> str:
    """Generate timestamp-based run ID with microsecond granularity.

    Returns:
        Run ID in format: run_YYYYMMDD_HHMMSS_ffffff
    """
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.microsecond
    return f"run_{timestamp}_{microseconds:06d}"


def ensure_run_id(run_id: str | None, verbosity: int = 0, logger=None) -> str:
    """Ensure a run_id exists, generating one if needed and logging if appropriate."""
    if run_id is None:
        run_id = generate_run_id()
        if logger is None:
            logger = get_logger(__name__)
        log_info_v(logger, "Generated run ID: %s", run_id, verbosity=verbosity)
    return run_id


def save_run_dir_record(
    run_dir: str,
    run_id: str,
    record: dict[str, Any] | None = None,
) -> None:
    """Save run directory record to ``metadata.json``.

    ``record`` should carry ``path_key`` (the component-aware directory
    identity); it is required for the record to be readable back via
    :func:`load_run_dir_record`.
    """
    os.makedirs(run_dir, exist_ok=True)

    composition = record.get("composition") if record else None
    formula = record.get("formula") if record else None
    if composition and not formula:
        formula = get_cluster_formula(composition)

    record_obj = RunDirRecord(
        run_id=run_id,
        path_key=record.get("path_key") if record else None,
        composition=composition,
        formula=formula,
        params=record.get("params") if record else None,
    )

    payload = {**output_json_provenance(), **record_obj.to_dict()}

    metadata_file = os.path.join(run_dir, "metadata.json")
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_run_",
        suffix=".json",
        dir=run_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, cls=RunDirJSONEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, metadata_file)
    except OSError as e:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise SCGOFileError(f"Failed to write {metadata_file}: {e}") from e
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def load_run_dir_record(run_dir: str) -> RunDirRecord | None:
    """Load run directory record from ``metadata.json``.

    Returns ``None`` only when the file is missing or is not parseable JSON.
    Schema violations (missing ``run_id`` / ``path_key``) propagate from
    :meth:`RunDirRecord.from_dict`.
    """
    metadata_file = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse run dir record %s: %s", metadata_file, e)
        return None
    return RunDirRecord.from_dict(data)


def get_run_directories(base_output_dir: str) -> list[str]:
    """Get sorted paths of canonically named ``run_*`` directories.

    Directories whose name does not parse as a run ID (see
    :func:`get_run_id_from_dir`) are ignored.
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


def resolve_run_id_from_db_path(
    db_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> str | None:
    """Resolve GO run ID from a database path (``run_*`` segment when present).

    Returns ``None`` (with a warning) when no ``run_*`` segment is found, so
    callers can skip the database instead of poisoning provenance with the
    filename basename.
    """
    db_path_str = os.path.abspath(str(db_path))
    if base_dir is not None:
        base_s = os.path.abspath(str(base_dir))
        try:
            rel = os.path.relpath(db_path_str, base_s)
            parts = rel.split(os.sep)
        except ValueError:
            parts = Path(db_path_str).parts
    else:
        parts = Path(db_path_str).parts

    for part in parts:
        resolved = get_run_id_from_dir(part)
        if resolved is not None:
            return resolved
        if part.startswith("run_"):
            return part

    parent_name = Path(db_path_str).parent.name
    resolved = get_run_id_from_dir(parent_name)
    if resolved is not None:
        return resolved
    if parent_name.startswith("run_"):
        return parent_name

    logger.warning(
        "Could not resolve run_id from path %s; leaving run_id unset",
        db_path,
    )
    return None


def get_run_id_from_dir(run_dir: str) -> str | None:
    """Extract run ID from directory name, or None if it is not canonical."""
    dir_name = os.path.basename(run_dir)
    if dir_name.startswith("run_") and len(dir_name) == 26:
        parts = dir_name.split("_")
        if (
            len(parts) == 4
            and len(parts[1]) == 8
            and len(parts[2]) == 6
            and len(parts[3]) == 6
        ):
            return dir_name
    return None
