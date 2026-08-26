r"""Structure tags on ASE Atoms (single bag: ``key_value_pairs``).

All structure annotations go through this API. ASE persists
``atoms.info['key_value_pairs']`` into SQLite ``systems.key_value_pairs``.

Encoding (ASE requires scalars only):

- ``bool`` / ``int`` / ``float`` (incl. numpy scalars) stored as-is
- Everything else stored as ``j:`` + ``json.dumps(value)`` when the raw value
  is rejected by ASE or is a non-scalar (lists, dicts, ambiguous strings like
  ``\"0_1\"``)

:func:`get_tag` / :func:`get_tags` reverse ``j:`` encoding only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

import numpy as np
from ase import Atoms
from ase.db.core import check as ase_check_kvp

from scgo.utils.logging import get_logger

logger = get_logger(__name__)

_TAGS_KEY = "key_value_pairs"
_JSON_PREFIX = "j:"


def _tags_dict(atoms: Atoms) -> dict[str, Any]:
    tags = atoms.info.get(_TAGS_KEY)
    if not isinstance(tags, dict):
        tags = {}
        atoms.info[_TAGS_KEY] = tags
    return tags


def _coerce_numpy_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python types ASE/JSON accept."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _encode_tag_value(key: str, value: Any) -> Any:
    """Coerce *value* to an ASE ``key_value_pairs``-compatible scalar."""
    value = _coerce_numpy_scalar(value)

    if isinstance(value, (bool, int, float)):
        ase_check_kvp({key: value})
        return value

    if isinstance(value, str):
        try:
            ase_check_kvp({key: value})
            return value
        except ValueError:
            wrapped = f"{_JSON_PREFIX}{json.dumps(value)}"
            ase_check_kvp({key: wrapped})
            return wrapped

    wrapped = f"{_JSON_PREFIX}{json.dumps(value)}"
    ase_check_kvp({key: wrapped})
    return wrapped


def _decode_tag_value(value: Any) -> Any:
    """Reverse :func:`_encode_tag_value` for values read back from tags."""
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        try:
            return json.loads(value[len(_JSON_PREFIX) :])
        except (json.JSONDecodeError, TypeError, ValueError):
            return value
    return value


def set_tags(atoms: Atoms, **tags: Any) -> None:
    """Write structure tags into ``atoms.info['key_value_pairs']``.

    Values are coerced to ASE-compatible scalars (see module docstring).
    Keys whose value is ``None`` are skipped (not stored or cleared).
    """
    bag = _tags_dict(atoms)
    for key, value in tags.items():
        if value is None:
            continue
        bag[key] = _encode_tag_value(key, value)

    logger.trace("Set tags on atoms: %s", list(tags.keys()))


def get_tag(atoms: Atoms, key: str, default: Any = None) -> Any:
    """Return one structure tag, or *default* if missing."""
    tags = atoms.info.get(_TAGS_KEY)
    if isinstance(tags, dict) and key in tags:
        return _decode_tag_value(tags[key])
    return default


def get_tags(atoms: Atoms) -> dict[str, Any]:
    """Return a shallow copy of all structure tags (decoded)."""
    tags = atoms.info.get(_TAGS_KEY)
    if isinstance(tags, dict):
        return {k: _decode_tag_value(v) for k, v in tags.items()}
    return {}


def filter_by_tags(
    structures: list[Atoms],
    **filters: Any,
) -> list[Atoms]:
    """Return structures whose tags match all provided filters."""
    return [
        atoms
        for atoms in structures
        if all(get_tag(atoms, key) == value for key, value in filters.items())
    ]


def compute_final_id(atoms: Atoms, energy: float | None) -> str:
    """Compute a deterministic identifier for a final structure.

    SHA256 over the centered copy's chemical symbols, positions rounded to 8
    decimals, and the energy when one is given.
    """
    a = atoms.copy()
    with contextlib.suppress(AttributeError, TypeError):
        a.center()

    symbols = a.get_chemical_symbols()
    pos = a.get_positions()
    pos_rounded = [[f"{x:.8f}" for x in triple] for triple in pos]

    parts = ["|".join(symbols)] + [";".join(p) for p in pos_rounded]
    if energy is not None:
        parts.append(f"E={energy:.12e}")
    payload = "::".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_final_id(atoms: Atoms, energy: float | None = None) -> str:
    """Return stable ``final_id``, assigning the tag when missing.

    Relaxed DB rows and final-minima tagging use the same identifier so
    :func:`scgo.metadata.persist.mark_final_minima_in_db` can match by stored
    ``final_id``.
    """
    existing = get_tag(atoms, "final_id")
    if existing:
        return str(existing)
    if energy is None:
        raw = get_tag(atoms, "raw_score")
        if raw is not None:
            energy = -float(raw)
        else:
            try:
                energy = atoms.get_potential_energy()
            except (RuntimeError, AttributeError):
                energy = None
    final_id = compute_final_id(atoms, energy)
    set_tags(atoms, final_id=final_id)
    return final_id
