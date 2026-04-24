"""Shared names for SCGO / ASE ``ase.db`` layout.

SCGO uses ASE's ``systems`` table; JSON for flags and GA fields lives in a
single column (not a separate ``metadata`` SQLite column on ``systems``).
"""

from __future__ import annotations

# Column on ``systems`` holding JSON (relaxed, raw_score, run_id, etc.)
SYSTEMS_JSON_COLUMN: str = "key_value_pairs"
