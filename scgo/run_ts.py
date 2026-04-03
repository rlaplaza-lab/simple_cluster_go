"""High-level transition state API exposed at top-level.

This module re-exports the transition state campaign and search
functions implemented in the internal `ts_search` package. Users
should import from `scgo.run_ts` or use the convenience exports
available in `scgo.__init__`.
"""

from __future__ import annotations

from scgo.ts_search import run_transition_state_campaign, run_transition_state_search

__all__ = ["run_transition_state_search", "run_transition_state_campaign"]
