"""Tests for database path pre-filtering during seed candidate discovery."""

from collections import Counter

from scgo.initialization.candidate_discovery import (
    _could_path_contain_relevant_candidates,
)
from scgo.utils.helpers import get_composition_counts


def test_rejects_path_when_parsed_count_exceeds_target():
    """Paths hinting at a larger stoichiometry than the target cannot be sub-compositions."""
    target = get_composition_counts(["Pt", "Pt"])
    db_path = "runs/Pt4_searches/trial_1/cluster.db"
    assert not _could_path_contain_relevant_candidates(db_path, target)


def test_accepts_path_when_parsed_is_subset_of_target():
    target = get_composition_counts(["Pt"] * 4)
    db_path = "runs/Pt2_searches/trial_1/cluster.db"
    assert _could_path_contain_relevant_candidates(db_path, target)


def test_accepts_path_when_composition_cannot_be_parsed_from_path():
    target = Counter({"Pt": 2})
    db_path = "some/flat/dir/minima.db"
    assert _could_path_contain_relevant_candidates(db_path, target)
