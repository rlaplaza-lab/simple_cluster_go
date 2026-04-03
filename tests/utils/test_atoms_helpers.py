"""Tests for atoms_helpers module.

This module tests utility functions for working with ASE Atoms objects.
"""

import pytest

from scgo.utils.atoms_helpers import parse_energy_from_xyz_comment


class TestParseEnergyFromXyzComment:
    """Tests for parse_energy_from_xyz_comment function."""

    def test_valid_energy_float(self):
        """Test parsing valid float energy from comment."""
        comment = {"energy": -123.456}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(-123.456), (
            "Float energy should be parsed correctly"
        )

    def test_valid_energy_string(self):
        """Test parsing valid string energy from comment."""
        comment = {"E": "-123.456"}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(-123.456), (
            "String energy should be parsed correctly"
        )

    def test_multiple_values_uses_last(self):
        """Test that function uses last value when multiple exist."""
        comment = {"key1": "100.0", "key2": "200.0", "energy": "-50.5"}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(-50.5), (
            "Should use 'energy' key when multiple keys exist"
        )

    def test_empty_dict(self):
        """Test handling of empty dictionary."""
        comment = {}
        result = parse_energy_from_xyz_comment(comment)
        assert result is None, "Empty dict should return None"

    def test_non_numeric_value(self):
        """Test handling of non-numeric energy value."""
        comment = {"energy": "not_a_number"}
        result = parse_energy_from_xyz_comment(comment)
        assert result is None, "Non-numeric value should return None"

    def test_none_value(self):
        """Test handling of None value."""
        comment = {"energy": None}
        result = parse_energy_from_xyz_comment(comment)
        assert result is None, "None value should return None"

    def test_none_input(self):
        """Test handling of None input."""
        result = parse_energy_from_xyz_comment(None)
        assert result is None, "None input should return None"

    def test_non_dict_input(self):
        """Test handling of non-dict input."""
        result = parse_energy_from_xyz_comment("not_a_dict")
        assert result is None, "Non-dict input should return None"

    def test_list_input(self):
        """Test handling of list input."""
        result = parse_energy_from_xyz_comment([-123.456])
        assert result is None, "List input should return None"

    def test_positive_energy(self):
        """Test parsing positive energy value."""
        comment = {"E": "999.123"}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(999.123), (
            "Positive energy should be parsed correctly"
        )

    def test_zero_energy(self):
        """Test parsing zero energy value."""
        comment = {"energy": "0.0"}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(0.0), "Zero energy should be parsed correctly"

    def test_scientific_notation(self):
        """Test parsing energy in scientific notation."""
        comment = {"E": "-1.23e-10"}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(-1.23e-10), (
            "Scientific notation should be parsed correctly"
        )

    def test_integer_energy(self):
        """Test parsing integer energy value."""
        comment = {"energy": 42}
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(42.0), (
            "Integer energy should be converted to float"
        )

    @pytest.mark.parametrize(
        "comment,expected",
        [
            ({"energy": -100.0}, -100.0),
            ({"E": "-200.5"}, -200.5),
            ({"energy": "300.75"}, 300.75),
            ({"E": 400}, 400.0),
        ],
    )
    def test_various_energy_formats(self, comment, expected):
        """Test parsing various energy formats using parametrization."""
        result = parse_energy_from_xyz_comment(comment)
        assert result == pytest.approx(expected), (
            f"Should parse {comment} as {expected}"
        )
