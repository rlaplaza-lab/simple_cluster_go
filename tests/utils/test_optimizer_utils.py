"""Tests for optimizer utility functions.

This module tests utility functions for resolving optimizer class names
to their corresponding ASE optimizer classes.
"""

import pytest

from scgo.utils.optimizer_utils import get_optimizer_class


class TestGetOptimizerClass:
    """Tests for get_optimizer_class function."""

    def test_fire_optimizer(self):
        """Test getting FIRE optimizer class."""
        optimizer_class = get_optimizer_class("FIRE")
        assert optimizer_class.__name__ == "FIRE"

    def test_bfgs_optimizer(self):
        """Test getting BFGS optimizer class."""
        optimizer_class = get_optimizer_class("BFGS")
        assert optimizer_class.__name__ == "BFGS"

    def test_lbfgs_optimizer(self):
        """Test getting LBFGS optimizer class."""
        optimizer_class = get_optimizer_class("LBFGS")
        assert optimizer_class.__name__ == "LBFGS"

    def test_case_insensitive_fire(self):
        """Test case-insensitive optimizer name."""
        optimizer1 = get_optimizer_class("fire")
        optimizer2 = get_optimizer_class("FIRE")
        optimizer3 = get_optimizer_class("Fire")

        assert optimizer1 == optimizer2 == optimizer3

    def test_case_insensitive_lbfgs(self):
        """Test case-insensitive LBFGS."""
        optimizer1 = get_optimizer_class("lbfgs")
        optimizer2 = get_optimizer_class("LBFGS")

        assert optimizer1 == optimizer2

    def test_none_input_raises_error(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="optimizer_name cannot be None"):
            get_optimizer_class(None)

    def test_unknown_optimizer_raises_error(self):
        """Test that unknown optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer_class("UNKNOWN_OPTIMIZER")

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer_class("")

    def test_numeric_string_raises_error(self):
        """Test that numeric string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer_class("123")

    def test_error_message_suggests_supported_optimizers(self):
        """Test that error message lists supported optimizers."""
        with pytest.raises(ValueError) as exc_info:
            get_optimizer_class("INVALID")

        error_msg = str(exc_info.value)
        assert "Supported optimizers" in error_msg
        assert "FIRE" in error_msg
        assert "BFGS" in error_msg
        assert "LBFGS" in error_msg

    def test_returned_class_is_callable(self):
        """Test that returned optimizer class is callable."""
        optimizer_class = get_optimizer_class("FIRE")
        assert callable(optimizer_class)

    def test_optimizer_instance_creation(self):
        """Test creating optimizer instance from returned class."""
        from ase import Atoms

        optimizer_class = get_optimizer_class("FIRE")
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        # Should be able to create optimizer instance
        optimizer = optimizer_class(atoms)
        assert optimizer is not None
