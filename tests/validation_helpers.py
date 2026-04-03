"""Shared utilities for parameter validation testing.

This module provides reusable test patterns and fixtures for validation
testing to eliminate duplicated test patterns across test files.
"""

from typing import Any

import pytest

# Validation test patterns for common parameter types
VALIDATION_PATTERNS = {
    "type_checks": [
        ("atoms", "not atoms", TypeError, "must be an ASE Atoms object"),
        ("atoms", None, TypeError, "must be an ASE Atoms object"),
        ("atoms", [1, 2, 3], TypeError, "must be an ASE Atoms object"),
    ],
    "range_checks": [
        ("positive", 0, ValueError, "positive"),
        ("positive", -1, ValueError, "positive"),
        ("positive", 0.001, None, None),  # Valid
        ("range_0_1", -0.1, ValueError, "range"),
        ("range_0_1", 1.1, ValueError, "range"),
        ("range_0_1", 0.5, None, None),  # Valid
    ],
    "integer_checks": [
        ("integer", 1.5, TypeError, "integer"),
        ("integer", 5, None, None),  # Valid
        ("integer_positive", 0, ValueError, "positive"),
        ("integer_positive", 5, None, None),  # Valid
    ],
}


@pytest.fixture(params=VALIDATION_PATTERNS["type_checks"])
def type_check_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized fixture for type validation test cases.

    Provides a dictionary with keys:
    - 'name': parameter name being validated
    - 'value': invalid value to test
    - 'error': expected exception type
    - 'msg': expected message substring
    """
    param_name, invalid_value, expected_error, error_msg = request.param
    return {
        "name": param_name,
        "value": invalid_value,
        "error": expected_error,
        "msg": error_msg,
    }


@pytest.fixture(params=VALIDATION_PATTERNS["range_checks"])
def range_check_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized fixture for range validation test cases.

    Provides a dictionary with keys:
    - 'type': check type (e.g., 'positive', 'range_0_1')
    - 'value': value to test
    - 'error': expected exception type (None if should be valid)
    - 'msg': expected message substring
    """
    check_type, value, expected_error, error_msg = request.param
    return {
        "type": check_type,
        "value": value,
        "error": expected_error,
        "msg": error_msg,
    }


def create_type_validation_test(
    validator_func, param_name, valid_example, invalid_values
):
    """Factory for creating type validation test classes.

    Automatically generates a test class with test_valid and test_invalid
    methods for type validation testing.

    Args:
        validator_func: The validation function to test
        param_name: Name of the parameter being validated
        valid_example: An example valid value
        invalid_values: List of invalid values to test

    Returns:
        A pytest test class with valid/invalid tests

    Example:
        >>> TestValidateAtoms = create_type_validation_test(
        ...     validate_atoms,
        ...     "atoms",
        ...     Atoms("Pt"),
        ...     ["not atoms", None, [1, 2, 3]]
        ... )
    """

    class TypeValidationTest:
        def test_valid(self):
            """Valid type should not raise."""
            validator_func(valid_example)

        @pytest.mark.parametrize("invalid_value", invalid_values)
        def test_invalid(self, invalid_value):
            """Invalid type should raise appropriate error."""
            with pytest.raises(TypeError):
                validator_func(invalid_value)

    TypeValidationTest.__name__ = f"TestValidate{param_name.title()}"
    return TypeValidationTest


def create_range_validation_test(
    validator_func, param_name, valid_values, invalid_values
):
    """Factory for creating range validation test classes.

    Automatically generates a test class with test_valid and test_invalid
    methods for range/value validation testing.

    Args:
        validator_func: The validation function to test
        param_name: Name of the parameter being validated
        valid_values: List of valid values to test
        invalid_values: List of invalid values to test

    Returns:
        A pytest test class with valid/invalid tests

    Example:
        >>> TestValidatePositive = create_range_validation_test(
        ...     validate_positive,
        ...     "positive",
        ...     [0.001, 1, 100],
        ...     [0, -1, -100]
        ... )
    """

    class RangeValidationTest:
        @pytest.mark.parametrize("valid_value", valid_values)
        def test_valid(self, valid_value):
            """Values in valid range should not raise."""
            validator_func(valid_value)

        @pytest.mark.parametrize("invalid_value", invalid_values)
        def test_invalid(self, invalid_value):
            """Values outside range should raise ValueError."""
            with pytest.raises(ValueError):
                validator_func(invalid_value)

    RangeValidationTest.__name__ = f"TestValidate{param_name.title()}"
    return RangeValidationTest


def parametrize_invalid_cases(cases):
    """Decorator to parametrize tests with multiple invalid cases.

    Reduces boilerplate for tests that check multiple invalid inputs.

    Args:
        cases: List of (value, expected_error_type) tuples

    Example:
        >>> @parametrize_invalid_cases([
        ...     (None, TypeError),
        ...     (-1, ValueError),
        ...     ("invalid", TypeError),
        ... ])
        ... def test_validation(value, expected_error):
        ...     with pytest.raises(expected_error):
        ...         some_validator(value)
    """
    return pytest.mark.parametrize(
        "invalid_value,expected_error",
        cases,
        ids=[f"invalid_{i}" for i in range(len(cases))],
    )
