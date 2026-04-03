"""Shared validation utilities for SCGO algorithms.

This module provides common validation functions to reduce code duplication
across different optimization algorithm implementations.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator


def validate_atoms(atoms: Atoms) -> None:
    """Validate that atoms is a valid ASE Atoms object.

    Args:
        atoms: Object to validate.

    Raises:
        TypeError: If atoms is not an Atoms instance.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("Input 'atoms' must be an ASE Atoms object.")


def validate_calculator_attached(
    atoms: Atoms,
    algorithm_name: str,
) -> Calculator:
    """Validate that atoms has a calculator attached.

    Args:
        atoms: Atoms object to check.
        algorithm_name: Name of algorithm for error message (e.g., "basin hopping",
            "genetic algorithm").

    Returns:
        The calculator instance.

    Raises:
        ValueError: If calculator is not attached.
    """
    calculator = atoms.calc
    if calculator is None:
        raise ValueError(
            f"The input 'atoms' object for {algorithm_name} must have a calculator attached.",
        )
    return calculator


def validate_positive(name: str, value: float, strict: bool = False) -> None:
    """Validate that a numeric value is positive.

    Args:
        name: Name of parameter for error message.
        value: Value to validate.
        strict: If True, value must be > 0. If False, value must be >= 0.

    Raises:
        ValueError: If value is not positive.
    """
    if strict and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    elif not strict and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_in_range(
    name: str,
    value: float,
    min_val: float,
    max_val: float,
) -> None:
    """Validate that a value is within a specified range.

    Args:
        name: Name of parameter for error message.
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Raises:
        ValueError: If value is outside the range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {value}",
        )


def validate_integer(name: str, value: Any, strict: bool = True) -> None:
    """Validate that a value is an integer.

    Args:
        name: Name of parameter for error message.
        value: Value to validate.
        strict: If True, must be exactly int. If False, allows int-like
            (float with .0).

    Raises:
        TypeError: If value is not an integer.
    """
    if isinstance(value, int):
        return
    if strict:
        raise TypeError(
            f"Input '{name}' must be an integer, got {type(value).__name__}"
        )
    if not (isinstance(value, float) and value.is_integer()):
        raise TypeError(
            f"Input '{name}' must be an integer, got {type(value).__name__}"
        )


def validate_in_choices(
    name: str,
    value: Any,
    choices: list[str] | tuple[str, ...],
) -> None:
    """Validate that a value is one of the allowed choices.

    Args:
        name: Name of parameter for error message.
        value: Value to validate.
        choices: List or tuple of allowed string values.

    Raises:
        ValueError: If value is not in choices.
    """
    if value not in choices:
        choices_str = ", ".join(f"'{c}'" for c in choices)
        raise ValueError(
            f"{name} must be one of {choices_str}, got '{value}'",
        )


def validate_composition(
    composition: Any,
    allow_empty: bool = False,
    allow_tuple: bool = False,
) -> None:
    """Validate that composition is a valid list/tuple of atomic symbols.

    Args:
        composition: Composition to validate.
        allow_empty: If True, empty composition is valid. If False, raises
            ValueError.
        allow_tuple: If True, accepts both list and tuple. If False, only
            accepts list.

    Raises:
        TypeError: If composition is None or not a list/tuple.
        ValueError: If composition is empty (when allow_empty=False) or contains
            non-string elements.
    """
    if composition is None:
        raise TypeError("composition cannot be None")

    valid_types = (list, tuple) if allow_tuple else (list,)
    if not isinstance(composition, valid_types):
        type_names = "list or tuple" if allow_tuple else "list"
        raise TypeError(
            f"composition must be a {type_names} of atomic symbols, got {type(composition).__name__}",
        )

    if not allow_empty and not composition:
        raise ValueError(
            "Composition cannot be empty. "
            "Provide a list of atomic symbols, e.g., ['Pt', 'Pt', 'Pt'] for Pt₃.",
        )

    if not all(isinstance(s, str) for s in composition):
        raise TypeError("composition must contain only string element symbols")
