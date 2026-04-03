"""Comprehensive tests for validation functionality - consolidated version.

This module provides tests for:
- Unit tests for validation utility functions (scgo.utils.validation)
- Integration tests for parameter validation at API level
- Initialization-specific parameter validation
- Cluster structure validation (geometry validation)
"""

import time

import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from scgo.algorithms.basinhopping_go import bh_go
from scgo.algorithms.geneticalgorithm_go import ga_go
from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim
from scgo.initialization import create_initial_cluster, random_spherical
from scgo.initialization.geometry_helpers import validate_cluster_structure
from scgo.minima_search import run_trials, scgo
from scgo.utils.helpers import (
    auto_niter,
    auto_population_size,
    perform_local_relaxation,
)
from scgo.utils.validation import (
    validate_atoms,
    validate_calculator_attached,
    validate_composition,
    validate_in_choices,
    validate_in_range,
    validate_integer,
    validate_positive,
)

# ============================================================================
# Section 1: Unit Tests for Validation Utilities (Consolidated)
# ============================================================================

# Parametrized test cases for atoms validation
ATOMS_INVALID_CASES = [
    ("not atoms", TypeError),
    (None, TypeError),
    ([1, 2, 3], TypeError),
]


class TestValidateAtoms:
    """Tests for validate_atoms function - consolidated."""

    def test_valid_atoms(self, pt2_atoms):
        """Test validate_atoms accepts valid Atoms object."""
        validate_atoms(pt2_atoms)

    @pytest.mark.parametrize("invalid_value,expected_error", ATOMS_INVALID_CASES)
    def test_invalid_atoms(self, invalid_value, expected_error):
        """Test validate_atoms raises error for invalid input."""
        with pytest.raises(expected_error, match="must be an ASE Atoms object"):
            validate_atoms(invalid_value)


# Parametrized test cases for calculator validation
CALCULATOR_ALGORITHM_CASES = [
    ("test algorithm", "test algorithm"),
    ("basin hopping", "basin hopping"),
    ("genetic algorithm", "genetic algorithm"),
]


class TestValidateCalculatorAttached:
    """Tests for validate_calculator_attached function - consolidated."""

    def test_valid_calculator_attached(self, pt2_with_calc):
        """Test validate_calculator_attached accepts atoms with calculator."""
        calc = validate_calculator_attached(pt2_with_calc, "test algorithm")
        assert calc is pt2_with_calc.calc

    def test_none_calculator_raises(self, pt2_atoms):
        """Test validate_calculator_attached raises for None calculator."""
        atoms = pt2_atoms.copy()
        atoms.calc = None

        with pytest.raises(ValueError, match="must have a calculator attached"):
            validate_calculator_attached(atoms, "basin hopping")

    @pytest.mark.parametrize("algorithm", [c[0] for c in CALCULATOR_ALGORITHM_CASES])
    def test_error_message_includes_algorithm(self, algorithm, pt2_atoms):
        """Test error message includes algorithm name."""
        atoms = pt2_atoms.copy()
        atoms.calc = None

        with pytest.raises(ValueError) as exc_info:
            validate_calculator_attached(atoms, algorithm)

        assert algorithm in str(exc_info.value)


# Parametrized test cases for positive validation
POSITIVE_TEST_CASES = [
    # (value, strict, should_pass, expected_error_msg)
    (1.0, True, True, None),
    (0.1, True, True, None),
    (100.0, True, True, None),
    (0.0, True, False, "must be positive"),
    (-1.0, True, False, "must be positive"),
    (0.0, False, True, None),
    (1.0, False, True, None),
    (0.1, False, True, None),
    (-1.0, False, False, "must be non-negative"),
]


class TestValidatePositive:
    """Consolidated tests for validate_positive function."""

    @pytest.mark.parametrize("value,strict,should_pass,error_msg", POSITIVE_TEST_CASES)
    def test_validate_positive(self, value, strict, should_pass, error_msg):
        """Test validate_positive with various value/strict combinations."""
        if should_pass:
            validate_positive("test_param", value, strict=strict)
        else:
            with pytest.raises(ValueError, match=error_msg):
                validate_positive("test_param", value, strict=strict)

    def test_error_includes_name(self):
        """Test error message includes parameter name."""
        with pytest.raises(ValueError, match="my_param"):
            validate_positive("my_param", -5.0, strict=True)


# Parametrized test cases for range validation
RANGE_VALID_CASES = [(5.0, 0.0, 10.0), (0.0, 0.0, 10.0), (10.0, 0.0, 10.0)]
RANGE_INVALID_CASES = [(-1.0, 0.0, 10.0), (11.0, 0.0, 10.0)]


class TestValidateInRangeValid:
    """Tests for validate_in_range valid cases - consolidated."""

    @pytest.mark.parametrize("value,min_val,max_val", RANGE_VALID_CASES)
    def test_valid_in_range(self, value, min_val, max_val):
        """Test validate_in_range accepts values within range."""
        validate_in_range("test_param", value, min_val, max_val)


class TestValidateInRangeInvalid:
    """Tests for validate_in_range invalid cases - consolidated."""

    @pytest.mark.parametrize("value,min_val,max_val", RANGE_INVALID_CASES)
    def test_invalid_in_range_raises(self, value, min_val, max_val):
        """Test validate_in_range raises error for out-of-range values."""
        with pytest.raises(ValueError, match="must be between"):
            validate_in_range("test_param", value, min_val, max_val)

    @pytest.mark.parametrize("value,min_val,max_val", RANGE_INVALID_CASES)
    def test_error_includes_range(self, value, min_val, max_val):
        """Test error message includes range values."""
        with pytest.raises(ValueError) as exc_info:
            validate_in_range("test_param", value, min_val, max_val)

        error_msg = str(exc_info.value)
        assert str(min_val) in error_msg
        assert str(max_val) in error_msg
        assert str(value) in error_msg


# Parametrized test cases for integer validation
INTEGER_VALID_STRICT = [0, 1, -5, 100]
INTEGER_INVALID_STRICT = [(5.0, TypeError), ("5", TypeError)]
INTEGER_VALID_NONSTRICT = [(5.0, True), (10.0, True)]
INTEGER_INVALID_NONSTRICT = [(5.5, TypeError)]


class TestValidateIntegerStrict:
    """Tests for validate_integer strict mode - consolidated."""

    @pytest.mark.parametrize("value", INTEGER_VALID_STRICT)
    def test_valid_integer_strict(self, value):
        """Test validate_integer accepts integers in strict mode."""
        validate_integer("test_param", value, strict=True)

    def test_error_includes_type(self):
        """Test error message includes type name."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_integer("test_param", "5", strict=True)


class TestValidateIntegerStrictInvalid:
    """Tests for validate_integer strict mode invalid - consolidated."""

    @pytest.mark.parametrize("value,expected_error", INTEGER_INVALID_STRICT)
    def test_invalid_strict_raises(self, value, expected_error):
        """Test validate_integer raises error for invalid strict values."""
        with pytest.raises(expected_error, match="must be an integer"):
            validate_integer("test_param", value, strict=True)


class TestValidateIntegerNonStrict:
    """Tests for validate_integer non-strict mode - consolidated."""

    @pytest.mark.parametrize("value,_", INTEGER_VALID_NONSTRICT)
    def test_valid_non_strict(self, value, _):
        """Test validate_integer accepts int-like float in non-strict mode."""
        validate_integer("test_param", value, strict=False)


class TestValidateIntegerNonStrictInvalid:
    """Tests for validate_integer non-strict invalid - consolidated."""

    @pytest.mark.parametrize("value,expected_error", INTEGER_INVALID_NONSTRICT)
    def test_invalid_non_strict_raises(self, value, expected_error):
        """Test validate_integer raises error for non-int-like float."""
        with pytest.raises(expected_error, match="must be an integer"):
            validate_integer("test_param", value, strict=False)


# Parametrized test cases for choice validation
CHOICES_VALID_CASES = [
    ("option1", ["option1", "option2", "option3"]),
    ("option2", ["option1", "option2", "option3"]),
    ("a", ("a", "b", "c")),
]
CHOICES_INVALID_CASES = [
    ("invalid", ["option1", "option2"]),
    ("x", ["a", "b", "c"]),
]


class TestValidateInChoicesValid:
    """Tests for validate_in_choices valid cases - consolidated."""

    @pytest.mark.parametrize("value,choices", CHOICES_VALID_CASES)
    def test_valid_choice(self, value, choices):
        """Test validate_in_choices accepts valid choice."""
        validate_in_choices("test_param", value, choices)


class TestValidateInChoicesInvalid:
    """Tests for validate_in_choices invalid cases - consolidated."""

    @pytest.mark.parametrize("value,choices", CHOICES_INVALID_CASES)
    def test_invalid_choice_raises(self, value, choices):
        """Test validate_in_choices raises error for invalid choice."""
        with pytest.raises(ValueError, match="must be one of"):
            validate_in_choices("test_param", value, choices)

    @pytest.mark.parametrize("value,choices", CHOICES_INVALID_CASES)
    def test_error_includes_choices(self, value, choices):
        """Test error message includes all valid choices."""
        with pytest.raises(ValueError) as exc_info:
            validate_in_choices("test_param", value, choices)

        error_msg = str(exc_info.value)
        for choice in choices:
            assert str(choice) in error_msg


# Parametrized test cases for composition validation
COMPOSITION_VALID_CASES = [
    (["Pt", "Pt", "Au"], False, False),
    (["H", "H"], False, False),
    ([], True, False),  # allow_empty=True
    (("Pt", "Pt"), False, True),  # allow_tuple=True
]
COMPOSITION_INVALID_CASES = [
    (None, TypeError, "cannot be None"),
    ([], ValueError, "cannot be empty"),
    (("Pt", "Pt"), TypeError, "must be a list"),  # allow_tuple=False
    (["Pt", 5, "Au"], TypeError, "must contain only string"),
    (["Pt", None, "Au"], TypeError, "must contain only string"),
    ("PtPt", TypeError, "must be a list"),
    ({"Pt": 2}, TypeError, "must be a list"),
]


@pytest.mark.parametrize("comp,allow_empty,allow_tuple", COMPOSITION_VALID_CASES)
class TestValidateCompositionValid:
    """Tests for validate_composition valid cases - consolidated."""

    def test_valid_composition(self, comp, allow_empty, allow_tuple):
        """Test validate_composition accepts valid input."""
        validate_composition(comp, allow_empty=allow_empty, allow_tuple=allow_tuple)


@pytest.mark.parametrize("comp,expected_error,error_msg", COMPOSITION_INVALID_CASES)
class TestValidateCompositionInvalid:
    """Tests for validate_composition invalid cases - consolidated."""

    def test_invalid_composition_raises(self, comp, expected_error, error_msg):
        """Test validate_composition raises error for invalid input."""
        with pytest.raises(expected_error, match=error_msg):
            validate_composition(comp)


# ============================================================================
# Section 2: Integration Tests - Parameter Validation (Consolidated)
# ============================================================================

# Parametrized test cases for composition validation
COMPOSITION_INVALID_TYPES = [
    ("Pt2", "string"),
    ([1, 2, 3], "non-string"),
    (None, "None"),
]


@pytest.mark.parametrize("comp,desc", COMPOSITION_INVALID_TYPES)
class TestCompositionValidationIntegration:
    """Tests for composition parameter validation - consolidated."""

    def test_invalid_composition_raises(self, comp, desc, rng):
        """Test handling of invalid composition types."""
        with pytest.raises((TypeError, ValueError)):
            create_initial_cluster(comp, rng=rng)

    def test_empty_composition_validation(self, comp, desc, rng):
        """Test that empty composition is handled correctly."""
        result = create_initial_cluster([], rng=rng)
        assert len(result) == 0
        assert isinstance(result, Atoms)


# Parametrized test cases for numeric parameters
NUMERIC_PARAM_CASES = [
    ("placement_radius_scaling", [-1.0, 0.0]),
    ("min_distance_factor", [-0.1, -1.0]),
    ("vacuum", [-1.0, -0.5]),
]


@pytest.mark.parametrize("param_name,invalid_values", NUMERIC_PARAM_CASES)
class TestNumericParameterValidationIntegration:
    """Tests for numeric parameter validation - consolidated."""

    def test_negative_parameters_raise(self, param_name, invalid_values):
        """Test that negative parameters raise appropriate errors."""
        for invalid_val in invalid_values:
            with pytest.raises(ValueError):
                create_initial_cluster(
                    ["Pt", "Pt"], rng=None, **{param_name: invalid_val}
                )


# Parametrized test cases for optimization parameters
OPTIMIZER_PARAM_CASES = [
    ("niter", [-1, 0, "invalid"]),
    ("population_size", [-1, 0, "invalid"]),
    ("temperature", [-1.0, "invalid"]),
    ("dr", [-1.0, 0.0, "invalid"]),
]


@pytest.mark.parametrize("param_name,invalid_values", OPTIMIZER_PARAM_CASES)
class TestOptimizationParameterValidation:
    """Tests for optimization parameter validation - consolidated."""

    def test_optimization_parameters_raise(self, param_name, invalid_values):
        """Test optimization parameter validation."""
        for invalid_val in invalid_values:
            with pytest.raises((ValueError, TypeError)):
                bh_go(
                    composition=["Pt", "Pt"],
                    niter=invalid_val if param_name == "niter" else 10,
                    population_size=(
                        invalid_val if param_name == "population_size" else 20
                    ),
                    temperature=invalid_val if param_name == "temperature" else 0.1,
                    dr=invalid_val if param_name == "dr" else 0.3,
                    calculator_for_global_optimization=EMT(),
                )


def test_ga_offspring_fraction_validation(tmp_path, rng):
    """GA should validate `offspring_fraction` is (0, 1]."""
    calc = EMT()

    # Invalid: zero or negative
    with pytest.raises(ValueError):
        ga_go(
            composition=["Pt", "Pt"],
            output_dir=str(tmp_path / "bad_off_0"),
            calculator=calc,
            rng=rng,
            niter=1,
            population_size=2,
            offspring_fraction=0.0,
            niter_local_relaxation=1,
        )

    # Invalid: > 1
    with pytest.raises(ValueError):
        ga_go(
            composition=["Pt", "Pt"],
            output_dir=str(tmp_path / "bad_off_1"),
            calculator=calc,
            rng=rng,
            niter=1,
            population_size=2,
            offspring_fraction=1.1,
            niter_local_relaxation=1,
        )

    # TorchSim variant should also validate
    class LocalMockRelaxer:
        def __init__(self, max_steps: int | None = None):
            self.max_steps = max_steps

        def relax_batch(self, batch: list):
            return [(0.0, a.copy()) for a in batch]

    with pytest.raises(ValueError):
        ga_go_torchsim(
            composition=["Pt", "Pt"],
            output_dir=str(tmp_path / "bad_off_ts"),
            calculator=calc,
            rng=rng,
            niter=1,
            population_size=2,
            offspring_fraction=-0.2,
            niter_local_relaxation=1,
            relaxer=LocalMockRelaxer(max_steps=1),
        )

    # Valid edge cases
    ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(tmp_path / "ok_off"),
        calculator=calc,
        rng=rng,
        niter=1,
        population_size=2,
        offspring_fraction=1.0,
        niter_local_relaxation=1,
    )


# Parametrized test cases for calculator validation
CALCULATOR_INVALID_CASES = [
    ("invalid_calculator", "string"),
    (None, "None"),
]


@pytest.mark.parametrize("calc,desc", CALCULATOR_INVALID_CASES)
class TestCalculatorValidationIntegration:
    """Tests for calculator parameter validation - consolidated."""

    def test_invalid_calculator_raises(self, calc, desc):
        """Test handling of invalid calculator."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            bh_go(
                composition=["Pt", "Pt"],
                calculator_for_global_optimization=calc,
                niter=1,
            )


class TestHelperFunctionValidationIntegration:
    """Tests for helper function parameter validation - consolidated."""

    def test_auto_niter_none_raises(self):
        """Test auto_niter with None raises TypeError."""
        with pytest.raises(TypeError):
            auto_niter(None)

    @pytest.mark.parametrize("comp", [["Pt", "Pt"], "string"])
    def test_auto_niter_valid(self, comp):
        """Test auto_niter with valid inputs."""
        assert auto_niter(comp) > 0

    def test_auto_population_size_none_raises(self):
        """Test auto_population_size with None raises TypeError."""
        with pytest.raises(TypeError):
            auto_population_size(None)

    @pytest.mark.parametrize("comp", [["Pt", "Pt"], "string"])
    def test_auto_population_size_valid(self, comp):
        """Test auto_population_size with valid inputs."""
        assert auto_population_size(comp) > 0

    def test_perform_local_relaxation_invalid_calc(self, pt2_atoms):
        """Test perform_local_relaxation with invalid calculator."""
        atoms = pt2_atoms.copy()
        with pytest.raises((TypeError, ValueError, AttributeError)):
            perform_local_relaxation(atoms, calculator="invalid", fmax=0.01, steps=100)

    @pytest.mark.parametrize("fmax", [-0.01])
    def test_perform_local_relaxation_invalid_fmax(self, fmax, pt2_atoms):
        """Test perform_local_relaxation with invalid fmax."""
        atoms = pt2_atoms.copy()
        with pytest.raises((ValueError, TypeError)):
            perform_local_relaxation(atoms, calculator=EMT(), fmax=fmax, steps=100)

    @pytest.mark.parametrize("steps", [-1])
    def test_perform_local_relaxation_invalid_steps(self, steps, pt2_atoms):
        """Test perform_local_relaxation with invalid steps."""
        atoms = pt2_atoms.copy()
        with pytest.raises((ValueError, TypeError)):
            perform_local_relaxation(atoms, calculator=EMT(), fmax=0.01, steps=steps)


class TestGlobalOptimizerValidationIntegration:
    """Tests for global optimizer parameter validation - consolidated."""

    def test_bh_go_invalid_composition(self):
        """Test bh_go with invalid composition."""
        with pytest.raises((TypeError, ValueError)):
            bh_go(
                composition="invalid", calculator_for_global_optimization=EMT(), niter=1
            )

    @pytest.mark.parametrize("niter", [-1])
    def test_bh_go_invalid_niter(self, niter):
        """Test bh_go with invalid niter."""
        with pytest.raises((ValueError, TypeError)):
            bh_go(
                composition=["Pt", "Pt"],
                calculator_for_global_optimization=EMT(),
                niter=niter,
            )

    @pytest.mark.parametrize("temp", [-1.0])
    def test_bh_go_invalid_temperature(self, temp):
        """Test bh_go with invalid temperature."""
        with pytest.raises((ValueError, TypeError)):
            bh_go(
                composition=["Pt", "Pt"],
                calculator_for_global_optimization=EMT(),
                niter=1,
                temperature=temp,
            )

    def test_ga_go_invalid_composition(self):
        """Test ga_go with invalid composition."""
        with pytest.raises((TypeError, ValueError)):
            ga_go(
                composition="invalid",
                calculator_for_global_optimization=EMT(),
                niter=1,
                population_size=20,
            )

    @pytest.mark.parametrize("pop_size", [-1])
    def test_ga_go_invalid_population_size(self, pop_size):
        """Test ga_go with invalid population_size."""
        with pytest.raises((ValueError, TypeError)):
            ga_go(
                composition=["Pt", "Pt"],
                calculator_for_global_optimization=EMT(),
                niter=1,
                population_size=pop_size,
            )

    @pytest.mark.parametrize("mut_prob", [1.5])
    def test_ga_go_invalid_mutation_probability(self, mut_prob):
        """Test ga_go with invalid mutation_probability."""
        with pytest.raises((ValueError, TypeError)):
            ga_go(
                composition=["Pt", "Pt"],
                calculator_for_global_optimization=EMT(),
                niter=1,
                population_size=20,
                mutation_probability=mut_prob,
            )


class TestCampaignFunctionValidationIntegration:
    """Tests for campaign function parameter validation - consolidated."""

    def test_run_trials_invalid_composition(self):
        """Test run_trials with invalid composition."""
        with pytest.raises((TypeError, ValueError)):
            run_trials(
                composition="invalid",
                global_optimizer="bh",
                calculator_for_global_optimization=EMT(),
            )

    def test_run_trials_invalid_optimizer(self):
        """Test run_trials with invalid global_optimizer."""
        with pytest.raises((ValueError, TypeError)):
            run_trials(
                composition=["Pt", "Pt"],
                global_optimizer="invalid",
                calculator_for_global_optimization=EMT(),
            )

    @pytest.mark.parametrize("n_trials", [-1])
    def test_run_trials_invalid_n_trials(self, n_trials):
        """Test run_trials with invalid n_trials."""
        with pytest.raises((ValueError, TypeError)):
            run_trials(
                composition=["Pt", "Pt"],
                global_optimizer="bh",
                calculator_for_global_optimization=EMT(),
                n_trials=n_trials,
            )

    def test_scgo_invalid_composition(self):
        """Test scgo with invalid composition."""
        with pytest.raises((TypeError, ValueError)):
            scgo(
                composition="invalid",
                global_optimizer="bh",
                calculator_for_global_optimization=EMT(),
            )

    def test_scgo_invalid_optimizer(self):
        """Test scgo with invalid global_optimizer."""
        with pytest.raises((ValueError, TypeError)):
            scgo(
                composition=["Pt", "Pt"],
                global_optimizer="invalid",
                calculator_for_global_optimization=EMT(),
            )

    @pytest.mark.parametrize("n_trials", [-1])
    def test_scgo_invalid_n_trials(self, n_trials):
        """Test scgo with invalid n_trials."""
        with pytest.raises((ValueError, TypeError)):
            scgo(
                composition=["Pt", "Pt"],
                global_optimizer="bh",
                calculator_for_global_optimization=EMT(),
                n_trials=n_trials,
            )


class TestEdgeCaseValidationIntegration:
    """Tests for edge case parameter validation - consolidated."""

    def test_extreme_parameter_values(self, rng):
        """Test handling of extreme parameter values."""
        result = create_initial_cluster(["Pt", "Pt"], rng=rng)
        assert len(result) == 2

        result = create_initial_cluster(
            ["Pt", "Pt"], rng=rng, placement_radius_scaling=1.5, vacuum=2.0
        )
        assert len(result) == 2

    @pytest.mark.parametrize("invalid_param", ["1.0", "invalid"])
    def test_mixed_type_parameters(self, invalid_param):
        """Test handling of mixed type parameters."""
        with pytest.raises((TypeError, ValueError)):
            create_initial_cluster(
                ["Pt", "Pt"], rng=None, placement_radius_scaling=invalid_param
            )

    def test_none_parameters_raise(self):
        """Test handling of None parameters."""
        with pytest.raises((TypeError, ValueError)):
            create_initial_cluster(None, rng=None)

        with pytest.raises((TypeError, ValueError)):
            create_initial_cluster(
                ["Pt", "Pt"], rng=None, placement_radius_scaling=None
            )


# ============================================================================
# Section 3: Initialization Parameter Validation (Consolidated)
# ============================================================================

# Parametrized test cases for initialization parameters
INIT_PARAM_INVALID_CASES = [
    ("placement_radius_scaling", 0, "positive"),
    ("placement_radius_scaling", -1, "positive"),
    ("min_distance_factor", -0.1, "non-negative"),
    ("cell_side", 0, "positive"),
    ("vacuum", -1.0, "non-negative"),
]


@pytest.mark.parametrize("param_name,value,error_match", INIT_PARAM_INVALID_CASES)
class TestInitializationParameterValidationIntegration:
    """Tests for initialization parameter validation - consolidated."""

    def test_invalid_init_params_raise(self, param_name, value, error_match, rng):
        """Test initialization parameter validation."""
        if param_name == "cell_side":
            with pytest.raises(ValueError, match=error_match):
                random_spherical(["Pt", "Pt"], cell_side=value, rng=rng)
        else:
            with pytest.raises(ValueError, match=error_match):
                create_initial_cluster(["Pt", "Pt"], **{param_name: value}, rng=rng)


class TestInitializationCompositionValidation:
    """Tests for initialization composition validation - consolidated."""

    def test_composition_none_raises(self, rng):
        """Test composition = None."""
        with pytest.raises(TypeError, match="cannot be None"):
            create_initial_cluster(None, rng=rng)

    def test_composition_not_list_raises(self, rng):
        """Test composition that is not list/tuple."""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            create_initial_cluster("Pt", rng=rng)

    def test_composition_non_string_elements_raises(self, rng):
        """Test composition with non-string elements."""
        with pytest.raises(TypeError, match="must contain only string"):
            create_initial_cluster([1, 2, 3], rng=rng)

    def test_unknown_mode_raises(self, rng):
        """Test unknown mode parameter."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            create_initial_cluster(["Pt", "Pt"], mode="invalid_mode", rng=rng)

    def test_placement_error_message(self, rng):
        """Test that placement errors include diagnostic information."""
        comp = ["Pt"] * 20
        with pytest.raises(ValueError) as exc_info:
            random_spherical(
                comp,
                cell_side=5.0,
                placement_radius_scaling=0.01,
                rng=rng,
            )

        error_msg = str(exc_info.value)
        assert (
            "placement_radius_scaling" in error_msg
            or "suggestions" in error_msg.lower()
        )


# ============================================================================
# Section 4: Cluster Structure Validation (Consolidated)
# ============================================================================


class TestClusterStructureValidationUnit:
    """Unit tests for cluster structure validation - consolidated."""

    TEST_CONNECTIVITY_FACTOR = 1.5

    @pytest.mark.parametrize("n_atoms", [0, 1])
    def test_empty_or_single_atom_valid(self, n_atoms, single_atom):
        """Test that empty/single atom clusters are valid."""
        atoms = Atoms() if n_atoms == 0 else single_atom.copy()

        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is True
        assert msg == ""

    def test_valid_cluster_passes(self, pt2_atoms):
        """Test that a valid cluster passes validation."""
        atoms = pt2_atoms.copy()
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is True
        assert msg == ""

    def test_clashing_atoms_detected(self):
        """Test that clashing atoms are detected."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.0, 0, 0]])
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is False
        assert "Atomic clashes detected" in msg
        assert "min_distance_factor=0.5" in msg

    def test_disconnected_cluster_detected(self):
        """Test that disconnected clusters are detected."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [20, 0, 0]])
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is False
        assert "Cluster is not connected" in msg
        assert f"connectivity_factor={self.TEST_CONNECTIVITY_FACTOR}" in msg

    def test_both_errors_reported(self):
        """Test validation when both clashes and disconnection occur."""
        atoms = Atoms(
            "Pt4",
            positions=[
                [0, 0, 0],
                [1.0, 0, 0],  # Clash
                [20, 0, 0],  # Disconnect
                [21, 0, 0],
            ],
        )
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is False
        assert "Atomic clashes detected" in msg
        assert "Cluster is not connected" in msg

    @pytest.mark.parametrize(
        "check_clashes,check_conn,should_be_valid",
        [(True, False, False), (False, True, True), (False, False, True)],
    )
    def test_selective_checks(self, check_clashes, check_conn, should_be_valid):
        """Test validation with selective checks enabled/disabled."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.0, 0, 0]])

        is_valid, _ = validate_cluster_structure(
            atoms,
            0.5,
            self.TEST_CONNECTIVITY_FACTOR,
            check_clashes=check_clashes,
            check_connectivity=check_conn,
        )
        assert is_valid is should_be_valid

    @pytest.mark.parametrize(
        "min_dist,conn_fact,expected", [(0.8, 1.0, False), (0.3, 2.0, True)]
    )
    def test_different_parameters(self, min_dist, conn_fact, expected):
        """Test validation with different parameter values."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.0, 0, 0]])
        is_valid, _ = validate_cluster_structure(atoms, min_dist, conn_fact)
        assert is_valid is expected

    def test_error_message_format(self):
        """Test that error messages contain expected information."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.0, 0, 0]])
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )

        assert is_valid is False
        assert "Validation failed for Pt2 cluster (2 atoms):" in msg
        assert "Pt(0)-Pt(1):" in msg

    def test_large_cluster_performance(self):
        """Test validation performance on large clusters."""
        atoms = bulk("Pt", "fcc", a=4.0).repeat((3, 3, 3))
        atoms.center()

        start = time.time()
        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Validation took {elapsed:.3f}s, should be <5s"
        assert is_valid is True
        assert msg == ""

    def test_mixed_elements_valid(self):
        """Test validation with mixed element types."""
        atoms = Atoms(
            "PtAu",
            positions=[[0, 0, 0], [2.8, 0, 0]],
        )

        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is True
        assert msg == ""


class TestValidationIntegrationFinal:
    """Tests for validation integration with entry points - consolidated."""

    TEST_CONNECTIVITY_FACTOR = 1.5

    def test_create_initial_cluster_validation(self, rng):
        """Test that create_initial_cluster calls validation when enabled."""
        try:
            atoms = create_initial_cluster(
                composition=["Pt", "Pt"],
                rng=rng,
                mode="random_spherical",
                min_distance_factor=0.5,
                connectivity_factor=self.TEST_CONNECTIVITY_FACTOR,
            )
            assert len(atoms) == 2
            assert atoms.get_chemical_formula() == "Pt2"
        except ValueError as e:
            assert "Final validation failed" in str(e)

    def test_random_spherical_validation(self, rng):
        """Test that random_spherical calls validation."""
        try:
            atoms = random_spherical(
                composition=["Pt", "Pt"],
                cell_side=10.0,
                rng=rng,
                min_distance_factor=0.5,
                connectivity_factor=self.TEST_CONNECTIVITY_FACTOR,
            )
            assert len(atoms) == 2
            assert atoms.get_chemical_formula() == "Pt2"
        except ValueError as e:
            assert "Final validation failed in random_spherical" in str(e)

    def test_validation_catches_invalid_structures(self):
        """Test that final validation catches invalid structures."""
        atoms = Atoms(
            "Pt3",
            positions=[
                [0, 0, 0],
                [1.0, 0, 0],  # Too close
                [20, 0, 0],  # Too far
            ],
        )

        is_valid, msg = validate_cluster_structure(
            atoms, 0.5, self.TEST_CONNECTIVITY_FACTOR
        )
        assert is_valid is False
        assert "Atomic clashes detected" in msg
        assert "Cluster is not connected" in msg
