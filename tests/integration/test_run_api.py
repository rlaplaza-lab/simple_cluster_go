import pytest

from scgo import parse_composition_arg
from scgo.minima_search import run_trials
from scgo.param_presets import get_default_params, get_testing_params
from scgo.run_minima import (
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_campaign_one_element,
    run_scgo_campaign_two_elements,
    run_scgo_trials,
)


def test_parse_composition_arg_formats():
    assert parse_composition_arg("Pt,Pt,Au") == ["Pt", "Pt", "Au"]
    assert parse_composition_arg("Pt3Au") == ["Pt", "Pt", "Pt", "Au"]
    assert parse_composition_arg("Pt10") == ["Pt"] * 10


@pytest.mark.parametrize(
    "fn,args",
    [
        pytest.param(
            run_scgo_campaign_one_element,
            ("", 2, 4),
            id="one_element_empty_symbol",
        ),
        pytest.param(
            run_scgo_campaign_one_element,
            ("Pt", 0, 3),
            id="one_element_min_atoms_zero",
        ),
        pytest.param(
            run_scgo_campaign_one_element,
            ("Pt", 5, 3),
            id="one_element_min_gt_max",
        ),
        pytest.param(
            run_scgo_campaign_two_elements,
            ("", "Pt", 2, 4),
            id="two_elements_empty_first_symbol",
        ),
        pytest.param(
            run_scgo_campaign_two_elements,
            ("Pt", "", 2, 4),
            id="two_elements_empty_second_symbol",
        ),
        pytest.param(
            run_scgo_campaign_two_elements,
            ("Pt", "Au", 0, 3),
            id="two_elements_min_atoms_zero",
        ),
        pytest.param(
            run_scgo_campaign_two_elements,
            ("Pt", "Au", 5, 3),
            id="two_elements_min_gt_max",
        ),
        pytest.param(
            run_scgo_campaign_arbitrary_compositions,
            ([],),
            id="arbitrary_compositions_empty",
        ),
    ],
)
def test_run_campaign_invalid_inputs(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_rng_in_optimizer_params_raises():
    params = get_default_params()
    # inject forbidden 'rng' into optimizer params
    params["optimizer_params"]["ga"] = params["optimizer_params"].get("ga", {})
    params["optimizer_params"]["ga"]["rng"] = "not-allowed"

    with pytest.raises(ValueError):
        run_scgo_trials(["Pt"] * 4, params=params)


def test_scgo_validations(rng):
    # Use deterministic rng fixture from conftest

    # Invalid RNG
    with pytest.raises(ValueError):
        from scgo.minima_search import scgo

        scgo(["Pt"], "ga", {}, "out_dir", None)

    # Invalid optimizer name
    with pytest.raises(ValueError):
        scgo(["Pt"], "invalid_optimizer", {}, "out_dir", rng)

    # Invalid trial id
    with pytest.raises(ValueError):
        scgo(["Pt"], "ga", {}, "out_dir", rng, trial_id=0)


def test_run_trials_validations(rng):
    # Use deterministic rng fixture from conftest

    with pytest.raises(ValueError):
        run_trials([], "ga", {}, 1, "out", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], 123, {}, 1, "out", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, 0, "out", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, 1, "", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, 1, "out", None)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, 1, "out", rng, verbosity=5)


def test_parse_composition_arg_case_insensitive():
    assert parse_composition_arg("pt3") == ["Pt", "Pt", "Pt"]
    assert parse_composition_arg("pt3au") == ["Pt", "Pt", "Pt", "Au"]
    assert parse_composition_arg("pt,pt,au") == ["Pt", "Pt", "Au"]


def test_parse_composition_arg_zero_count():
    # Zero counts in compact formula should be rejected
    with pytest.raises(ValueError):
        parse_composition_arg("Pt0")
    with pytest.raises(ValueError):
        parse_composition_arg("pt0au")
    with pytest.raises(ValueError):
        parse_composition_arg("AuPt0")


def test_seed_in_params_respected():
    params = get_testing_params()
    params["seed"] = 12345

    comp = ["Pt", "Pt"]  # Pt2 small test

    # Run twice with same params (no explicit seed argument) -> results deterministic
    res1 = run_scgo_trials(comp, params=params, verbosity=0)
    res2 = run_scgo_trials(comp, params=params, verbosity=0)

    assert res1 == res2


def test_campaign_respects_params_seed():
    params = get_testing_params()
    params["seed"] = 54321

    # Run a tiny campaign (Pt2 only) twice and ensure deterministic results
    res_a = run_scgo_campaign_one_element("Pt", 2, 2, params=params, verbosity=0)
    res_b = run_scgo_campaign_one_element("Pt", 2, 2, params=params, verbosity=0)

    assert res_a == res_b
