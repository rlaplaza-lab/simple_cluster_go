import pytest
from ase import Atoms

from scgo import parse_composition_arg
from scgo.minima_search import run_trials
from scgo.param_presets import (
    get_default_params,
    get_testing_params,
    get_ts_search_params,
)
from scgo.run_minima import (
    run_scgo_campaign_arbitrary_compositions,
    run_scgo_campaign_one_element,
    run_scgo_campaign_two_elements,
    run_scgo_one_element_go_ts_pipeline,
    run_scgo_trials,
)
from scgo.runner_api import (
    log_go_ts_summary,
    run_go,
    run_go_campaign,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)
from scgo.system_types import get_system_policy
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs


def _emt_ts_gasc() -> dict:
    return {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_parallel_neb": False,
        "use_torchsim": False,
    }


def _emt_ts_surf_ads() -> dict:
    return {
        **get_ts_search_params(
            system_type="surface_cluster_adsorbate",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_parallel_neb": False,
        "use_torchsim": False,
    }


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


def test_run_scgo_one_element_go_ts_pipeline_smoke(monkeypatch, tmp_path):
    import scgo.run_minima as run_minima_module
    import scgo.ts_search as ts_search_module

    def _fake_trials(*args, **kwargs):
        return []

    def _fake_ts(*args, **kwargs):
        return [{"status": "success"}, {"status": "failed"}]

    monkeypatch.setattr(
        run_minima_module,
        "run_scgo_trials",
        _fake_trials,
    )
    monkeypatch.setattr(
        ts_search_module,
        "run_transition_state_search",
        _fake_ts,
    )

    flat_ts = {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
        "max_pairs": 2,
    }
    summary = run_scgo_one_element_go_ts_pipeline(
        "Pt",
        5,
        go=get_testing_params(),
        ts_kwargs=coerce_ts_params_to_runner_kwargs(flat_ts),
        seed=42,
        verbosity=0,
        output_dir=tmp_path / "pt5_gas",
    )
    assert summary["formula"] == "Pt5"
    assert summary["ts_success_count"] == 1
    assert summary["ts_total_count"] == 2
    assert summary["output_dir"] == (tmp_path / "pt5_gas").resolve()


def test_run_go_atoms_matches_explicit_list(monkeypatch):
    captured: dict[str, list] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr("scgo.runner_api.run_scgo_trials", _fake_trials)

    run_go(Atoms("Pt3"), params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]

    run_go(["Pt", "Pt", "Pt"], params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]

    run_go("Pt3", params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]


def test_run_go_system_type_wires_optimizer_params(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api.run_scgo_trials", _fake_trials)
    run_go(
        "Pt5O2H2",
        params={"optimizer_params": {"ga": {}, "bh": {}}},
        verbosity=0,
        system_type="surface_cluster_adsorbate",
    )
    params = captured["params"]
    assert (
        params["optimizer_params"]["ga"]["system_type"] == "surface_cluster_adsorbate"
    )
    assert (
        params["optimizer_params"]["bh"]["system_type"] == "surface_cluster_adsorbate"
    )
    assert (
        params["optimizer_params"]["simple"]["system_type"]
        == "surface_cluster_adsorbate"
    )
    assert params["optimizer_params"]["ga"]["write_timing_json"] is False
    assert params["optimizer_params"]["bh"]["write_timing_json"] is False


def test_run_go_profile_toggle(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api.run_scgo_trials", _fake_trials)
    run_go(
        "Pt3",
        params={"optimizer_params": {"ga": {}}},
        verbosity=0,
        system_type="gas_cluster",
        profile_ga=False,
    )
    params = captured["params"]
    assert params["optimizer_params"]["ga"]["write_timing_json"] is False


@pytest.mark.parametrize(
    "system_type",
    [
        "gas_cluster",
        "surface_cluster",
        "gas_cluster_adsorbate",
        "surface_cluster_adsorbate",
    ],
)
def test_run_go_system_type_matrix(monkeypatch, system_type):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api.run_scgo_trials", _fake_trials)
    run_go(
        "Pt3",
        params={"optimizer_params": {"simple": {}, "ga": {}, "bh": {}}},
        verbosity=0,
        system_type=system_type,
    )
    params = captured["params"]
    assert params["optimizer_params"]["simple"]["system_type"] == system_type
    assert params["optimizer_params"]["ga"]["system_type"] == system_type
    assert params["optimizer_params"]["bh"]["system_type"] == system_type


def test_system_policy_surface_neb_defaults():
    gas = get_system_policy("gas_cluster")
    surf = get_system_policy("surface_cluster_adsorbate")
    assert gas.neb_force_mic is False
    assert gas.neb_disable_alignment is False
    assert surf.neb_force_mic is True
    assert surf.neb_disable_alignment is True


def test_run_go_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go("Pt3", params=None, verbosity=0)


def test_run_go_campaign_normalizes_items(monkeypatch):
    captured: list[list[str]] = []

    def _fake_campaign(compositions, *args, **kwargs):
        captured.extend(compositions)
        return {}

    monkeypatch.setattr(
        "scgo.runner_api.run_scgo_campaign_arbitrary_compositions",
        _fake_campaign,
    )

    run_go_campaign(
        [Atoms("Au2"), "Pt", ["Cu", "Cu"]],
        params=None,
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"], ["Cu", "Cu"]]


def test_run_go_campaign_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        run_go_campaign([], params=None, verbosity=0, system_type="gas_cluster")


def test_run_go_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go_campaign(["Pt2"], params=None, verbosity=0)


def test_run_ts_search_normalizes_composition(monkeypatch):
    captured: dict[str, list] = {}

    def _fake(composition, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr(
        "scgo.runner_api._ts_search",
        _fake,
    )

    run_ts_search(
        "Cu2",
        params={"calculator": "EMT"},
        ts=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]

    run_ts_search(
        Atoms("Cu2"),
        params={"calculator": "EMT"},
        ts=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]

    run_ts_search(
        ["Cu", "Cu"],
        params={"calculator": "EMT"},
        ts=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]


def test_run_ts_search_passes_system_type(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_api._ts_search", _fake)
    run_ts_search(
        "Pt5O2H2",
        params={"calculator": "EMT"},
        ts=_emt_ts_surf_ads(),
        verbosity=0,
        system_type="surface_cluster_adsorbate",
    )
    assert captured["kwargs"]["system_type"] == "surface_cluster_adsorbate"


def test_run_ts_search_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_ts_search(
            "Pt2", params={"calculator": "EMT"}, ts=_emt_ts_gasc(), verbosity=0
        )


def test_run_ts_search_requires_ts_dict():
    with pytest.raises(ValueError, match="ts is required"):
        run_ts_search(
            "Pt2",
            params={"calculator": "EMT"},
            ts={},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_ts_campaign_normalizes_items(monkeypatch):
    captured: list[list[str]] = []

    def _fake(compositions, **kwargs):
        captured.extend(compositions)
        return {}

    monkeypatch.setattr(
        "scgo.runner_api._ts_campaign",
        _fake,
    )

    run_ts_campaign(
        [Atoms("Au2"), "Pt"],
        params={"calculator": "EMT"},
        ts=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"]]


def test_run_ts_campaign_requires_ts_dict():
    with pytest.raises(ValueError, match="ts is required"):
        run_ts_campaign(
            [Atoms("Au2"), "Pt"],
            params={"calculator": "EMT"},
            ts={},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_ts_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_ts_campaign(
            [Atoms("Au2"), "Pt"],
            params={"calculator": "EMT"},
            ts=_emt_ts_gasc(),
            verbosity=0,
        )


def test_run_go_ts_campaign_paths(monkeypatch, tmp_path):
    calls: list[tuple[list[str], object]] = []

    def _fake_pipeline(composition, **kwargs):
        calls.append((list(composition), kwargs.get("output_dir")))
        return {"formula": "x", "ts_total_count": 0}

    monkeypatch.setattr("scgo.runner_api.run_scgo_go_ts_pipeline", _fake_pipeline)

    root = tmp_path / "camp"
    run_go_ts_campaign(
        ["Pt2", ["Au", "Au"]],
        go={},
        ts=_emt_ts_gasc(),
        verbosity=0,
        output_dir=root,
        system_type="gas_cluster",
    )
    assert len(calls) == 2
    assert calls[0][0] == ["Pt", "Pt"]
    assert calls[0][1] == root / "Pt2_campaign"
    assert calls[1][0] == ["Au", "Au"]
    assert calls[1][1] == root / "Au2_campaign"


def test_run_go_ts_wires_profile_toggle(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, **kwargs):
        captured["go"] = kwargs["go"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api.run_scgo_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        "Pt2",
        go={"optimizer_params": {"ga": {}}},
        ts=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
        profile_ga=False,
    )
    go = captured["go"]
    assert go["optimizer_params"]["ga"]["write_timing_json"] is False


def test_run_go_ts_campaign_no_output_dir(
    monkeypatch,
):
    calls: list[object] = []

    def _fake_pipeline(composition, **kwargs):
        calls.append(kwargs.get("output_dir"))
        return {}

    monkeypatch.setattr("scgo.runner_api.run_scgo_go_ts_pipeline", _fake_pipeline)

    run_go_ts_campaign(
        ["H2"],
        go={},
        ts=_emt_ts_gasc(),
        verbosity=0,
        output_dir=None,
        system_type="gas_cluster",
    )
    assert len(calls) == 1
    assert calls[0] is not None
    assert "H2_campaign" in str(calls[0])


def test_run_go_ts_campaign_requires_ts_dict():
    with pytest.raises(ValueError, match="ts is required"):
        run_go_ts_campaign(
            ["H2"],
            go={},
            ts={},
            verbosity=0,
            output_dir=None,
            system_type="gas_cluster",
        )


def test_run_go_ts_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go_ts_campaign(
            ["H2"],
            go={},
            ts=_emt_ts_gasc(),
            verbosity=0,
            output_dir=None,
        )


def test_run_go_ts_requires_ts_dict():
    with pytest.raises(ValueError, match="ts is required"):
        run_go_ts(
            "H2",
            go={},
            ts={},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_log_go_ts_summary():
    class _Log:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, fmt: str, *args: object) -> None:
            self.messages.append(fmt % args if args else fmt)

    log = _Log()
    log_go_ts_summary(
        log,
        {"ts_results": [{"status": "success"}, {"status": "failed"}]},
        wall_time_s=3.25,
    )
    assert log.messages[0] == "Successful NEBs: 1/2"
    assert log.messages[1] == "Total wall time: 3.25 s"
