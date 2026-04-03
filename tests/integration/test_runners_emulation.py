"""Runner emulation integration tests."""

import pytest

from scgo import run_scgo_campaign_one_element
from scgo.calculators.orca_helpers import prepare_orca_calculations
from scgo.database.registry import clear_registry_cache
from scgo.param_presets import get_testing_params
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula


@pytest.mark.slow
def test_emulate_run_Pt4_6_orca_lowe(tmp_path):
    """Emulate GA → ORCA → TS workflow."""
    out_root = tmp_path / "orca_calcs"

    # Run a compact, deterministic GA campaign that writes *_searches/run_*/candidates.db
    ga_params = get_testing_params()
    minima_by_formula = run_scgo_campaign_one_element(
        "Pt", 4, 6, params=ga_params, seed=42, verbosity=0, output_dir=tmp_path
    )
    assert minima_by_formula, "Expected minima from the GA campaign"

    # Prepare ORCA inputs for the produced minima (replicates runner behaviour)
    for formula, minima in minima_by_formula.items():
        prepare_orca_calculations(minima, str(out_root / formula), {})
        formula_dir = out_root / formula
        assert formula_dir.exists()
        min_dir = next(formula_dir.glob("minimum_*_" + formula))
        assert (min_dir / "orca.inp").exists()

    # Run TS search reusing the GA databases (base_dir points to *_searches)
    for n in range(4, 7):
        comp = ["Pt"] * n
        formula = get_cluster_formula(comp)
        base_dir = tmp_path / f"{formula}_searches"

        results = run_transition_state_search(
            comp,
            base_dir=base_dir,
            params={"calculator": "EMT"},
            seed=42,
            verbosity=0,
            max_pairs=10,
            neb_n_images=3,
            neb_fmax=0.3,
            neb_steps=10,
        )

        assert isinstance(results, list)
        assert len(results) <= 10

        final_dir = base_dir / f"ts_results_{formula}" / "final_unique_ts"
        assert final_dir.exists()
        summary_file = final_dir / f"final_unique_ts_summary_{formula}.json"
        assert summary_file.exists()


@pytest.mark.slow
def test_emulate_run_Pt4_6_ts_search(tmp_path):
    """Produce minima DBs and run capped TS campaign reusing them."""
    compositions = [["Pt"] * n for n in range(4, 7)]

    # Run lightweight GA to produce databases under tmp_path/{formula}_searches
    ga_params = get_testing_params()
    minima_by_formula = run_scgo_campaign_one_element(
        "Pt", 4, 6, params=ga_params, seed=42, verbosity=0, output_dir=tmp_path
    )
    assert minima_by_formula, "Expected GA campaign to produce minima"

    # Sanity-check DB files were created under the GA output directory
    clear_registry_cache()
    for comp in compositions:
        formula = get_cluster_formula(comp)
        search_dir = tmp_path / f"{formula}_searches"
        db_files = list(search_dir.rglob("*.db"))
        assert db_files, f"no .db files under {search_dir} (expected GA output)"

    # Run TS search per-composition reusing the GA DBs (cap 10 pairs)
    results = {}
    for comp in compositions:
        formula = get_cluster_formula(comp)
        base_dir = tmp_path / f"{formula}_searches"
        res = run_transition_state_search(
            comp,
            base_dir=base_dir,
            params={"calculator": "EMT"},
            verbosity=0,
            seed=42,
            max_pairs=10,
            neb_n_images=3,
            neb_fmax=0.3,
            neb_steps=10,
        )
        results[formula] = res

    # Verify results and outputs
    for comp in compositions:
        formula = get_cluster_formula(comp)
        assert formula in results
        assert isinstance(results[formula], list)
        assert 0 < len(results[formula]) <= 10

        final_dir = (
            tmp_path
            / f"{formula}_searches"
            / f"ts_results_{formula}"
            / "final_unique_ts"
        )
        assert final_dir.exists()
        summary_file = final_dir / f"final_unique_ts_summary_{formula}.json"
        assert summary_file.exists()
