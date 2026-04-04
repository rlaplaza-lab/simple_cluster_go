"""Runner emulation integration tests."""

import pytest

from scgo import run_scgo_campaign_one_element
from scgo.calculators.orca_helpers import prepare_orca_calculations
from scgo.database.registry import clear_registry_cache
from scgo.param_presets import get_testing_params
from scgo.ts_search import run_transition_state_search
from scgo.utils.helpers import get_cluster_formula


@pytest.mark.slow
def test_emulate_run_Pt4_6_orca_lowe_and_ts_search(tmp_path):
    """Emulate GA → ORCA → TS workflow and DB reuse (single GA campaign).

    Merges the former split tests that each ran an identical GA campaign.
    """
    out_root = tmp_path / "orca_calcs"
    compositions = [["Pt"] * n for n in range(4, 7)]

    ga_params = get_testing_params()
    minima_by_formula = run_scgo_campaign_one_element(
        "Pt", 4, 6, params=ga_params, seed=42, verbosity=0, output_dir=tmp_path
    )
    assert minima_by_formula, "Expected minima from the GA campaign"

    for formula, minima in minima_by_formula.items():
        prepare_orca_calculations(minima, str(out_root / formula), {})
        formula_dir = out_root / formula
        assert formula_dir.exists()
        min_dir = next(formula_dir.glob("minimum_*_" + formula))
        assert (min_dir / "orca.inp").exists()

    clear_registry_cache()
    for comp in compositions:
        formula = get_cluster_formula(comp)
        search_dir = tmp_path / f"{formula}_searches"
        db_files = list(search_dir.rglob("*.db"))
        assert db_files, f"no .db files under {search_dir} (expected GA output)"

    for comp in compositions:
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
        assert 0 < len(results) <= 10

        final_dir = base_dir / f"ts_results_{formula}" / "final_unique_ts"
        assert final_dir.exists()
        summary_file = final_dir / f"final_unique_ts_summary_{formula}.json"
        assert summary_file.exists()
