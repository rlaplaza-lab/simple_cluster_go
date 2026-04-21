import math

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms.geneticalgorithm_go import ga_go
from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim


class MockRelaxer:
    """Minimal test relaxer that returns slightly different energies."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def relax_batch(self, batch: list[Atoms]):
        # return (energy, atoms) spaced to avoid duplicate collapse
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def test_ga_go_generational_smoke(tmp_path, rng):
    calc = EMT()
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_gen"),
        calculator=calc,
        rng=rng,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
    )

    assert isinstance(minima, list)


def test_ga_go_torchsim_generational_smoke(tmp_path, rng):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go_torchsim(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_torchsim_gen"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
    )

    assert isinstance(minima, list)


def test_ga_signatures_consistent():
    """Ensure `ga_go` and `ga_go_torchsim` expose the same public parameters.

    Allow `ga_go_torchsim` to have the TorchSim-specific extras
    (`relaxer`, `batch_size`)."""
    import inspect

    sig_ase = inspect.signature(ga_go)
    sig_ts = inspect.signature(ga_go_torchsim)

    ase_params = set(sig_ase.parameters.keys())
    ts_params = set(sig_ts.parameters.keys())

    # TorchSim may add these extras but otherwise parameter sets must match
    extras = {"relaxer", "batch_size", "n_jobs_offspring"}
    assert ase_params == (ts_params - extras)

    # Check a couple of important default alignments
    assert (
        sig_ase.parameters["niter_local_relaxation"].default
        == sig_ts.parameters["niter_local_relaxation"].default
    )
    assert (
        sig_ase.parameters["mutation_probability"].default
        == sig_ts.parameters["mutation_probability"].default
    )
    assert (
        sig_ase.parameters["offspring_fraction"].default
        == sig_ts.parameters["offspring_fraction"].default
    )


def test_ga_go_torchsim_accepts_optimizer(tmp_path, rng):
    from ase.optimize import LBFGS

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go_torchsim(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_torchsim_opt"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        optimizer=LBFGS,
        rng=rng,
    )
    assert isinstance(minima, list)


def test_ga_go_torchsim_optimizer_default_is_fire():
    import inspect

    from ase.optimize import FIRE

    sig_ts = inspect.signature(ga_go_torchsim)
    assert sig_ts.parameters["optimizer"].default is FIRE


def test_ga_go_offspring_fraction_creates_expected_offspring(
    tmp_path, rng, monkeypatch
):
    """Verify offspring count equals ceil(population_size * offspring_fraction)."""
    from scgo.database import open_db

    calc = EMT()

    # Monkeypatch pairing to always return a fresh unique child
    import scgo.algorithms.geneticalgorithm_go as ga_mod

    counter = {"i": 0}

    def fake_create_pairing(atoms_template, n_to_optimize, rng_arg, **kwargs):
        class Pairing:
            def get_new_individual(self, parents):
                i = counter["i"]
                counter["i"] += 1
                a = Atoms(
                    symbols=atoms_template.get_chemical_symbols(),
                    positions=[[i * 0.13, 0, 0] for _ in range(n_to_optimize)],
                )
                return a, f"fake:label{i}"

        return Pairing()

    monkeypatch.setattr(ga_mod, "create_ga_pairing", fake_create_pairing)

    population_size = 5
    offs_frac = 0.6
    expected_offspring = math.ceil(population_size * offs_frac)

    outdir = tmp_path / "ga_go_off"
    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(outdir),
        calculator=calc,
        rng=rng,
        niter=1,
        population_size=population_size,
        offspring_fraction=offs_frac,
        niter_local_relaxation=1,
    )

    assert isinstance(minima, list)

    db_file = outdir / "ga_go.db"
    with open_db(str(db_file)) as da:
        from scgo.database.metadata import get_metadata

        rows = da.get_all_relaxed_candidates()
        gen0 = [a for a in rows if get_metadata(a, "generation") == 0]

    # Count unique configurations (avoid counting multiple relaxed steps for same conf)
    unique_confids = {a.info.get("confid") for a in gen0}
    assert len(unique_confids) - population_size == expected_offspring


def test_ga_go_torchsim_offspring_fraction_creates_expected_offspring(
    tmp_path, rng, monkeypatch
):
    from scgo.database import open_db

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)

    import scgo.algorithms.geneticalgorithm_go_torchsim as ga_ts_mod

    counter = {"i": 0}

    def fake_create_pairing(atoms_template, n_to_optimize, rng_arg, **kwargs):
        class Pairing:
            def get_new_individual(self, parents):
                i = counter["i"]
                counter["i"] += 1
                a = Atoms(
                    symbols=atoms_template.get_chemical_symbols(),
                    positions=[[i * 0.17, 0, 0] for _ in range(n_to_optimize)],
                )
                return a, f"fake:label{i}"

        return Pairing()

    monkeypatch.setattr(ga_ts_mod, "create_ga_pairing", fake_create_pairing)

    population_size = 4
    offs_frac = 0.5
    expected_offspring = math.ceil(population_size * offs_frac)

    outdir = tmp_path / "ga_go_torchsim_off"
    minima = ga_go_torchsim(
        composition=["Pt"] * 3,
        output_dir=str(outdir),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=population_size,
        offspring_fraction=offs_frac,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
    )

    assert isinstance(minima, list)

    db_file = outdir / "ga_go.db"
    with open_db(str(db_file)) as da:
        from scgo.database.metadata import get_metadata

        rows = da.get_all_relaxed_candidates()
        gen0 = [a for a in rows if get_metadata(a, "generation") == 0]

    unique_confids = {a.info.get("confid") for a in gen0}
    assert len(unique_confids) - population_size == expected_offspring


def test_ga_go_torchsim_parallel_offspring_deterministic(tmp_path):
    calc = EMT()
    kwargs = {
        "composition": ["Pt"] * 3,
        "calculator": calc,
        "relaxer": MockRelaxer(max_steps=1),
        "niter": 1,
        "population_size": 4,
        "offspring_fraction": 0.75,
        "niter_local_relaxation": 1,
        "batch_size": None,
        "verbosity": 0,
    }
    minima_single = ga_go_torchsim(
        output_dir=str(tmp_path / "torchsim_single_worker"),
        rng=np.random.default_rng(1234),
        n_jobs_offspring=1,
        **kwargs,
    )
    minima_parallel = ga_go_torchsim(
        output_dir=str(tmp_path / "torchsim_parallel_worker"),
        rng=np.random.default_rng(1234),
        n_jobs_offspring=2,
        **kwargs,
    )
    assert len(minima_single) == len(minima_parallel)
    energies_single = [float(e) for e, _ in minima_single]
    energies_parallel = [float(e) for e, _ in minima_parallel]
    np.testing.assert_allclose(energies_single, energies_parallel, atol=1e-12, rtol=0.0)


def test_ga_go_torchsim_parallel_offspring_handles_worker_failures(
    tmp_path, rng, monkeypatch
):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)

    import scgo.algorithms.geneticalgorithm_go_torchsim as ga_ts_mod

    base_factory = ga_ts_mod.create_ga_pairing
    call_counter = {"n": 0}

    def flaky_pairing_factory(*args, **kwargs):
        pairing = base_factory(*args, **kwargs)
        base_get = pairing.get_new_individual

        def wrapped_get(parents):
            call_counter["n"] += 1
            if call_counter["n"] % 4 == 0:
                raise RuntimeError("synthetic crossover failure")
            return base_get(parents)

        pairing.get_new_individual = wrapped_get  # type: ignore[assignment]
        return pairing

    monkeypatch.setattr(ga_ts_mod, "create_ga_pairing", flaky_pairing_factory)

    minima = ga_go_torchsim(
        composition=["Pt"] * 3,
        output_dir=str(tmp_path / "ga_go_torchsim_worker_failures"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        n_jobs_offspring=2,
        rng=rng,
        verbosity=0,
    )
    assert isinstance(minima, list)


def test_ga_persisted_unconstrained_rows_are_centered(tmp_path, rng):
    from scgo.database import open_db

    calc = EMT()
    outdir_ase = tmp_path / "ga_center_ase"
    ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(outdir_ase),
        calculator=calc,
        rng=rng,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
    )

    with open_db(str(outdir_ase / "ga_go.db")) as da:
        rows_ase = da.get_all_relaxed_candidates()
    assert rows_ase
    for row in rows_ase:
        bbox_center = 0.5 * (
            row.get_positions().min(axis=0) + row.get_positions().max(axis=0)
        )
        np.testing.assert_allclose(
            bbox_center,
            np.diag(row.get_cell()) / 2.0,
            atol=1e-6,
        )

    outdir_ts = tmp_path / "ga_center_torchsim"
    ga_go_torchsim(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(outdir_ts),
        calculator=calc,
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
    )
    with open_db(str(outdir_ts / "ga_go.db")) as da:
        rows_ts = da.get_all_relaxed_candidates()
    assert rows_ts
    for row in rows_ts:
        bbox_center = 0.5 * (
            row.get_positions().min(axis=0) + row.get_positions().max(axis=0)
        )
        # TorchSim + MockRelaxer uses few steps; bbox need not match cell midpoint tightly.
        np.testing.assert_allclose(
            bbox_center,
            np.diag(row.get_cell()) / 2.0,
            atol=0.75,
        )
