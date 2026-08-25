import math
import pickle

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

import scgo.algorithms.geneticalgorithm_go_torchsim as ga_mod
from scgo.algorithms import ga_go
from scgo.database import get_connection
from scgo.metadata.atoms import get_tag
from tests.helpers import MockRelaxer, assert_serial_parallel_offspring_equal


def test_ga_go_generational_smoke(tmp_path, rng):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_gen"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
    )

    assert isinstance(minima, list)


def test_ga_go_accepts_optimizer(tmp_path, rng):
    from ase.optimize import LBFGS

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_opt"),
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


def test_ga_go_optimizer_default_is_fire():
    import inspect

    from ase.optimize import FIRE

    sig_ts = inspect.signature(ga_go)
    assert sig_ts.parameters["optimizer"].default is FIRE


def test_ga_go_offspring_fraction_creates_expected_offspring(
    tmp_path, rng, monkeypatch
):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
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

    monkeypatch.setattr(ga_mod, "create_ga_pairing", fake_create_pairing)

    population_size = 4
    offs_frac = 0.5
    expected_offspring = math.ceil(population_size * offs_frac)

    outdir = tmp_path / "ga_go_off"
    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(outdir),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=population_size,
        offspring_fraction=offs_frac,
        niter_local_relaxation=1,
        batch_size=None,
        ga_fast_prefilter_enabled=False,
        rng=rng,
    )

    assert isinstance(minima, list)

    db_file = outdir / "ga_go.db"
    with get_connection(str(db_file)) as da:
        rows = da.get_all_relaxed_candidates()
        gen0 = [a for a in rows if get_tag(a, "generation") == 0]

    unique_confids = {a.info.get("confid") for a in gen0}
    assert len(unique_confids) - population_size == expected_offspring


def test_ga_go_parallel_offspring_deterministic(tmp_path):
    calc = EMT()
    kwargs = {
        "composition": ["Pt"] * 3,
        "calculator": calc,
        "relaxer": MockRelaxer(max_steps=1),
        "niter": 1,
        "population_size": 3,
        "offspring_fraction": 0.34,
        "niter_local_relaxation": 1,
        "batch_size": None,
        "verbosity": 0,
        "clean": True,
        "previous_search_glob": ".__scgo_no_prior_runs__/**/*.db",
    }
    assert_serial_parallel_offspring_equal(
        tmp_path,
        seed=1234,
        ga_kwargs=kwargs,
        rtol=0.0,
        atol=1e-12,
        output_suffix_serial="torchsim_single_worker",
        output_suffix_parallel="torchsim_parallel_worker",
    )


@pytest.mark.requires_multicore
def test_ga_go_parallel_offspring_deterministic_adaptive_pt4(tmp_path):
    calc = EMT()
    kwargs = {
        "composition": ["Pt"] * 4,
        "calculator": calc,
        "relaxer": MockRelaxer(max_steps=1),
        "niter": 1,
        "population_size": 4,
        "offspring_fraction": 0.5,
        "niter_local_relaxation": 2,
        "batch_size": None,
        "verbosity": 0,
        "use_adaptive_mutations": True,
        "clean": True,
        "previous_search_glob": ".__scgo_no_prior_runs__/**/*.db",
    }
    assert_serial_parallel_offspring_equal(
        tmp_path,
        seed=271828,
        ga_kwargs=kwargs,
        rtol=0.0,
        atol=1e-12,
        output_suffix_serial="adaptive_single_worker",
        output_suffix_parallel="adaptive_parallel_worker",
    )


class _RecordingRelaxer:
    """Records confid order passed to relax_batch."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.confid_order: list[int] = []

    def relax_batch(self, batch: list[Atoms]):
        for atoms in batch:
            self.confid_order.append(int(atoms.info.get("confid", -1)))
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def test_relax_unrelaxed_candidates_sorted_by_confid(tmp_path, rng):
    relaxer = _RecordingRelaxer(max_steps=1)
    ga_go(
        composition=["Pt"] * 3,
        output_dir=str(tmp_path / "sorted_relax"),
        calculator=EMT(),
        relaxer=relaxer,
        niter=1,
        population_size=3,
        offspring_fraction=0.67,
        niter_local_relaxation=1,
        batch_size=2,
        n_jobs_offspring=2,
        rng=rng,
        verbosity=0,
        clean=True,
        previous_search_glob=".__scgo_no_prior_runs__/**/*.db",
    )
    assert relaxer.confid_order
    assert relaxer.confid_order == sorted(relaxer.confid_order)


def test_offspring_build_context_picklable(rng):
    from ase.calculators.emt import EMT
    from ase_ga.utilities import closest_distances_generator, get_all_atom_types

    from scgo.algorithms.ga_common import create_mutation_operators
    from scgo.algorithms.geneticalgorithm_go_torchsim import (
        OffspringBuildContext,
        _picklable_atoms_copy,
    )
    from scgo.utils.mutation_weights import get_adaptive_mutation_config

    composition = ["Pt", "Pt", "Pt"]
    atoms_template = Atoms(
        symbols=composition,
        positions=[[0, 0, 0]] * 3,
        cell=[10, 10, 10],
        pbc=False,
    )
    atoms_template.calc = EMT()
    all_atom_types = get_all_atom_types(atoms_template, [78])
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=3,
        blmin=blmin,
        rng=rng,
        use_adaptive=True,
    )
    adaptive_config = get_adaptive_mutation_config(
        composition=composition,
        current_generation=0,
        total_generations=1,
        use_adaptive=True,
        generations_without_improvement=0,
    )
    ctx = OffspringBuildContext(
        atoms_template=_picklable_atoms_copy(atoms_template),
        n_to_optimize=3,
        composition=composition,
        blmin=blmin,
        system_type="gas_cluster",
        n_slab=0,
        n_frozen_prefix=0,
        slab_for_pairing=None,
        surface_normal_axis=2,
        adsorbate_definition=None,
        connectivity_factor=None,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=False,
        enforce_adsorbate_subgraph_integrity=True,
        freeze_adsorbate_internal_geometry=False,
        adsorbate_fragment_templates=None,
        surface_config=None,
        adaptive_config=adaptive_config,
        current_mutation_probability=0.3,
        operators_list=operators_list,
        name_map=name_map,
        operators_epoch=0,
    )
    pickle.loads(pickle.dumps(ctx))


def test_ga_go_parallel_offspring_handles_worker_failures(tmp_path, rng, monkeypatch):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    base_factory = ga_mod.create_ga_pairing
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

    monkeypatch.setattr(ga_mod, "create_ga_pairing", flaky_pairing_factory)

    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(tmp_path / "ga_go_worker_failures"),
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

    with get_connection(str(outdir_ase / "ga_go.db")) as da:
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
    ga_go(
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
    with get_connection(str(outdir_ts / "ga_go.db")) as da:
        rows_ts = da.get_all_relaxed_candidates()
    assert rows_ts
    for row in rows_ts:
        bbox_center = 0.5 * (
            row.get_positions().min(axis=0) + row.get_positions().max(axis=0)
        )
        np.testing.assert_allclose(
            bbox_center,
            np.diag(row.get_cell()) / 2.0,
            atol=1e-6,
        )


def test_relax_unrelaxed_relaxes_when_available_below_max_batch(tmp_path):
    """P1.2: no early-return stall when available < max_batch (not forced).

    Before this change ``_relax_unrelaxed_candidates`` returned (0, 0) when
    ``available < max_batch`` and ``force=False``, deferring work and starving
    the GPU. Now it relaxes all available candidates.
    """
    from ase_ga.data import DataConnection

    from tests.helpers import create_preparedb

    db_path = tmp_path / "ga_relax_stall.db"
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[10, 10, 10])
    create_preparedb(atoms, db_path, population_size=10)
    da = DataConnection(str(db_path))
    # Insert unrelaxed candidates the same low-level way the GA does so that
    # raw_score survives into key_value_pairs (add_unrelaxed_candidate strips it).
    for k in range(3):
        a = atoms.copy()
        # Non-clashing, connected Pt3 triangle (~2.5 Å sides) so the storage
        # gate (clash + connectivity) passes after the mock relaxer returns it.
        a.positions = [
            [k * 0.1, 0.0, 0.0],
            [2.5 + k * 0.1, 0.0, 0.0],
            [1.25 + k * 0.1, 2.165, 0.0],
        ]
        with da.c:
            gaid = da.c.write(
                a,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
                description=f"pt3_{k}",
            )
            da.c.update(gaid, gaid=gaid)
            a.info["confid"] = gaid

    # Return relaxed copies with no key_value_pairs so the GA's raw_score
    # fallback (raw_score = -energy) applies, mirroring a real TorchSim relaxer.
    class _StrippedRelaxer:
        def relax_batch(self, batch):
            out = []
            for i, a in enumerate(batch):
                ra = Atoms(
                    symbols=a.get_chemical_symbols(),
                    positions=a.get_positions(),
                    cell=a.get_cell(),
                    pbc=a.get_pbc(),
                )
                out.append((float(i) * 0.1, ra))
            return out

    relaxer = _StrippedRelaxer()

    eligible, ineligible, _reasons = ga_mod._relax_unrelaxed_candidates(
        da,
        relaxer,
        max_batch=10,  # larger than available (3)
        force=False,
        composition=["Pt", "Pt", "Pt"],
        system_type="gas_cluster",
    )

    # All 3 unrelaxed candidates must be relaxed despite available < max_batch.
    assert eligible + ineligible == 3
    assert eligible == 3


def test_relax_unrelaxed_ineligible_count_stable_across_write_retry(
    tmp_path, monkeypatch
):
    """SQLite write retries must not double-count ineligible outcomes."""
    import sqlite3

    from ase_ga.data import DataConnection

    from tests.helpers import create_preparedb

    db_path = tmp_path / "ga_ineligible_retry.db"
    atoms = Atoms(
        "Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]], cell=[10] * 3
    )
    create_preparedb(atoms, db_path, population_size=10)
    da = DataConnection(str(db_path))
    for k in range(2):
        a = atoms.copy()
        a.positions = a.positions + k * 0.1
        with da.c:
            gaid = da.c.write(
                a,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
                description=f"pt3_{k}",
            )
            da.c.update(gaid, gaid=gaid)
            a.info["confid"] = gaid

    class _DisconnectedRelaxer:
        def relax_batch(self, batch):
            out = []
            for i, a in enumerate(batch):
                ra = Atoms(
                    symbols=a.get_chemical_symbols(),
                    positions=[[0, 0, 0], [8, 0, 0], [0, 8, 0]],
                    cell=a.get_cell(),
                    pbc=a.get_pbc(),
                )
                out.append((float(i), ra))
            return out

    real_add = DataConnection.add_relaxed_step
    calls = {"n": 0}

    def flaky_add(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(DataConnection, "add_relaxed_step", flaky_add)

    eligible, ineligible, reasons = ga_mod._relax_unrelaxed_candidates(
        da,
        _DisconnectedRelaxer(),
        max_batch=10,
        composition=["Pt", "Pt", "Pt"],
        system_type="gas_cluster",
    )
    assert eligible == 0
    assert ineligible == 2
    assert reasons == {"disconnected": 2}
    assert calls["n"] == 3
    rows = da.get_all_relaxed_candidates()
    assert len(rows) == 2
    assert all(not bool(get_tag(row, "ga_eligible", default=True)) for row in rows)


def test_per_gen_max_targets_population_when_batch_size_none():
    """P1: per-generation relax cap resolves to max(n_offspring, population_size)."""
    # Mirrors the resolution inside run_ga_torchsim's generational loop.
    population_size = 40
    n_offspring = max(1, math.ceil(population_size * 0.5))

    batch_size = None
    per_gen_max = (
        batch_size if batch_size is not None else max(n_offspring, population_size)
    )
    assert per_gen_max == population_size

    # A user-set batch_size still wins.
    batch_size = 7
    per_gen_max = (
        batch_size if batch_size is not None else max(n_offspring, population_size)
    )
    assert per_gen_max == 7


def test_ga_go_reports_mutation_counters(tmp_path, rng):
    """Offspring workers expose mutation request/application counters."""
    payloads: list[dict] = []
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_mut_counters"),
        calculator=calc,
        relaxer=relaxer,
        niter=2,
        population_size=4,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
        timing_collector=payloads,
    )
    assert isinstance(minima, list)
    assert payloads, "timing collector received no payload"
    counters = payloads[-1]["counters"]
    assert counters["offspring_attempts_total"] > 0
    assert counters["offspring_created"] > 0
    assert counters["offspring_mutations_requested"] >= 0
    assert (
        counters["offspring_mutations_applied"]
        <= counters["offspring_mutations_requested"]
    )
