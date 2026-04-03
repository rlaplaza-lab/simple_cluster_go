from ase.calculators.emt import EMT

from scgo.algorithms.geneticalgorithm_go_torchsim import ga_go_torchsim


class MockRelaxer:
    """Minimal test relaxer that returns slightly different energies."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def relax_batch(self, batch: list):
        # return (energy, atoms) spaced to avoid duplicate collapse
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def test_ga_go_torchsim_offspring_logging_levels(tmp_path, rng, capfd):
    """Debug (2) and Trace (3) show per-generation summaries, not per-offspring spam."""
    from scgo.utils.logging import configure_logging

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    outdir_debug = tmp_path / "ga_go_torchsim_log_debug"

    # Run in debug mode: should contain one concise summary per generation but
    # not the per-offspring detailed lines.
    configure_logging(2)
    minima = ga_go_torchsim(
        composition=["Pt"] * 3,
        output_dir=str(outdir_debug),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
        verbosity=2,
    )
    assert isinstance(minima, list)
    captured = capfd.readouterr().out
    assert "Generation 0 offspring loop: n_offspring=" in captured
    assert "Selected parents for pairing" not in captured
    assert "Pairing produced child" not in captured
    assert "Queued unrelaxed candidate for generation" not in captured

    # Run in trace mode: detailed per-offspring messages should be present.
    configure_logging(3)
    outdir_trace = tmp_path / "ga_go_torchsim_log_trace"
    minima = ga_go_torchsim(
        composition=["Pt"] * 3,
        output_dir=str(outdir_trace),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
        verbosity=3,
    )
    assert isinstance(minima, list)
    captured2 = capfd.readouterr().out
    # Trace enables DEBUG+; we still expect the per-generation summary (not per-offspring spam).
    assert "Generation 0 offspring loop: n_offspring=" in captured2
    assert "Selected parents for pairing" not in captured2
    assert "Pairing produced child" not in captured2
