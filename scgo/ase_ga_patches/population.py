"""Implementation of a population for maintaining a GA population and
proposing structures to pair.
"""

# fmt: off

from __future__ import annotations

from math import sqrt, tanh

import numpy as np
from ase.db.core import now

from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import get_tag
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    calculate_fitness,
    set_fitness_in_atoms,
    validate_fitness_strategy,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _raw_score(a):
    """Return GA raw_score from structure tags.

    Raises:
        SCGOValidationError: If the candidate carries no ``raw_score`` tag.
    """
    raw = get_tag(a, "raw_score", default=None)
    if raw is None:
        raise SCGOValidationError(
            "Population candidate missing raw_score in key_value_pairs",
        )
    return float(raw)


def _population_candidate_sort_key(a):
    """Stable ordering for population candidates with tied raw_score."""
    return (
        -_raw_score(a),
        a.info.get("relax_id", 0),
        a.info.get("confid", 0),
    )


def count_looks_like(a, all_cand, comp):
    """Utility method for counting occurrences."""
    n = 0
    for b in all_cand:
        if a.info.get("confid") == b.info.get("confid"):
            continue
        if comp.looks_like(a, b):
            n += 1
    return n


from scgo.utils.rng_helpers import ensure_rng_or_create


class Population:
    """Population class which maintains the current population
    and proposes which candidates to pair together.

    Parameters
    ----------
    data_connection: DataConnection object
        ASE database connection for reading and writing candidates.

    population_size: int
        The number of candidates in the population.

    comparator: Comparator object
        this will tell if two configurations are equal.
        Default compare atoms objects directly.

    logfile: str
        Text file that contains information about the population
        The format is::

            timestamp: generation(if available): id1,id2,id3...

        Using this file greatly speeds up convergence checks.
        Default None meaning that no file is written.

    use_extinct: boolean
        Set this to True if mass extinction and the extinct key
        are going to be used. Default is False.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    elite_fraction: float
        Fraction of the population protected from replacement, counted
        from the best raw_score downwards.

    run_id: str or None
        When given, only candidates tagged with this run_id are considered.

    """

    def __init__(self, data_connection, population_size,
                 comparator=None, logfile=None, use_extinct=False,
                 rng=None, elite_fraction=0.1, run_id: str | None = None):
        self.dc = data_connection
        self.population_size = population_size
        if comparator is None:
            from ase_ga.standard_comparators import AtomsComparator
            comparator = AtomsComparator()
        self.comparator = comparator
        self.logfile = logfile
        self.use_extinct = use_extinct
        self.rng = ensure_rng_or_create(rng)
        self.elite_fraction = elite_fraction
        self.elite_size = max(1, int(self.population_size * self.elite_fraction))
        self.run_id = run_id
        self.pop = []
        self.pairs = None
        self.all_cand = None
        self.__initialize_pop__()

    def _filter_candidates_by_run_id(self, candidates):
        if self.run_id is None:
            return candidates
        return [
            cand
            for cand in candidates
            if get_tag(cand, "run_id", default=None) == self.run_id
        ]

    def _filter_candidates_by_ga_eligibility(self, candidates):
        """Exclude candidates explicitly marked as ineligible for GA evolution."""
        return [
            cand
            for cand in candidates
            if bool(get_tag(cand, "ga_eligible", default=True))
        ]

    def _get_all_relaxed_candidates(self, *, only_new=False, use_extinct=False):
        candidates = self.dc.get_all_relaxed_candidates(
            only_new=only_new,
            use_extinct=use_extinct,
        )
        candidates = self._filter_candidates_by_run_id(candidates)
        return self._filter_candidates_by_ga_eligibility(candidates)

    def __initialize_pop__(self):
        """Private method that initializes the population when
        the population is created.
        """
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self._get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=_population_candidate_sort_key)

        # Fill the population with the fittest unique candidates.
        # Each new candidate is checked only against the already-accepted pop
        # (O(pop) per candidate).  When a duplicate is found the matching
        # pop member's rediscovery counter is incremented in-place so the
        # full O(history) rescan below is no longer needed.
        i = 0
        while i < len(all_cand) and len(self.pop) < self.population_size:
            c = all_cand[i]
            i += 1
            duplicate_of = None
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    duplicate_of = a
                    break
            if duplicate_of is None:
                c.info["looks_like"] = 0
                self.pop.append(c)
            else:
                duplicate_of.info["looks_like"] = (
                    int(duplicate_of.info.get("looks_like", 0)) + 1
                )

        # Any remaining all_cand (beyond population_size) that were never
        # checked in the loop above are also counted as duplicates only if
        # they are identical to a pop member.  These come after the worst
        # pop member in score order so they would not have been accepted.
        # Count them to keep looks_like consistent with the historical meaning.
        while i < len(all_cand):
            c = all_cand[i]
            i += 1
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    a.info["looks_like"] = int(a.info.get("looks_like", 0)) + 1
                    break

        self.all_cand = all_cand
        self.__calc_participation__()

    def __calc_participation__(self):
        """Determines, from the database, how many times each
        candidate has been used to generate new candidates.
        """
        (participation, pairs) = self.dc.get_participation_in_pairing()
        for a in self.pop:
            if a.info.get("confid") in participation:
                a.info["n_paired"] = participation[a.info.get("confid")]
            else:
                a.info["n_paired"] = 0
        self.pairs = pairs

    def update(self, new_cand=None):
        """New candidates can be added to the database
        after the population object has been created.
        This method extracts these new candidates from the
        database and includes them in the population.

        When ``new_cand`` is provided the database is not read; the caller
        supplies already-in-memory Atoms (e.g. from a just-completed relax
        batch).  Passing an empty list is valid and skips the round-trip
        entirely.  ``None`` falls back to the database path.
        """
        if len(self.pop) == 0 and new_cand is None:
            self.__initialize_pop__()

        if new_cand is None:
            ue = self.use_extinct
            new_cand = self._get_all_relaxed_candidates(only_new=True, use_extinct=ue)
        else:
            new_cand = self._filter_candidates_by_run_id(new_cand)
            new_cand = self._filter_candidates_by_ga_eligibility(new_cand)
            # Sync already_returned so a later update(new_cand=None) call
            # (e.g. from get_two_candidates) does not re-fetch these gaids.
            already = getattr(self.dc, "already_returned", None)
            if already is not None:
                for a in new_cand:
                    gaid = a.info.get("confid")
                    if gaid is not None:
                        already.add(int(gaid))

        new_cand.sort(key=_population_candidate_sort_key)

        for a in new_cand:
            self.__add_candidate__(a)
            self.all_cand.append(a)
        self.__calc_participation__()
        self._write_log()

    def __add_candidate__(self, a):
        """Adds a single candidate to the population."""
        # An empty population accepts the candidate unconditionally.
        if not self.pop:
            a.info["looks_like"] = 0
            self.pop.append(a)
            return

        # check if the structure is too low in raw score
        raw_score_a = _raw_score(a)
        raw_score_worst = _raw_score(self.pop[-1])
        if raw_score_a < raw_score_worst \
                and len(self.pop) == self.population_size:
            return

        # check if the new candidate should
        # replace a similar structure in the population
        for (i, b) in enumerate(self.pop):
            if self.comparator.looks_like(a, b):
                # Replace a duplicate only when the newcomer is strictly better;
                # elites are no exception, otherwise a better copy of the best
                # candidate would be discarded and the population could never
                # improve past it. Ties keep the incumbent.
                if _raw_score(b) < raw_score_a:
                    # Newcomer inherits the incumbent's rediscovery count + 1
                    # so the fitness penalty is not reset on replacement.
                    a.info["looks_like"] = int(b.info.get("looks_like", 0)) + 1
                    del self.pop[i]
                    self.pop.append(a)
                    self.pop.sort(key=_raw_score, reverse=True)
                else:
                    # Incumbent keeps its seat; count this as another rediscovery.
                    b.info["looks_like"] = int(b.info.get("looks_like", 0)) + 1
                return

        # the new candidate needs to be added, so ensure we have room
        # Always keep top elite_size candidates
        if len(self.pop) == self.population_size:
            # Remove the worst candidate to make room (the population is
            # kept sorted by raw_score, best first).
            del self.pop[-1]

        # add the new candidate
        a.info["looks_like"] = 0
        self.pop.append(a)
        self.pop.sort(key=_raw_score, reverse=True)

    def __get_fitness__(self, indecies, with_history=True):
        """Calculates the fitness using the formula from
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816

        Sign change on the fitness compared to the formulation in the
        abovementioned paper due to maximizing raw_score instead of
        minimizing energy. (Set raw_score=-energy to optimize the energy)
        """
        scores = [_raw_score(x) for x in self.pop]
        min_s = min(scores)
        max_s = max(scores)
        T = min_s - max_s
        if isinstance(indecies, int):
            indecies = [indecies]

        if abs(T) < 1e-12:
            # All candidates have the same score; assign uniform fitness
            f = [1.0 for _ in indecies]
        else:
            f = [0.5 * (1. - tanh(2. * (scores[i] - max_s) / T - 1.))
                 for i in indecies]
        if with_history:
            M = [float(self.pop[i].info.get("n_paired", 0)) for i in indecies]
            L = [float(self.pop[i].info.get("looks_like", 0)) for i in indecies]
            f = [f[i] * 1. / sqrt(1. + M[i]) * 1. / sqrt(1. + L[i])
                 for i in range(len(f))]
        return f

    def get_two_candidates(self, with_history=True):
        """Returns two candidates for pairing employing the
        fitness criteria from
        L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        and the roulette wheel selection scheme described in
        R.L. Johnston Dalton Transactions,
        Vol. 22, No. 22. (2003), pp. 4193-4207

        Returns None if the population holds fewer than two candidates.
        """
        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        fit = self.__get_fitness__(range(len(self.pop)), with_history)
        fmax = max(fit)
        if fmax < 1e-12:
            # All fitness values are near-zero; fall back to uniform random selection
            idx = self.rng.choice(len(self.pop), size=2, replace=False)
            return (self.pop[idx[0]].copy(), self.pop[idx[1]].copy())

        # Fitness-proportional selection without replacement: the two indices
        # are distinct by construction, so a confid collision is impossible.
        p = np.clip(np.asarray(fit, dtype=float), 0.0, None)
        p = p / p.sum()
        c1 = self.pop[0]
        c2 = self.pop[1]
        # Bounded retry: only the pairing history can reject a drawn pair.
        for _attempt in range(100):
            idx = self.rng.choice(len(self.pop), size=2, replace=False, p=p)
            c1 = self.pop[int(idx[0])]
            c2 = self.pop[int(idx[1])]
            c1id = c1.info.get("confid")
            c2id = c2.info.get("confid")
            used_before = (min([c1id, c2id]), max([c1id, c2id])) in self.pairs
            if not used_before:
                break
        return (c1.copy(), c2.copy())

    def _write_log(self):
        """Writes the population to a logfile.

        The format is::

            timestamp: generation(if available): id1,id2,id3...
        """
        if self.logfile is not None:
            ids = [str(a.info["relax_id"]) for a in self.pop]
            # Always touch the logfile so it exists even if the population is empty.
            # If we have IDs, write a log entry; otherwise just create the file.
            try:
                gen_nums = [get_tag(c, "generation", default=None)
                            for c in self.all_cand]
                max_gen = max(gen_nums) if gen_nums else " "
            except (TypeError, ValueError):
                logger.debug("Could not determine max generation for logfile entry")
                max_gen = " "
            # Opening in append mode will create the file if it doesn't exist.
            with open(self.logfile, "a") as fd:
                if ids:
                    fd.write(f"{now()}: {max_gen}: {','.join(ids)}\n")


class FitnessStrategyPopulation(Population):
    """Population class with configurable fitness strategies.

    Extends the base Population class to support different fitness calculation
    strategies beyond simple energy minimization, with efficient diversity scoring.

    Parameters
    ----------
    data_connection: DataConnection object
        Database connection for population management.
    population_size: int
        Number of individuals in population.
    fitness_strategy: str
        Strategy name ("low_energy", "high_energy", "diversity").
    diversity_scorer: DiversityScorer or None
        Scorer for diversity calculations. Required for diversity strategy.
    diversity_update_interval: int
        Number of generations between reference updates (for diversity strategy).
    comparator: Comparator object
        Comparator for structure similarity checks.
    logfile: str or None
        Optional log file for population tracking.
    use_extinct: bool
        Whether to use extinct flag for mass extinction.
    rng: Random number generator
        Random number generator for stochastic operations.
    elite_fraction: float
        Fraction of population to preserve as elite (top performers).
    run_id: str or None
        When given, only candidates tagged with this run_id are considered.
    """

    def __init__(
        self,
        data_connection,
        population_size,
        fitness_strategy: str = "low_energy",
        diversity_scorer=None,
        diversity_update_interval: int = 5,
        comparator=None,
        logfile=None,
        use_extinct=False,
        rng=None,
        elite_fraction=0.1,
        run_id: str | None = None,
    ):
        """Initialize fitness strategy population."""
        validate_fitness_strategy(fitness_strategy)
        if isinstance(fitness_strategy, str):
            fitness_strategy = FitnessStrategy(fitness_strategy)
        self.fitness_strategy = fitness_strategy
        self.diversity_scorer = diversity_scorer
        self.diversity_update_interval = diversity_update_interval
        self._generation_count = 0

        # Call parent constructor
        super().__init__(
            data_connection,
            population_size,
            comparator=comparator,
            logfile=logfile,
            use_extinct=use_extinct,
            rng=rng,
            elite_fraction=elite_fraction,
            run_id=run_id,
        )

    def __get_fitness__(self, indecies, with_history=True):
        """Calculate fitness based on configured strategy.

        Args:
            indecies: Index or list of indices to calculate fitness for.
            with_history: If True, include history-based fitness penalties.

        Returns:
            List of fitness values (higher is better).
        """
        if isinstance(indecies, int):
            indecies = [indecies]

        # For low_energy, use parent class implementation
        if self.fitness_strategy == FitnessStrategy.LOW_ENERGY:
            return super().__get_fitness__(indecies, with_history)

        # Calculate fitness for each candidate
        fitness_values = []
        for i in indecies:
            atoms = self.pop[i]
            energy = -_raw_score(atoms)

            # Calculate fitness based on strategy
            fitness = calculate_fitness(
                energy=energy,
                atoms=atoms,
                strategy=self.fitness_strategy,
                diversity_scorer=self.diversity_scorer,
            )

            # Store fitness in atoms for later retrieval
            set_fitness_in_atoms(atoms, fitness, self.fitness_strategy)

            fitness_values.append(fitness)

        # Normalize fitness to 0-1 range for selection probability calculation
        if len(fitness_values) > 1:
            min_f = min(fitness_values)
            max_f = max(fitness_values)
            if max_f > min_f:
                fitness_values = [(f - min_f) / (max_f - min_f) for f in fitness_values]

        # Apply history-based penalties if requested
        if with_history:
            M = [float(self.pop[i].info.get("n_paired", 0)) for i in indecies]
            L = [float(self.pop[i].info.get("looks_like", 0)) for i in indecies]
            fitness_values = [
                fitness_values[j] * 1. / sqrt(1. + M[j]) * 1. / sqrt(1. + L[j])
                for j in range(len(fitness_values))
            ]

        return fitness_values

    def update(self, new_cand=None):
        """Update population and periodically add new references."""
        super().update(new_cand=new_cand)

        # Periodic reference update for diversity strategy
        if (
            self.fitness_strategy == FitnessStrategy.DIVERSITY
            and self.diversity_scorer
        ):
            self._generation_count += 1

            if (
                self._generation_count % self.diversity_update_interval == 0
                and self.pop
            ):
                # Add best new structure to references
                best = min(self.pop, key=lambda x: -_raw_score(x))
                self.diversity_scorer.add_reference(best)

# fmt: on
