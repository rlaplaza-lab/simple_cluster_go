# fmt: off

from __future__ import annotations

"""Implementation of a population for maintaining a GA population and
proposing structures to pair.
"""
from math import sqrt, tanh

import numpy as np
from ase.db.core import now

from scgo.database.metadata import get_metadata
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    calculate_fitness,
    set_fitness_in_atoms,
    validate_fitness_strategy,
)


def _raw_score(a):
    """Return GA raw_score from SCGO metadata or ASE ``key_value_pairs``."""
    raw = get_metadata(a, "raw_score", default=None)
    if raw is None:
        raise ValueError(
            "Population candidate missing raw_score in metadata or key_value_pairs",
        )
    return float(raw)


def count_looks_like(a, all_cand, comp):
    """Utility method for counting occurrences."""
    n = 0
    for b in all_cand:
        if a.info["confid"] == b.info["confid"]:
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
            if get_metadata(cand, "run_id", default=None) == self.run_id
        ]

    def _get_all_relaxed_candidates(self, *, only_new=False, use_extinct=False):
        candidates = self.dc.get_all_relaxed_candidates(
            only_new=only_new,
            use_extinct=use_extinct,
        )
        return self._filter_candidates_by_run_id(candidates)

    def __initialize_pop__(self):
        """Private method that initializes the population when
        the population is created.
        """
        # Get all relaxed candidates from the database
        ue = self.use_extinct
        all_cand = self._get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=_raw_score, reverse=True)

        # Fill up the population with the fittest unique candidates.
        i = 0
        while i < len(all_cand) and len(self.pop) < self.population_size:
            c = all_cand[i]
            i += 1
            eq = False
            for a in self.pop:
                if self.comparator.looks_like(a, c):
                    eq = True
                    break
            if not eq:
                self.pop.append(c)

        for a in self.pop:
            a.info["looks_like"] = count_looks_like(a, all_cand,
                                                    self.comparator)

        self.all_cand = all_cand
        self.__calc_participation__()

    def __calc_participation__(self):
        """Determines, from the database, how many times each
        candidate has been used to generate new candidates.
        """
        (participation, pairs) = self.dc.get_participation_in_pairing()
        for a in self.pop:
            if a.info["confid"] in participation:
                a.info["n_paired"] = participation[a.info["confid"]]
            else:
                a.info["n_paired"] = 0
        self.pairs = pairs

    def update(self, new_cand=None):
        """New candidates can be added to the database
        after the population object has been created.
        This method extracts these new candidates from the
        database and includes them in the population.
        """
        if len(self.pop) == 0:
            self.__initialize_pop__()

        if new_cand is None:
            ue = self.use_extinct
            new_cand = self._get_all_relaxed_candidates(only_new=True, use_extinct=ue)

        for a in new_cand:
            self.__add_candidate__(a)
            self.all_cand.append(a)
        self.__calc_participation__()
        self._write_log()

    def get_current_population(self):
        """Returns a copy of the current population."""
        self.update()
        return [a.copy() for a in self.pop]

    def get_population_after_generation(self, gen):
        """Returns a copy of the population as it where
        after generation gen
        """
        if self.logfile is not None:
            with open(self.logfile) as fd:
                gens = {}
                for line in fd:
                    _, no, popul = line.split(":")
                    gens[int(no)] = [int(i) for i in popul.split(",")]
            return [c.copy() for c in self.all_cand[::-1]
                    if c.info["relax_id"] in gens[gen]]

        all_candidates = [c for c in self.all_cand
                          if get_metadata(c, "generation", default=float("inf")) <= gen]
        cands = [all_candidates[0]]
        for b in all_candidates:
            if b not in cands:
                for a in cands:
                    if self.comparator.looks_like(a, b):
                        break
                else:
                    cands.append(b)
        pop = cands[: self.population_size]
        return [a.copy() for a in pop]

    def __add_candidate__(self, a):
        """Adds a single candidate to the population."""
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
                if _raw_score(b) < raw_score_a:
                    # Only replace if the structure being removed is not elite
                    # Elite candidates are the top elite_size by raw_score
                    if i < self.elite_size:
                        # Trying to replace an elite candidate - keep the elite
                        return
                    del self.pop[i]
                    a.info["looks_like"] = count_looks_like(a,
                                                            self.all_cand,
                                                            self.comparator)
                    self.pop.append(a)
                    self.pop.sort(key=_raw_score, reverse=True)
                return

        # the new candidate needs to be added, so ensure we have room
        # Always keep top elite_size candidates
        if len(self.pop) == self.population_size:
            # Remove worst candidate to make room (it can't be elite since population is sorted)
            del self.pop[-1]

        # add the new candidate
        a.info["looks_like"] = count_looks_like(a,
                                                self.all_cand,
                                                self.comparator)
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

        f = [0.5 * (1. - tanh(2. * (scores[i] - max_s) / T - 1.))
             for i in indecies]
        if with_history:
            M = [float(self.pop[i].info["n_paired"]) for i in indecies]
            L = [float(self.pop[i].info["looks_like"]) for i in indecies]
            f = [f[i] * 1. / sqrt(1. + M[i]) * 1. / sqrt(1. + L[i])
                 for i in range(len(f))]
        return f

    def get_two_candidates(self, with_history=True):
        """Returns two candidates for pairing employing the
        fitness criteria from
        L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        and the roulete wheel selection scheme described in
        R.L. Johnston Dalton Transactions,
        Vol. 22, No. 22. (2003), pp. 4193-4207
        """
        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        fit = self.__get_fitness__(range(len(self.pop)), with_history)
        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        used_before = False
        while c1.info["confid"] == c2.info["confid"] and not used_before:
            nnf = True
            while nnf:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c2 = self.pop[t]
                    nnf = False

            c1id = c1.info["confid"]
            c2id = c2.info["confid"]
            used_before = (min([c1id, c2id]), max([c1id, c2id])) in self.pairs
        return (c1.copy(), c2.copy())

    def get_one_candidate(self, with_history=True):
        """Returns one candidate for mutation employing the
        fitness criteria from
        L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
        and the roulete wheel selection scheme described in
        R.L. Johnston Dalton Transactions,
        Vol. 22, No. 22. (2003), pp. 4193-4207
        """
        if len(self.pop) < 1:
            self.update()

        if len(self.pop) < 1:
            return None

        fit = self.__get_fitness__(range(len(self.pop)), with_history)
        fmax = max(fit)
        nnf = True
        while nnf:
            t = self.rng.integers(len(self.pop))
            if fit[t] > self.rng.random() * fmax:
                c1 = self.pop[t]
                nnf = False

        return c1.copy()

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
                gen_nums = [get_metadata(c, "generation", default=None)
                            for c in self.all_cand]
                max_gen = max(gen_nums) if gen_nums else " "
            except (TypeError, ValueError):
                max_gen = " "
            # Opening in append mode will create the file if it doesn't exist.
            with open(self.logfile, "a") as fd:
                if ids:
                    fd.write(f"{now()}: {max_gen}: {','.join(ids)}\n")

    def is_uniform(self, func, min_std, pop=None):
        """Tests whether the current population is uniform or diverse.
        Returns True if uniform, False otherwise.

        Parameters
        ----------
        func: function
            that takes one argument an atoms object and returns a value that
            will be used for testing against the rest of the population.

        min_std: int or float
            The minimum standard deviation, if the population has a lower
            std dev it is uniform.

        pop: list, optional
            use this list of Atoms objects instead of the current population.

        """
        if pop is None:
            pop = self.pop
        vals = [func(a) for a in pop]
        stddev = np.std(vals)
        return stddev < min_std

    def mass_extinction(self, ids):
        """Kills every candidate in the database with gaid in the
        supplied list of ids. Typically used on the main part of the current
        population if the diversity is to small.

        Parameters
        ----------
        ids: list
            list of ids of candidates to be killed.

        """
        for confid in ids:
            self.dc.kill_candidate(confid)
        self.pop = []


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
        rng=np.random,
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
            M = [float(self.pop[i].info["n_paired"]) for i in indecies]
            L = [float(self.pop[i].info["looks_like"]) for i in indecies]
            fitness_values = [
                fitness_values[j] * 1. / sqrt(1. + M[j]) * 1. / sqrt(1. + L[j])
                for j in range(len(fitness_values))
            ]

        return fitness_values

    def update(self):
        """Update population and periodically add new references."""
        super().update()

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
