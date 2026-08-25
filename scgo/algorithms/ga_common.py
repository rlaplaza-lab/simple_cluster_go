"""Shared components for Genetic Algorithm implementations.

This module contains code shared by the TorchSim Genetic Algorithm and the other
SCGO drivers that reuse its building blocks (Basin Hopping, transition-state
search) to reduce duplication.
"""

from __future__ import annotations

import logging
import math
import typing

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms as ASEFixAtoms
from ase.constraints import FixBondLengths as ASEFixBondLengths
from ase_ga.offspring_creator import OperationSelector
from ase_ga.startgenerator import StartGenerator
from ase_ga.utilities import get_all_atom_types
from numpy.random import Generator

from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
)
from scgo.ase_ga_patches.mutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    CustomPermutationMutation,
    FlatteningMutation,
    InPlaneRotateMutation,
    InPlaneSlideMutation,
    MirrorMutation,
    OverlapReliefMutation,
    RattleMutation,
    RotationalMutation,
    ShellSwapMutation,
)
from scgo.ase_ga_patches.population import FitnessStrategyPopulation, Population
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.helpers import parse_positive_fragment_lengths
from scgo.cluster_adsorbate.hierarchical import (
    build_hierarchical_core_fragment_cluster,
    build_hierarchical_core_fragment_cluster_batch,
)
from scgo.cluster_adsorbate.reposition import FragmentRepositionMutation

# Prefer tag reader for raw_score and other fields
from scgo.database import SCGODatabaseManager
from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.initialization import create_initial_cluster, create_initial_cluster_batch
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.initialization_config import (
    BLMIN_RATIO_DEFAULT,
    CONNECTIVITY_FACTOR,
)
from scgo.metadata.atoms import get_tag
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import (
    create_deposited_cluster,
    create_deposited_cluster_batch,
)
from scgo.surface.partition import resolve_slab_search_partition
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    SystemType,
    get_system_policy,
    uses_surface,
    validate_composition_against_adsorbate,
    validate_minimum_structure,
)
from scgo.utils.comparators import (
    ComparatorBlocks,
    EnergyAndStructureComparator,
    UniquenessSettings,
    create_geometry_comparator,
)
from scgo.utils.diversity_scorer import DiversityScorer
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    get_fitness_from_atoms,
)
from scgo.utils.helpers import canonicalize_relaxed_for_storage
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs, validate_n_jobs
from scgo.utils.phase_logging import (
    log_phase_header,
)
from scgo.utils.rng_helpers import (
    create_child_rng,
    ensure_rng_or_create,
    get_child_rng_or_none,
)
from scgo.utils.validation import (
    validate_in_range,
    validate_integer,
    validate_positive,
)

logger = get_logger(__name__)


def _copy_adsorbate_fragment_template(
    fragment_template: AdsorbateFragmentInput | None,
) -> AdsorbateFragmentInput | None:
    """Clone an adsorbate fragment template without duplicating the caller state."""
    if isinstance(fragment_template, list):
        return [frag.copy() for frag in fragment_template]
    if fragment_template is not None:
        return fragment_template.copy()
    return None


def slab_ga_metadata_extras(
    surface_config: SurfaceSystemConfig | None, n_slab: int, system_type: SystemType
) -> dict[str, int | str | list[str]]:
    """Extra tags for slab+adsorbate GA (atom order: slab indices 0..n_slab-1)."""
    metadata: dict[str, int | str | list[str]] = {"system_type": system_type}
    if uses_surface(system_type) and surface_config is not None and n_slab > 0:
        metadata["n_slab_atoms"] = n_slab
        metadata["slab_chemical_symbols_json"] = list(
            surface_config.slab.get_chemical_symbols()
        )
        if get_system_policy(system_type).slab_is_search_target:
            part = resolve_slab_search_partition(surface_config)
            metadata["n_fixed_slab_atoms"] = part.n_fixed
            metadata["n_mobile_slab_atoms"] = part.n_mobile_slab
    return metadata


def adsorbate_partition_metadata(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
) -> dict[str, int | str | list[str] | list[int]]:
    """Store core vs adsorbate mobile prefix for has_adsorbate system types (GA DB round-trip)."""
    if not get_system_policy(system_type).has_adsorbate:
        return {}
    if adsorbate_definition is None:
        return {}
    core_list, ads_list = validate_composition_against_adsorbate(
        composition, adsorbate_definition, context="adsorbate_partition_metadata"
    )
    n_core = len(core_list)
    n_ads = len(ads_list)
    fragment_lengths = parse_positive_fragment_lengths(
        adsorbate_definition.adsorbate_fragment_lengths
    )
    if sum(fragment_lengths) != n_ads and n_ads > 0:
        fragment_lengths = [n_ads]
    return {
        "n_core_atoms": n_core,
        "n_adsorbate_fragment_atoms": n_ads,
        "n_adsorbate_fragments": len(fragment_lengths),
        "core_chemical_symbols_json": list(core_list),
        "adsorbate_fragment_chemical_symbols_json": list(ads_list),
        "adsorbate_fragment_lengths_json": list(fragment_lengths),
    }


def ga_run_metadata_extras(
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None = None,
    fix_atoms_indices: list[int] | None = None,
    fix_bond_lengths_pairs: list[list[int]] | None = None,
) -> dict[str, int | str]:
    """Slab + optional core/adsorbate mobile partition for GA written structures.

    When ``fix_atoms_indices`` / ``fix_bond_lengths_pairs`` are provided (the
    relaxation-time constraint index lists), they are persisted as JSON-encoded
    tags so the constraint state can be rebuilt on load even when a consumer
    relies on the metadata backstop rather than the native DB constraint
    round-trip.
    """
    out = slab_ga_metadata_extras(surface_config, n_slab, system_type)
    out.update(
        adsorbate_partition_metadata(system_type, composition, adsorbate_definition)
    )
    if fix_atoms_indices is not None:
        out["fix_atoms_indices_json"] = [int(i) for i in fix_atoms_indices]
    if fix_bond_lengths_pairs is not None:
        out["fix_bond_lengths_pairs_json"] = [
            [int(a), int(b)] for a, b in fix_bond_lengths_pairs
        ]
    return out


def extract_constraint_index_lists(atoms: Atoms) -> dict[str, typing.Any]:
    """Extract FixAtoms indices and FixBondLengths pairs for DB round-trip.

    Covers both constraint families (similar to
    ``collect_ase_fixatoms_indices`` in ``torchsim_helpers``).

    Returns a JSON-serializable dict with:
      - ``fix_atoms_indices``: sorted unique positive indices fixed by any
        ``FixAtoms`` constraint (negative indices normalized via ``i % len(atoms)``).
      - ``fix_bond_lengths_pairs``: list of ``[i, j]`` pairs (each sorted,
        negatives normalized) from every ``FixBondLengths`` constraint.

    Empty lists are returned when ``atoms`` carries no matching constraints.
    """
    n = len(atoms)
    fix_atoms: set[int] = set()
    bond_pairs: list[list[int]] = []
    for c in atoms.constraints:
        if isinstance(c, ASEFixAtoms):
            for i in c.index:
                fix_atoms.add(int(i) % n)
        elif isinstance(c, ASEFixBondLengths):
            for a, b in c.pairs:
                aa, bb = int(a) % n, int(b) % n
                bond_pairs.append([min(aa, bb), max(aa, bb)])
    return {
        "fix_atoms_indices": sorted(fix_atoms),
        "fix_bond_lengths_pairs": bond_pairs,
    }


def reconstruct_constraints_from_index_lists(
    atoms: Atoms,
    *,
    fix_atoms_indices: list[int] | None = None,
    fix_bond_lengths_pairs: list[list[int]] | None = None,
) -> bool:
    """Reattach FixAtoms / FixBondLengths on *atoms* when missing (additive).

    Rebuilds each constraint type only if the loaded ``Atoms`` does not already
    carry one of that type, so an existing (e.g. native round-tripped)
    constraint is never overwritten or duplicated.

    ``FixBondLengths`` is rebuilt from the stored pairs; ASE recomputes the
    target bond lengths from the loaded (relaxed) positions, which is correct
    for downstream reuse (e.g. TS NEB setups). Returns ``True`` when any
    constraint was (re)built.
    """
    existing = list(atoms.constraints)
    has_fix_atoms = any(isinstance(c, ASEFixAtoms) for c in existing)
    has_fix_bonds = any(isinstance(c, ASEFixBondLengths) for c in existing)
    new: list = list(existing)
    if fix_atoms_indices and not has_fix_atoms:
        new.append(ASEFixAtoms(indices=list(fix_atoms_indices)))
    if fix_bond_lengths_pairs and not has_fix_bonds:
        pairs = [tuple(int(x) for x in p) for p in fix_bond_lengths_pairs]
        new.append(ASEFixBondLengths(pairs))
    if len(new) != len(existing):
        atoms.set_constraint(new)
        return True
    return False


def validate_structure_for_ga_storage(
    atoms: Atoms,
    *,
    surface_mode: bool,
    n_slab: int,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: (
        ConnectivityFactorInput | NormalizedConnectivityFactor | None
    ) = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    n_slab_deposit: int | None = None,
) -> str | None:
    """Validate ``atoms`` in the GA database storage frame.

    Applies :func:`~scgo.utils.helpers.canonicalize_relaxed_for_storage` and then
    :func:`~scgo.system_types.validate_minimum_structure`. Returns
    ``None`` when the structure is eligible for GA evolution; otherwise the
    validation error message.

    ``n_slab`` is the full slab prefix used for canonicalization, symbol
    matching, and adsorbate tag partition. For slab-as-search-target systems
    pass ``n_slab_deposit`` as the frozen prefix so deposit/connectivity sees
    the search-mobile region (top layers + adsorbate).

    All TorchSim GA code paths that assign ``ga_eligible`` after relaxation
    (or before initial unrelaxed insert) must use this helper so pre- and
    post-relax checks see the same canonical frame.
    """
    try:
        canonicalize_relaxed_for_storage(
            atoms,
            surface_mode=surface_mode,
            n_slab=n_slab,
        )
        validate_minimum_structure(
            atoms,
            system_type=system_type,
            surface_config=surface_config,
            n_slab=n_slab if surface_mode else None,
            adsorbate_definition=adsorbate_definition,
            connectivity_factor=connectivity_factor,
            cluster_adsorbate_config=cluster_adsorbate_config,
            allow_cluster_fragmentation=allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            n_slab_deposit=n_slab_deposit,
        )
    except (ValueError, SCGOValidationError) as exc:
        return str(exc)
    return None


def core_adsorbate_partition_counts(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    *,
    allow_empty_core: bool = False,
) -> tuple[int, int] | None:
    """(n_core, n_ads) for the mobile region, or None if not a two-block adsorbate run.

    When ``allow_empty_core`` is True (TS empty-core adsorbates), only the
    adsorbate block must be nonempty.
    """
    if not get_system_policy(system_type).has_adsorbate or adsorbate_definition is None:
        return None
    try:
        core_list, ads_list = validate_composition_against_adsorbate(
            composition,
            adsorbate_definition,
            context="core_adsorbate_partition_counts",
        )
    except (ValueError, SCGOValidationError) as exc:
        logger.debug("Validation of core_adsorbate_partition_counts failed: %s", exc)
        return None
    if len(ads_list) == 0:
        return None
    if len(core_list) == 0 and not allow_empty_core:
        return None
    return (len(core_list), len(ads_list))


def core_adsorbate_partition_details(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    *,
    allow_empty_core: bool = False,
) -> tuple[int, list[int]] | None:
    """Mobile partition details as ``(n_core, ads_fragment_lengths)``."""
    counts = core_adsorbate_partition_counts(
        system_type,
        composition,
        adsorbate_definition,
        allow_empty_core=allow_empty_core,
    )
    if counts is None or adsorbate_definition is None:
        return None
    n_core, n_ads = counts
    lengths = parse_positive_fragment_lengths(
        adsorbate_definition.adsorbate_fragment_lengths
    )
    if sum(lengths) != n_ads:
        lengths = [n_ads]
    return (n_core, lengths)


def resolve_neb_mobile_dims(
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    *,
    neb_align_endpoints: bool,
) -> tuple[int | None, int | None, list[int] | None]:
    """Raise-style NEB block dims: ``(n_core, n_ads, fragment_lengths)``.

    Empty-core adsorbates are allowed (``n_core=0``). Bare surface / no-align /
    missing definition returns ``(None, None, None)``. Invalid definitions raise.
    """
    if (
        not neb_align_endpoints
        or not get_system_policy(system_type).has_adsorbate
        or adsorbate_definition is None
    ):
        return None, None, None
    core_list, ads_list = validate_composition_against_adsorbate(
        composition,
        adsorbate_definition,
        context="resolve_neb_mobile_dims",
    )
    if len(ads_list) == 0:
        return None, None, None
    n_core, n_ads = len(core_list), len(ads_list)
    frag_lengths = list(adsorbate_definition.adsorbate_fragment_lengths)
    return n_core, n_ads, frag_lengths


def apply_mobile_core_ads_tags(
    atoms: Atoms, n_slab: int, n_core: int, ads_fragment_lengths: list[int]
) -> None:
    """Tag mobile atoms: core=0; adsorbate fragments get tags 1..N."""
    n = len(atoms)
    n_ads = int(sum(int(x) for x in ads_fragment_lengths))
    if n_slab + n_core + n_ads != n:
        raise SCGOValidationError(
            f"apply_mobile_core_ads_tags: len(atoms)={n}, n_slab={n_slab}, "
            f"n_core={n_core}, n_ads={n_ads} (must sum to len)"
        )
    tags = np.zeros(n, dtype=int)
    offset = n_slab + n_core
    for frag_idx, frag_len in enumerate(ads_fragment_lengths):
        next_offset = offset + int(frag_len)
        if next_offset > n:
            raise SCGOValidationError("adsorbate fragment lengths exceed atom count")
        tags[offset:next_offset] = frag_idx + 1
        offset = next_offset
    atoms.set_tags(tags)


def maybe_apply_mobile_core_ads_tags(
    atoms: Atoms,
    n_slab: int,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    system_type: SystemType,
) -> None:
    part = core_adsorbate_partition_details(
        system_type,
        composition,
        adsorbate_definition,
        allow_empty_core=get_system_policy(system_type).has_adsorbate,
    )
    if part is None:
        return
    n_core, ads_fragment_lengths = part
    apply_mobile_core_ads_tags(atoms, n_slab, n_core, ads_fragment_lengths)


def validate_ga_common_params(
    niter: int,
    population_size: int,
    n_jobs_population_init: int,
    calculator: typing.Any,
    mutation_probability: float,
    offspring_fraction: float,
    vacuum: float,
    fmax: float | None = None,
) -> None:
    """Validate parameters shared by GA and GA TorchSim implementations."""
    validate_integer("niter", niter)
    validate_positive("niter", niter, strict=True)
    validate_integer("population_size", population_size)
    validate_positive("population_size", population_size, strict=True)
    validate_n_jobs(n_jobs_population_init, "n_jobs_population_init")
    if calculator is None:
        raise SCGOValidationError("calculator is required for genetic algorithm")
    if fmax is not None:
        validate_positive("fmax", fmax, strict=True)
    validate_in_range("mutation_probability", mutation_probability, 0.0, 1.0)
    validate_in_range("offspring_fraction", offspring_fraction, 0.0, 1.0)
    validate_positive("offspring_fraction", offspring_fraction, strict=True)
    validate_positive("vacuum", vacuum, strict=True)


class ClusterStartGenerator(StartGenerator):
    """StartGenerator creating initial clusters.

    Uses :func:`scgo.initialization.create_initial_cluster_batch` to produce
    starting candidates. When population_size is provided, pre-generates the
    entire population up front.

    For ``gas_cluster_adsorbate``, ``adsorbate_definition`` and
    ``adsorbate_fragment_template`` are required (``cluster_adsorbate_config`` is
    optional); candidates come from
    :func:`scgo.cluster_adsorbate.hierarchical.build_hierarchical_core_fragment_cluster`
    (same as surface hierarchical seeds without a slab). Plain ``gas_cluster`` must
    not pass these keyword arguments.
    """

    def __init__(
        self,
        composition: list[str],
        vacuum: float,
        rng: np.random.Generator | None = None,
        calculator: Calculator | None = None,
        population_size: int | None = None,
        mode: str = "smart",
        previous_search_glob: str = "**/*.db",
        n_jobs: int | None = None,
        *,
        system_type: SystemType = "gas_cluster",
        adsorbate_definition: AdsorbateDefinition | None = None,
        adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
        cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
        max_hierarchical_attempts: int = 200,
        verbosity: int = 1,
    ) -> None:
        """Initialize ClusterStartGenerator.

        Args:
            composition: List of atomic symbols defining the cluster.
            vacuum: Amount of vacuum to add around the cluster.
            rng: Optional numpy random number generator for reproducibility.
            calculator: Optional calculator to assign to generated atoms.
            population_size: Optional total population size. If provided, pre-generates
                the entire population up front. If None, generates on demand.
            mode: Initialization mode. Default "smart".
            previous_search_glob: Glob pattern used to find prior databases for
                seed-based initialization. Defaults to ``"**/*.db"``.
            n_jobs: Number of parallel workers for batch initialization.
                ``None`` uses the project default (single worker; opt in with
                -1/-2 for parallelism). Special values: -1 (all CPUs),
                -2 (all except one). Only used when population_size is provided.
            system_type: ``gas_cluster`` or ``gas_cluster_adsorbate``; used to reject
                spurious adsorbate kwargs for plain gas clusters.
            adsorbate_definition: Required for hierarchical gas seeds; optional for
                monolithic (validated at runner; still used for metadata in GA).
            adsorbate_fragment_template: Fragment geometry for the hierarchical
                layout; required for ``system_type=gas_cluster_adsorbate``.
            cluster_adsorbate_config: Optional placement/validation for the fragment.
            max_hierarchical_attempts: Max inner tries in hierarchical core+fragment
                build (per candidate).
        """
        st_pol = get_system_policy(system_type)
        if st_pol.uses_surface:
            raise SCGOValidationError(
                "ClusterStartGenerator is for gas-phase runs only"
            )
        if not st_pol.has_adsorbate and (
            adsorbate_definition is not None
            or adsorbate_fragment_template is not None
            or cluster_adsorbate_config is not None
        ):
            raise SCGOValidationError(
                "adsorbate_definition, adsorbate_fragment_template, and "
                "cluster_adsorbate_config are only valid for system_type=gas_cluster_adsorbate"
            )
        if (
            adsorbate_fragment_template is not None
            or cluster_adsorbate_config is not None
        ) and adsorbate_definition is None:
            raise SCGOValidationError(
                "adsorbate_fragment_template and cluster_adsorbate_config require "
                "adsorbate_definition"
            )
        if st_pol.has_adsorbate and adsorbate_definition is None:
            raise SCGOValidationError(
                "adsorbate_definition is required in ClusterStartGenerator for "
                "system_type=gas_cluster_adsorbate"
            )

        # Normalize RNG if provided; allow None (falls back to default RNG later)
        self.rng: Generator | None = (
            ensure_rng_or_create(rng) if rng is not None else None
        )
        self.composition: list[str] = composition
        self.vacuum: float = vacuum
        self.calculator: Calculator | None = calculator
        self.population_size: int | None = population_size
        self.mode: str = mode
        self.previous_search_glob: str = previous_search_glob
        self.n_jobs: int = resolve_n_jobs(n_jobs)
        self.system_type: SystemType = system_type
        self.adsorbate_definition = adsorbate_definition
        self.adsorbate_fragment_template = (
            [frag.copy() for frag in adsorbate_fragment_template]
            if isinstance(adsorbate_fragment_template, list)
            else adsorbate_fragment_template.copy()
            if adsorbate_fragment_template is not None
            else None
        )
        self.cluster_adsorbate_config = cluster_adsorbate_config
        self.max_hierarchical_attempts: int = max_hierarchical_attempts
        self._hierarchical: bool = bool(adsorbate_definition is not None)
        self.verbosity: int = verbosity
        self._batch_site_type_counts: dict[str, int] = {
            "vertex": 0,
            "edge": 0,
            "facet": 0,
        }
        if st_pol.has_adsorbate and self.adsorbate_fragment_template is None:
            raise SCGOValidationError(
                "adsorbate_fragment_template is required for hierarchical "
                "gas_cluster_adsorbate initialization."
            )
        self._candidate_count = 0
        self._candidate_batch: list[Atoms] | None = None

        if population_size is not None and self.rng is not None:
            log_phase_header(
                logger,
                "Population initialization",
                verbosity=verbosity,
            )
            if self._hierarchical and adsorbate_definition is not None:
                self._candidate_batch = build_hierarchical_core_fragment_cluster_batch(
                    adsorbate_definition,
                    self.rng,
                    self.previous_search_glob,
                    self.adsorbate_fragment_template,
                    self.cluster_adsorbate_config,
                    cluster_init_vacuum=self.vacuum,
                    init_mode=self.mode,
                    n_structures=population_size,
                    max_placement_attempts=self.max_hierarchical_attempts,
                    batch_site_counts=self._batch_site_type_counts,
                    n_jobs=self.n_jobs,
                    verbosity=verbosity,
                )
            else:
                self._candidate_batch = create_initial_cluster_batch(
                    composition=composition,
                    n_structures=population_size,
                    rng=self.rng,
                    vacuum=vacuum,
                    previous_search_glob=previous_search_glob,
                    mode=mode,
                    n_jobs=self.n_jobs,
                    verbosity=verbosity,
                )

    def get_new_candidate(self) -> Atoms:
        """Generate a single new, random cluster candidate.

        If population_size was provided, serves candidates from pre-generated batch.
        Otherwise, generates structures on-demand.
        """
        atoms: Atoms | None = None
        if self._candidate_batch is not None and self._candidate_count < len(
            self._candidate_batch
        ):
            atoms = self._candidate_batch[self._candidate_count]
            self._candidate_count += 1

        if atoms is None:
            if self._hierarchical and self.adsorbate_definition is not None:
                atoms = build_hierarchical_core_fragment_cluster(
                    self.adsorbate_definition,
                    ensure_rng_or_create(self.rng),
                    self.previous_search_glob,
                    self.adsorbate_fragment_template,
                    self.cluster_adsorbate_config,
                    cluster_init_vacuum=self.vacuum,
                    init_mode=self.mode,
                    max_placement_attempts=self.max_hierarchical_attempts,
                    batch_site_counts=self._batch_site_type_counts,
                    verbosity=self.verbosity,
                )
                if atoms is None:
                    raise SCGORuntimeError(
                        "ClusterStartGenerator: hierarchical gas seed could not be placed; "
                        "increase max_hierarchical_attempts or relax ClusterAdsorbateConfig."
                    )
                site_type = get_tag(atoms, "adsorbate_site_type")
                if (
                    isinstance(site_type, str)
                    and site_type in self._batch_site_type_counts
                ):
                    self._batch_site_type_counts[site_type] += 1
            else:
                atoms = create_initial_cluster(
                    self.composition,
                    vacuum=self.vacuum,
                    rng=ensure_rng_or_create(self.rng),
                    previous_search_glob=self.previous_search_glob,
                    mode=self.mode,
                    verbosity=self.verbosity,
                )

        if atoms is None:
            raise SCGORuntimeError(
                "StartGenerator failed to produce a valid Atoms object"
            )
        if self.calculator is not None:
            atoms.calc = self.calculator
        return atoms


class SurfaceClusterStartGenerator(StartGenerator):
    """StartGenerator for slab + adsorbate cluster using :mod:`scgo.surface.deposition`."""

    def __init__(
        self,
        composition: list[str],
        slab: Atoms,
        surface_config: SurfaceSystemConfig,
        blmin: dict,
        rng: np.random.Generator | None = None,
        calculator: Calculator | None = None,
        population_size: int | None = None,
        previous_search_glob: str = "**/*.db",
        n_jobs: int | None = None,
        adsorbate_definition: AdsorbateDefinition | None = None,
        adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
        cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
        verbosity: int = 1,
    ) -> None:
        self.rng: Generator | None = (
            ensure_rng_or_create(rng) if rng is not None else None
        )
        self.composition = composition
        self.slab = slab.copy()
        self.surface_config = surface_config
        self.blmin = blmin
        self.calculator = calculator
        self.population_size = population_size
        self.previous_search_glob = previous_search_glob
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.adsorbate_definition = adsorbate_definition
        self.adsorbate_fragment_template = _copy_adsorbate_fragment_template(
            adsorbate_fragment_template
        )
        self.cluster_adsorbate_config = cluster_adsorbate_config
        self.verbosity: int = verbosity
        self._batch_site_type_counts: dict[str, int] = {
            "vertex": 0,
            "edge": 0,
            "facet": 0,
        }
        self._candidate_count = 0
        self._candidate_batch: list[Atoms] | None = None

        if population_size is not None and self.rng is not None:
            log_phase_header(
                logger,
                "Population initialization",
                verbosity=verbosity,
            )
            self._candidate_batch = create_deposited_cluster_batch(
                composition=composition,
                slab=self.slab,
                blmin=blmin,
                n_structures=population_size,
                rng=self.rng,
                config=surface_config,
                previous_search_glob=previous_search_glob,
                n_jobs=self.n_jobs,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=self.adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
                batch_site_counts=self._batch_site_type_counts,
                verbosity=verbosity,
            )

    def get_new_candidate(self) -> Atoms:
        atoms = None
        if self._candidate_batch is not None and self._candidate_count < len(
            self._candidate_batch
        ):
            atoms = self._candidate_batch[self._candidate_count]
            self._candidate_count += 1

        if atoms is None:
            atoms = create_deposited_cluster(
                self.composition,
                self.slab,
                self.blmin,
                ensure_rng_or_create(self.rng),
                self.surface_config,
                previous_search_glob=self.previous_search_glob,
                adsorbate_definition=self.adsorbate_definition,
                adsorbate_fragment_template=self.adsorbate_fragment_template,
                cluster_adsorbate_config=self.cluster_adsorbate_config,
                batch_site_counts=self._batch_site_type_counts,
                verbosity=self.verbosity,
            )
            if atoms is None:
                raise SCGORuntimeError(
                    "SurfaceClusterStartGenerator could not place a valid structure; "
                    "increase max_placement_attempts or height range."
                )
            site_type = get_tag(atoms, "adsorbate_site_type")
            if isinstance(site_type, str) and site_type in self._batch_site_type_counts:
                self._batch_site_type_counts[site_type] += 1

        if self.calculator is not None:
            atoms.calc = self.calculator
        return atoms


class SurfaceSlabStartGenerator(StartGenerator):
    """StartGenerator for bare slab search: rattle top-layer atoms of a fixed slab."""

    def __init__(
        self,
        slab: Atoms,
        *,
        n_fixed: int,
        rattle_strength: float = 0.35,
        rng: np.random.Generator | None = None,
        calculator: Calculator | None = None,
        population_size: int | None = None,
        verbosity: int = 1,
    ) -> None:
        self.rng: Generator = ensure_rng_or_create(
            rng if rng is not None else np.random.default_rng()
        )
        self.slab = slab.copy()
        self.n_fixed = int(n_fixed)
        self.rattle_strength = float(rattle_strength)
        self.calculator = calculator
        self._candidate_count = 0
        self._candidate_batch: list[Atoms] = []
        if self.n_fixed < 0 or self.n_fixed >= len(self.slab):
            raise SCGOValidationError(
                f"SurfaceSlabStartGenerator: n_fixed={self.n_fixed} invalid for "
                f"len(slab)={len(self.slab)}"
            )
        n_pop = int(population_size) if population_size is not None else 1
        log_phase_header(
            logger,
            "Population initialization",
            verbosity=verbosity,
        )
        for _ in range(max(n_pop, 1)):
            self._candidate_batch.append(self._make_candidate())

    def _make_candidate(self) -> Atoms:
        atoms = self.slab.copy()
        pos = atoms.get_positions()
        noise = self.rng.normal(
            0.0, self.rattle_strength, size=pos[self.n_fixed :].shape
        )
        pos[self.n_fixed :] += noise
        atoms.set_positions(pos)
        return atoms

    def get_new_candidate(self) -> Atoms:
        if self._candidate_count < len(self._candidate_batch):
            atoms = self._candidate_batch[self._candidate_count]
            self._candidate_count += 1
        else:
            atoms = self._make_candidate()
        if self.calculator is not None:
            atoms.calc = self.calculator
        return atoms


def create_ga_pairing(
    atoms_template: Atoms,
    n_to_optimize: int,
    rng: np.random.Generator | None = None,
    slab_atoms: Atoms | None = None,
    system_type: SystemType = "gas_cluster",
    *,
    composition: list[str] | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    exploratory_crossover_probability: float = 0.2,
    exploratory_minfrac: float | None = None,
) -> CutAndSplicePairing | DualCutAndSplicePairing:
    """Create a cut-and-splice pairing operator for GA evolution.

    Accepts an optional RNG; if provided it will be used as a parent RNG
    for creating child RNGs for internal operators.

    Args:
        atoms_template: Template Atoms object with cell and pbc settings.
        n_to_optimize: Number of atoms to optimize (trailing ``n_top`` atoms).
        rng: Random number generator.
        slab_atoms: If provided, real slab atoms for adsorbate GA (non-empty).
            If None, an empty slab with the template cell/pbc is used (gas-phase GA).
        composition, adsorbate_definition: If both set for a two-block ``*_adsorbate``
            run, pairing uses ``use_tags`` (rigid core/fragment groups).
        exploratory_crossover_probability: When > 0 and exploratory ``minfrac``
            differs from the primary, a dual wrapper uses this probability to
            pick the more asymmetric cut-and-splice variant.
        exploratory_minfrac: Lower ``minfrac`` for the exploratory variant.
            Default ``max(0.1, primary_minfrac - 0.15)``.

    Returns:
        :class:`~scgo.ase_ga_patches.cutandsplicepairing.CutAndSplicePairing` or
        :class:`~scgo.ase_ga_patches.cutandsplicepairing.DualCutAndSplicePairing`.
    """
    if not uses_surface(system_type) and slab_atoms is not None and len(slab_atoms) > 0:
        raise SCGOValidationError(
            f"Received non-empty slab_atoms with non-surface system_type={system_type!r}. "
            "Use a surface_* system type."
        )
    n_template = len(atoms_template)
    if uses_surface(system_type):
        if slab_atoms is None or len(slab_atoms) == 0:
            raise SCGOValidationError(
                "Surface system types require slab_atoms for pairing."
            )
        if n_template != len(slab_atoms) + n_to_optimize:
            raise SCGOValidationError(
                "atoms_template length must equal len(slab_atoms) + n_to_optimize "
                f"for surface GA, got {n_template}, slab={len(slab_atoms)}, "
                f"n_to_optimize={n_to_optimize}"
            )
        idx_top = range(len(slab_atoms), n_template)
    else:
        if n_template != n_to_optimize:
            raise SCGOValidationError(
                "atoms_template length must equal n_to_optimize for gas-phase GA"
            )
        idx_top = range(n_to_optimize)

    # ``ase_ga.utilities.get_all_atom_types`` expects atomic numbers for the
    # second argument, not template indices (large slab indices would crash
    # ``closest_distances_generator``).
    top_z = list({int(atoms_template[i].number) for i in idx_top})
    all_atom_types = get_all_atom_types(atoms_template, top_z)
    blmin = build_blmin_from_zs(all_atom_types, ratio=BLMIN_RATIO_DEFAULT)

    if uses_surface(system_type):
        slab = slab_atoms.copy()
    else:
        slab = Atoms(cell=atoms_template.get_cell(), pbc=atoms_template.get_pbc())
    min_parent_fraction: float = min(0.5, max(0.3, 2.0 / max(1, n_to_optimize)))
    child_rng_primary = get_child_rng_or_none(rng)

    use_partition_tags = False
    pairing_target_tags: list[int] | None = None
    if composition is not None:
        use_partition_tags = (
            core_adsorbate_partition_counts(
                system_type,
                composition,
                adsorbate_definition,
                allow_empty_core=get_system_policy(system_type).has_adsorbate,
            )
            is not None
        )
        if use_partition_tags:
            if system_type == "surface_adsorbate":
                part = core_adsorbate_partition_details(
                    system_type,
                    composition,
                    adsorbate_definition,
                    allow_empty_core=True,
                )
                if part is not None:
                    _n_core, ads_fragment_lengths = part
                    ads_tags = list(range(1, len(ads_fragment_lengths) + 1))
                    pairing_target_tags = ads_tags if ads_tags else None
            else:
                pairing_target_tags = None

    def _cut_and_splice(
        minfrac: float, *, pairing_rng: Generator | None
    ) -> CutAndSplicePairing:
        return CutAndSplicePairing(  # type: ignore[arg-type]
            slab,
            n_to_optimize,
            blmin,
            minfrac=minfrac,
            rng=pairing_rng,
            system_type=system_type,
            use_tags=use_partition_tags,
            target_tags=pairing_target_tags,
        )

    expl_minfrac = (
        float(exploratory_minfrac)
        if exploratory_minfrac is not None
        else max(0.1, min_parent_fraction - 0.15)
    )
    if exploratory_crossover_probability <= 0.0 or math.isclose(
        expl_minfrac, min_parent_fraction
    ):
        return _cut_and_splice(min_parent_fraction, pairing_rng=child_rng_primary)

    primary = _cut_and_splice(min_parent_fraction, pairing_rng=child_rng_primary)
    exploratory = _cut_and_splice(expl_minfrac, pairing_rng=get_child_rng_or_none(rng))
    return DualCutAndSplicePairing(
        primary,
        exploratory,
        exploratory_crossover_probability,
        rng=get_child_rng_or_none(rng),
    )


# How root weight-table keys fan out onto partitioned operator variants
# (``_core`` / ``_ads``) registered by ``_append_partitioned_mutation``.
OPERATOR_PARTITION_SPECS: dict[str, tuple[str, float]] = {
    "flattening_core": ("flattening", 0.65),
    "flattening_ads": ("flattening", 0.35),
    "breathing_core": ("breathing", 0.65),
    "breathing_ads": ("breathing", 0.35),
    "in_plane_slide_core": ("in_plane_slide", 0.15),
    "in_plane_slide_ads": ("in_plane_slide", 0.15),
}


def _effective_operator_weight(
    name: str | None,
    operator_weights: dict[str, float],
    name_map: dict[str, int],
) -> float:
    """Map base adaptive weights onto partitioned adsorbate operator names."""
    if not name:
        return 0.0

    slide_peers = [n for n in name_map if n.startswith("in_plane_slide_")]
    slide_root = "in_plane_slide"
    if name == slide_root and name in operator_weights and slide_peers:
        allocated = float(operator_weights[slide_root])
        unscoped_fraction = 0.70
        peer_fraction = 0.15
        active_fraction = unscoped_fraction + peer_fraction * len(slide_peers)
        return allocated * (unscoped_fraction / active_fraction)

    if name in operator_weights:
        return float(operator_weights[name])

    partition_specs = OPERATOR_PARTITION_SPECS
    if name not in partition_specs:
        return 0.0

    root, fraction = partition_specs[name]
    if root not in operator_weights:
        return 0.0

    peer_names = [n for n in name_map if n.startswith(f"{root}_")]
    if not peer_names or name not in peer_names:
        return 0.0

    allocated = float(operator_weights[root])
    if root in name_map:
        unscoped_fraction = 0.70
        peer_fraction = 0.15
        active_fraction = unscoped_fraction + peer_fraction * len(peer_names)
        return allocated * (peer_fraction / active_fraction)

    active_fraction = sum(
        partition_specs[n][1] for n in peer_names if n in partition_specs
    )
    if active_fraction <= 0.0:
        return allocated / len(peer_names)
    return allocated * (fraction / active_fraction)


def unmatched_operator_weight_keys(
    operator_weights: dict[str, float],
    name_map: dict[str, int],
) -> list[str]:
    """Return weight-table keys that resolve to no registered operator.

    A key is matched when it names a registered operator, or when it is the
    root of a registered partitioned variant (e.g. ``flattening`` covers
    ``flattening_core``/``flattening_ads``). Unmatched keys silently carry no
    selector mass, so they usually indicate a typo or a stale table entry.
    """
    covered: set[str] = set(name_map)
    for name in name_map:
        spec = OPERATOR_PARTITION_SPECS.get(name)
        if spec is not None:
            covered.add(spec[0])
    return sorted(key for key in operator_weights if key not in covered)


def _append_partitioned_mutation(
    operators: list,
    name_map: dict[str, int],
    *,
    base_name: str,
    mutation_cls: type,
    use_partition_tags: bool,
    ads_tags: list[int],
    include_ads_variant: bool,
    kwargs_for: typing.Callable[[str], dict[str, typing.Any]],
) -> None:
    r"""Register a mutation as plain or ``_core`` / ``_ads`` partition variants.

    ``kwargs_for`` receives ``\"plain\"``, ``\"core\"``, or ``\"ads\"`` and must
    return constructor kwargs for that variant (including ``target_tags`` when
    partitioned).
    """
    if not use_partition_tags:
        operators.append(mutation_cls(**kwargs_for("plain")))
        name_map[base_name] = len(operators) - 1
        return
    operators.append(mutation_cls(**kwargs_for("core")))
    name_map[f"{base_name}_core"] = len(operators) - 1
    if include_ads_variant and ads_tags:
        operators.append(mutation_cls(**kwargs_for("ads")))
        name_map[f"{base_name}_ads"] = len(operators) - 1


def create_mutation_operators(
    composition: list[str],
    n_to_optimize: int,
    blmin: dict,
    rng: np.random.Generator | None = None,
    use_adaptive: bool = True,
    system_type: SystemType = "gas_cluster",
    *,
    n_slab: int = 0,
    surface_normal_axis: int = 2,
    flattening_thickness_factor: float = 0.5,
    flattening_max_inner_attempts: int = 12,
    rotational_max_inner_attempts: int = 24,
    mirror_max_tries: int = 12,
    breathing_max_inner_attempts: int = 1000,
    in_plane_slide_max_inner_attempts: int = 1000,
    in_plane_slide_max_displacement: float = 10.0,
    breathing_scale_min: float = 0.82,
    breathing_scale_max: float = 1.22,
    adsorbate_definition: AdsorbateDefinition | None = None,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
) -> tuple[list, dict[str, int]]:
    """Create mutation operators once at start of GA.

    Accepts an optional RNG (parent); child RNGs will be derived when needed.

    Args:
        composition: List of atomic symbols.
        n_to_optimize: Number of atoms to optimize.
        blmin: Bond length minimums dictionary.
        rng: Random number generator.
        use_adaptive: Whether to include adaptive mutation operators.
        adsorbate_definition: Enables the two-block core/adsorbate mobile
            partition: tag-aware rattle plus ``_core``/``_ads`` variants of the
            flattening, breathing, and in-plane slide mutations, and (for
            cluster searches) a fragment reposition operator.
        n_slab: Number of fixed slab atoms; when > 0, registers in-plane slide.
        surface_normal_axis: Slab normal (0, 1, or 2) for in-plane slide.
        flattening_thickness_factor: Passed to :class:`~scgo.ase_ga_patches.mutations.FlatteningMutation`
            (larger values relax post-projection thickness, helping large clusters).
        flattening_max_inner_attempts: Max ranked flattening candidates per call.
        rotational_max_inner_attempts: Max ranked rotation candidates per call.
        mirror_max_tries: Max ranked mirror cutting-plane candidates per call.
            Mirror is registered only for surface systems and tagged
            adsorbate partitions; untagged gas-phase clusters omit it.
        breathing_max_inner_attempts: Max radial-scale trials per breathing call.
        in_plane_slide_max_inner_attempts: Max slide trials per slide call.
        in_plane_slide_max_displacement: Maximum displacement magnitude (Å) per
            in-plane direction for in-plane slide mutation.
        breathing_scale_min: Lower bound for radial scale factors (about the fragment CoM).
        breathing_scale_max: Upper bound for radial scale factors.
        connectivity_factor: Stamped onto each operator for mutation connectivity gates.

    Returns:
        Tuple of (operators_list, operator_name_to_index_map).
    """
    if not uses_surface(system_type) and n_slab > 0:
        raise SCGOValidationError(
            f"Received n_slab > 0 with non-surface system_type={system_type!r}. "
            "Use a surface_* system type."
        )
    operators = []
    name_map = {}
    policy = get_system_policy(system_type)
    partition_composition = list(composition)
    if policy.slab_is_search_target and policy.has_adsorbate and adsorbate_definition:
        ads = adsorbate_definition.adsorbate_symbols
        core = adsorbate_definition.core_symbols
        if isinstance(ads, list) and isinstance(core, list):
            partition_composition = [str(s) for s in core] + [str(s) for s in ads]
    part = core_adsorbate_partition_details(
        system_type,
        partition_composition,
        adsorbate_definition,
        allow_empty_core=policy.has_adsorbate,
    )
    use_partition_tags = part is not None
    ads_tags: list[int] = []
    if part is not None:
        _n_core, ads_fragment_lengths = part
        ads_tags = list(range(1, len(ads_fragment_lengths) + 1))

    # Adsorbate scale only throttles ads-targeted ops when core/ads are partitioned;
    # shared and core ops keep full strength. Non-partitioned constrained systems
    # (ads-only) keep the global scale.
    ads_move_scale = (
        policy.adsorbate_move_scale if policy.constrain_adsorbate_moves else 1.0
    )
    shared_move_scale = 1.0 if use_partition_tags else ads_move_scale
    core_move_scale = 1.0 if use_partition_tags else ads_move_scale

    core_only_tags = [0] if use_partition_tags else None
    include_overlap_relief = not (
        freeze_adsorbate_internal_geometry and use_partition_tags
    )
    # Crystalline slab search: skip cluster-shape operators.
    include_cluster_shape_ops = not policy.slab_is_search_target

    # Estimate a physically meaningful max displacement for in-plane slide
    # based on the expected cluster size. Use 3 * estimated cluster radius,
    # where radius is approximated as n_to_optimize**(1/3) * avg_blmin.
    # This ensures sufficient displacement range even when clusters are
    # placed close to slab atoms.
    avg_blmin = np.mean(list(blmin.values())) if blmin else 1.0
    estimated_cluster_radius = n_to_optimize ** (1.0 / 3.0) * avg_blmin
    default_max_displacement = 3.0 * estimated_cluster_radius

    rattle: RattleMutation = RattleMutation(
        blmin,
        n_to_optimize,
        rattle_strength=0.8 * shared_move_scale,
        rattle_prop=min(0.4, 0.4 * shared_move_scale),
        use_tags=use_partition_tags,
        system_type=system_type,
        surface_normal_axis=surface_normal_axis,
        rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
    )
    operators.append(rattle)
    name_map["rattle"] = 0

    overlap_relief: OverlapReliefMutation = OverlapReliefMutation(
        blmin,
        n_to_optimize,
        system_type=system_type,
        use_tags=use_partition_tags,
        rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
    )
    if include_overlap_relief:
        operators.append(overlap_relief)
        name_map["overlap_relief"] = len(operators) - 1

    if len(set(composition)) > 1 and policy.allow_composition_permutations:
        permutation: CustomPermutationMutation = CustomPermutationMutation(
            n_to_optimize,
            rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
            blmin=blmin,
            test_dist_to_slab=uses_surface(system_type),
            system_type=system_type,
        )
        operators.append(permutation)
        name_map["permutation"] = len(operators) - 1

        if include_cluster_shape_ops:
            shell_swap: ShellSwapMutation = ShellSwapMutation(
                n_to_optimize,
                rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
                blmin=blmin,
                test_dist_to_slab=uses_surface(system_type),
                system_type=system_type,
            )
            operators.append(shell_swap)
            name_map["shell_swap"] = len(operators) - 1

    if use_adaptive and include_cluster_shape_ops:

        def _flattening_kwargs(variant: str) -> dict[str, typing.Any]:
            kw: dict[str, typing.Any] = {
                "blmin": blmin,
                "n_top": n_to_optimize,
                "thickness_factor": flattening_thickness_factor,
                "rng": get_child_rng_or_none(rng),
                "max_inner_attempts": flattening_max_inner_attempts,
                "system_type": system_type,
                "surface_normal_axis": surface_normal_axis,
            }
            if variant == "core":
                kw["target_tags"] = [0]
            elif variant == "ads":
                kw["target_tags"] = ads_tags
            return kw

        _append_partitioned_mutation(
            operators,
            name_map,
            base_name="flattening",
            mutation_cls=FlatteningMutation,
            use_partition_tags=use_partition_tags,
            ads_tags=ads_tags,
            include_ads_variant=not freeze_adsorbate_internal_geometry,
            kwargs_for=_flattening_kwargs,
        )

        rotational: RotationalMutation = RotationalMutation(
            blmin,
            system_type=system_type,
            n_top=n_to_optimize,
            target_tags=core_only_tags,
            use_tags=use_partition_tags,
            rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
            max_inner_attempts=rotational_max_inner_attempts,
            surface_normal_axis=surface_normal_axis,
        )
        operators.append(rotational)
        name_map["rotational"] = len(operators) - 1

        # Untagged gas-phase clusters have no leftover reference, so a
        # full-cluster mirror is an isometry. Omit it from the factory
        # (and mutate() still returns None as a safety net).
        if use_partition_tags or uses_surface(system_type):
            mirror: MirrorMutation = MirrorMutation(
                blmin,
                n_to_optimize,
                reflect=True,
                system_type=system_type,
                target_tags=core_only_tags,
                rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
                max_tries=mirror_max_tries,
                surface_normal_axis=surface_normal_axis,
            )
            operators.append(mirror)
            name_map["mirror"] = len(operators) - 1

        anisotropic: AnisotropicRattleMutation = AnisotropicRattleMutation(
            blmin,
            n_to_optimize,
            in_plane_strength=1.0 * shared_move_scale,
            normal_strength=0.2 * shared_move_scale,
            rattle_prop=min(0.5, 0.5 * shared_move_scale),
            use_tags=use_partition_tags,
            system_type=system_type,
            surface_normal_axis=surface_normal_axis,
            rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
        )
        operators.append(anisotropic)
        name_map["anisotropic_rattle"] = len(operators) - 1

        def _breathing_kwargs(variant: str) -> dict[str, typing.Any]:
            if variant == "ads":
                scale = ads_move_scale
            elif variant == "core":
                scale = core_move_scale
            else:
                scale = shared_move_scale
            kw: dict[str, typing.Any] = {
                "blmin": blmin,
                "n_top": n_to_optimize,
                "scale_min": 1.0 - (1.0 - breathing_scale_min) * scale,
                "scale_max": 1.0 + (breathing_scale_max - 1.0) * scale,
                "system_type": system_type,
                "rng": get_child_rng_or_none(rng),
                "max_inner_attempts": breathing_max_inner_attempts,
                "surface_normal_axis": surface_normal_axis,
            }
            if variant == "core":
                kw["target_tags"] = [0]
            elif variant == "ads":
                kw["target_tags"] = ads_tags
            return kw

        _append_partitioned_mutation(
            operators,
            name_map,
            base_name="breathing",
            mutation_cls=BreathingMutation,
            use_partition_tags=use_partition_tags,
            ads_tags=ads_tags,
            include_ads_variant=not freeze_adsorbate_internal_geometry,
            kwargs_for=_breathing_kwargs,
        )

    if use_adaptive and uses_surface(system_type) and n_slab > 0:
        slide_max_disp = max(in_plane_slide_max_displacement, default_max_displacement)

        def _slide_kwargs(variant: str) -> dict[str, typing.Any]:
            kw: dict[str, typing.Any] = {
                "blmin": blmin,
                "n_top": n_to_optimize,
                "surface_normal_axis": surface_normal_axis,
                "system_type": system_type,
                "rng": get_child_rng_or_none(rng),
                "max_inner_attempts": in_plane_slide_max_inner_attempts,
                "max_displacement": slide_max_disp,
            }
            if variant == "core":
                kw["target_tags"] = [0]
            elif variant == "ads":
                kw["target_tags"] = ads_tags
            return kw

        if use_partition_tags:
            operators.append(InPlaneSlideMutation(**_slide_kwargs("plain")))
            name_map["in_plane_slide"] = len(operators) - 1
            operators.append(InPlaneSlideMutation(**_slide_kwargs("core")))
            name_map["in_plane_slide_core"] = len(operators) - 1
            if ads_tags:
                operators.append(InPlaneSlideMutation(**_slide_kwargs("ads")))
                name_map["in_plane_slide_ads"] = len(operators) - 1
        else:
            _append_partitioned_mutation(
                operators,
                name_map,
                base_name="in_plane_slide",
                mutation_cls=InPlaneSlideMutation,
                use_partition_tags=False,
                ads_tags=ads_tags,
                include_ads_variant=True,
                kwargs_for=_slide_kwargs,
            )

        # Bare slab-target searches keep in-plane rotation registered with
        # zero table weight (pinned by tests): unavailable moves stay
        # reachable to future weight schedules without recreating operators.
        in_plane_rotate = InPlaneRotateMutation(
            blmin,
            n_to_optimize,
            system_type=system_type,
            surface_normal_axis=surface_normal_axis,
            rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
        )
        operators.append(in_plane_rotate)
        name_map["in_plane_rotate"] = len(operators) - 1

    if (
        use_partition_tags
        and adsorbate_definition is not None
        and (include_cluster_shape_ops or policy.slab_is_search_target)
    ):
        reposition = FragmentRepositionMutation(
            blmin,
            n_to_optimize,
            system_type=system_type,
            adsorbate_definition=adsorbate_definition,
            fragment_templates=adsorbate_fragment_template,
            cluster_adsorbate_config=cluster_adsorbate_config,
            rng=get_child_rng_or_none(rng),  # type: ignore[arg-type]
            surface_normal_axis=surface_normal_axis,
        )
        operators.append(reposition)
        name_map["fragment_reposition"] = len(operators) - 1

    resolved_cf: ConnectivityFactorInput | NormalizedConnectivityFactor = (
        CONNECTIVITY_FACTOR if connectivity_factor is None else connectivity_factor
    )
    for op in operators:
        op.connectivity_factor = resolved_cf

    return operators, name_map


def reseed_mutation_operator_rngs(
    operators: list,
    rng: np.random.Generator,
) -> None:
    """Assign fresh child RNGs to mutation operators in deterministic list order."""
    for op in operators:
        if hasattr(op, "rng"):
            op.rng = create_child_rng(rng)


def update_mutation_weights(
    operators_list: list,
    name_map: dict[str, int],
    adaptive_config: dict,
    rng: Generator,
) -> OperationSelector:
    """Update operator weights without recreating operators.

    Args:
        operators_list: List of operator instances.
        name_map: Mapping from operator names to list indices.
        adaptive_config: Config dict with operator_weights.
        rng: Explicit RNG for reproducible operator selection. Required so
            serial and parallel offspring paths never fall back to global state.

    Returns:
        Updated OperationSelector with new weights.
    """
    operator_weights = adaptive_config["operator_weights"]
    unmatched = unmatched_operator_weight_keys(operator_weights, name_map)
    if unmatched:
        logger.warning(
            "Adaptive operator weights reference unregistered operators; "
            "these keys carry no selector mass: %s",
            unmatched,
        )
    index_to_name = {idx: name for name, idx in name_map.items()}

    weights: list[float] = []
    for i in range(len(operators_list)):
        name = index_to_name.get(i)
        weights.append(_effective_operator_weight(name, operator_weights, name_map))

    s = float(sum(weights))
    if s > 0.0:
        weights = [w / s for w in weights]
    elif weights:
        # All-zero (or negative) weights make OperationSelector.__get_index__
        # return None -> ``oplist[None]`` TypeError. Fall back to uniform.
        logger.warning(
            "All operator weights are non-positive; falling back to uniform "
            "selection over %d operators",
            len(weights),
        )
        weights = [1.0 / len(weights)] * len(weights)

    if "rattle" in name_map:
        rattle_idx: int = name_map["rattle"]
        rattle_op = operators_list[rattle_idx]
        rattle_op.rattle_strength = adaptive_config["rattle_strength"]
        rattle_op.rattle_prop = adaptive_config["rattle_prop"]

    if "anisotropic_rattle" in name_map:
        anisotropic_idx: int = name_map["anisotropic_rattle"]
        anisotropic_op = operators_list[anisotropic_idx]
        if "anisotropic_in_plane_strength" in adaptive_config:
            anisotropic_op.in_plane_strength = adaptive_config[
                "anisotropic_in_plane_strength"
            ]
        if "anisotropic_normal_strength" in adaptive_config:
            anisotropic_op.normal_strength = adaptive_config[
                "anisotropic_normal_strength"
            ]
        if "anisotropic_rattle_prop" in adaptive_config:
            anisotropic_op.rattle_prop = adaptive_config["anisotropic_rattle_prop"]

    return OperationSelector(weights, operators_list, rng=rng)


def create_structure_comparator(
    n_atoms: int,
    energy_tolerance: float,
    settings: UniquenessSettings | None = None,
    *,
    mic: bool = False,
    blocks: ComparatorBlocks | None = None,
) -> EnergyAndStructureComparator:
    resolved = settings if settings is not None else UniquenessSettings()
    geometry = create_geometry_comparator(
        n_top=n_atoms,
        mic=mic,
        settings=resolved,
        blocks=blocks,
    )
    return EnergyAndStructureComparator(energy_tolerance, geometry)


def _as_fitness_strategy(fitness_strategy: str | FitnessStrategy) -> FitnessStrategy:
    return (
        FitnessStrategy(fitness_strategy)
        if isinstance(fitness_strategy, str)
        else fitness_strategy
    )


def update_early_stopping_state_unified(
    population: Population,
    fitness_strategy: str | FitnessStrategy,
    best_value: float | None,
    generations_without_improvement: int,
    early_stopping_niter: int,
) -> tuple[float | None, int, bool]:
    """Update early stopping for energy-based and fitness-based strategies."""
    fitness_strategy = _as_fitness_strategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        if len(population.pop) == 0:
            return best_value, generations_without_improvement, False

        current_best_fitness: float = max(
            (
                get_fitness_from_atoms(atoms_obj, default=-float("inf"))
                for atoms_obj in population.pop
            ),
            default=-float("inf"),
        )

        if best_value is None or current_best_fitness > best_value:
            return current_best_fitness, 0, False

        updated_generations: int = generations_without_improvement + 1
        should_stop: bool = updated_generations >= early_stopping_niter
        return best_value, updated_generations, should_stop

    if len(population.pop) == 0:
        return best_value, generations_without_improvement, False

    current_best_energy = -float(get_tag(population.pop[0], "raw_score", default=0.0))

    if best_value is None or current_best_energy < best_value:
        return current_best_energy, 0, False

    updated_generations = generations_without_improvement + 1
    should_stop = updated_generations >= early_stopping_niter
    return best_value, updated_generations, should_stop


def setup_diversity_scorer(
    fitness_strategy: str | FitnessStrategy,
    diversity_reference_db: str | None,
    composition: list[str],
    n_to_optimize: int,
    diversity_max_references: int,
    logger,
    *,
    base_dir: str,
    mic: bool = False,
    uniqueness: UniquenessSettings | None = None,
    blocks: ComparatorBlocks | None = None,
) -> DiversityScorer | None:
    """Setup DiversityScorer for diversity fitness strategy.

    Args:
        fitness_strategy: Fitness strategy name.
        diversity_reference_db: Glob pattern for reference structure databases.
        composition: List of atomic symbols.
        n_to_optimize: Number of atoms to optimize.
        diversity_max_references: Maximum number of reference structures to load.
        logger: Logger instance for logging messages.
        base_dir: Base directory for resolving reference DB glob patterns.
        mic: Whether the diversity comparator uses the minimum-image convention.
        uniqueness: Geometry tolerances for diversity scoring (GO defaults when omitted).
        blocks: Optional block-aware partition mirroring the uniqueness comparator.

    Returns:
        DiversityScorer instance when fitness_strategy is "diversity" and at least
        one reference structure was loaded; None otherwise.

    Raises:
        SCGOValidationError: If diversity_reference_db is None when
            fitness_strategy is "diversity".
    """
    fitness_strategy = _as_fitness_strategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.DIVERSITY:
        return None

    if diversity_reference_db is None:
        raise SCGOValidationError(
            "diversity_reference_db is required when fitness_strategy='diversity'"
        )

    if logger.isEnabledFor(logging.INFO):
        logger.info("Loading reference structures from: %s", diversity_reference_db)
    with SCGODatabaseManager(base_dir=base_dir, enable_caching=True) as db_manager:
        reference_structures: list[Atoms] = db_manager.load_reference_structures(
            db_glob_pattern=diversity_reference_db,
            composition=composition,
            max_structures=diversity_max_references,
        )
    if logger.isEnabledFor(logging.INFO):
        logger.info("Loaded %d reference structures", len(reference_structures))

    if not reference_structures:
        logger.warning(
            "No reference structures found for diversity strategy; "
            "diversity optimization may be ineffective"
        )
        return None

    comparator_for_diversity = create_geometry_comparator(
        n_top=n_to_optimize,
        mic=mic,
        settings=uniqueness,
        blocks=blocks,
    )
    return DiversityScorer(reference_structures, comparator_for_diversity)


def select_population_class(
    fitness_strategy: str | FitnessStrategy,
    diversity_scorer: DiversityScorer | None,
    diversity_update_interval: int,
    logger,
) -> tuple[type, dict]:
    """Select appropriate Population class based on fitness strategy.

    Args:
        fitness_strategy: Fitness strategy name.
        diversity_scorer: DiversityScorer instance (if using diversity strategy).
        diversity_update_interval: Number of generations between reference updates.
        logger: Logger instance for logging messages.

    Returns:
        Tuple of (PopulationClass, population_kwargs).
    """
    fitness_strategy = _as_fitness_strategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        PopulationClass = FitnessStrategyPopulation
        population_kwargs = {
            "fitness_strategy": fitness_strategy,
            "diversity_scorer": diversity_scorer,
            "diversity_update_interval": diversity_update_interval,
        }
        logger.info(
            f"Using FitnessStrategyPopulation with fitness_strategy='{fitness_strategy}'"
        )
    else:
        PopulationClass = Population
        population_kwargs = {}

    return PopulationClass, population_kwargs


def log_early_stopping_info(
    verbosity: int,
    fitness_strategy: str | FitnessStrategy,
    early_stopping_niter: int,
    niter: int,
    logger,
) -> None:
    """Log early stopping configuration information.

    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        fitness_strategy: Fitness strategy name.
        early_stopping_niter: Number of generations without improvement to trigger stopping.
        niter: Total number of generations to run.
        logger: Logger instance for logging messages.
    """
    if verbosity < 1:
        return

    fitness_strategy = _as_fitness_strategy(fitness_strategy)

    if logger.isEnabledFor(logging.INFO):
        logger.info("Starting GA evolution with %d generations", niter)
        logger.info("Using fitness_strategy='%s'", fitness_strategy)
    if early_stopping_niter > 0:
        stopping_metric: str = (
            "fitness" if fitness_strategy != FitnessStrategy.LOW_ENERGY else "energy"
        )
        logger.info(
            f"Early stopping enabled: will stop after {early_stopping_niter} "
            f"generations with no {stopping_metric} improvement"
        )


def sort_minima_by_fitness(
    all_minima: list[tuple[float, Atoms]],
    fitness_strategy: str | FitnessStrategy,
    logger,
) -> None:
    """Sort minima by fitness for non-low_energy strategies.

    Mutates input list in place. For low_energy, list remains sorted by energy
    (lowest first). For other strategies, sorts by fitness (highest first).

    Args:
        all_minima: List of (energy, Atoms) tuples to sort.
        fitness_strategy: Fitness strategy name.
        logger: Logger instance.
    """
    fitness_strategy = _as_fitness_strategy(fitness_strategy)

    if fitness_strategy != FitnessStrategy.LOW_ENERGY:
        all_minima.sort(
            key=lambda x: get_fitness_from_atoms(x[1], default=-float("inf")),
            reverse=True,  # Higher fitness first
        )
        logger.info(
            "Sorted %d minima by %s fitness",
            len(all_minima),
            fitness_strategy,
        )
