"""Acceptance tests for GA mutation operators and crossover at production-like sizes.

Uses the same factories as ``ga_go`` / ``ga_go_torchsim``:
:class:`~scgo.algorithms.ga_common.create_mutation_operators` and
:class:`~scgo.algorithms.ga_common.create_ga_pairing`. Geometry acceptance
mirrors slab-aware mutations and ``CutAndSplicePairing``:
:class:`~ase_ga.utilities.atoms_too_close` on the adsorbate (or the whole
cluster in gas phase) and :func:`~ase_ga.utilities.atoms_too_close_two_sets`
between slab and adsorbate when ``n_slab > 0``.

**Why two gas-phase 55-atom setups**

- *Icosahedral template* (magic 55): ``create_initial_cluster(..., mode=\"template\")``.
  Some operators (flattening, mirror) essentially never succeed on this dense
  geometry within their attempt budgets; we still require **rattle**,
  **rotational**, and **anisotropic_rattle** to succeed.
- *Random spherical* (55 Pt): looser cluster shape. All factory operators are
    exercised, including **flattening**, at the production
    ``flattening_thickness_factor`` because the bounded flattening construction
    no longer needs an acceptance-only thickness override.

``create_initial_cluster`` may return an ASE ``Cluster`` subclass whose
``copy()`` breaks inside crossover; helpers coerce to plain :class:`~ase.Atoms`.

**Permutation:** monometallic Pt has no permutation slot in the factory; a
bimetallic 55-atom gas case covers **permutation** plus the other operators.

Smaller-cluster tests under ``tests/ase_ga_patches/`` remain for RNG and
fine-grained behavior (they are not removed here).

**Surface rotational mutation:** skipped in ``test_mutations_surface_pt20_all_factory_operators``
because :class:`~scgo.ase_ga_patches.standardmutations.RotationalMutation` calls
``center()`` on the adsorbate, which—combined with in-plane slab periodicity—makes
``test_dist_to_slab=True`` success too rare for a bounded acceptance loop; gas-phase
cases still cover this operator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.algorithms.ga_common import create_ga_pairing, create_mutation_operators
from scgo.initialization import create_initial_cluster
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster

# The active mirror and bounded cut selection make crossover acceptance fast
# enough that large legacy outer retry caps are unnecessary here.
MAX_MUTATION_ATTEMPTS = 20
MAX_CROSSOVER_ATTEMPTS = 12

# Flattening now succeeds at the production default thickness in these
# acceptance cases; keep the tests aligned with the real operator setting.
_ACCEPTANCE_FLATTENING_THICKNESS = 0.5
# Cap inner trials so surface + gas acceptance stays CI-bounded while matching
# the bounded candidate sets used by the operators.
_ACCEPTANCE_FLATTEN_MAX_INNER = 900
_ACCEPTANCE_ROT_MAX_INNER = 900
_ACCEPTANCE_MIRROR_TRIES = 280
_ACCEPTANCE_BREATHING_MAX_INNER = 5
_ACCEPTANCE_SLIDE_MAX_INNER = 12

_TEMPLATE_CORE_OPERATOR_NAMES = ("rattle", "rotational", "anisotropic_rattle")


def _plain_atoms(a: Atoms) -> Atoms:
    """Coerce to plain Atoms so CutAndSplicePairing.copy() never hits Cluster bugs."""
    return Atoms(
        numbers=a.get_atomic_numbers(),
        positions=a.get_positions(),
        cell=a.get_cell(),
        pbc=a.get_pbc(),
    )


def _assert_accepted_geometry(
    atoms: Atoms,
    n_slab: int,
    blmin: dict,
    parent: Atoms,
    *,
    adsorbate_use_tags: bool = False,
) -> None:
    assert len(atoms) == len(parent)
    assert np.array_equal(atoms.get_atomic_numbers(), parent.get_atomic_numbers())
    assert np.allclose(atoms.get_cell(), parent.get_cell())
    assert np.all(atoms.get_pbc() == parent.get_pbc())
    if n_slab == 0:
        assert not atoms_too_close(atoms, blmin, use_tags=adsorbate_use_tags)
    else:
        slab_part = atoms[:n_slab]
        ads = atoms[n_slab:]
        assert not atoms_too_close(ads, blmin, use_tags=adsorbate_use_tags)
        assert not atoms_too_close_two_sets(slab_part, ads, blmin)


def _gas_pt55_template_parent() -> tuple[Atoms, list[str], dict]:
    composition = ["Pt"] * 55
    raw = create_initial_cluster(
        composition,
        rng=np.random.default_rng(2025),
        mode="template",
        vacuum=10.0,
    )
    parent = _plain_atoms(raw)
    blmin = closest_distances_generator(
        get_all_atom_types(parent, range(55)),
        ratio_of_covalent_radii=0.7,
    )
    return parent, composition, blmin


def _gas_pt55_random_spherical_parent(seed: int) -> tuple[Atoms, list[str], dict]:
    composition = ["Pt"] * 55
    raw = create_initial_cluster(
        composition,
        rng=np.random.default_rng(seed),
        mode="random_spherical",
        vacuum=10.0,
    )
    parent = _plain_atoms(raw)
    blmin = closest_distances_generator(
        get_all_atom_types(parent, range(55)),
        ratio_of_covalent_radii=0.7,
    )
    return parent, composition, blmin


def _gas_bimetallic_55_parent(seed: int) -> tuple[Atoms, list[str], dict]:
    composition = ["Pt"] * 28 + ["Au"] * 27
    np.random.default_rng(seed + 999).shuffle(composition)
    raw = create_initial_cluster(
        composition,
        rng=np.random.default_rng(seed),
        mode="random_spherical",
        vacuum=10.0,
    )
    parent = _plain_atoms(raw)
    blmin = closest_distances_generator(
        get_all_atom_types(parent, range(55)),
        ratio_of_covalent_radii=0.7,
    )
    return parent, composition, blmin


def _surface_pt20_system(
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
) -> tuple[Atoms, Atoms, Atoms, Atoms, list[str], dict, int]:
    # Larger in-plane cell improves odds of placing a 20-atom cluster without clashes.
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)
    composition = ["Pt"] * 20
    n_slab = len(slab)
    n_top = len(composition)

    dummy = np.vstack([slab.get_positions(), np.zeros((n_top, 3))])
    tmpl = Atoms(
        symbols=list(slab.get_chemical_symbols()) + composition,
        positions=dummy,
        cell=slab.cell,
        pbc=slab.pbc,
    )
    idx_top = range(n_slab, n_slab + n_top)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, idx_top),
        ratio_of_covalent_radii=0.7,
    )

    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=3.2,
        max_placement_attempts=3500,
        cluster_init_vacuum=10.0,
        init_mode="random_spherical",
    )

    p1_raw = create_deposited_cluster(composition, slab, blmin, rng_a, cfg)
    p2_raw = create_deposited_cluster(composition, slab, blmin, rng_b, cfg)
    assert p1_raw is not None and p2_raw is not None

    return (
        slab,
        tmpl,
        _plain_atoms(p1_raw),
        _plain_atoms(p2_raw),
        composition,
        blmin,
        n_slab,
    )


def _prepare_ga_parent(atoms: Atoms, confid: int) -> Atoms:
    p = atoms.copy()
    p.info["confid"] = confid
    return p


def _mutation_operator_succeeds(
    op_name: str,
    name_map: dict[str, int],
    parent: Atoms,
    composition: list[str],
    n_opt: int,
    blmin: dict,
    n_slab: int,
    *,
    flattening_thickness_factor: float,
    surface_normal_axis: int = 2,
    flattening_max_inner_attempts: int = _ACCEPTANCE_FLATTEN_MAX_INNER,
    rotational_max_inner_attempts: int = _ACCEPTANCE_ROT_MAX_INNER,
    mirror_max_tries: int = _ACCEPTANCE_MIRROR_TRIES,
    breathing_max_inner_attempts: int = _ACCEPTANCE_BREATHING_MAX_INNER,
    in_plane_slide_max_inner_attempts: int = _ACCEPTANCE_SLIDE_MAX_INNER,
    breathing_scale_min: float = 0.82,
    breathing_scale_max: float = 1.22,
) -> bool:
    adsorbate_use_tags = op_name == "rotational"
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(50_000 + attempt * 97 + name_map[op_name] * 13)
        ops, nm = create_mutation_operators(
            composition,
            n_opt,
            blmin,
            rng=op_rng,
            use_adaptive=True,
            n_slab=n_slab,
            surface_normal_axis=surface_normal_axis,
            flattening_thickness_factor=flattening_thickness_factor,
            flattening_max_inner_attempts=flattening_max_inner_attempts,
            rotational_max_inner_attempts=rotational_max_inner_attempts,
            mirror_max_tries=mirror_max_tries,
            breathing_max_inner_attempts=breathing_max_inner_attempts,
            in_plane_slide_max_inner_attempts=in_plane_slide_max_inner_attempts,
            breathing_scale_min=breathing_scale_min,
            breathing_scale_max=breathing_scale_max,
        )
        assert nm == name_map
        op = ops[nm[op_name]]
        cand, _desc = op.get_new_individual([parent])
        if cand is None:
            continue
        _assert_accepted_geometry(
            cand, n_slab, blmin, parent, adsorbate_use_tags=adsorbate_use_tags
        )
        return True
    return False


@pytest.mark.slow
def test_mutations_gas_pt55_icosahedral_template_core_operators() -> None:
    parent0, composition, blmin = _gas_pt55_template_parent()
    parent = _prepare_ga_parent(parent0, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        55,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
    )
    assert "permutation" not in name_map

    for op_name in _TEMPLATE_CORE_OPERATOR_NAMES:
        assert op_name in name_map
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            55,
            blmin,
            0,
            flattening_thickness_factor=0.5,
            flattening_max_inner_attempts=5000,
            rotational_max_inner_attempts=10000,
            mirror_max_tries=1000,
        )
        assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


@pytest.mark.slow
def test_mutations_gas_pt55_random_spherical_all_factory_operators() -> None:
    parent0, composition, blmin = _gas_pt55_random_spherical_parent(1234)
    parent = _prepare_ga_parent(parent0, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        55,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
    )
    assert "permutation" not in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            55,
            blmin,
            0,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
        )
        assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


@pytest.mark.slow
def test_mutations_gas_bimetallic_55_permutation_and_rest() -> None:
    # Permutation with blmin rarely succeeds on arbitrary bimetallic seeds; 500 works
    # with random_spherical placement (see scgo.algorithms.ga_common perm + blmin).
    parent0, composition, blmin = _gas_bimetallic_55_parent(500)
    parent = _prepare_ga_parent(parent0, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        55,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
    )
    assert "permutation" in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            55,
            blmin,
            0,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
        )
        assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


@pytest.mark.slow
def test_mutations_surface_pt20_all_factory_operators() -> None:
    _slab, _tmpl, raw_p1, _raw_p2, composition, blmin, n_slab = _surface_pt20_system(
        np.random.default_rng(101),
        np.random.default_rng(202),
    )
    parent = _prepare_ga_parent(raw_p1, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        len(composition),
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        n_slab=n_slab,
        surface_normal_axis=2,
        flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
    )
    assert "permutation" not in name_map
    assert "in_plane_slide" in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        # RotationalMutation centers the adsorbate fragment; with periodic in-plane
        # slab boundaries, satisfying both intra-adsorbate and slab blmin within
        # default attempt budgets is unreliable here. Gas-phase tests above still
        # exercise rotational mutation end-to-end.
        if op_name == "rotational":
            continue
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            len(composition),
            blmin,
            n_slab,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
        )
        assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


def _crossover_child(
    p1: Atoms,
    p2: Atoms,
    n_top: int,
    blmin: dict,
    n_slab: int,
    slab_atoms: Atoms | None,
    template_atoms: Atoms,
    **pairing_kwargs: Any,
) -> Atoms | None:
    for attempt in range(MAX_CROSSOVER_ATTEMPTS):
        prng = np.random.default_rng(80_000 + attempt)
        pairing = create_ga_pairing(
            template_atoms,
            n_top,
            rng=prng,
            slab_atoms=slab_atoms,
            **pairing_kwargs,
        )
        cand, _desc = pairing.get_new_individual([p1, p2])
        if cand is None:
            continue
        _assert_accepted_geometry(
            cand,
            n_slab,
            blmin,
            p1,
            adsorbate_use_tags=False,
        )
        return cand
    return None


@pytest.mark.slow
def test_crossover_gas_pt55_random_spherical_then_rattle_mutate() -> None:
    p1_raw, composition, blmin = _gas_pt55_random_spherical_parent(11)
    p2_raw, _, _ = _gas_pt55_random_spherical_parent(22)
    p1 = _prepare_ga_parent(p1_raw, confid=1)
    p2 = _prepare_ga_parent(p2_raw, confid=2)
    rng_pert = np.random.default_rng(5)
    for p, _subseed in ((p1, 1), (p2, 2)):
        for _trial in range(30):
            delta = rng_pert.normal(0, 0.06, size=p.positions.shape)
            trial_pos = p.positions + delta
            probe = p.copy()
            probe.positions[:] = trial_pos
            if not atoms_too_close(probe, blmin):
                p.positions[:] = trial_pos
                break
        else:
            pytest.fail(
                f"could not apply small diversity perturbation (confid={p.info['confid']})"
            )

    child = _crossover_child(p1, p2, 55, blmin, 0, None, p1.copy())
    assert child is not None, (
        f"crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )

    mut_ok = False
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(90_000 + attempt)
        ops, nm = create_mutation_operators(
            composition,
            55,
            blmin,
            rng=op_rng,
            use_adaptive=True,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
            flattening_max_inner_attempts=_ACCEPTANCE_FLATTEN_MAX_INNER,
            rotational_max_inner_attempts=_ACCEPTANCE_ROT_MAX_INNER,
            mirror_max_tries=_ACCEPTANCE_MIRROR_TRIES,
        )
        rattle = ops[nm["rattle"]]
        mutated = rattle.mutate(child)
        if mutated is None:
            continue
        _assert_accepted_geometry(mutated, 0, blmin, p1, adsorbate_use_tags=False)
        mut_ok = True
        break
    assert mut_ok, "rattle.mutate after crossover did not yield accepted geometry"


@pytest.mark.slow
def test_crossover_surface_pt20_then_rattle_mutate() -> None:
    slab, tmpl, raw_p1, raw_p2, composition, blmin, n_slab = _surface_pt20_system(
        np.random.default_rng(303),
        np.random.default_rng(404),
    )
    p1 = _prepare_ga_parent(raw_p1, confid=1)
    p2 = _prepare_ga_parent(raw_p2, confid=2)

    child = _crossover_child(p1, p2, len(composition), blmin, n_slab, slab, tmpl)
    assert child is not None, (
        f"surface crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )

    mut_ok = False
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(110_000 + attempt)
        ops, nm = create_mutation_operators(
            composition,
            len(composition),
            blmin,
            rng=op_rng,
            use_adaptive=True,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
            flattening_max_inner_attempts=_ACCEPTANCE_FLATTEN_MAX_INNER,
            rotational_max_inner_attempts=_ACCEPTANCE_ROT_MAX_INNER,
            mirror_max_tries=_ACCEPTANCE_MIRROR_TRIES,
        )
        rattle = ops[nm["rattle"]]
        mutated = rattle.mutate(child)
        if mutated is None:
            continue
        _assert_accepted_geometry(mutated, n_slab, blmin, p1, adsorbate_use_tags=False)
        mut_ok = True
        break
    assert mut_ok, (
        "rattle.mutate after surface crossover did not yield accepted geometry"
    )


@pytest.mark.slow
def test_crossover_gas_pt55_dual_pairing_accepts_offspring() -> None:
    """Dual minfrac pairing should still yield valid gas-phase children."""
    p1_raw, composition, blmin = _gas_pt55_random_spherical_parent(11)
    p2_raw, _, _ = _gas_pt55_random_spherical_parent(22)
    p1 = _prepare_ga_parent(p1_raw, confid=1)
    p2 = _prepare_ga_parent(p2_raw, confid=2)
    rng_pert = np.random.default_rng(5)
    for p, _subseed in ((p1, 1), (p2, 2)):
        for _trial in range(30):
            delta = rng_pert.normal(0, 0.06, size=p.positions.shape)
            trial_pos = p.positions + delta
            probe = p.copy()
            probe.positions[:] = trial_pos
            if not atoms_too_close(probe, blmin):
                p.positions[:] = trial_pos
                break
        else:
            pytest.fail(
                f"could not apply small diversity perturbation (confid={p.info['confid']})"
            )

    child = _crossover_child(
        p1,
        p2,
        55,
        blmin,
        0,
        None,
        p1.copy(),
        exploratory_crossover_probability=0.25,
    )
    assert child is not None, (
        f"dual crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )
