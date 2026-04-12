# fmt: off

from __future__ import annotations

"""Implementation of the cut-and-splice paring operator."""
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
)

from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng


def _random_unit_vector(rng, fallback=None):
    vector = rng.normal(0.0, 1.0, 3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _append_unique_unit_vector(candidates, vector, tol=0.995):
    unit = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(unit)
    if norm <= 1e-12:
        return
    unit /= norm
    for existing in candidates:
        if float(np.dot(unit, existing)) > tol:
            return
    candidates.append(unit)


class Positions:
    """Helper object to simplify the pairing process.

    Parameters
    ----------
    scaled_positions: (Nx3) array
        Positions in scaled coordinates
    cop: (1x3) array
        Center-of-positions (also in scaled coordinates)
    symbols: str
        String with the atomic symbols
    distance: float
        Signed distance to the cutting plane
    origin: int (0 or 1)
        Determines at which side of the plane the position should be.

    """

    def __init__(self, scaled_positions, cop, symbols, distance, origin):
        self.scaled_positions = scaled_positions
        self.cop = cop
        self.symbols = symbols
        self.distance = distance
        self.origin = origin

    def to_use(self):
        """Tells whether this position is at the right side."""
        return bool(self.distance > 0.0 and self.origin == 0 or self.distance < 0.0 and self.origin == 1)


class CutAndSplicePairing(OffspringCreator):
    """The Cut and Splice operator by Deaven and Ho.

    Creates offspring from two parent structures using
    a randomly generated cutting plane.

    The parents may have different unit cells, in which
    case the offspring unit cell will be a random combination
    of the parent cells.

    The basic implementation (for fixed unit cells) is
    described in:

    :doi:`L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)
    <10.1103/PhysRevLett.108.126101`>

    The extension to variable unit cells is similar to:

    * :doi:`Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720
      <10.1016/j.cpc.2006.07.020>`

    * :doi:`Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387
      <10.1016/j.cpc.2010.07.048>`

    The operator can furthermore preserve molecular identity
    if desired (see the *use_tags* kwarg). Atoms with the same
    tag will then be considered as belonging to the same molecule,
    and their internal geometry will not be changed by the operator.

    If use_tags is enabled, the operator will also conserve the
    number of molecules of each kind (in addition to conserving
    the overall stoichiometry). Currently, molecules are considered
    to be of the same kind if their chemical symbol strings are
    identical. In rare cases where this may not be sufficient
    (i.e. when desiring to keep the same ratio of isomers), the
    different isomers can be differentiated by giving them different
    elemental orderings (e.g. 'XY2' and 'Y2X').

    Parameters
    ----------
    slab: Atoms object
        Specifies the cell vectors and periodic boundary conditions
        to be applied to the randomly generated structures.
        Any included atoms (e.g. representing an underlying slab)
        are copied to these new structures.

    n_top: int
        The number of atoms to optimize

    blmin: dict
        Dictionary with minimal interatomic distances.
        Note: when preserving molecular identity (see use_tags),
        the blmin dict will (naturally) only be applied
        to intermolecular distances (not the intramolecular
        ones).

    number_of_variable_cell_vectors: int (default 0)
        The number of variable cell vectors (0, 1, 2 or 3).
        To keep things simple, it is the 'first' vectors which
        will be treated as variable, i.e. the 'a' vector in the
        univariate case, the 'a' and 'b' vectors in the bivariate
        case, etc.

    p1: float or int between 0 and 1
        Probability that a parent is shifted over a random
        distance along the normal of the cutting plane
        (only operative if number_of_variable_cell_vectors > 0).

    p2: float or int between 0 and 1
        Same as p1, but for shifting along the directions
        in the cutting plane (only operative if
        number_of_variable_cell_vectors > 0).

    minfrac: float between 0 and 1, or None (default)
        Minimal fraction of atoms a parent must contribute
        to the child. If None, each parent must contribute
        at least one atom.

    cellbounds: ase.ga.utilities.CellBounds instance
        Describing limits on the cell shape, see
        :class:`~ase.ga.utilities.CellBounds`.
        Note that it only make sense to impose conditions
        regarding cell vectors which have been marked as
        variable (see number_of_variable_cell_vectors).

    use_tags: bool
        Whether to use the atomic tags to preserve
        molecular identity.

    test_dist_to_slab: bool (default True)
        Whether to make sure that the distances between
        the atoms and the slab satisfy the blmin.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, slab, n_top, blmin, number_of_variable_cell_vectors=0,
                 p1=1, p2=0.05, minfrac=None, cellbounds=None,
                 test_dist_to_slab=True, use_tags=False, rng=None,
                 verbose=False):

        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.slab = slab
        self.n_top = n_top
        self.blmin = blmin
        assert number_of_variable_cell_vectors in range(4)
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.p1 = p1
        self.p2 = p2
        self.minfrac = minfrac
        self.cellbounds = cellbounds
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.last_attempt_count = 0
        self.last_cell_attempt_count = 0

        self.scaling_volume = None
        self.descriptor = "CutAndSplicePairing"
        self.min_inputs = 2

    def _group_centers_and_weights(self, atoms):
        tags = atoms.get_tags() if self.use_tags else np.arange(len(atoms))
        centers = []
        weights = []
        for tag in np.unique(tags):
            indices = np.where(tags == tag)[0]
            centers.append(np.mean(atoms.positions[indices], axis=0))
            weights.append(len(indices))
        return np.asarray(centers, dtype=float), np.asarray(weights, dtype=int)

    def _candidate_cut_normals(self, a1, a2):
        combined = np.vstack([a1.get_positions(), a2.get_positions()])
        centered = combined - np.mean(combined, axis=0)
        candidates = []

        _append_unique_unit_vector(
            candidates,
            np.mean(a2.get_positions(), axis=0) - np.mean(a1.get_positions(), axis=0),
        )

        if len(centered) > 1:
            covariance = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            axes = [eigenvectors[:, index] for index in order]
            for axis in axes:
                _append_unique_unit_vector(candidates, axis)
            if len(axes) >= 2:
                _append_unique_unit_vector(candidates, axes[0] + axes[-1])

        parent_delta = a2.get_positions() - a1.get_positions()
        if len(parent_delta) > 0:
            _append_unique_unit_vector(
                candidates,
                parent_delta[int(np.argmax(np.linalg.norm(parent_delta, axis=1)))],
            )

        attempts = 0
        while len(candidates) < 6 and attempts < 100:
            _append_unique_unit_vector(candidates, _random_unit_vector(self.rng))
            attempts += 1

        return candidates[:6]

    def _estimate_cut_balance(
        self,
        centers_a1,
        centers_a2,
        weights,
        cut_point_cart,
        cut_normal_cart,
    ):
        distances_a1 = np.dot(centers_a1 - cut_point_cart, cut_normal_cart)
        distances_a2 = np.dot(centers_a2 - cut_point_cart, cut_normal_cart)

        count1 = int(np.sum(weights[distances_a1 > 0.0]))
        count2 = int(np.sum(weights[distances_a2 < 0.0]))
        nmin = 1 if self.minfrac is None else max(1, int(np.ceil(self.minfrac * self.n_top)))
        target = 0.5 * self.n_top
        deficit = max(0, nmin - count1) + max(0, nmin - count2)
        return (
            abs(count1 - target)
            + abs(count2 - target)
            + 10.0 * deficit
        )

    def _candidate_cut_configurations(self, a1, a2, cell):
        if self.number_of_variable_cell_vectors != 0:
            cosp1 = np.average(a1.get_scaled_positions(), axis=0)
            cosp2 = np.average(a2.get_scaled_positions(), axis=0)
            cut_p = np.zeros((1, 3))
            for i in range(3):
                if i < self.number_of_variable_cell_vectors:
                    cut_p[0, i] = self.rng.random()
                else:
                    cut_p[0, i] = 0.5 * (cosp1[i] + cosp2[i])
            cut_n = self.rng.choice(self.number_of_variable_cell_vectors)
            return [(0.0, cut_p, cut_n)]

        centers_a1, weights = self._group_centers_and_weights(a1)
        centers_a2, _ = self._group_centers_and_weights(a2)
        base_cut_point = 0.5 * (
            np.mean(a1.get_positions(), axis=0) + np.mean(a2.get_positions(), axis=0)
        )
        cell_array = np.asarray(cell)
        cut_candidates = []

        for cut_normal_cart in self._candidate_cut_normals(a1, a2):
            projections = np.concatenate(
                [
                    np.dot(centers_a1 - base_cut_point, cut_normal_cart),
                    np.dot(centers_a2 - base_cut_point, cut_normal_cart),
                ]
            )
            shifts = [0.0]
            if len(projections) > 0:
                for quantile in (0.35, 0.5, 0.65):
                    shifts.append(float(np.quantile(projections, quantile)))

            cut_normal = np.linalg.solve(cell_array.T, cut_normal_cart)
            for shift in shifts:
                cut_point_cart = base_cut_point + shift * cut_normal_cart
                cut_point = np.linalg.solve(cell_array.T, cut_point_cart)[np.newaxis, :]
                score = self._estimate_cut_balance(
                    centers_a1,
                    centers_a2,
                    weights,
                    cut_point_cart,
                    cut_normal_cart,
                )
                cut_candidates.append((score, cut_point, cut_normal))

        cut_candidates.sort(key=lambda item: item[0])
        return cut_candidates[:12]

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        """Updates the scaling volume that is used in the pairing

        w_adapt: weight of the new vs the old scaling volume
        n_adapt: number of best candidates in the population that
                 are used to calculate the new scaling volume
        """
        if not n_adapt:
            # take best 20% of the population
            n_adapt = int(np.ceil(0.2 * len(population)))
        v_new = np.mean([a.get_volume() for a in population[:n_adapt]])

        if not self.scaling_volume:
            self.scaling_volume = v_new
        else:
            volumes = [self.scaling_volume, v_new]
            weights = [1 - w_adapt, w_adapt]
            self.scaling_volume = np.average(volumes, weights=weights)

    def get_new_individual(self, parents):
        """The method called by the user that
        returns the paired structure.
        """
        f, m = parents

        indi = self.cross(f, m)
        desc = f"pairing: {f.info['confid']} {m.info['confid']}"
        # It is ok for an operator to return None
        # It means that it could not make a legal offspring
        # within a reasonable amount of time
        if indi is None:
            return indi, desc
        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"],
                                        m.info["confid"]]

        return self.finalize_individual(indi), desc

    def cross(self, a1, a2):
        """Crosses the two atoms objects and returns one"""
        if len(a1) != len(self.slab) + self.n_top:
            raise ValueError("Wrong size of structure to optimize")
        if len(a1) != len(a2):
            raise ValueError("The two structures do not have the same length")

        N = self.n_top

        # Only consider the atoms to optimize
        a1 = a1[len(a1) - N: len(a1)]
        a2 = a2[len(a2) - N: len(a2)]

        if not np.array_equal(a1.numbers, a2.numbers):
            err = "Trying to pair two structures with different stoichiometry"
            raise ValueError(err)

        if self.use_tags and not np.array_equal(a1.get_tags(), a2.get_tags()):
            err = "Trying to pair two structures with different tags"
            raise ValueError(err)

        cell1 = a1.get_cell()
        cell2 = a2.get_cell()
        for i in range(self.number_of_variable_cell_vectors, 3):
            err = "Unit cells are supposed to be identical in direction %d"
            assert np.allclose(cell1[i], cell2[i]), (err % i, cell1, cell2)

        counter = 0
        maxcount = 1000
        a1_copy = a1.copy()
        a2_copy = a2.copy()
        self.last_attempt_count = 0

        # Run until a valid pairing is made or maxcount pairings are tested.
        while counter < maxcount:
            counter += 1

            newcell = self.generate_unit_cell(cell1, cell2)
            if newcell is None:
                # No valid unit cell could be generated.
                # This strongly suggests that it is near-impossible
                # to generate one from these parent cells and it is
                # better to abort now.
                break

            cut_n = None
            if self.number_of_variable_cell_vectors != 0:
                # Pick one of the 'variable' cell vectors.
                cut_n = self.rng.choice(self.number_of_variable_cell_vectors)

            # Randomly translate parent structures
            for a_copy, a in zip([a1_copy, a2_copy], [a1, a2], strict=False):
                a_copy.set_positions(a.get_positions())

                cell = a_copy.get_cell()
                for i in range(self.number_of_variable_cell_vectors):
                    r = self.rng.random()
                    cond1 = i == cut_n and r < self.p1
                    cond2 = i != cut_n and r < self.p2
                    if cond1 or cond2:
                        a_copy.positions += self.rng.random() * cell[i]

                if self.use_tags:
                    # For correct determination of the center-
                    # of-position of the multi-atom blocks,
                    # we need to group their constituent atoms
                    # together
                    gather_atoms_by_tag(a_copy)
                else:
                    # For clusters, just center without wrapping. For slab+adsorbate
                    # the top fragment must stay in the parent frame (no global center).
                    if len(self.slab) == 0:
                        a_copy.center()

            for _score, cut_p, cut_normal in self._candidate_cut_configurations(
                a1_copy,
                a2_copy,
                newcell,
            ):
                self.last_attempt_count += 1

                child = self._get_pairing(a1_copy, a2_copy, cut_p, cut_normal, newcell)
                if child is None:
                    continue

                if atoms_too_close(child, self.blmin, use_tags=self.use_tags):
                    continue

                if (
                    self.test_dist_to_slab
                    and len(self.slab) > 0
                    and atoms_too_close_two_sets(self.slab, child, self.blmin)
                ):
                    continue

                child = self.slab + child
                child.set_cell(newcell, scale_atoms=False)
                if len(self.slab) == 0:
                    child.center()
                return child

        return None

    def generate_unit_cell(self, cell1, cell2, maxcount=10000):
        """Generates a new unit cell by a random linear combination
        of the parent cells. The new cell must satisfy the
        self.cellbounds constraints. Returns None if no such cell
        was generated within a given number of attempts.

        Parameters
        ----------
        maxcount: int
            The maximal number of attempts.

        """
        # First calculate the scaling volume
        if not self.scaling_volume:
            v1 = np.abs(np.linalg.det(cell1))
            v2 = np.abs(np.linalg.det(cell2))
            r = self.rng.random()
            v_ref = r * v1 + (1 - r) * v2
        else:
            v_ref = self.scaling_volume

        # Now the cell vectors
        if self.number_of_variable_cell_vectors == 0:
            assert np.allclose(cell1, cell2), "Parent cells are not the same"
            newcell = np.copy(cell1)
            self.last_cell_attempt_count = 1
        else:
            count = 0
            while count < maxcount:
                r = self.rng.random()
                newcell = r * cell1 + (1 - r) * cell2

                vol = abs(np.linalg.det(newcell))
                scaling = v_ref / vol
                scaling **= 1. / self.number_of_variable_cell_vectors
                newcell[:self.number_of_variable_cell_vectors] *= scaling

                found = True
                if self.cellbounds is not None:
                    found = self.cellbounds.is_within_bounds(newcell)
                if found:
                    break

                count += 1
            else:
                # Did not find acceptable cell
                newcell = None
            self.last_cell_attempt_count = count if newcell is None else count + 1

        return newcell

    def _get_pairing(self, a1, a2, cutting_point, cutting_normal, cell):
        """Creates a child from two parents using the given cut.

        Returns None if the generated structure does not contain
        a large enough fraction of each parent (see self.minfrac).

        Does not check whether atoms are too close.

        Assumes the 'slab' parts have been removed from the parent
        structures and that these have been checked for equal
        lengths, stoichiometries, and tags (if self.use_tags).

        Parameters
        ----------
        cutting_normal: int or (1x3) array

        cutting_point: (1x3) array
            In fractional coordinates

        cell: (3x3) array
            The unit cell for the child structure

        """
        symbols = a1.get_chemical_symbols()
        tags = a1.get_tags() if self.use_tags else np.arange(len(a1))

        # Generate list of all atoms / atom groups:
        p1, p2, sym = [], [], []
        for i in np.unique(tags):
            indices = np.where(tags == i)[0]
            s = "".join([symbols[j] for j in indices])
            sym.append(s)

            for i, (a, p) in enumerate(zip([a1, a2], [p1, p2], strict=False)):
                c = a.get_cell()
                cop = np.mean(a.positions[indices], axis=0)
                cut_p = np.dot(cutting_point, c)
                if isinstance(cutting_normal, int):
                    vecs = [c[j] for j in range(3) if j != cutting_normal]
                    cut_n = np.cross(vecs[0], vecs[1])
                else:
                    cut_n = np.dot(cutting_normal, c)
                d = np.dot(cop - cut_p, cut_n)
                spos = a.get_scaled_positions()[indices]
                scop = np.mean(spos, axis=0)
                p.append(Positions(spos, scop, s, d, i))

        all_points = p1 + p2
        unique_sym = np.unique(sym)
        types = {s: sym.count(s) for s in unique_sym}

        # Sort these by chemical symbols:
        all_points.sort(key=lambda x: x.symbols, reverse=True)

        # For each atom type make the pairing
        unique_sym.sort()
        use_total = {}
        for s in unique_sym:
            used = []
            not_used = []
            # The list is looked trough in
            # reverse order so atoms can be removed
            # from the list along the way.
            for i in reversed(range(len(all_points))):
                # If there are no more atoms of this type
                if all_points[i].symbols != s:
                    break
                # Check if the atom should be included
                if all_points[i].to_use():
                    used.append(all_points.pop(i))
                else:
                    not_used.append(all_points.pop(i))

            assert len(used) + len(not_used) == types[s] * 2

            # While we have too few of the given atom type
            while len(used) < types[s]:
                index = self.rng.integers(len(not_used))
                used.append(not_used.pop(index))

            # While we have too many of the given atom type
            while len(used) > types[s]:
                # remove randomly:
                index = self.rng.integers(len(used))
                not_used.append(used.pop(index))

            use_total[s] = used

        n_tot = sum(len(ll) for ll in use_total.values())
        assert n_tot == len(sym)

        # check if the generated structure contains
        # atoms from both parents:
        count1, count2, N = 0, 0, len(a1)
        for x in use_total.values():
            count1 += sum(y.origin == 0 for y in x)
            count2 += sum(y.origin == 1 for y in x)

        nmin = 1 if self.minfrac is None else max(1, int(np.ceil(self.minfrac * N)))
        if count1 < nmin or count2 < nmin:
            return None

        # Construct the cartesian positions and reorder the atoms
        # to follow the original order
        newpos = []
        pbc = a1.get_pbc()
        for s in sym:
            p = use_total[s].pop()
            c = a1.get_cell() if p.origin == 0 else a2.get_cell()
            pos = np.dot(p.scaled_positions, c)
            cop = np.dot(p.cop, c)
            vectors, _lengths = find_mic(pos - cop, c, pbc)
            newcop = np.dot(p.cop, cell)
            pos = newcop + vectors
            newpos.extend(pos)

        newpos = np.reshape(newpos, (N, 3))
        num = a1.get_atomic_numbers()
        child = Atoms(numbers=num, positions=newpos, pbc=pbc, cell=cell,
                      tags=tags)
        if len(self.slab) == 0:
            child.center()
        return child


class DualCutAndSplicePairing:
    """Wrapper that probabilistically delegates to one of two pairing operators.

    With probability ``exploratory_probability`` the exploratory (more
    asymmetric) cut-and-splice variant is used; otherwise the primary one.

    Parameters
    ----------
    primary : CutAndSplicePairing
        The default pairing operator.
    exploratory : CutAndSplicePairing
        A more exploratory variant (e.g. lower ``minfrac``).
    exploratory_probability : float
        Probability of choosing *exploratory* per call.
    rng : numpy.random.Generator or None
        Random number generator for the selection coin flip.
    """

    def __init__(
        self,
        primary: CutAndSplicePairing,
        exploratory: CutAndSplicePairing,
        exploratory_probability: float = 0.2,
        *,
        rng=None,
    ):
        self.primary = primary
        self.exploratory = exploratory
        self.exploratory_probability = exploratory_probability
        self.rng = _ensure_rng(rng)

    def get_new_individual(self, parents):
        if self.rng.random() < self.exploratory_probability:
            return self.exploratory.get_new_individual(parents)
        return self.primary.get_new_individual(parents)

    def update_scaling_volume(self, population, **kwargs):
        self.primary.update_scaling_volume(population, **kwargs)
        self.exploratory.update_scaling_volume(population, **kwargs)


# fmt: on
