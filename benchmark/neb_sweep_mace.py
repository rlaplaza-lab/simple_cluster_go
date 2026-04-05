#!/usr/bin/env python3
"""Sweep NEB hyperparameters with TorchSim+MACE on gas and/or surface endpoint pairs.

**Synthetic endpoints** (default gas / fallback surface): Cu₃ triangle–linear for gas;
a small fcc111+adsorbate toy pair for surface (often fails to converge — for real
surface–nanoparticle work use ``--surface-searches-dir``).

**Real minima** (recommended): point ``--surface-searches-dir`` and/or
``--gas-searches-dir`` at an SCGO campaign with ``run_*/`` databases and
``final_unique_minima/*.xyz``, same layout as :func:`run_transition_state_search`.
Use ``--pair-index`` to choose the starting pair and ``--primary-pair-count N`` to
sweep the same hyperparameter grid over the first N consecutive pairs (capped by
available pairs) for more representative tuning.

**Surface examples** (after running the graphene TS runners):

- ``ts_search_graphene_results/<ELEMENT><n>_searches`` — e.g. Cu₄ on graphene
  (from ``runners/run_scgo_with_ts_search_graphene.py``).
- ``ts_search_graphene_with_oh_results/<composition>_searches`` — Cu₄OH on
  graphene (from ``runners/run_scgo_with_ts_search_graphene_with_oh.py``). Use
  this as an extra validation campaign; :func:`scgo.param_presets.get_ts_search_params`
  still exposes a single ``regime="surface"`` unless a dedicated OH sweep drives
  separate defaults.

Run in conda env ``scgo`` on a CUDA GPU, e.g.::

    CUDA_VISIBLE_DEVICES=0 python benchmark/neb_sweep_mace.py \\
        --regime surface --surface-searches-dir ts_search_graphene_results/Cu4_searches \\
        --output benchmark/neb_mace_surface_real.jsonl

Each JSONL line is one ``find_transition_state`` call with ``use_torchsim=True``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import fcc111

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
)
from scgo.runner_surface import read_full_composition_from_first_xyz
from scgo.ts_search.transition_state import find_transition_state
from scgo.ts_search.transition_state_io import (
    load_minima_by_composition,
    select_structure_pairs,
)
from scgo.utils.helpers import (
    auto_niter_ts,
    filter_unique_minima,
    get_cluster_formula,
)
from scgo.utils.logging import configure_logging, get_logger

# Primary grid covers interpolation × images × climb × spring × fmax × tangent.
# Use ``--list-followup-sweeps`` for axes worth exploring next (pairing, robustness, ML).
FOLLOWUP_SWEEP_AXES = """
Suggested follow-up sweeps (not in the primary product grid; add ad hoc loops or extend the script):

  Pairing / coverage
    - --pair-index 1,2,… on the same searches dir (different minima pairs)
    - --energy-gap-threshold and --similarity-tolerance / --similarity-pair-cor-max
    - Multiple campaigns (different slab sizes, adsorbates, metal identities)

  NEB path / ASE knobs
    - neb_tangent_method: use --include-aseneb-tangent vs default improvedtangent (primary grid)
    - neb_align_endpoints: use ``--grid extended`` to compare True/False (surface presets use False)
    - neb_perturb_sigma > 0 on interior images (rarely; can help stuck bands)
    - neb_retry_on_endpoint: use ``--sweep-retry-on-endpoint`` with ``--grid extended``

  Optimizer / budget
    - --neb-steps as a multiple of auto_niter_ts (e.g. 1.5×, 2×) for large slabs
    - FIRE vs other optimizers (requires a small code hook in find_transition_state)

  ML backend
    - --models mace_matpes_0,mace_mp_small (repeat full grid per model)
    - TorchSim relaxer float64 vs float32 (trade startup + throughput vs force noise)

  Production paths
    - use_parallel_neb True with several pairs (memory vs throughput)
    - validate_ts_by_frequency + imag_freq_threshold after NEB
"""


def print_followup_sweeps() -> None:
    print(FOLLOWUP_SWEEP_AXES.strip())


def resolve_searches_composition(
    searches_dir: Path,
    composition: list[str] | None,
) -> list[str]:
    """Return ``composition`` if given; else infer from ``final_unique_minima/*.xyz``."""
    if composition is not None:
        return list(composition)
    final_minima = searches_dir / "final_unique_minima"
    if final_minima.is_dir() and any(final_minima.glob("*.xyz")):
        return read_full_composition_from_first_xyz(final_minima)
    raise SystemExit(
        f"Could not infer composition: pass --composition / --surface-composition "
        f"or add xyz files under {final_minima}"
    )


@dataclass(frozen=True)
class EndpointBundle:
    """Reactant/product structures and metadata for the sweep."""

    reactant: Atoms
    product: Atoms
    composition: list[str]
    neb_steps: int
    meta: dict[str, Any]


def _prepare_minima_and_pairs(
    searches_dir: Path,
    composition: list[str] | None,
    *,
    energy_gap_threshold: float | None,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
    dedupe_minima: bool,
    minima_energy_tolerance: float,
) -> tuple[list[str], str, list[tuple[float, Atoms]], list[tuple[int, int]]]:
    """Return composition, formula, deduped minima, and pair index list."""
    comp = resolve_searches_composition(searches_dir, composition)
    formula = get_cluster_formula(comp)

    minima_by_formula = load_minima_by_composition(
        str(searches_dir.resolve()),
        composition=comp,
        prefer_final_unique=True,
    )
    minima = minima_by_formula.get(formula, [])
    if len(minima) < 2:
        raise SystemExit(
            f"Need at least two minima under {searches_dir} for formula {formula!r}; "
            f"found {len(minima)}."
        )

    if dedupe_minima:
        minima = filter_unique_minima(minima, minima_energy_tolerance)

    if len(minima) < 2:
        raise SystemExit(
            f"After deduplication, fewer than 2 minima remain for {searches_dir}."
        )

    pairs = select_structure_pairs(
        minima,
        max_pairs=None,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
    )
    return comp, formula, minima, pairs


def load_endpoint_bundle_from_searches(
    searches_dir: Path,
    composition: list[str] | None,
    *,
    pair_index: int,
    energy_gap_threshold: float | None,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
    dedupe_minima: bool,
    minima_energy_tolerance: float,
    neb_steps_override: int | None,
) -> EndpointBundle:
    """Load first (or indexed) minima pair the same way as ``run_transition_state_search``."""
    comp, formula, minima, pairs = _prepare_minima_and_pairs(
        searches_dir,
        composition,
        energy_gap_threshold=energy_gap_threshold,
        similarity_tolerance=similarity_tolerance,
        similarity_pair_cor_max=similarity_pair_cor_max,
        dedupe_minima=dedupe_minima,
        minima_energy_tolerance=minima_energy_tolerance,
    )
    if not pairs:
        raise SystemExit(
            f"No structure pairs passed filters under {searches_dir}. "
            "Try increasing --energy-gap-threshold or loosening --similarity-tolerance."
        )
    if pair_index < 0 or pair_index >= len(pairs):
        raise SystemExit(
            f"--pair-index {pair_index} out of range (0..{len(pairs) - 1} for "
            f"{len(pairs)} pair(s))."
        )

    i, j = pairs[pair_index]
    e_i, atoms_i = minima[i]
    e_j, atoms_j = minima[j]
    neb_steps = (
        int(neb_steps_override)
        if neb_steps_override is not None
        else auto_niter_ts(comp)
    )

    meta: dict[str, Any] = {
        "endpoint_source": "minima",
        "searches_dir": str(searches_dir.resolve()),
        "pair_index": int(pair_index),
        "pair_indices": [int(i), int(j)],
        "formula": formula,
        "reactant_energy_ev": float(e_i),
        "product_energy_ev": float(e_j),
        "energy_gap_ev": float(abs(e_j - e_i)),
        "n_minima_loaded": len(minima),
        "n_pairs_available": len(pairs),
        "neb_steps": neb_steps,
    }
    return EndpointBundle(
        reactant=atoms_i.copy(),
        product=atoms_j.copy(),
        composition=comp,
        neb_steps=neb_steps,
        meta=meta,
    )


def _synthetic_gas_bundle(neb_steps: int) -> EndpointBundle:
    d = 2.5
    tri_pos = [
        [0, 0, 0],
        [d, 0, 0],
        [d / 2, d * np.sqrt(3) / 2, 0],
    ]
    lin_pos = [[0, 0, 0], [d, 0, 0], [2 * d, 0, 0]]
    a = Atoms("Cu3", positions=tri_pos)
    a.center(vacuum=5.0)
    b = Atoms("Cu3", positions=lin_pos)
    b.center(vacuum=5.0)
    meta = {
        "endpoint_source": "synthetic",
        "formula": "Cu3",
        "neb_steps": neb_steps,
    }
    return EndpointBundle(
        reactant=a,
        product=b,
        composition=["Cu", "Cu", "Cu"],
        neb_steps=neb_steps,
        meta=meta,
    )


def _synthetic_surface_bundle(neb_steps: int) -> EndpointBundle:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    z0 = float(slab.get_positions()[:, 2].max() + 1.5)
    a = slab.copy() + Atoms("Pt", positions=[[1.0, 1.0, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[2.0, 2.0, z0]])
    n_pt = len(slab) + 1
    comp = ["Pt"] * n_pt
    meta = {
        "endpoint_source": "synthetic_surface_probe",
        "formula": get_cluster_formula(comp),
        "neb_steps": neb_steps,
    }
    return EndpointBundle(
        reactant=a,
        product=b,
        composition=comp,
        neb_steps=neb_steps,
        meta=meta,
    )


@dataclass(frozen=True)
class SweepCell:
    regime: str
    neb_interpolation_method: str
    neb_interpolation_mic: bool
    n_images: int
    neb_climb: bool
    spring_constant: float
    fmax: float
    neb_steps: int
    neb_tangent_method: str
    align_endpoints: bool = True
    perturb_sigma: float = 0.0
    neb_retry_on_endpoint: bool = True


def _primary_grid(
    *,
    regime: str,
    gas_fmax: list[float],
    surface_fmax: list[float],
    tangent_methods: list[str],
    neb_steps: int,
) -> Iterator[SweepCell]:
    methods = ("idpp", "linear")
    n_images_list = (3, 5, 7)
    climbs = (True, False)
    springs = (0.05, 0.1)

    if regime == "gas":
        fmax_opts = gas_fmax
        mic = False
    elif regime == "surface":
        fmax_opts = surface_fmax
        mic = True
    else:
        raise ValueError(regime)

    for combo in itertools.product(
        methods,
        n_images_list,
        climbs,
        springs,
        fmax_opts,
        tangent_methods,
    ):
        interp, n_img, climb, k, fmax, tang = combo
        yield SweepCell(
            regime=regime,
            neb_interpolation_method=interp,
            neb_interpolation_mic=mic,
            n_images=n_img,
            neb_climb=climb,
            spring_constant=k,
            fmax=fmax,
            neb_steps=neb_steps,
            neb_tangent_method=tang,
            align_endpoints=True,
            perturb_sigma=0.0,
            neb_retry_on_endpoint=True,
        )


def _extended_grid(
    *,
    regime: str,
    neb_steps: int,
    sweep_retry_on_endpoint: bool,
) -> Iterator[SweepCell]:
    """Winner settings from primary MACE sweeps × alignment × perturb × tangent."""
    mic = regime == "surface"
    climb = regime == "gas"
    fmax = 0.05 if regime == "gas" else 0.08
    retry_opts = (True, False) if sweep_retry_on_endpoint else (True,)
    for align, pert, tang, retry in itertools.product(
        (True, False),
        (0.0, 0.03),
        (DEFAULT_NEB_TANGENT_METHOD, "aseneb"),
        retry_opts,
    ):
        yield SweepCell(
            regime=regime,
            neb_interpolation_method="idpp",
            neb_interpolation_mic=mic,
            n_images=3,
            neb_climb=climb,
            spring_constant=0.1,
            fmax=fmax,
            neb_steps=neb_steps,
            neb_tangent_method=tang,
            align_endpoints=align,
            perturb_sigma=pert,
            neb_retry_on_endpoint=retry,
        )


def _row(
    cell: SweepCell,
    *,
    backend: str,
    mace_model_name: str,
    wall_s: float,
    result: dict[str, Any],
    endpoint_meta: dict[str, Any] | None = None,
    sweep_grid_tag: str = "primary_mace",
) -> dict[str, Any]:
    err = result.get("error")
    if err is not None and not isinstance(err, str):
        err = str(err)
    row: dict[str, Any] = {
        "backend": backend,
        "mace_model_name": mace_model_name,
        "regime": cell.regime,
        "spring_constant": cell.spring_constant,
        "fmax": cell.fmax,
        "neb_steps": cell.neb_steps,
        "neb_interpolation_method": cell.neb_interpolation_method,
        "neb_interpolation_mic": cell.neb_interpolation_mic,
        "neb_climb": cell.neb_climb,
        "n_images": cell.n_images,
        "neb_tangent_method": cell.neb_tangent_method,
        "neb_align_endpoints": cell.align_endpoints,
        "neb_perturb_sigma": cell.perturb_sigma,
        "neb_retry_on_endpoint": cell.neb_retry_on_endpoint,
        "wall_s": wall_s,
        "status": result.get("status"),
        "neb_converged": result.get("neb_converged"),
        "final_fmax": result.get("final_fmax"),
        "steps_taken": result.get("steps_taken"),
        "barrier_height": result.get("barrier_height"),
        "ts_image_index": result.get("ts_image_index"),
        "error": err,
        "sweep_grid": sweep_grid_tag,
    }
    if endpoint_meta:
        row.update(endpoint_meta)
    return row


def _barrier_suspicious(
    row: dict[str, Any],
    *,
    max_barrier_ev: float,
) -> bool:
    if not row.get("neb_converged"):
        return False
    bh = row.get("barrier_height")
    if bh is None:
        return False
    try:
        return float(bh) > max_barrier_ev
    except (TypeError, ValueError):
        return False


def summarize_jsonl(
    path: Path,
    *,
    max_barrier_ev: float = 3.0,
    physics_check: bool = False,
    top_n: int = 25,
) -> None:
    """Print best-effort ranking: converged first, then lowest final_fmax."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    def sort_key(r: dict[str, Any]) -> tuple[int, float]:
        failed = 1 if not r.get("neb_converged") else 0
        ff = r.get("final_fmax")
        ff_v = float(ff) if ff is not None else float("inf")
        return (failed, ff_v)

    rows.sort(key=sort_key)
    suspicious = sum(
        1 for r in rows if _barrier_suspicious(r, max_barrier_ev=max_barrier_ev)
    )
    if physics_check:
        print(
            f"(physics check: {suspicious} converged row(s) with barrier_height > "
            f"{max_barrier_ev} eV — likely misaligned periodic path or bad pair)",
            file=sys.stderr,
        )
    for r in rows[:top_n]:
        src = r.get("endpoint_source", "?")
        pair_ix = r.get("pair_index", "?")
        mark = ""
        if physics_check and _barrier_suspicious(r, max_barrier_ev=max_barrier_ev):
            mark = "[SUSPICIOUS_BARRIER] "
        retry = r.get("neb_retry_on_endpoint", True)
        print(
            f"{mark}{r.get('regime')} pair={pair_ix} src={src} model={r.get('mace_model_name')} "
            f"interp={r.get('neb_interpolation_method')} "
            f"nimg={r.get('n_images')} climb={r.get('neb_climb')} k={r.get('spring_constant')} "
            f"fmax={r.get('fmax')} tang={r.get('neb_tangent_method')} "
            f"align={r.get('neb_align_endpoints')} pert={r.get('neb_perturb_sigma')} "
            f"retry={retry} "
            f"conv={r.get('neb_converged')} final_fmax={r.get('final_fmax')} "
            f"barrier={r.get('barrier_height')} status={r.get('status')}"
        )


def _parse_composition(s: str | None) -> list[str] | None:
    if s is None or not s.strip():
        return None
    s = s.strip()
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return s.split()


def _parse_models(s: str | None, fallback: str) -> list[str]:
    if s is None or not str(s).strip():
        return [fallback]
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return parts if parts else [fallback]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-followup-sweeps",
        action="store_true",
        help="Print suggested sweep axes not covered by the primary grid, then exit",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        metavar="JSONL",
        default=None,
        help="Load JSONL, print ranked rows, and exit (no GPU run)",
    )
    parser.add_argument(
        "--summarize-physics-check",
        action="store_true",
        help="With --summarize: flag converged rows whose barrier_height exceeds "
        "--summarize-max-barrier-ev (stderr summary + [SUSPICIOUS_BARRIER] prefix)",
    )
    parser.add_argument(
        "--summarize-max-barrier-ev",
        type=float,
        default=3.0,
        metavar="EV",
        help="With --summarize-physics-check: barrier threshold in eV (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/neb_mace_primary.jsonl"),
        help="JSONL path to append results",
    )
    parser.add_argument(
        "--grid",
        choices=("primary", "extended"),
        default="primary",
        help="primary: full factorial (interpolation×images×…); extended: winner NEB settings "
        "× align_endpoints × perturb_sigma × tangent × multiple pairs",
    )
    parser.add_argument(
        "--extended-pair-count",
        type=int,
        default=3,
        metavar="N",
        help="With --grid extended: sweep N consecutive minima pairs per regime starting at "
        "--pair-index (capped by available pairs)",
    )
    parser.add_argument(
        "--primary-pair-count",
        type=int,
        default=1,
        metavar="N",
        help="With --grid primary and real searches dirs: repeat the primary factorial over N "
        "consecutive pairs starting at --pair-index (default: 1)",
    )
    parser.add_argument(
        "--regime",
        choices=("gas", "surface", "all"),
        default="all",
        help="Which endpoint systems to sweep",
    )
    parser.add_argument(
        "--gas-searches-dir",
        type=Path,
        default=None,
        help="SCGO searches dir for gas-phase real minima (run_*/ + DBs)",
    )
    parser.add_argument(
        "--surface-searches-dir",
        type=Path,
        default=None,
        help="SCGO searches dir for slab+nanoparticle real minima (required for meaningful surface tuning)",
    )
    parser.add_argument(
        "--gas-composition",
        type=str,
        default=None,
        help="Element symbols for gas minima (e.g. 'Cu Cu Cu'); inferred from final_unique_minima if omitted",
    )
    parser.add_argument(
        "--surface-composition",
        type=str,
        default=None,
        help="Element symbols for surface minima; inferred from final_unique_minima if omitted",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Starting pair index from select_structure_pairs (0-based); with "
        "--primary-pair-count / --extended-pair-count >1, pairs index, index+1, … are used",
    )
    parser.add_argument(
        "--energy-gap-threshold",
        type=float,
        default=1.0,
        help="eV: only pair minima with |ΔE| below this (None = any gap)",
    )
    parser.add_argument(
        "--similarity-tolerance",
        type=float,
        default=DEFAULT_COMPARATOR_TOL,
        help="Cumulative distance comparator tolerance (Å) for pairing",
    )
    parser.add_argument(
        "--similarity-pair-cor-max",
        type=float,
        default=0.1,
        help="Max single pair distance difference (Å) for pairing (matches TS runner default)",
    )
    parser.add_argument(
        "--no-energy-gap-filter",
        action="store_true",
        help="Pair minima regardless of energy gap (sets energy_gap_threshold=None)",
    )
    parser.add_argument(
        "--no-dedupe-minima",
        action="store_true",
        help="Skip filter_unique_minima before pairing",
    )
    parser.add_argument(
        "--minima-energy-tolerance",
        type=float,
        default=DEFAULT_ENERGY_TOLERANCE,
        help="Dedup energy tolerance (eV) when deduping minima",
    )
    parser.add_argument(
        "--neb-steps",
        type=int,
        default=None,
        help="Override NEB optimizer steps (default: 500 synthetic, auto_niter_ts for real minima)",
    )
    parser.add_argument(
        "--model",
        default="mace_matpes_0",
        help="MACE foundation model id for TorchSimBatchRelaxer (used if --models omitted)",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="IDS",
        help="Comma-separated MACE model ids; repeats the full cell grid once per model "
        "(large wall-time multiplier; pair with --max-cells for smoke tests)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for MACE (e.g. cuda:0)",
    )
    parser.add_argument(
        "--include-aseneb-tangent",
        action="store_true",
        help="Also run cells with neb_tangent_method='aseneb'",
    )
    parser.add_argument(
        "--sweep-retry-on-endpoint",
        action="store_true",
        help="With --grid extended only: also sweep neb_retry_on_endpoint True/False (doubles "
        "extended cell count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid size and endpoint summary, then exit",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        metavar="N",
        help="Run at most N cells (truncate grid; default: all)",
    )
    parser.add_argument(
        "--allow-synthetic-surface",
        action="store_true",
        help="If set, use toy slab pair when --surface-searches-dir is omitted",
    )
    args = parser.parse_args()

    if args.sweep_retry_on_endpoint and args.grid != "extended":
        raise SystemExit("--sweep-retry-on-endpoint requires --grid extended")

    if args.list_followup_sweeps:
        print_followup_sweeps()
        return

    if args.summarize is not None:
        summarize_jsonl(
            args.summarize,
            max_barrier_ev=args.summarize_max_barrier_ev,
            physics_check=args.summarize_physics_check,
        )
        return

    configure_logging(1)
    logger = get_logger(__name__)

    tangent_methods = [DEFAULT_NEB_TANGENT_METHOD]
    if args.include_aseneb_tangent and args.grid == "primary":
        tangent_methods.append("aseneb")

    regimes: list[str]
    if args.regime == "all":
        regimes = ["gas", "surface"]
    else:
        regimes = [args.regime]

    gas_fmax = [0.05]
    surface_fmax = [0.05, 0.08, 0.15]

    # Resolve endpoint bundles per regime (may differ in neb_steps when from minima)
    gas_comp_arg = _parse_composition(args.gas_composition)
    surf_comp_arg = _parse_composition(args.surface_composition)
    energy_gap: float | None = (
        None if args.no_energy_gap_filter else args.energy_gap_threshold
    )

    bundles: dict[str, EndpointBundle] = {}
    cells: list[tuple[SweepCell, EndpointBundle]] = []
    sweep_grid_tag = "primary_mace"

    if args.grid == "extended":
        sweep_grid_tag = "extended_mace"
        pair_kw = dict(
            energy_gap_threshold=energy_gap,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            dedupe_minima=not args.no_dedupe_minima,
            minima_energy_tolerance=args.minima_energy_tolerance,
            neb_steps_override=args.neb_steps,
        )
        for reg in regimes:
            if reg == "gas":
                sdir = args.gas_searches_dir
                comp_arg = gas_comp_arg
                if sdir is None:
                    raise SystemExit(
                        "--grid extended requires --gas-searches-dir when regime includes gas"
                    )
            else:
                sdir = args.surface_searches_dir
                comp_arg = surf_comp_arg
                if sdir is None:
                    raise SystemExit(
                        "--grid extended requires --surface-searches-dir when regime includes surface"
                    )

            _comp, _formula, _minima, pairs = _prepare_minima_and_pairs(
                sdir,
                comp_arg,
                energy_gap_threshold=energy_gap,
                similarity_tolerance=args.similarity_tolerance,
                similarity_pair_cor_max=args.similarity_pair_cor_max,
                dedupe_minima=not args.no_dedupe_minima,
                minima_energy_tolerance=args.minima_energy_tolerance,
            )
            if not pairs:
                raise SystemExit(
                    f"No structure pairs for extended sweep under {sdir}. "
                    "Loosen filters or use --no-energy-gap-filter."
                )
            start_pi = args.pair_index
            if start_pi < 0 or start_pi >= len(pairs):
                raise SystemExit(
                    f"--pair-index {start_pi} out of range (0..{len(pairs) - 1} for "
                    f"{len(pairs)} pair(s))."
                )
            n_pairs_try = min(max(1, args.extended_pair_count), len(pairs) - start_pi)
            ext_cells_per_pair = 8 * (2 if args.sweep_retry_on_endpoint else 1)
            logger.info(
                "Extended sweep %s: %d pair(s) × %d cells (align×pert×tangent%s)",
                reg,
                n_pairs_try,
                ext_cells_per_pair,
                "×retry" if args.sweep_retry_on_endpoint else "",
            )
            for offset in range(n_pairs_try):
                pi = start_pi + offset
                bundle = load_endpoint_bundle_from_searches(
                    sdir,
                    comp_arg,
                    pair_index=pi,
                    **pair_kw,
                )
                bundles[f"{reg}_pair{pi}"] = bundle
                for cell in _extended_grid(
                    regime=reg,
                    neb_steps=bundle.neb_steps,
                    sweep_retry_on_endpoint=args.sweep_retry_on_endpoint,
                ):
                    cells.append((cell, bundle))
    else:
        pair_load_kw = dict(
            energy_gap_threshold=energy_gap,
            similarity_tolerance=args.similarity_tolerance,
            similarity_pair_cor_max=args.similarity_pair_cor_max,
            dedupe_minima=not args.no_dedupe_minima,
            minima_energy_tolerance=args.minima_energy_tolerance,
            neb_steps_override=args.neb_steps,
        )
        for reg in regimes:
            if reg == "gas":
                if args.gas_searches_dir is not None:
                    _gc, _gf, _gmin, pairs = _prepare_minima_and_pairs(
                        args.gas_searches_dir,
                        gas_comp_arg,
                        energy_gap_threshold=energy_gap,
                        similarity_tolerance=args.similarity_tolerance,
                        similarity_pair_cor_max=args.similarity_pair_cor_max,
                        dedupe_minima=not args.no_dedupe_minima,
                        minima_energy_tolerance=args.minima_energy_tolerance,
                    )
                    if not pairs:
                        raise SystemExit(
                            f"No structure pairs under {args.gas_searches_dir}. "
                            "Loosen filters or use --no-energy-gap-filter."
                        )
                    start_pi = args.pair_index
                    if start_pi < 0 or start_pi >= len(pairs):
                        raise SystemExit(
                            f"--pair-index {start_pi} out of range (0..{len(pairs) - 1} for "
                            f"{len(pairs)} pair(s))."
                        )
                    n_try = min(max(1, args.primary_pair_count), len(pairs) - start_pi)
                    for offset in range(n_try):
                        pi = start_pi + offset
                        bundle = load_endpoint_bundle_from_searches(
                            args.gas_searches_dir,
                            gas_comp_arg,
                            pair_index=pi,
                            **pair_load_kw,
                        )
                        bundles[f"{reg}_pair{pi}"] = bundle
                        logger.info(
                            "Gas endpoints from minima (pair %s): %s", pi, bundle.meta
                        )
                        for cell in _primary_grid(
                            regime=reg,
                            gas_fmax=gas_fmax,
                            surface_fmax=surface_fmax,
                            tangent_methods=tangent_methods,
                            neb_steps=bundle.neb_steps,
                        ):
                            cells.append((cell, bundle))
                else:
                    step = args.neb_steps if args.neb_steps is not None else 500
                    bundles[reg] = _synthetic_gas_bundle(step)
                    bundle = bundles[reg]
                    for cell in _primary_grid(
                        regime=reg,
                        gas_fmax=gas_fmax,
                        surface_fmax=surface_fmax,
                        tangent_methods=tangent_methods,
                        neb_steps=bundle.neb_steps,
                    ):
                        cells.append((cell, bundle))
            else:
                if args.surface_searches_dir is not None:
                    _sc, _sf, _smin, pairs = _prepare_minima_and_pairs(
                        args.surface_searches_dir,
                        surf_comp_arg,
                        energy_gap_threshold=energy_gap,
                        similarity_tolerance=args.similarity_tolerance,
                        similarity_pair_cor_max=args.similarity_pair_cor_max,
                        dedupe_minima=not args.no_dedupe_minima,
                        minima_energy_tolerance=args.minima_energy_tolerance,
                    )
                    if not pairs:
                        raise SystemExit(
                            f"No structure pairs under {args.surface_searches_dir}. "
                            "Loosen filters or use --no-energy-gap-filter."
                        )
                    start_pi = args.pair_index
                    if start_pi < 0 or start_pi >= len(pairs):
                        raise SystemExit(
                            f"--pair-index {start_pi} out of range (0..{len(pairs) - 1} for "
                            f"{len(pairs)} pair(s))."
                        )
                    n_try = min(max(1, args.primary_pair_count), len(pairs) - start_pi)
                    for offset in range(n_try):
                        pi = start_pi + offset
                        bundle = load_endpoint_bundle_from_searches(
                            args.surface_searches_dir,
                            surf_comp_arg,
                            pair_index=pi,
                            **pair_load_kw,
                        )
                        bundles[f"{reg}_pair{pi}"] = bundle
                        logger.info(
                            "Surface endpoints from minima (pair %s): %s",
                            pi,
                            bundle.meta,
                        )
                        for cell in _primary_grid(
                            regime=reg,
                            gas_fmax=gas_fmax,
                            surface_fmax=surface_fmax,
                            tangent_methods=tangent_methods,
                            neb_steps=bundle.neb_steps,
                        ):
                            cells.append((cell, bundle))
                else:
                    if not args.allow_synthetic_surface:
                        raise SystemExit(
                            "Surface regime requires real minima for meaningful tuning. "
                            "Pass --surface-searches-dir PATH (SCGO output with run_*/ DBs), "
                            "or pass --allow-synthetic-surface to use the old toy slab pair."
                        )
                    warnings.warn(
                        "Using synthetic surface probe; use --surface-searches-dir for "
                        "surface-adsorbed nanoparticle minima.",
                        stacklevel=1,
                    )
                    step = args.neb_steps if args.neb_steps is not None else 500
                    bundles[reg] = _synthetic_surface_bundle(step)
                    bundle = bundles[reg]
                    for cell in _primary_grid(
                        regime=reg,
                        gas_fmax=gas_fmax,
                        surface_fmax=surface_fmax,
                        tangent_methods=tangent_methods,
                        neb_steps=bundle.neb_steps,
                    ):
                        cells.append((cell, bundle))

    model_names = _parse_models(args.models, args.model)
    base_cell_count = len(cells)
    cells = [(c, b, m) for m in model_names for c, b in cells]

    if args.max_cells is not None:
        cells = cells[: max(0, args.max_cells)]

    if args.dry_run:
        by_reg: defaultdict[str, int] = defaultdict(int)
        for cell, _b, _m in cells:
            by_reg[cell.regime] += 1
        print(
            f"Would run {len(cells)} NEB evaluations (--grid {args.grid}, "
            f"{len(model_names)} model(s))",
            file=sys.stderr,
        )
        print(
            f"  base cells (before model repeat): {base_cell_count}",
            file=sys.stderr,
        )
        for reg, n in sorted(by_reg.items()):
            print(f"  cells with regime={reg!r}: {n}", file=sys.stderr)
        for bk, b in sorted(bundles.items()):
            print(f"  bundle {bk}: {b.meta}", file=sys.stderr)
        return

    dev = torch.device(args.device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "This benchmark expects a CUDA device; set --device cuda:0 and run on a GPU box.",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i, (cell, bundle, model_name) in enumerate(cells, 1):
        atoms_a = bundle.reactant.copy()
        atoms_b = bundle.product.copy()

        torchsim_params: dict[str, Any] = {
            "mace_model_name": model_name,
            "device": dev,
            "dtype": torch.float32,
            "force_tol": cell.fmax,
            "max_steps": cell.neb_steps,
        }

        endpoint_meta = dict(bundle.meta)
        endpoint_meta["sweep_regime"] = cell.regime
        endpoint_meta["n_atoms"] = int(len(bundle.reactant))
        endpoint_meta["pbc"] = [bool(x) for x in bundle.reactant.pbc]

        rng = (
            np.random.default_rng(42)
            if cell.perturb_sigma and cell.perturb_sigma > 0.0
            else None
        )

        with tempfile.TemporaryDirectory(prefix="neb_sweep_mace_") as tmp:
            t0 = time.perf_counter()
            result = find_transition_state(
                atoms_a,
                atoms_b,
                calculator=None,
                output_dir=tmp,
                pair_id=f"sweep_{cell.regime}_{i}",
                rng=rng,
                n_images=cell.n_images,
                spring_constant=cell.spring_constant,
                fmax=cell.fmax,
                neb_steps=cell.neb_steps,
                verbosity=0,
                use_torchsim=True,
                torchsim_params=torchsim_params,
                climb=cell.neb_climb,
                interpolation_method=cell.neb_interpolation_method,
                align_endpoints=cell.align_endpoints,
                perturb_sigma=cell.perturb_sigma,
                neb_interpolation_mic=cell.neb_interpolation_mic,
                neb_retry_on_endpoint=cell.neb_retry_on_endpoint,
                neb_tangent_method=cell.neb_tangent_method,
            )
            wall_s = time.perf_counter() - t0

        row = _row(
            cell,
            backend="torchsim_mace",
            mace_model_name=model_name,
            wall_s=wall_s,
            result=result,
            endpoint_meta=endpoint_meta,
            sweep_grid_tag=sweep_grid_tag,
        )
        with args.output.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[{i}/{len(cells)}] model={model_name} {cell.regime} {cell.neb_interpolation_method} "
            f"nimg={cell.n_images} climb={cell.neb_climb} k={cell.spring_constant} "
            f"fmax={cell.fmax} tang={cell.neb_tangent_method} "
            f"align={cell.align_endpoints} pert={cell.perturb_sigma} "
            f"retry={cell.neb_retry_on_endpoint} -> "
            f"conv={row['neb_converged']} status={row['status']}",
            flush=True,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
