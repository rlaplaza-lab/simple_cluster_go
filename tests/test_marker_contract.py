"""AST guards for pytest marker contracts used by CI.

CPU CI fast excludes ``integration``; slow selects ``slow and not benchmark``.
``integration`` without ``slow`` is never collected.

Kaggle runs MACE and UPET kernels separately. A ``requires_cuda`` test without
an MLIP marker (``requires_mace`` / ``requires_upet`` / ``requires_uma``) is
selected by **both** kernels and wastes GPU minutes.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

_MLIP_MARKERS = frozenset({"requires_mace", "requires_upet", "requires_uma"})


def _pytest_mark_name(node: ast.AST) -> str | None:
    """Return ``X`` from ``pytest.mark.X`` / ``pytest.mark.X(...)``, else None."""
    if isinstance(node, ast.Call):
        node = node.func
    if not isinstance(node, ast.Attribute):
        return None
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if (
        isinstance(cur, ast.Name)
        and cur.id == "pytest"
        and len(parts) >= 2
        and parts[-1] == "mark"
    ):
        return parts[-2]
    return None


def _markers_from_expr(expr: ast.AST) -> set[str]:
    name = _pytest_mark_name(expr)
    if name is not None:
        return {name}
    if isinstance(expr, ast.Call):
        names = _markers_from_expr(expr.func)
        for arg in expr.args:
            names |= _markers_from_expr(arg)
        for kw in expr.keywords:
            if kw.arg == "marks":
                names |= _markers_from_expr(kw.value)
        return names
    if isinstance(expr, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for elt in expr.elts:
            names |= _markers_from_expr(elt)
        return names
    return set()


def _collect_test_items(path: Path) -> list[tuple[str, set[str]]]:
    """Return ``(label, markers)`` for each test function in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_marks: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    module_marks |= _markers_from_expr(node.value)

    items: list[tuple[str, set[str]]] = []
    rel = path.relative_to(TESTS_ROOT)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                marks = module_marks.copy()
                for dec in node.decorator_list:
                    marks |= _markers_from_expr(dec)
                items.append((f"{rel}::{node.name}", marks))
        elif isinstance(node, ast.ClassDef):
            class_marks = module_marks.copy()
            for dec in node.decorator_list:
                class_marks |= _markers_from_expr(dec)
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not item.name.startswith("test_"):
                    continue
                marks = class_marks.copy()
                for dec in item.decorator_list:
                    marks |= _markers_from_expr(dec)
                items.append((f"{rel}::{node.name}::{item.name}", marks))
    return items


def test_integration_tests_are_also_marked_slow() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        for label, marks in _collect_test_items(path):
            if "integration" in marks and "slow" not in marks:
                offenders.append(label)
    assert not offenders, (
        "integration without slow (never selected by CPU CI):\n"
        + "\n".join(f"  - {o}" for o in offenders)
    )


def test_requires_cuda_tests_pin_an_mlip_suite() -> None:
    """CUDA tests must carry requires_mace / requires_upet / requires_uma.

    Otherwise both Kaggle kernels collect them. Parametrize marks on
    ``pytest.param(..., marks=...)`` are included via AST.
    """
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        for label, marks in _collect_test_items(path):
            if "requires_cuda" in marks and not (marks & _MLIP_MARKERS):
                offenders.append(label)
    assert not offenders, (
        "requires_cuda without an MLIP marker (runs on both Kaggle suites):\n"
        + "\n".join(f"  - {o}" for o in offenders)
    )


def _module_level_imported_modules(path: Path) -> set[str]:
    """Return modules imported at the top of ``path`` (not inside functions)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


_MACE_HELPERS_MODULE = "scgo.calculators.mace_helpers"
_MACE_ROOT_PACKAGE = "mace"


def _imports_mace_at_module_level(names: set[str]) -> set[str]:
    """Return imported module names that pull ``mace`` in at import time.

    Matches both the exact helpers module and any dotted path ending in it,
    plus direct ``mace`` / ``mace.*`` imports.
    """
    hits: set[str] = set()
    for name in names:
        is_helpers = name == _MACE_HELPERS_MODULE or name.endswith(
            "." + _MACE_HELPERS_MODULE
        )
        is_mace = name == _MACE_ROOT_PACKAGE or name.startswith(
            _MACE_ROOT_PACKAGE + "."
        )
        if is_helpers or is_mace:
            hits.add(name)
    return hits


def test_mace_helpers_not_imported_at_collection_time() -> None:
    """``mace_helpers`` imports ``mace`` at module load.

    A top-level test import fails UMA/UPET collection even when tests are
    marked ``requires_mace`` (``pytest -m`` still imports the module).
    Direct ``mace`` / ``mace.*`` imports are equally fatal.
    """
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        hits = _imports_mace_at_module_level(_module_level_imported_modules(path))
        if hits:
            offenders.append(f"{path.relative_to(TESTS_ROOT)}: {sorted(hits)}")
    assert not offenders, (
        "module-level import of mace machinery "
        "(breaks UMA/UPET collection):\n" + "\n".join(f"  - {o}" for o in offenders)
    )
