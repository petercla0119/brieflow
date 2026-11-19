"""Guard the tests/direct/ stubs against the runners' import list drifting away.

The tests in `tests/direct/` fake `lib.shared.*` so they can import a runner without
the heavy scientific stack. Each fake is built from an explicit kwarg list, so adding
a name to a runner's `from lib.shared.resource_monitor import ...` line silently
breaks every one of those tests -- at *collection* time, with

    ImportError: cannot import name '<new>' from 'lib.shared.resource_monitor'

That is how `plates_from_combos` broke four test modules at once: it was added to all
four runners' import line and to none of the stubs. Same family as the `proc_gpu`
drift guarded by test_direct_runner_monitor_kwargs.py -- code and stub disagreeing,
caught only when something actually imports.

Static (AST only), so it needs none of the runners' heavy imports.
"""

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflow"))

MODULE = "lib.shared.resource_monitor"
RUNNERS = sorted((_REPO_ROOT / "scripts" / "direct").glob("run_*_direct.py"))
DIRECT_TESTS = sorted((_REPO_ROOT / "tests" / "direct").glob("test_*.py"))


def _imported_names(source, module):
    """Names imported from `module` by `source` (ast.ImportFrom)."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(a.name for a in node.names)
    return names


def _stubbed_names(source, module):
    """Kwarg names passed to a _stub("<module>", ...) call, or None if not stubbed."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "_stub" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == module:
            return {kw.arg for kw in node.keywords if kw.arg}
    return None


def test_runners_import_something_from_resource_monitor():
    """Sanity: the guard below is vacuous if nothing imports from the module."""
    assert RUNNERS, "no run_*_direct.py found"
    combined = set()
    for r in RUNNERS:
        combined |= _imported_names(r.read_text(), MODULE)
    assert combined, f"no runner imports from {MODULE}; this guard would be vacuous"


@pytest.mark.parametrize("test_file", DIRECT_TESTS, ids=lambda p: p.name)
def test_direct_stubs_cover_every_imported_name(test_file):
    stubbed = _stubbed_names(test_file.read_text(), MODULE)
    if stubbed is None:
        pytest.skip(f"{test_file.name} does not stub {MODULE}")

    needed = set()
    for r in RUNNERS:
        needed |= _imported_names(r.read_text(), MODULE)

    missing = needed - stubbed
    assert not missing, (
        f"{test_file.name} stubs {MODULE} but omits {sorted(missing)}, which "
        f"run_*_direct.py imports. Importing the runner in that test will fail at "
        f"collection with ImportError. Add the name to the _stub(...) call."
    )
