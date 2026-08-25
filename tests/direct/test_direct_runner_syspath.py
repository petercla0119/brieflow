"""Regression test: each direct runner's sys.path insert must resolve to this
repo's real workflow/ dir, never a stale/nonexistent layout.

Guards GitHub issue #6: run_preprocess_direct.py inserted
`SCRIPT_DIR.parents[1] / "brieflow" / "workflow"` -- the `brieflow/` segment is
the OLD nested layout that no longer exists, so the runner silently imported the
env-installed main-branch lib instead of this worktree's. Correct is
`parents[1] / "workflow"`. The sibling runners were already correct; this test
covers all four so any future runner with the same class of bug is caught too.

Pure static/path checks: no data I/O, no pipeline exec, no heavy lib imports,
so it is also immune to the "env-fallthrough" gotcha (the env has main-branch
brieflow installed editable, which would mask the bug at import time).

Run: pytest tests/direct/test_direct_runner_syspath.py
"""
import ast
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIRECT_DIR = _REPO_ROOT / "scripts" / "direct"
_WORKFLOW = _REPO_ROOT / "workflow"

RUNNERS = sorted(_DIRECT_DIR.glob("run_*_direct.py"))


def _inserted_paths(runner: Path):
    """Statically evaluate the sys.path.insert(...) targets of a runner.

    Executes only module-level assignments and `sys.path.insert(...)` calls in a
    sandbox (real `Path`, a fake `sys`, the runner's real `__file__`), so path
    exprs like `SCRIPT_DIR.parents[1] / "workflow"` resolve exactly as at
    runtime -- without importing the runner's heavy lib deps.
    """
    tree = ast.parse(runner.read_text())
    fake_sys = types.SimpleNamespace(path=[])
    ns = {"Path": Path, "sys": fake_sys, "__file__": str(runner)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module([node], []), str(runner), "exec"), ns)
            except Exception:
                pass  # skip assignments needing deps we deliberately didn't load
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "insert"
            and isinstance(node.value.func.value, ast.Attribute)
            and node.value.func.value.attr == "path"  # i.e. sys.path.insert(...)
        ):
            exec(compile(ast.Module([node], []), str(runner), "exec"), ns)
    return [Path(p).resolve() for p in fake_sys.path]


def test_runners_discovered():
    # sanity: all four direct runners present and the repo workflow lib exists
    assert {r.name for r in RUNNERS} == {
        "run_preprocess_direct.py",
        "run_sbs_direct.py",
        "run_phenotype_direct.py",
        "run_merge_direct.py",
    }, {r.name for r in RUNNERS}
    assert (_WORKFLOW / "lib").is_dir(), _WORKFLOW


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.name)
def test_syspath_inserts_exist(runner):
    # the core guard: a spurious segment (e.g. `brieflow/`) -> nonexistent dir
    paths = _inserted_paths(runner)
    assert paths, f"{runner.name}: no sys.path.insert found"
    for p in paths:
        assert p.is_dir(), f"{runner.name}: sys.path insert points at missing dir: {p}"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.name)
def test_syspath_points_at_repo_workflow(runner):
    paths = _inserted_paths(runner)
    ok = {_WORKFLOW.resolve(), (_WORKFLOW / "lib").resolve()}
    assert ok & set(paths), (
        f"{runner.name}: no sys.path insert resolves to repo workflow/. got {paths}"
    )
    for p in paths:  # nothing may escape this worktree into site-packages/env
        assert _REPO_ROOT == p or _REPO_ROOT in p.parents, (
            f"{runner.name}: sys.path insert escapes worktree: {p}"
        )


def test_preprocess_target_module_present():
    # stronger, still hermetic: the module the preprocess runner imports
    # convert_to_array from must exist under a resolved insert AND actually
    # define that symbol -- proves the right copy is on the path, no env leak.
    paths = _inserted_paths(_DIRECT_DIR / "run_preprocess_direct.py")
    mod = None
    for base in paths:
        for cand in (base / "lib" / "preprocess" / "preprocess.py",
                     base / "preprocess" / "preprocess.py"):
            if cand.is_file():
                mod = cand
                break
        if mod:
            break
    assert mod is not None, f"preprocess.py not found under any insert: {paths}"
    assert _REPO_ROOT in mod.parents, f"resolves outside worktree: {mod}"
    assert "def convert_to_array" in mod.read_text(), mod
