"""Guard the direct runners' monitor_step(...) call against signature drift.

`run_parallel` in every direct runner wraps its pool in `monitor_step(...)`, and
`monitor_step` forwards **kw straight into `ResourceMonitor.__init__`. A kwarg
that the monitor does not accept is therefore a TypeError raised on the *first*
parallel step of a real run -- not at import, not at lint, not in any unit test.

That is exactly how `proc_gpu=` shipped: added to `run_parallel` and passed
through, never added to `ResourceMonitor`, so `run_sbs_direct.py` and
`run_phenotype_direct.py` both died on their first task batch. Nothing caught it
because nothing in CI ran the direct runners (#54).

Static, so it costs milliseconds and needs none of the runners' heavy imports.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflow"))

from lib.shared.resource_monitor import ResourceMonitor  # noqa: E402

RUNNERS = sorted((_REPO_ROOT / "scripts" / "direct").glob("run_*_direct.py"))


def monitor_step_kwargs(source):
    """Every keyword name passed to a monitor_step(...) call in `source`."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "id", None) == "monitor_step" or getattr(fn, "attr", "") == (
            "monitor_step"
        ):
            names.update(kw.arg for kw in node.keywords if kw.arg)
    return names


def test_runners_were_found():
    # A glob that silently matches nothing would make every test below vacuous.
    assert len(RUNNERS) >= 4, [p.name for p in RUNNERS]


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_monitor_step_kwargs_are_accepted_by_resource_monitor(runner):
    accepted = set(inspect.signature(ResourceMonitor.__init__).parameters) - {"self"}
    passed = monitor_step_kwargs(runner.read_text())
    unknown = passed - accepted
    assert not unknown, (
        f"{runner.name} passes {sorted(unknown)} to monitor_step(), which "
        f"ResourceMonitor.__init__ does not accept {sorted(accepted)}. This is a "
        f"TypeError on the first parallel step of a real run."
    )
