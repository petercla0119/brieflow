"""Structural guard on .github/workflows/test_analysis.yml.

CI coverage is easy to delete by accident -- dropping a matrix value or a thread
cap is a one-line edit that leaves the workflow valid and green. These are the
invariants from #52/#53/#54 that must survive future edits.
"""

from pathlib import Path

import yaml

WORKFLOW_FP = (
    Path(__file__).resolve().parents[1] / ".github/workflows/test_analysis.yml"
)
WORKFLOW = yaml.safe_load(WORKFLOW_FP.read_text())
JOB = WORKFLOW["jobs"]["test-analysis"]
STEPS = JOB["steps"]

THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)
DIRECT_RUNNERS = (
    "run_preprocess_direct.py",
    "run_sbs_direct.py",
    "run_phenotype_direct.py",
    "run_merge_direct.py",
)


def step(substring):
    """The one step whose `run` mentions `substring`."""
    matches = [s for s in STEPS if substring in s.get("run", "")]
    assert len(matches) == 1, f"expected 1 step running {substring!r}, got {matches}"
    return matches[0]


def test_matrix_has_all_three_legs():
    assert JOB["strategy"]["matrix"]["leg"] == ["tiff", "zarr", "direct"]


def test_fail_fast_is_off():
    # A zarr-only failure must still report the tiff result.
    assert JOB["strategy"]["fail-fast"] is False


def test_zarr_leg_passes_the_zarr_flag():
    run = step("run_brieflow.sh")["run"]
    assert "matrix.leg == 'zarr' && '--zarr'" in run


def test_snakemake_step_skips_the_direct_leg():
    assert step("run_brieflow.sh")["if"] == "matrix.leg != 'direct'"


def test_direct_leg_invokes_all_four_runners():
    run = step("run_preprocess_direct.py")["run"]
    for runner in DIRECT_RUNNERS:
        assert runner in run, f"{runner} missing from the direct leg"


def test_direct_leg_aborts_on_the_first_runner_failure():
    # `shell: bash -l {0}` is not -e; without this only the last runner's exit
    # code reaches GitHub.
    assert "set -e" in step("run_preprocess_direct.py")["run"]


def test_direct_leg_sets_thread_caps():
    env = step("run_preprocess_direct.py")["env"]
    for var in THREAD_CAPS:
        # 2-core hosted runner / --workers 2 -> floor(2/2) == 1.
        assert env.get(var) == 1, f"{var} not capped to 1 (got {env.get(var)!r})"


def test_every_leg_verifies_its_outputs():
    # Exit 0 is not a test; each leg asserts on artifacts.
    verify = step("verify_outputs.py")
    assert "if" not in verify, "output verification must run on every leg"
    assert "${{ matrix.leg }}" in verify["run"]


def test_pytest_runs_once_and_reports_skips():
    pytest_step = step("pytest -rs")
    assert pytest_step["if"] == "matrix.leg == 'tiff'"
    # -rs keeps the broad-cpu-only post-seg E2E skip legible in the run summary
    # instead of letting it read as coverage (#55).
    assert "-rs" in pytest_step["run"]


def test_concurrency_group_is_per_run_not_per_leg():
    concurrency = WORKFLOW["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    assert (
        concurrency["group"]
        == "${{ github.workflow }}-${{ github.head_ref || github.ref }}"
    )
    # Adding matrix values to the group would let a stale leg outlive the run
    # that superseded it.
    assert "matrix" not in concurrency["group"]
