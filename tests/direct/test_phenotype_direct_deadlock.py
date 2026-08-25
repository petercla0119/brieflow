"""Regression test for the phenotype merge fork-after-threading deadlock (refs #5).

Background
----------
``scripts/direct/run_phenotype_direct.py::run_parallel()`` runs its pool workers.
Before the fix it used the default ``ProcessPoolExecutor``, which FORKS on Linux.
Earlier pipeline steps start threads (skimage / scipy / mahotas / numexpr / BLAS);
a fork inherits those threads' locked mutexes WITHOUT the owning thread, so the
forked merge workers deadlock forever in ``futex_wait`` -- the merge step hung.

The fix (commit a57fe11): a module-level
``_MP = multiprocessing.get_context("spawn")`` passed as ``mp_context=_MP`` so
every worker starts from a clean interpreter with no inherited locks.

What this guards
----------------
PRIMARY -- deterministic, ZERO timing dependence. The property the fix
guarantees is that run_parallel workers start from a CLEAN interpreter (spawn),
not a fork of the already-mutated parent. We mutate a module-level attribute of
the runner in the PARENT *after* import, then run a worker that reports whether
it observes that mutation.
  - fork  (pre-fix): child inherits parent state -> worker sees it   -> FAIL
  - spawn (fixed):   child re-imports fresh       -> worker unaware  -> PASS

SECONDARY -- timeout-guarded faithful repro. A background thread holds a
module-global lock across the pool submit, and each worker tries to acquire it.
  - spawn (fixed): worker re-imports -> fresh unlocked lock -> completes -> PASS
  - fork  (pre-fix): worker inherits the lock LOCKED by a thread that does not
    exist in the child -> deadlock. The whole repro runs inside a spawn-isolated
    child process with a hard wall-clock watchdog that KILLS the process group on
    timeout, so a genuine deadlock FAILS the test but can never wedge pytest.

Run: pytest tests/direct/test_phenotype_direct_deadlock.py
 or: python tests/direct/test_phenotype_direct_deadlock.py
"""
import multiprocessing
import os
import queue as _queue
import signal
import sys
import threading
import types
from pathlib import Path


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


# Stub the heavy lib.shared.* imports the runner does at module load so this test
# is fast and env-independent (same trick as test_phenotype_direct_niceties.py).
# This block re-runs on every spawn/fork re-import of this module -- exactly what
# makes the PRIMARY test valid: a fresh import never carries the parent's runtime
# mutation.
_stub("lib")
_stub("lib.shared")
_stub("lib.shared.file_utils", get_data_output_path=lambda *a, **k: "",
      get_image_output_path=lambda *a, **k: "", validate_dtypes=lambda df: df)
_stub("lib.shared.image_io", read_image=lambda *a, **k: None, save_image=lambda *a, **k: None)
_stub("lib.shared.illumination_correction", apply_ic_field=lambda *a, **k: None)
_stub("lib.shared.parquet_io", write_parquet=lambda df, p: df.to_parquet(p),
      read_parquets=lambda *a, **k: None)
_stub("lib.shared.rule_utils", get_alignment_params=lambda *a, **k: {},
      get_segmentation_params=lambda *a, **k: {})

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_phenotype_direct as rpd  # noqa: E402


# ---------------------------------------------------------------------------
# PRIMARY: deterministic fork-vs-spawn interpreter-cleanliness probe
# ---------------------------------------------------------------------------
# run_parallel expects fn(task) -> (status, msg) and returns the count of "err".
# We encode the observation in the status so a fork is a nonzero err count.
def _probe_sees_parent_mutation(task):
    # _FORK_SENTINEL is set on the runner module in the PARENT at runtime only;
    # it is NOT set at module load. A forked child inherits it; a spawned child
    # re-imports the runner fresh and never sees it.
    if getattr(rpd, "_FORK_SENTINEL", None) is not None:
        return ("err", "worker inherited parent-only state => pool FORKED")
    return ("ok", "clean interpreter => pool spawned")


def test_run_parallel_workers_start_from_clean_interpreter():
    # Complementary (tautological) sanity: the fix is textually present.
    assert rpd._MP.get_start_method() == "spawn"

    # The real, behavioral guard -- deterministic, no timing.
    rpd._FORK_SENTINEL = os.getpid()
    try:
        errs = rpd.run_parallel(
            tasks=[0, 1, 2, 3],
            fn=_probe_sees_parent_mutation,
            workers=2,
            label="fork-vs-spawn probe",
        )
    finally:
        del rpd._FORK_SENTINEL
    assert errs == 0, (
        "run_parallel workers observed parent-only state -> the pool forked; "
        "the spawn fix for the merge deadlock (issue #5) has regressed"
    )


# ---------------------------------------------------------------------------
# SECONDARY: timeout-guarded faithful fork-after-threading deadlock repro
# ---------------------------------------------------------------------------
_GLOBAL_LOCK = threading.Lock()


def _worker_acquire_global_lock(task):
    # A forked child inherits _GLOBAL_LOCK in whatever state the parent left it
    # (LOCKED, by a holder thread that does not exist in the child) -> deadlock.
    # A spawned child re-imports this module -> a fresh unlocked lock -> returns.
    with _GLOBAL_LOCK:
        return ("ok", "acquired")


def _threaded_parent_body(result_q):
    # Runs inside the spawn-isolated watchdog child. Detach into its own process
    # group so the watchdog can reap any deadlocked grandchild pool workers.
    os.setsid()
    holder_ready = threading.Event()
    release = threading.Event()

    def _holder():
        _GLOBAL_LOCK.acquire()
        holder_ready.set()
        release.wait()          # keep the lock held across the pool submit
        _GLOBAL_LOCK.release()

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    holder_ready.wait()         # lock is now definitely held by the bg thread
    try:
        errs = rpd.run_parallel(
            tasks=[0, 1, 2, 3],
            fn=_worker_acquire_global_lock,
            workers=2,
            label="fork-after-threading repro",
        )
        result_q.put(("done", errs))
    finally:
        release.set()


def _run_with_watchdog(target, timeout=45):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        p.join(5)
        raise AssertionError(
            f"run_parallel did not complete within {timeout}s under a threaded "
            "parent holding a lock -> fork-after-threading deadlock (issue #5) "
            "has regressed"
        )
    assert p.exitcode == 0, f"watchdog child exited abnormally (exitcode={p.exitcode})"
    try:
        status, errs = q.get(timeout=5)
    except _queue.Empty:
        raise AssertionError("watchdog child produced no result")
    assert status == "done" and errs == 0, f"repro reported failure: {status} errs={errs}"


def test_run_parallel_survives_fork_after_threading():
    _run_with_watchdog(_threaded_parent_body, timeout=45)


if __name__ == "__main__":
    test_run_parallel_workers_start_from_clean_interpreter()
    test_run_parallel_survives_fork_after_threading()
    print("OK: fork-vs-spawn probe + fork-after-threading watchdog pass")
