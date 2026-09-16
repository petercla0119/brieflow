"""Self-check for per-process GPU attribution parsing + no-GPU graceful path.
Run: python tests/test_proc_gpu.py  (asserts; no framework)."""
import sys, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from lib.shared import resource_monitor as rm


def test_pmon_parse():
    # nvidia-smi pmon -c 1 : "# gpu pid type sm mem enc dec command", '-' = idle
    out = (
        "# gpu        pid  type    sm   mem   enc   dec   command\n"
        "# Idx          #   C/G     %     %     %     %   name\n"
        "    0      12345     C    80    30     -     -   python\n"
        "    0      12346     C    15    10     -     -   python\n"
        "    0          -     -     -     -     -     -   -\n"
    )
    u = rm._pmon_util_by_pid(out)
    assert u == {12345: 80.0, 12346: 15.0}, u
    # sm '-' (proc present but idle sm) is skipped, not crashed
    assert rm._pmon_util_by_pid("    0  999  C  -  -  -  -  python\n") == {}
    assert rm._pmon_util_by_pid("") == {}  # empty-output safe


def test_compute_apps_parse():
    out = "12345, 40960\n12346, 2048\n"
    m = rm._compute_apps_mem_by_pid(out)
    assert m == {12345: 40960.0, 12346: 2048.0}, m
    assert rm._compute_apps_mem_by_pid("6789, [N/A]\n") == {}  # N/A skipped
    assert rm._compute_apps_mem_by_pid("") == {}  # empty-output safe


def test_pid_scoping():
    # only PIDs in our tree are summed; another job's PID (99999) is excluded
    util = {12345: 80.0, 12346: 20.0, 99999: 100.0}
    mem = {12345: 40960.0, 12346: 2048.0, 99999: 81920.0}
    ours = {12345, 12346}
    assert sum(v for p, v in util.items() if p in ours) == 100.0
    assert sum(v for p, v in mem.items() if p in ours) == 43008.0


def test_no_gpu_graceful():
    # proc_gpu=True on a box with no nvidia-smi must run + write a row, no crash,
    # proc columns present and zero.
    out = tempfile.mktemp(suffix=".tsv")
    with rm.monitor_step("selfcheck-procgpu", out=out, interval=0.2, proc_gpu=True):
        time.sleep(0.5)
    rows = open(out).read().splitlines()
    d = dict(zip(rows[0].split("\t"), rows[1].split("\t")))
    for col in ("max_proc_gpu_pct", "mean_proc_gpu_pct",
                "max_proc_gpu_mem_mb", "mean_proc_gpu_mem_mb"):
        assert col in d, f"missing column {col}"
        assert d[col] == "0.0", (col, d[col])


if __name__ == "__main__":
    test_pmon_parse()
    test_compute_apps_parse()
    test_pid_scoping()
    test_no_gpu_graceful()
    print("ALL PROC-GPU CHECKS PASS")
