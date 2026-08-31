"""Low-overhead per-step resource monitor for brieflow direct runners.

Samples the current process tree (parent + ProcessPoolExecutor workers) on a
background thread and records peak RSS and peak CPU% for a labeled step. One
row is appended per step to a shared TSV, so the direct runners produce
per-step resource data comparable to Snakemake's native ``benchmark:`` output.

Three ways to use it:

1. Context manager (inside a runner)::

       from lib.shared.resource_monitor import monitor_step
       with monitor_step("Segment SBS"):
           run_parallel(tasks, _segment_one, workers, "Segment SBS")

2. Standalone, wrapping any command (e.g. a GPU submit script)::

       python -m lib.shared.resource_monitor --label "Segment GPU" -- \
           python run_phenotype_direct.py --config config.yml --step segment

3. Standalone, attaching to an already-running PID::

       python -m lib.shared.resource_monitor --label foo --pid 12345

Output location (first match wins):
    --out FILE  >  $BRIEFLOW_BENCHMARK_DIR/direct_benchmarks.tsv  >  ./benchmarks/direct/direct_benchmarks.tsv

Sampling is cheap: one ``memory_info().rss`` + ``cpu_percent()`` per process
per tick (default 2 s), guarded so a monitoring error can never crash the step.
"""

import os
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # monitoring must never be a hard dependency of a step
    psutil = None

# ponytail: RSS-only (memory_info), not USS/PSS — USS/PSS reads /proc smaps
# every tick and defeats "low resource". Add memory_full_info() if per-step
# shared-memory accounting is ever actually needed.

COLUMNS = [
    "iso_time", "stage", "step", "plate", "machine_type", "n_workers",
    "wall_s", "max_rss_mb", "max_vms_mb",
    "max_cpu_pct", "mean_cpu_pct", "cpu_time_s", "read_mb", "write_mb",
    "peak_nproc",
    # GPU: nvidia-smi sampled on the same tick. gpu_count=0 (blank cols) off-GPU.
    # gpu_pct is fleet-mean util (mean across all GPUs) — peak fleet-mean near
    # 100/gpu_count means only ~1 GPU busy, i.e. paying for idle GPUs.
    "gpu_count", "max_gpu_pct", "mean_gpu_pct", "max_gpu_mem_mb",
    "n_samples",
]

# nvidia-smi path resolved once; None => no GPU tooling, GPU sampling skipped.
import shutil
_NVIDIA_SMI = shutil.which("nvidia-smi")

_DEFAULT_INTERVAL = float(os.environ.get("BRIEFLOW_MONITOR_INTERVAL", "2.0"))


def _detect_machine_type():
    """GCP machine type (e.g. 'n2-standard-64'), '' off-GCP. Env override wins.

    Queries the metadata server once; result is cached in the env by
    set_benchmark_context so this network hit happens at most once per run.
    """
    mt = os.environ.get("BRIEFLOW_MACHINE_TYPE")
    if mt is not None:
        return mt
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=1.0) as r:
            return r.read().decode().rsplit("/", 1)[-1]  # projects/.../machineTypes/X -> X
    except Exception:
        return ""  # not on GCP or metadata unreachable — leave blank


def _resolve_out(out):
    if out:
        return Path(out)
    env_dir = os.environ.get("BRIEFLOW_BENCHMARK_DIR")
    base = Path(env_dir) if env_dir else Path.cwd() / "benchmarks" / "direct"
    return base / "direct_benchmarks.tsv"


def _iso(t):
    # local time, no timezone math needed for a run log
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


class ResourceMonitor:
    """Sample a process tree's peak RSS/CPU until stopped, then append a row."""

    def __init__(self, step, pid=None, out=None, interval=None, stage=None,
                 n_workers=None, plate=None):
        self.step = step
        self.pid = pid or os.getpid()
        self.out = _resolve_out(out)
        self.interval = interval or _DEFAULT_INTERVAL
        self.stage = stage or os.environ.get("BRIEFLOW_STAGE", "")
        self.n_workers = n_workers if n_workers is not None else os.environ.get("BRIEFLOW_N_WORKERS", "")
        self.plate = plate if plate is not None else os.environ.get("BRIEFLOW_PLATE", "")
        self.machine_type = os.environ.get("BRIEFLOW_MACHINE_TYPE") or _detect_machine_type()
        self._stop = threading.Event()
        self._thread = None
        self._pcache = {}  # pid -> psutil.Process, reused so cpu_percent() deltas work
        self._io_baseline = {}  # pid -> (read_bytes, write_bytes) at first sight
        self._io_last = {}      # pid -> latest (read_bytes, write_bytes)
        self._io_ok = True      # cleared if io_counters unsupported (e.g. macOS)
        self.read_bytes = 0
        self.write_bytes = 0
        self.max_rss = 0
        self.max_vms = 0
        self.max_cpu = 0.0
        self._cpu_sum = 0.0
        self._samples = 0
        self.peak_nproc = 0
        self.cpu_time = 0.0
        self.t0 = None
        # GPU accumulators (populated only when nvidia-smi is present)
        self.gpu_count = 0
        self.max_gpu = 0.0       # peak fleet-mean util %
        self._gpu_sum = 0.0
        self._gpu_samples = 0
        self.max_gpu_mem = 0.0   # peak total mem used across GPUs (MB)
        self._gpu_ok = _NVIDIA_SMI is not None

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False  # never suppress the step's own exceptions

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if psutil is None:
            print(f"  [monitor] psutil unavailable — '{self.step}' not measured",
                  file=sys.stderr)
            return self
        self.t0 = time.time()
        self._prime_root_io()
        self._thread = threading.Thread(target=self._run, name="resmon", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if psutil is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        self._finalize_cpu_time()
        self._finalize_io()
        self._write()

    # -- internals -----------------------------------------------------------
    def _live_procs(self):
        """Current process tree, reusing cached Process objects per pid.

        cpu_percent(None) reports usage since the *same object's* last call,
        so objects must persist across ticks. New processes are primed (their
        first reading is 0) and dead ones are dropped.
        """
        try:
            root = self._pcache.get(self.pid) or psutil.Process(self.pid)
        except psutil.Error:
            return []
        self._pcache[self.pid] = root
        tree = [root]
        try:
            tree += root.children(recursive=True)
        except psutil.Error:
            pass
        result, seen = [], set()
        for p in tree:
            seen.add(p.pid)
            cached = self._pcache.get(p.pid)
            if cached is None:
                self._pcache[p.pid] = p
                try:
                    p.cpu_percent(None)  # prime; delta counts from here
                except psutil.Error:
                    pass
                result.append(p)
            else:
                result.append(cached)
        for pid in [k for k in self._pcache if k not in seen and k != self.pid]:
            del self._pcache[pid]
        return result

    def _run(self):
        # prime the root so the first real reading spans a full interval
        self._live_procs()
        while not self._stop.wait(self.interval):
            self._sample()
        # one final sample after stop so short steps still record something
        self._sample()

    def _sample(self):
        rss = vms = cpu = 0.0
        n = 0
        for p in self._live_procs():
            try:
                mi = p.memory_info()
                rss += mi.rss
                vms += mi.vms
                cpu += p.cpu_percent(None)
                n += 1
            except psutil.Error:
                continue  # process vanished mid-sample; skip it
            self._sample_io(p)
        if n == 0:
            return
        self.max_rss = max(self.max_rss, rss)
        self.max_vms = max(self.max_vms, vms)
        self.max_cpu = max(self.max_cpu, cpu)
        self.peak_nproc = max(self.peak_nproc, n)
        self._cpu_sum += cpu
        self._samples += 1
        self._sample_gpu()

    def _sample_gpu(self):
        # One nvidia-smi call per tick (~2s) — negligible next to a GPU step.
        # Records fleet-mean util and total mem used across all visible GPUs.
        # ponytail: node-wide (all GPUs on the box), not scoped to this PID's
        # CUDA context — fine for a dedicated single-step GPU VM; scope via
        # nvidia-smi pmon if a box ever runs concurrent GPU jobs.
        if not self._gpu_ok:
            return
        import subprocess
        try:
            out = subprocess.run(
                [_NVIDIA_SMI,
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            self._gpu_ok = False  # smi vanished/hung — stop trying
            return
        utils, mem_total = [], 0.0
        for line in out.splitlines():
            try:
                u, m = line.split(",")
                utils.append(float(u))
                mem_total += float(m)
            except ValueError:
                continue
        if not utils:
            return
        self.gpu_count = max(self.gpu_count, len(utils))
        fleet_mean = sum(utils) / len(utils)
        self.max_gpu = max(self.max_gpu, fleet_mean)
        self._gpu_sum += fleet_mean
        self._gpu_samples += 1
        self.max_gpu_mem = max(self.max_gpu_mem, mem_total)

    def _prime_root_io(self):
        # Baseline the root process's disk counters at step start so I/O done
        # before the first sample tick still counts (single-process steps do all
        # their I/O in root; pool workers are born after this and baseline at
        # first-sight instead, undercounting only their first ~interval of I/O).
        if not self._io_ok:
            return
        try:
            io = psutil.Process(self.pid).io_counters()
            self._io_baseline[self.pid] = (io.read_bytes, io.write_bytes)
        except (psutil.Error, AttributeError, NotImplementedError):
            self._io_ok = False

    def _sample_io(self, p):
        # Disk read/write bytes (block-device level, what iostat reports) per
        # step. io_counters() is cumulative per process; track first-seen as the
        # baseline and sum (last - baseline) across the tree at finalize.
        # ponytail: first-seen baseline undercounts a fresh worker's I/O in its
        # first sample interval (~2s) but never over-counts across steps; drop to
        # a (0,0) worker baseline if that undercount ever matters.
        if not self._io_ok:
            return
        try:
            io = p.io_counters()
        except (psutil.Error, AttributeError, NotImplementedError):
            self._io_ok = False  # unsupported on this platform - stop trying
            return
        rw = (io.read_bytes, io.write_bytes)
        self._io_baseline.setdefault(p.pid, rw)
        self._io_last[p.pid] = rw

    def _finalize_io(self):
        read = write = 0
        for pid, (r0, w0) in self._io_baseline.items():
            r1, w1 = self._io_last.get(pid, (r0, w0))
            read += max(0, r1 - r0)   # max: guard counter reset on pid reuse
            write += max(0, w1 - w0)
        self.read_bytes = read
        self.write_bytes = write

    def _finalize_cpu_time(self):
        # best-effort: parent's cumulative CPU time (children that already
        # exited are not fully counted — that's the documented ceiling)
        try:
            ct = psutil.Process(self.pid).cpu_times()
            self.cpu_time = round(ct.user + ct.system, 1)
        except psutil.Error:
            self.cpu_time = 0.0

    def _write(self):
        wall = time.time() - self.t0 if self.t0 else 0.0
        mean_cpu = self._cpu_sum / self._samples if self._samples else 0.0
        row = {
            "iso_time": _iso(self.t0 or time.time()),
            "stage": self.stage,
            "step": self.step,
            "plate": self.plate,
            "machine_type": self.machine_type,
            "n_workers": self.n_workers,
            "wall_s": round(wall, 1),
            "max_rss_mb": round(self.max_rss / 1e6, 1),
            "max_vms_mb": round(self.max_vms / 1e6, 1),
            "max_cpu_pct": round(self.max_cpu, 1),
            "mean_cpu_pct": round(mean_cpu, 1),
            "cpu_time_s": self.cpu_time,
            "read_mb": round(self.read_bytes / 1e6, 1),
            "write_mb": round(self.write_bytes / 1e6, 1),
            "peak_nproc": self.peak_nproc,
            "gpu_count": self.gpu_count,
            "max_gpu_pct": round(self.max_gpu, 1),
            "mean_gpu_pct": round(self._gpu_sum / self._gpu_samples, 1) if self._gpu_samples else 0.0,
            "max_gpu_mem_mb": round(self.max_gpu_mem, 1),
            "n_samples": self._samples,
        }
        header = "\t".join(COLUMNS)
        try:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            # If an existing file has a different schema (e.g. an older run
            # without the plate/machine/workers columns), move it aside so we
            # never append mismatched rows into one un-parseable TSV.
            if self.out.exists():
                with open(self.out) as f:
                    old_header = f.readline().rstrip("\n")
                if old_header != header:
                    self.out.rename(self.out.with_suffix(self.out.suffix + ".oldschema.bak"))
            new = not self.out.exists()
            with open(self.out, "a") as f:
                if new:
                    f.write(header + "\n")
                f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        except OSError as e:
            print(f"  [monitor] could not write benchmark row: {e}", file=sys.stderr)
        gpu = (f"gpu={row['max_gpu_pct']}%x{row['gpu_count']} "
               f"gpu_mem={row['max_gpu_mem_mb']}MB " if row["gpu_count"] else "")
        print(f"  [monitor] {self.step}: peak_rss={row['max_rss_mb']}MB "
              f"peak_cpu={row['max_cpu_pct']}% "
              f"read={row['read_mb']}MB write={row['write_mb']}MB "
              f"{gpu}wall={row['wall_s']}s")


def monitor_step(step, **kw):
    """Convenience factory mirroring the class, for `with monitor_step(...):`."""
    return ResourceMonitor(step, **kw)


def set_benchmark_context(stage, root_fp, plate=None):
    """Point per-step benchmarks at <root_fp>/benchmarks/direct and tag rows.

    Call once at the top of a direct runner's main(). Honors a pre-set
    BRIEFLOW_BENCHMARK_DIR env var so the location can still be overridden.
    ``plate`` tags every row with the plate this run processes ('' if the run
    spans all plates). The GCP machine type is detected once and cached here.
    """
    os.environ["BRIEFLOW_STAGE"] = stage
    os.environ["BRIEFLOW_PLATE"] = "" if plate is None else str(plate)
    os.environ.setdefault("BRIEFLOW_MACHINE_TYPE", _detect_machine_type())
    os.environ.setdefault(
        "BRIEFLOW_BENCHMARK_DIR", str(Path(root_fp) / "benchmarks" / "direct"))


# ---------------------------------------------------------------------------
# Standalone CLI: wrap a command or attach to a PID
# ---------------------------------------------------------------------------

def _main(argv):
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(
        description="Monitor peak RSS/CPU of a process tree for one step.")
    ap.add_argument("--label", required=True, help="Step name recorded in the TSV")
    ap.add_argument("--stage", default=None, help="Optional stage tag column")
    ap.add_argument("--plate", default=None, help="Plate tag column")
    ap.add_argument("--workers", type=int, default=None, help="n_workers tag column")
    ap.add_argument("--out", default=None, help="Output TSV (default: env/cwd)")
    ap.add_argument("--interval", type=float, default=None, help="Sample seconds")
    ap.add_argument("--pid", type=int, default=None,
                    help="Attach to an existing PID and exit when it exits")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- COMMAND ARGS to launch and monitor")
    args = ap.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd

    if args.pid and cmd:
        ap.error("give either --pid or a command, not both")

    if cmd:
        proc = subprocess.Popen(cmd)
        mon = ResourceMonitor(args.label, pid=proc.pid, out=args.out,
                              interval=args.interval, stage=args.stage,
                              n_workers=args.workers, plate=args.plate)
        mon.start()
        rc = proc.wait()
        mon.stop()
        return rc

    if args.pid:
        if psutil is None:
            print("psutil unavailable", file=sys.stderr)
            return 1
        mon = ResourceMonitor(args.label, pid=args.pid, out=args.out,
                              interval=args.interval, stage=args.stage,
                              n_workers=args.workers, plate=args.plate)
        mon.start()
        try:
            while psutil.pid_exists(args.pid):
                time.sleep(mon.interval)
        except KeyboardInterrupt:
            pass
        mon.stop()
        return 0

    ap.error("provide --pid or a -- COMMAND to monitor")


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
