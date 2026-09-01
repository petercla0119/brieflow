"""Integration smoke test for the phenotype post-seg pipeline.

Runs the real `run_phenotype_direct.py --step post-seg` entry point on a couple
of REUSED plate-4 label tiles: the nuclei + cells masks are copied into a temp
scratch and the cytoplasm labels are stripped, so identify_cytoplasm_cellpose
actually re-runs. Asserts the whole post-seg chain

    identify_cytoplasm -> extract_phenotype_info -> extract_phenotype
    -> combine -> merge -> eval

completes and produces sane phenotype parquet.

This is a REGRESSION GUARD for breakages seen in this codebase's history, not a
correctness oracle (that is test_identify_cytoplasm_cellpose.py). It skips
cleanly when the plate-4 masks / real config / preprocess metadata fixtures are
absent, so it never hard-fails in an environment without fixtures.

Run:  pytest -q scripts/direct/tests/test_postseg_e2e_smoke.py
Uses 2 wells x 1 tile to also exercise the per-well combine/merge path. Fast
(~40s) once identify_cytoplasm_cellpose is vectorized; slower on the pre-merge
per-label loop.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

def _thread_cap(workers: int) -> str:
    # Fill idle cores instead of pinning BLAS/OMP at 2: size threads to this box.
    # ponytail: ceiling 12 — BLAS/OMP gains flatten past ~8-16; raise if a run shows headroom.
    # Override per-run with THREADS=N. Keep production Cheaha sbatch caps at 2 (admin-kill rule).
    cores = os.cpu_count() or 4
    return os.environ.get("THREADS") or str(max(2, min(12, cores // max(1, workers))))


# --- fixed paths (override via env) ------------------------------------------
WT = Path(os.environ.get("WT", str(Path(__file__).resolve().parents[3])))
RUNNER = WT / "scripts" / "direct" / "run_phenotype_direct.py"
sys.path.insert(0, str(WT / "workflow"))

ANALYSIS = Path(os.environ.get("ANALYSIS", "/mnt/work/broad-analysis/broad-tdp-gws/analysis"))
REAL_PHEN = ANALYSIS / "brieflow_output" / "phenotype"
REAL_META = ANALYSIS / "brieflow_output" / "preprocess" / "metadata"
REAL_CONFIG = ANALYSIS / "config" / "config.yml"
PLATE = 4
STORE = f"aligned_{PLATE}.zarr"

REQUIRED_MIN_COLS = {"label", "plate", "well", "tile", "cell_i", "cell_j"}


def _read_image_or_skip():
    try:
        from lib.shared.image_io import read_image
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot import lib.shared.image_io ({e})")
    return read_image


def _discover_well_tiles(n_wells=2):
    """One (row, col, tile) per well for up to n_wells distinct wells that have
    BOTH nuclei and cells label groups. Empty list if the store is absent."""
    root = REAL_PHEN / STORE
    if not root.is_dir():
        return []
    picks = []
    for row_dir in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[A-Za-z]+", p.name)):
        for col_dir in sorted((p for p in row_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            for t_dir in sorted((p for p in col_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
                lbl = t_dir / "labels"
                if (lbl / "nuclei").exists() and (lbl / "cells").exists():
                    picks.append((row_dir.name, col_dir.name, t_dir.name))
                    break
            if len(picks) >= n_wells:
                return picks
    return picks


@pytest.fixture(scope="module")
def postseg_run(tmp_path_factory):
    # Skip cleanly if any required fixture is missing.
    if not RUNNER.exists():
        pytest.skip(f"runner not found: {RUNNER}")
    if not REAL_CONFIG.exists():
        pytest.skip(f"real config not found: {REAL_CONFIG}")
    if not REAL_META.exists():
        pytest.skip(f"preprocess metadata not found: {REAL_META}")
    picks = _discover_well_tiles(2)
    if not picks:
        pytest.skip(f"no plate-{PLATE} nuclei+cells label tiles under {REAL_PHEN / STORE}")

    import yaml

    scratch = tmp_path_factory.mktemp("postseg_e2e")
    root_fp = scratch / "out"
    store_dst = root_fp / "phenotype" / STORE
    combo_rows = ["plate\tround\twell\ttile"]
    for row, col, tile in picks:
        src = REAL_PHEN / STORE / row / col / tile
        dst = store_dst / row / col / tile
        shutil.copytree(src, dst)
        # Strip cytoplasm so identify_cytoplasm_cellpose actually re-runs (else
        # out_exists() would skip it and the code under guard never executes).
        shutil.rmtree(dst / "labels" / "identified_cytoplasms", ignore_errors=True)
        combo_rows.append(f"{PLATE}\t1\t{row}{col}\t{tile}")

    # Symlink real preprocess metadata (read-only) so the eval steps have inputs.
    (root_fp / "preprocess").mkdir(parents=True, exist_ok=True)
    (root_fp / "preprocess" / "metadata").symlink_to(REAL_META)

    combo_fp = scratch / "combo.tsv"
    combo_fp.write_text("\n".join(combo_rows) + "\n")

    cfg = yaml.safe_load(REAL_CONFIG.read_text())
    cfg["all"]["root_fp"] = str(root_fp) + "/"
    cfg["all"]["image_format"] = "zarr"
    cfg["preprocess"]["phenotype_combo_fp"] = str(combo_fp)
    cfg_fp = scratch / "config.yml"
    cfg_fp.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = dict(os.environ)
    workers = 2  # 2 tiles -> 2 processes; each gets the rest of the box as threads
    tcap = _thread_cap(workers)
    env.update(OMP_NUM_THREADS=tcap, MKL_NUM_THREADS=tcap,
               OPENBLAS_NUM_THREADS=tcap, NUMEXPR_MAX_THREADS=tcap)
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(cfg_fp),
         "--step", "post-seg", "--plate-filter", str(PLATE), "--workers", str(workers)],
        cwd=str(WT / "scripts" / "direct"),
        env=env, capture_output=True, text=True, timeout=1800,
    )
    return picks, root_fp, proc


def test_postseg_runs_without_exception(postseg_run):
    # (a) The post-seg entry point completes cleanly (exit 0, "0 errors").
    #     Guards any per-step crash across identify -> extract -> combine ->
    #     merge -> eval.
    picks, root_fp, proc = postseg_run
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}\n--- stdout tail ---\n"
        f"{proc.stdout[-3000:]}\n--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert "0 errors" in proc.stdout, proc.stdout[-2000:]


def test_cytoplasm_labels_nontrivial(postseg_run):
    # (b) identify_cytoplasm_cellpose regenerated a non-trivial label image for
    #     every tile. A None return (mismatched nuclei/cells label sets) makes
    #     the runner write an all-zero cytoplasm -> unique == {0} -> this fails
    #     loudly, surfacing exactly that regression.
    read_image = _read_image_or_skip()
    picks, root_fp, proc = postseg_run
    for row, col, tile in picks:
        cyto_p = root_fp / "phenotype" / STORE / row / col / tile / "labels" / "identified_cytoplasms"
        assert cyto_p.exists(), f"cytoplasm labels missing for {row}{col}/{tile}"
        cyto = read_image(cyto_p)
        n_labels = len(set(np.unique(cyto).tolist()) - {0})
        assert n_labels > 0, f"cytoplasm all-background for {row}{col}/{tile} (gate returned None?)"


def test_phenotype_parquet_shape_and_dtypes(postseg_run):
    picks, root_fp, proc = postseg_run
    pq_dir = root_fp / "phenotype" / "parquets"
    parquets = sorted(pq_dir.rglob("*phenotype_cp.parquet"))  # excludes *_cp_min / _info
    assert parquets, f"no phenotype_cp.parquet under {pq_dir}"

    col_sets = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        # (c) Downstream parquet exists and is non-empty. Guards a zero-row /
        #     empty-concat silent write.
        assert len(df) > 0, f"empty phenotype parquet: {pq}"
        missing = REQUIRED_MIN_COLS - set(df.columns)
        assert not missing, f"{pq} missing columns {missing}"
        # plate/tile wildcard int-cast dtype (cp_emulator, ref 6421d94): a
        # regression there resurfaces as object/str plate|tile and silently
        # breaks downstream integer joins. Pin them to integer.
        assert np.issubdtype(df["plate"].dtype, np.integer), f"plate dtype {df['plate'].dtype} not int ({pq})"
        assert np.issubdtype(df["tile"].dtype, np.integer), f"tile dtype {df['tile'].dtype} not int ({pq})"
        col_sets.append(frozenset(df.columns))

    # Cross-well schema mismatch: every well's per-well parquet must share one
    # column schema, else the downstream cross-well concat/merge silently
    # coerces or drops columns.
    if len(col_sets) > 1:
        assert len(set(col_sets)) == 1, "phenotype parquet schema differs across wells"
