#!/usr/bin/env python3
"""Direct phenotype feature extraction runner — bypasses Snakemake DAG build.

Runs extract_phenotype for all tiles in parallel using ProcessPoolExecutor.
Assumes aligned images and segmentation masks already exist (output of upstream
Snakemake rules: align_phenotype, segment_phenotype, identify_cytoplasm).

Pool size: min(n_tiles, cpu_count - 4) so all cores stay productive after 1.1
lands (n_jobs=1 per tile, BLAS capped to 1 thread → each tile is single-threaded
and we can saturate the node without oversubscription).

Usage:
    python run_phenotype_direct.py --config config/config.yml
    python run_phenotype_direct.py --config config/config.yml --max-tiles 20
    python run_phenotype_direct.py --config config/config.yml --workers 40
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "workflow"))

from lib.shared.file_utils import get_data_output_path, get_image_output_path
from lib.shared.image_io import read_image
from lib.shared.parquet_io import write_parquet


# ---------------------------------------------------------------------------
# Path helpers (mirror targets/phenotype.smk logic)
# ---------------------------------------------------------------------------

def make_loc(img_fmt, plate, well=None, tile=None):
    loc = {"plate": str(plate)}
    if well is not None:
        w = str(well)
        if img_fmt == "zarr":
            m = re.match(r"^([A-Za-z]+)(\d+)$", w)
            loc["row"] = m.group(1) if m else w
            loc["col"] = m.group(2) if m else "0"
        else:
            loc["well"] = w
    if tile is not None:
        loc["tile"] = str(tile)
    return loc


def pheno_img_path(pheno_fp, fmt, plate, well, tile, info_type, subdirectory=None):
    loc = make_loc(fmt, plate, well, tile)
    return Path(pheno_fp) / get_image_output_path(loc, info_type, fmt, subdirectory=subdirectory)


def pheno_data_path(pheno_fp, fmt, plate, well, tile, info_type, ext):
    loc = make_loc(fmt, plate, well, tile)
    return Path(pheno_fp) / "parquets" / get_data_output_path(loc, info_type, ext, fmt)


def out_exists(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------

def run_parallel(tasks, fn, workers, label):
    n = len(tasks)
    if n == 0:
        print(f"  {label}: nothing to do")
        return 0
    ok = skip = err = 0
    t0 = time.time()
    print(f"\n  {label}: {n} tasks, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"    ERR: {msg}")
            total = ok + skip + err
            if total == n or total % max(1, n // 20) == 0:
                print(f"    [{total}/{n}] {time.time() - t0:.0f}s  new={ok} skip={skip} err={err}")
    print(f"  {label}: done in {time.time() - t0:.1f}s")
    return err


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------

def _extract_one(task):
    (aligned_path, nuclei_path, cells_path, cytoplasm_path,
     out_path, params) = task
    tag = Path(out_path).stem
    if out_exists(out_path):
        return "skip", tag
    try:
        # Cap BLAS threads — each tile worker is single-threaded; pool provides parallelism.
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1)
        except ImportError:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")

        from lib.shared.image_io import read_image
        from lib.shared.parquet_io import write_parquet
        from lib.phenotype.extract_phenotype_cp_emulator import (
            extract_phenotype_cp_emulator,
        )

        data = read_image(str(aligned_path))
        nuclei = read_image(str(nuclei_path))

        segment_cells = params.get("segment_cells", True)
        if segment_cells and cells_path and Path(cells_path).exists():
            cells = read_image(str(cells_path))
        else:
            cells = None

        if segment_cells and cytoplasm_path and Path(cytoplasm_path).exists():
            cytoplasms = read_image(str(cytoplasm_path))
        else:
            cytoplasms = None

        wildcards = params["wildcards"]
        result = extract_phenotype_cp_emulator(
            data_phenotype=data,
            nuclei=nuclei,
            cells=cells,
            cytoplasms=cytoplasms,
            foci_channel=params.get("foci_channel_index"),
            channel_names=params["channel_names"],
            wildcards=wildcards,
            n_jobs=1,  # tile-level parallelism — keep each worker single-threaded
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_parquet(result, str(out_path))
        return "ok", f"{tag} ({len(result)} cells)"
    except Exception as e:
        return "err", f"{tag}: {e}"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_phenotype(config, args):
    pheno_cfg = config["phenotype"]
    pp_cfg = config.get("preprocess", {})
    root = config["all"]["root_fp"]
    fmt = config.get("all", {}).get("image_format", "tiff")
    pheno_fp = Path(root) / "phenotype"

    # Load tile combos
    combo_fp = pp_cfg.get("phenotype_combo_fp") or pp_cfg.get("combo_fp")
    if combo_fp is None:
        # fall back: discover from existing aligned images
        combos = _discover_combos(pheno_fp, fmt)
    else:
        combos = pd.read_csv(combo_fp, sep="\t").astype(str)

    if args.plate_filter:
        combos = combos[combos["plate"] == str(args.plate_filter)]

    if args.max_tiles and "tile" in combos.columns:
        tiles = sorted(combos["tile"].unique(), key=lambda x: int(x))
        if len(tiles) > args.max_tiles:
            combos = combos[combos["tile"].isin(set(tiles[: args.max_tiles]))]

    # Unique tile combos (well-level, no cycle)
    tile_combos = combos[["plate", "well", "tile"]].drop_duplicates()

    if tile_combos.empty:
        print("  phenotype: no combos")
        return 0

    n_tiles = len(tile_combos)
    # Right-size pool: fill the node after 1.1 sets n_jobs=1 per tile
    workers = args.workers if args.workers else min(n_tiles, max(1, os.cpu_count() - 4))

    print(f"\n{'=' * 60}")
    print(f"  Phenotype: {n_tiles} tiles, {workers} workers (cpu_count={os.cpu_count()})")
    print(f"{'=' * 60}")

    channel_names = pheno_cfg["channel_names"]
    segment_cells = pheno_cfg.get("segment_cells", True)
    foci_channel_index = pheno_cfg.get("foci_channel_index")

    tasks = []
    for _, row in tile_combos.iterrows():
        p, we, ti = row["plate"], row["well"], row["tile"]
        aligned = pheno_img_path(pheno_fp, fmt, p, we, ti, "aligned")
        nuclei = pheno_img_path(pheno_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        cells = pheno_img_path(pheno_fp, fmt, p, we, ti, "cells", subdirectory="labels")
        cytoplasm = pheno_img_path(pheno_fp, fmt, p, we, ti, "identified_cytoplasms", subdirectory="labels")
        out = pheno_data_path(pheno_fp, fmt, p, we, ti, "phenotype_cp", "parquet")
        params = {
            "channel_names": channel_names,
            "segment_cells": segment_cells,
            "foci_channel_index": foci_channel_index,
            "wildcards": {"plate": p, "well": we, "tile": ti},
        }
        tasks.append((aligned, nuclei, cells, cytoplasm, out, params))

    return run_parallel(tasks, _extract_one, workers, "Extract phenotype")


def _discover_combos(pheno_fp, fmt):
    """Discover (plate, well, tile) from existing aligned images when no combo TSV is available."""
    import re as _re
    rows = []
    pattern = _re.compile(r"P-(\S+?)_W-(\S+?)_T-(\d+)__aligned")
    for p in Path(pheno_fp).rglob("*aligned*"):
        m = pattern.search(p.name)
        if m:
            rows.append({"plate": m.group(1), "well": m.group(2), "tile": m.group(3)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["plate", "well", "tile"])


def main():
    p = argparse.ArgumentParser(
        description="Direct phenotype feature extraction runner (bypasses Snakemake)"
    )
    p.add_argument("--config", required=True, help="Path to config.yml")
    p.add_argument("--max-tiles", type=int, default=None, help="Limit tiles processed")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: min(n_tiles, cpu_count - 4))",
    )
    p.add_argument("--plate-filter", type=int, default=None, help="Process only this plate")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    fmt = config.get("all", {}).get("image_format", "tiff")

    print(f"{'#' * 60}")
    print(f"  Direct Phenotype Runner | format={fmt}")
    print(f"  config={args.config}  max_tiles={args.max_tiles or 'all'}")
    print(f"  workers={args.workers or 'auto (cpu_count-4)'}  plate={args.plate_filter or 'all'}")
    print(f"{'#' * 60}")

    t0 = time.time()
    errs = process_phenotype(config, args)
    status = "DONE" if errs == 0 else "FAILED"
    print(f"\n{'#' * 60}")
    print(f"  {status}: {time.time() - t0:.1f}s total, {errs} errors")
    print(f"{'#' * 60}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
