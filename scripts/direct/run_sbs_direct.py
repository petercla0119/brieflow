#!/usr/bin/env python3
"""Direct SBS runner — bypasses Snakemake DAG build.

Runs all SBS processing steps (align → log_filter → std_dev → peaks →
max_filter → apply_IC → segment → extract_bases → call_reads → call_cells →
extract_sbs_info → combine → eval) with ProcessPoolExecutor parallelism.

Usage:
    python run_sbs_direct.py --config config/config.yml --max-tiles 100 --workers 8
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "workflow"))

from lib.shared.file_utils import get_data_output_path, get_image_output_path, validate_dtypes
from lib.shared.image_io import read_image, save_image
from lib.shared.illumination_correction import apply_ic_field, combine_ic_images
from lib.shared.parquet_io import write_parquet, read_parquets
from lib.shared.rule_utils import get_call_cells_params, get_segmentation_params, get_spot_detection_params
from lib.shared.resource_monitor import monitor_step, set_benchmark_context


# ---------------------------------------------------------------------------
# Path helpers (same as preprocessing script)
# ---------------------------------------------------------------------------

def make_loc(img_fmt, plate, well=None, tile=None, cycle=None):
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
    if cycle is not None:
        loc["cycle"] = str(cycle)
    return loc


def out_exists(path):
    p = Path(path)
    if p.name == "zarr.json":
        return p.exists()
    if p.suffix == ".zarr":
        return p.is_dir() and any(p.iterdir()) if p.exists() else False
    return p.exists() and p.stat().st_size > 0


def sbs_img_path(sbs_fp, fmt, plate, well, tile, info_type, subdirectory=None):
    loc = make_loc(fmt, plate, well, tile)
    return str(sbs_fp / get_image_output_path(loc, info_type, fmt, subdirectory=subdirectory))


def sbs_data_path(sbs_fp, fmt, plate, well, tile, info_type, ext):
    loc = make_loc(fmt, plate, well, tile)
    return str(sbs_fp / "tsvs" / get_data_output_path(loc, info_type, ext, fmt))


def sbs_well_path(sbs_fp, fmt, plate, well, info_type, ext):
    loc = make_loc(fmt, plate, well)
    return str(sbs_fp / "parquets" / get_data_output_path(loc, info_type, ext, fmt))


def sbs_plate_path(sbs_fp, fmt, plate, info_type, ext, subdir):
    loc = make_loc(fmt, plate)
    return str(sbs_fp / "eval" / subdir / get_data_output_path(loc, info_type, ext, fmt))


def preprocess_img_path(pp_fp, fmt, plate, well, tile, cycle):
    loc = make_loc(fmt, plate, well, tile, cycle)
    return str(pp_fp / get_image_output_path(loc, "image", fmt, image_subdir="sbs"))


def preprocess_ic_path(pp_fp, fmt, plate, well, cycle):
    loc = make_loc(fmt, plate, well, cycle=cycle)
    return str(pp_fp / "ic_fields" / "sbs" / get_data_output_path(loc, "ic_field", fmt, fmt))


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------

def _worker_init_gpu(num_gpus, omp_threads=4):
    """Worker initializer for GPU steps. Pins worker to a GPU via pid hash."""
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(os.getpid() % num_gpus)


def run_parallel(tasks, fn, workers, label, initializer=None, initargs=()):
    n = len(tasks)
    if n == 0:
        print(f"  {label}: nothing to do")
        return 0
    ok = skip = err = 0
    t0 = time.time()
    print(f"\n  {label}: {n} tasks, {workers} workers")
    with monitor_step(label), ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
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
    elapsed = time.time() - t0
    print(f"  {label}: done in {elapsed:.1f}s")
    print(f"  PERF: {label}: {n} tasks, {workers} workers, {elapsed:.1f}s")
    return err


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------

def _align_one(task):
    cycle_paths, output_path, sbs_cfg = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        images = [read_image(p) for p in cycle_paths]
        from lib.sbs.align_cycles import align_cycles
        aligned = align_cycles(
            images,
            channel_order=sbs_cfg["channel_names"],
            method=sbs_cfg.get("alignment_method"),
            upsample_factor=sbs_cfg.get("upsample_factor", 2),
            window=sbs_cfg.get("window", 2),
            skip_cycles=sbs_cfg.get("skip_cycles_indices"),
            manual_background_cycle=sbs_cfg.get("manual_background_cycle_index"),
            manual_channel_mapping=sbs_cfg.get("manual_channel_mapping"),
        )
        save_image(aligned, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _log_filter_one(task):
    input_path, output_path, skip_index = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.shared.log_filter import log_filter
        data = read_image(input_path)
        result = log_filter(aligned_image_data=data, skip_index=skip_index)
        save_image(result, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _std_dev_one(task):
    input_path, output_path, remove_index = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.sbs.compute_standard_deviation import compute_standard_deviation
        data = read_image(input_path)
        result = compute_standard_deviation(log_filtered_data=data, remove_index=remove_index)
        save_image(result, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _find_peaks_one(task):
    input_path, output_path, spot_params = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = read_image(input_path)
        method = spot_params.get("method", "standard")
        if method == "standard":
            from lib.sbs.find_peaks import find_peaks
            peaks = find_peaks(standard_deviation_data=data, width=spot_params["peak_width"])
        elif method == "spotiflow":
            from lib.sbs.find_peaks import find_peaks_spotiflow
            peaks, _ = find_peaks_spotiflow(
                aligned_images=data,
                cycle_idx=spot_params["spotiflow_cycle_index"],
                model=spot_params["spotiflow_model"],
                prob_thresh=spot_params["spotiflow_threshold"],
                min_distance=spot_params["spotiflow_min_distance"],
                remove_index=spot_params["remove_index"],
            )
        save_image(peaks, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _max_filter_one(task):
    input_path, output_path, width, remove_index = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.sbs.max_filter import max_filter
        data = read_image(input_path)
        result = max_filter(log_filtered_data=data, width=width, remove_index=remove_index)
        save_image(result, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _apply_ic_one(task):
    aligned_path, ic_dapi_path, ic_cyto_path, output_path, sbs_cfg = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        aligned = read_image(aligned_path)
        dapi_cycle = sbs_cfg["dapi_cycle"]
        cyto_cycle = sbs_cfg["cyto_cycle"]
        dapi_idx = sbs_cfg["dapi_cycle_index"]
        cyto_idx = sbs_cfg["cyto_cycle_index"]
        extra_ch = sbs_cfg.get("extra_channel_indices", [])

        if dapi_cycle != cyto_cycle:
            seg_data = combine_ic_images(
                [aligned[dapi_idx], aligned[cyto_idx]],
                [extra_ch, None],
            )
            ic_dapi = read_image(ic_dapi_path)
            ic_cyto = read_image(ic_cyto_path)
            ic_field = combine_ic_images([ic_dapi, ic_cyto], [extra_ch, None])
        else:
            seg_data = aligned[cyto_idx]
            ic_field = read_image(ic_dapi_path)

        corrected = apply_ic_field(seg_data, correction=ic_field)
        save_image(corrected, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _segment_one(task):
    input_path, nuclei_out, cells_out, stats_out, seg_params = task
    tag = Path(nuclei_out).stem
    if out_exists(nuclei_out) and out_exists(cells_out) and out_exists(stats_out):
        return "skip", tag
    try:
        for p in [nuclei_out, cells_out, stats_out]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        data = read_image(input_path)
        method = seg_params.get("segmentation_method", "cellpose")
        segment_cells = seg_params.get("segment_cells", True)

        if method == "cellpose":
            from lib.shared.segment_cellpose import segment_cellpose
            result = segment_cellpose(
                data=data,
                dapi_index=seg_params["dapi_index"],
                cyto_index=seg_params["cyto_index"],
                nuclei_diameter=seg_params["nuclei_diameter"],
                cell_diameter=seg_params["cell_diameter"],
                cellpose_model=seg_params["cellpose_model"],
                helper_index=seg_params.get("helper_index"),
                cellpose_kwargs=dict(
                    flow_threshold=seg_params.get("flow_threshold", 0.4),
                    cellprob_threshold=seg_params.get("cellprob_threshold", 0),
                    nuclei_flow_threshold=seg_params["nuclei_flow_threshold"],
                    nuclei_cellprob_threshold=seg_params["nuclei_cellprob_threshold"],
                    cell_flow_threshold=seg_params["cell_flow_threshold"],
                    cell_cellprob_threshold=seg_params["cell_cellprob_threshold"],
                ),
                reconcile=seg_params.get("reconcile"),
                return_counts=True,
                gpu=seg_params.get("gpu", False),
                cells=segment_cells,
            )
        elif method == "watershed":
            from lib.shared.segment_watershed import segment_watershed
            result = segment_watershed(
                data=data,
                nuclei_threshold=seg_params["threshold_dapi"],
                nuclei_area_min=seg_params["nuclei_area_min"],
                nuclei_area_max=seg_params["nuclei_area_max"],
                cell_threshold=seg_params["threshold_cell"],
                cells=segment_cells,
                reconcile=seg_params.get("reconcile"),
                return_counts=True,
            )
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        if segment_cells:
            nuclei, cells, counts = result
        else:
            nuclei, counts = result
            cells = np.zeros_like(nuclei)

        save_image(nuclei.astype(np.uint32), nuclei_out, is_label=True)
        save_image(cells.astype(np.uint32), cells_out, is_label=True)
        counts.to_csv(stats_out, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _extract_bases_one(task):
    peaks_path, maxfilt_path, seg_path, output_path, threshold_peaks, bases, wc = task
    tag = f"T{wc['tile']}"
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.sbs.extract_bases import extract_bases
        peaks = read_image(peaks_path)
        maxfilt = read_image(maxfilt_path)
        cells = read_image(seg_path)
        df = extract_bases(
            peaks_data=peaks, max_filtered_data=maxfilt, cells_data=cells,
            threshold_peaks=threshold_peaks, bases=bases, wildcards=wc,
        )
        df.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _call_reads_one(task):
    bases_path, peaks_path, output_path, method = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.sbs.call_reads import call_reads
        bases = pd.read_csv(bases_path, sep="\t")
        peaks = read_image(peaks_path)
        reads = call_reads(bases_data=bases, peaks_data=peaks, method=method)
        reads.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _call_cells_one(task):
    reads_path, output_path, cc_params = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.sbs.call_cells import call_cells, load_barcode_library
        reads = pd.read_csv(reads_path, sep="\t")
        barcode_lib = load_barcode_library(cc_params["df_barcode_library_fp"])
        barcode_type = cc_params.get("barcode_type", "simple")

        if barcode_type == "multi":
            cells = call_cells(
                reads_data=reads, df_barcode_library=barcode_lib,
                q_min=cc_params["q_min"], map_start=cc_params["map_start"],
                map_end=cc_params["map_end"], prefix_map=cc_params["prefix_map"],
                recomb_start=cc_params["recomb_start"], recomb_end=cc_params["recomb_end"],
                prefix_recomb=cc_params["prefix_recomb"],
                recomb_filter_col=cc_params["recomb_filter_col"],
                recomb_q_thresh=cc_params["recomb_q_thresh"],
                error_correct=cc_params["error_correct"],
                sort_calls=cc_params["sort_calls"],
                max_distance=cc_params["max_distance"],
                n_barcodes=cc_params["n_barcodes"],
                barcode_info_cols=cc_params.get("barcode_info_cols"),
            )
        else:
            cells = call_cells(
                reads_data=reads, df_barcode_library=barcode_lib,
                q_min=cc_params["q_min"], barcode_col=cc_params.get("barcode_col", "sgRNA"),
                prefix_col=cc_params.get("prefix_col"),
                error_correct=cc_params["error_correct"],
                sort_calls=cc_params["sort_calls"],
                max_distance=cc_params["max_distance"],
                n_barcodes=cc_params["n_barcodes"],
            )
        cells.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _extract_sbs_info_one(task):
    nuclei_path, output_path, wc = task
    tag = f"T{wc['tile']}"
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal
        nuclei = read_image(nuclei_path)
        df = extract_phenotype_minimal(phenotype_data=nuclei, nuclei_data=nuclei, wildcards=wc)
        df.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_sbs(config, args):
    sbs_cfg = config["sbs"]
    pp_cfg = config.get("preprocess", {})
    root = config["all"]["root_fp"]
    fmt = config.get("all", {}).get("image_format", "tiff")
    sbs_fp = Path(root) / "sbs"
    pp_fp = Path(root) / "preprocess"

    # Load combos (plate, well, tile, cycle)
    combos = pd.read_csv(pp_cfg["sbs_combo_fp"], sep="\t").astype(str)

    if args.plate_filter:
        combos = combos[combos["plate"] == str(args.plate_filter)]

    if args.max_tiles and "tile" in combos.columns:
        tiles = sorted(combos["tile"].unique(), key=lambda x: int(x))
        if len(tiles) > args.max_tiles:
            combos = combos[combos["tile"].isin(set(tiles[: args.max_tiles]))]

    # ponytail: SLURM array partitioning — slice sorted tile list by index range
    if args.tile_start is not None or args.tile_end is not None:
        tiles = sorted(combos["tile"].unique(), key=lambda x: int(x))
        start = args.tile_start or 0
        end = args.tile_end or len(tiles)
        combos = combos[combos["tile"].isin(set(tiles[start:end]))]

    if combos.empty:
        print("  SBS: no combos")
        return 0

    # Unique tile combos (drop cycle)
    tile_combos = combos[["plate", "well", "tile"]].drop_duplicates()
    cycles_by_tile = {}
    for _, r in combos.iterrows():
        key = (r["plate"], r["well"], r["tile"])
        cycles_by_tile.setdefault(key, []).append(r["cycle"])

    print(f"\n{'=' * 60}")
    print(f"  SBS: {len(tile_combos)} tiles, {len(combos['cycle'].unique()) if 'cycle' in combos.columns else 0} cycles")
    print(f"{'=' * 60}")

    errs = 0
    w = args.workers
    align_w = args.align_workers or w
    extra_ch = sbs_cfg.get("extra_channel_indices", [])
    spot_params = get_spot_detection_params(config)
    seg_params = get_segmentation_params("sbs", config)
    cc_params = get_call_cells_params(config)
    use_std_dev = sbs_cfg.get("spot_detection_method", "standard") == "standard"
    segment_cells_flag = sbs_cfg.get("segment_cells", True)

    run_tiles = args.step in ("tiles", "all")
    run_pre_seg = args.step in ("tiles", "pre-seg", "all")
    run_segment = args.step in ("tiles", "segment", "all")
    run_post_seg = args.step in ("tiles", "post-seg", "all")
    run_combine = args.step in ("combine", "all")

    # ponytail: gate tile_combos to empty for phases that don't run
    if not (run_pre_seg or run_segment or run_post_seg):
        tile_combos = tile_combos.iloc[0:0]

    # ponytail: per-phase tile views — out_exists() skips already-done; empty = no tasks
    pre_seg_tc = tile_combos if run_pre_seg else tile_combos.iloc[0:0]
    segment_tc = tile_combos if run_segment else tile_combos.iloc[0:0]
    post_seg_tc = tile_combos if run_post_seg else tile_combos.iloc[0:0]

    # --- Step 1: Align ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "aligned")
        cycles = sorted(cycles_by_tile[(p, we, ti)], key=int)
        cycle_paths = [preprocess_img_path(pp_fp, fmt, p, we, ti, c) for c in cycles]
        tasks.append((cycle_paths, out, sbs_cfg))
    errs += run_parallel(tasks, _align_one, align_w, "Align cycles")

    # --- Step 2: Log filter ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "aligned")
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "log_filtered")
        tasks.append((inp, out, extra_ch))
    errs += run_parallel(tasks, _log_filter_one, w, "Log filter")

    # --- Step 3: Std deviation ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "log_filtered")
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "standard_deviation")
        tasks.append((inp, out, extra_ch))
    errs += run_parallel(tasks, _std_dev_one, w, "Std deviation")

    # --- Step 4: Find peaks ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        if use_std_dev:
            inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "standard_deviation")
        else:
            inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "aligned")
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "peaks")
        tasks.append((inp, out, spot_params))
    errs += run_parallel(tasks, _find_peaks_one, w, "Find peaks")

    # --- Step 5: Max filter ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "log_filtered")
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "max_filtered")
        width = sbs_cfg.get("max_filter_width", 3)
        tasks.append((inp, out, width, extra_ch))
    errs += run_parallel(tasks, _max_filter_one, w, "Max filter")

    # --- Step 6: Apply IC field ---
    tasks = []
    dapi_cyc = str(sbs_cfg["dapi_cycle"])
    cyto_cyc = str(sbs_cfg["cyto_cycle"])
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        aligned_p = sbs_img_path(sbs_fp, fmt, p, we, ti, "aligned")
        ic_dapi = preprocess_ic_path(pp_fp, fmt, p, we, dapi_cyc)
        ic_cyto = preprocess_ic_path(pp_fp, fmt, p, we, cyto_cyc)
        out = sbs_img_path(sbs_fp, fmt, p, we, ti, "illumination_corrected")
        tasks.append((aligned_p, ic_dapi, ic_cyto, out, sbs_cfg))
    errs += run_parallel(tasks, _apply_ic_one, w, "Apply IC field")

    # --- Step 7: Segment ---
    seg_gpus = args.seg_gpus
    if args.gpu and seg_gpus > 1:
        try:
            import torch
            actual = torch.cuda.device_count()
            if actual < seg_gpus:
                print(f"  WARN: --seg-gpus {seg_gpus} but only {actual} GPU(s) detected, clamping")
                seg_gpus = max(1, actual)
        except Exception:
            pass
    if args.seg_workers:
        seg_workers = args.seg_workers
    elif args.gpu:
        seg_workers = seg_gpus
    else:
        seg_workers = min(w, 16)
    seg_init = (_worker_init_gpu, (seg_gpus,)) if args.gpu and seg_gpus > 1 else (None, ())
    tasks = []
    for _, r in segment_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = sbs_img_path(sbs_fp, fmt, p, we, ti, "illumination_corrected")
        n_out = sbs_img_path(sbs_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        c_out = sbs_img_path(sbs_fp, fmt, p, we, ti, "cells", subdirectory="labels")
        s_out = sbs_data_path(sbs_fp, fmt, p, we, ti, "segmentation_stats", "tsv")
        tasks.append((inp, n_out, c_out, s_out, seg_params))
    errs += run_parallel(tasks, _segment_one, seg_workers, "Segment SBS",
                         initializer=seg_init[0], initargs=seg_init[1])

    # --- Step 8: Extract bases ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        peaks_p = sbs_img_path(sbs_fp, fmt, p, we, ti, "peaks")
        maxf_p = sbs_img_path(sbs_fp, fmt, p, we, ti, "max_filtered")
        seg_p = sbs_img_path(sbs_fp, fmt, p, we, ti,
                             "cells" if segment_cells_flag else "nuclei",
                             subdirectory="labels")
        out = sbs_data_path(sbs_fp, fmt, p, we, ti, "bases", "tsv")
        wc = {"plate": p, "well": we, "tile": ti}
        tasks.append((peaks_p, maxf_p, seg_p, out,
                      sbs_cfg["threshold_peaks"], sbs_cfg["bases"], wc))
    errs += run_parallel(tasks, _extract_bases_one, w, "Extract bases")

    # --- Step 9: Call reads ---
    cr_method = sbs_cfg.get("call_reads_method", "median")
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        bases_p = sbs_data_path(sbs_fp, fmt, p, we, ti, "bases", "tsv")
        peaks_p = sbs_img_path(sbs_fp, fmt, p, we, ti, "peaks")
        out = sbs_data_path(sbs_fp, fmt, p, we, ti, "reads", "tsv")
        tasks.append((bases_p, peaks_p, out, cr_method))
    errs += run_parallel(tasks, _call_reads_one, w, "Call reads")

    # --- Step 10: Call cells ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        reads_p = sbs_data_path(sbs_fp, fmt, p, we, ti, "reads", "tsv")
        out = sbs_data_path(sbs_fp, fmt, p, we, ti, "cells", "tsv")
        tasks.append((reads_p, out, cc_params))
    errs += run_parallel(tasks, _call_cells_one, w, "Call cells")

    # --- Step 11: Extract SBS info ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        nuclei_p = sbs_img_path(sbs_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        out = sbs_data_path(sbs_fp, fmt, p, we, ti, "sbs_info", "tsv")
        wc = {"plate": p, "well": we, "tile": ti}
        tasks.append((nuclei_p, out, wc))
    errs += run_parallel(tasks, _extract_sbs_info_one, w, "Extract SBS info")

    if not run_combine:
        return errs

    # In combine-only mode, reload ALL tiles (not the SLURM array subset)
    if args.step == "combine":
        _all = pd.read_csv(pp_cfg["sbs_combo_fp"], sep="\t").astype(str)
        if args.plate_filter:
            _all = _all[_all["plate"] == str(args.plate_filter)]
        tile_combos = _all[["plate", "well", "tile"]].drop_duplicates()

    # --- Steps 12-14: Combine per well ---
    print(f"\n  Combine per well...")
    for info_type in ["reads", "cells", "sbs_info"]:
        for (plate, well), gdf in tile_combos.groupby(["plate", "well"]):
            out = sbs_well_path(sbs_fp, fmt, plate, well, info_type, "parquet")
            if out_exists(out):
                print(f"    SKIP combine {info_type} P{plate}/W{well}")
                continue

            input_paths = [
                sbs_data_path(sbs_fp, fmt, plate, well, str(tr["tile"]), info_type, "tsv")
                for _, tr in gdf.iterrows()
            ]
            dfs = []
            for f in input_paths:
                try:
                    dfs.append(pd.read_csv(f, sep="\t"))
                except Exception:
                    pass
            if not dfs:
                print(f"    WARN combine {info_type} P{plate}/W{well}: no inputs")
                continue

            combined = pd.concat(dfs, ignore_index=True)
            combined = validate_dtypes(combined)
            for col in combined.select_dtypes(include="object").columns:
                converted = pd.to_numeric(combined[col], errors="coerce")
                if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
                    combined[col] = converted

            Path(out).parent.mkdir(parents=True, exist_ok=True)
            write_parquet(combined, out)
            print(f"    OK combine {info_type} P{plate}/W{well} ({len(combined)} rows)")

    # --- Steps 15-16: Eval per plate ---
    print(f"\n  Eval per plate...")
    pp_meta_fp = Path(root) / "preprocess"
    for plate in sorted(tile_combos["plate"].unique()):
        # Eval segmentation
        overview_out = sbs_plate_path(sbs_fp, fmt, plate, "segmentation_overview", "tsv", "segmentation")
        if not out_exists(overview_out):
            try:
                from lib.shared.eval_segmentation import segmentation_overview, plot_cell_density_heatmap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                stats_paths = [
                    sbs_data_path(sbs_fp, fmt, plate, r["well"], r["tile"], "segmentation_stats", "tsv")
                    for _, r in tile_combos[tile_combos["plate"] == plate].iterrows()
                ]
                stats_paths = [p for p in stats_paths if Path(p).exists()]

                if stats_paths:
                    overview_df = segmentation_overview(stats_paths)
                    Path(overview_out).parent.mkdir(parents=True, exist_ok=True)
                    overview_df.to_csv(overview_out, sep="\t", index=False)

                    wells = tile_combos[tile_combos["plate"] == plate]["well"].unique()
                    cells_paths = [sbs_well_path(sbs_fp, fmt, plate, w, "cells", "parquet") for w in wells]
                    cells_paths = [p for p in cells_paths if Path(p).exists()]

                    md_loc = make_loc(fmt, plate)
                    md_paths = []
                    for w in wells:
                        mp = str(pp_meta_fp / "metadata" / "sbs" / get_data_output_path(
                            make_loc(fmt, plate, w), "combined_metadata", "parquet", fmt))
                        if Path(mp).exists():
                            md_paths.append(mp)

                    if cells_paths and md_paths:
                        cells_df = read_parquets(cells_paths)
                        md_df = pd.concat([pd.read_parquet(p) for p in md_paths], ignore_index=True)
                        md_df = md_df.drop_duplicates(subset=["well", "tile"])
                        summary, fig = plot_cell_density_heatmap(cells_df, metadata=md_df)
                        heatmap_tsv = sbs_plate_path(sbs_fp, fmt, plate, "cell_density_heatmap", "tsv", "segmentation")
                        heatmap_png = sbs_plate_path(sbs_fp, fmt, plate, "cell_density_heatmap", "png", "segmentation")
                        summary.to_csv(heatmap_tsv, index=False, sep="\t")
                        fig.savefig(heatmap_png, dpi=300, bbox_inches="tight", transparent=True)
                        plt.close(fig)
                    print(f"    OK eval_seg P{plate}")
            except Exception as e:
                print(f"    ERR eval_seg P{plate}: {e}")
                errs += 1

        # Eval mapping
        mapping_out = sbs_plate_path(sbs_fp, fmt, plate, "mapping_vs_threshold_peak", "png", "mapping")
        if not out_exists(mapping_out):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from lib.sbs.standardize_barcode_design import get_barcode_list
                from lib.sbs.eval_mapping import (
                    plot_mapping_vs_threshold, plot_read_mapping_heatmap,
                    plot_cell_mapping_heatmap, plot_cell_metric_histogram,
                    plot_gene_symbol_histogram, mapping_overview, plot_barcode_prefix_matching,
                )

                wells = tile_combos[tile_combos["plate"] == plate]["well"].unique()
                reads_paths = [sbs_well_path(sbs_fp, fmt, plate, w, "reads", "parquet") for w in wells]
                cells_paths = [sbs_well_path(sbs_fp, fmt, plate, w, "cells", "parquet") for w in wells]
                info_paths = [sbs_well_path(sbs_fp, fmt, plate, w, "sbs_info", "parquet") for w in wells]
                reads_paths = [p for p in reads_paths if Path(p).exists()]
                cells_paths = [p for p in cells_paths if Path(p).exists()]
                info_paths = [p for p in info_paths if Path(p).exists()]

                md_paths = []
                for w in wells:
                    mp = str(pp_meta_fp / "metadata" / "sbs" / get_data_output_path(
                        make_loc(fmt, plate, w), "combined_metadata", "parquet", fmt))
                    if Path(mp).exists():
                        md_paths.append(mp)

                if not (reads_paths and cells_paths and info_paths and md_paths):
                    print(f"    SKIP eval_mapping P{plate}: missing inputs")
                    continue

                barcode_lib = pd.read_csv(sbs_cfg["df_barcode_library_fp"], sep="\t")
                barcode_type = sbs_cfg.get("barcode_type", "simple")
                if barcode_type == "multi":
                    barcodes = get_barcode_list(barcode_lib, sequencing_order=sbs_cfg.get("sequencing_order", "map_recomb"))
                else:
                    barcodes = get_barcode_list(barcode_lib)

                reads = read_parquets(reads_paths)
                cells = read_parquets(cells_paths)
                sbs_info = read_parquets(info_paths)
                metadata = pd.concat([pd.read_parquet(p) for p in md_paths], ignore_index=True).drop_duplicates(subset=["well", "tile"])

                eval_dir = sbs_fp / "eval" / "mapping"
                Path(eval_dir).mkdir(parents=True, exist_ok=True)

                def _save(info, ext, fig_or_df):
                    path = sbs_plate_path(sbs_fp, fmt, plate, info, ext, "mapping")
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(fig_or_df, pd.DataFrame):
                        fig_or_df.to_csv(path, index=False, sep="\t")
                    else:
                        fig_or_df.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
                        plt.close(fig_or_df)

                _, fig = plot_mapping_vs_threshold(reads, barcodes, "peak", num_thresholds=10)
                _save("mapping_vs_threshold_peak", "png", fig)
                _, fig = plot_mapping_vs_threshold(reads, barcodes, "Q_min", num_thresholds=10)
                _save("mapping_vs_threshold_qmin", "png", fig)
                fig = plot_read_mapping_heatmap(reads, barcodes, metadata=metadata)
                _save("read_mapping_heatmap", "png", fig)

                sort_by = sbs_cfg.get("sort_calls", "count")
                df1, fig = plot_cell_mapping_heatmap(cells, sbs_info, barcodes, mapping_to="one", mapping_strategy="gene symbols", metadata=metadata, return_summary=True)
                _save("cell_mapping_heatmap_one", "tsv", df1)
                _save("cell_mapping_heatmap_one", "png", fig)
                df2, fig = plot_cell_mapping_heatmap(cells, sbs_info, barcodes, mapping_to="any", mapping_strategy="gene symbols", metadata=metadata, return_summary=True)
                _save("cell_mapping_heatmap_any", "tsv", df2)
                _save("cell_mapping_heatmap_any", "png", fig)

                _, fig = plot_cell_metric_histogram(cells, sort_by=sort_by)
                _save("cell_metric_histogram", "png", fig)
                _, fig = plot_gene_symbol_histogram(cells)
                _save("gene_symbol_histogram", "png", fig)
                mo = mapping_overview(sbs_info, cells, sort_by=sort_by)
                _save("mapping_overview", "tsv", mo)

                lib_col = (sbs_cfg.get("prefix_map", "prefix_map") if barcode_type == "multi"
                           else sbs_cfg.get("prefix_col", "prefix"))
                if barcode_type == "multi":
                    _, fig = plot_barcode_prefix_matching(reads, barcode_lib, library_col=lib_col,
                                                          library_col_recomb=sbs_cfg.get("prefix_recomb", "prefix_recomb"),
                                                          sequencing_order=sbs_cfg.get("sequencing_order", "map_recomb"))
                else:
                    _, fig = plot_barcode_prefix_matching(reads, barcode_lib, library_col=lib_col)
                _save("barcode_prefix_matching", "png", fig)

                print(f"    OK eval_mapping P{plate}")
            except Exception as e:
                print(f"    ERR eval_mapping P{plate}: {e}")
                errs += 1

    return errs


def main():
    p = argparse.ArgumentParser(description="Direct SBS runner (bypasses Snakemake)")
    p.add_argument("--config", required=True)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--plate-filter", type=int, default=None)
    p.add_argument("--gpu", action="store_true", help="Enable GPU for cellpose segmentation")
    p.add_argument("--tile-start", type=int, default=None,
                   help="Start index into sorted tile list (for SLURM array partitioning)")
    p.add_argument("--tile-end", type=int, default=None,
                   help="End index into sorted tile list (exclusive)")
    p.add_argument("--step", choices=["tiles", "pre-seg", "segment", "post-seg", "combine", "all"], default="all",
                   help="tiles=per-tile steps 1-11, pre-seg=steps 1-6 (CPU), segment=step 7 (GPU), post-seg=steps 8-11 (CPU), combine=merge+eval 12-16, all=everything")
    p.add_argument("--align-workers", type=int, default=None,
                   help="Workers for alignment step (default: same as --workers)")
    p.add_argument("--seg-workers", type=int, default=None,
                   help="Workers for segmentation step (default: seg-gpus if --gpu, else --workers)")
    p.add_argument("--seg-gpus", type=int, default=1,
                   help="Number of GPUs for segmentation (default: 1)")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    set_benchmark_context("sbs", config["all"]["root_fp"])
    if args.gpu:
        config.setdefault("sbs", {})["gpu"] = True
    fmt = config.get("all", {}).get("image_format", "tiff")

    print(f"{'#' * 60}")
    print(f"  Direct SBS Runner | format={fmt} gpu={args.gpu} step={args.step}")
    print(f"  config={args.config} workers={args.workers} max_tiles={args.max_tiles or 'all'}")
    tile_range = f" tiles[{args.tile_start}:{args.tile_end}]" if args.tile_start is not None or args.tile_end is not None else ""
    print(f"  plate_filter={args.plate_filter or 'none'}{tile_range}")
    print(f"{'#' * 60}")

    t0 = time.time()
    errs = process_sbs(config, args)
    status = "DONE" if errs == 0 else "FAILED"
    print(f"\n{'#' * 60}")
    print(f"  {status}: {time.time() - t0:.1f}s total, {errs} errors")
    print(f"{'#' * 60}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
