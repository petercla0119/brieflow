#!/usr/bin/env python3
"""Direct phenotype runner — bypasses Snakemake DAG build.

Runs all phenotype processing steps (apply_ic → align → segment →
identify_cytoplasm → extract_phenotype_info → extract_phenotype →
combine → merge → eval) with ProcessPoolExecutor parallelism.

Usage:
    python run_phenotype_direct.py --config config/config.yml --max-tiles 100 --workers 8
"""

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "brieflow" / "workflow"))

from lib.shared.file_utils import get_data_output_path, get_image_output_path, validate_dtypes
from lib.shared.image_io import read_image, save_image
from lib.shared.illumination_correction import apply_ic_field
from lib.shared.parquet_io import write_parquet, read_parquets
from lib.shared.rule_utils import get_alignment_params, get_segmentation_params


# ---------------------------------------------------------------------------
# Path helpers
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


def out_exists(path):
    p = Path(path)
    if p.name == "zarr.json":
        return p.exists()
    if p.suffix == ".zarr":
        return p.is_dir() and any(p.iterdir()) if p.exists() else False
    return p.exists() and p.stat().st_size > 0


def phen_img_path(phen_fp, fmt, plate, well, tile, info_type, subdirectory=None):
    loc = make_loc(fmt, plate, well, tile)
    return str(phen_fp / get_image_output_path(loc, info_type, fmt, subdirectory=subdirectory))


def phen_data_path(phen_fp, fmt, plate, well, tile, info_type, ext):
    loc = make_loc(fmt, plate, well, tile)
    return str(phen_fp / "tsvs" / get_data_output_path(loc, info_type, ext, fmt))


def phen_well_path(phen_fp, fmt, plate, well, info_type, ext):
    loc = make_loc(fmt, plate, well)
    return str(phen_fp / "parquets" / get_data_output_path(loc, info_type, ext, fmt))


def phen_plate_path(phen_fp, fmt, plate, info_type, ext, subdir):
    loc = make_loc(fmt, plate)
    return str(phen_fp / "eval" / subdir / get_data_output_path(loc, info_type, ext, fmt))


def preprocess_phen_img_path(pp_fp, fmt, plate, well, tile):
    loc = make_loc(fmt, plate, well, tile)
    return str(pp_fp / get_image_output_path(loc, "image", fmt, image_subdir="phenotype"))


def preprocess_phen_ic_path(pp_fp, fmt, plate, well):
    loc = make_loc(fmt, plate, well)
    return str(pp_fp / "ic_fields" / "phenotype" / get_data_output_path(loc, "ic_field", fmt, fmt))


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
# Per-tile workers
# ---------------------------------------------------------------------------

def _apply_ic_one(task):
    raw_path, ic_path, output_path = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        raw = read_image(raw_path)
        ic = read_image(ic_path)
        corrected = apply_ic_field(raw, correction=ic)
        save_image(corrected, output_path)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _align_one(task):
    input_path, output_path, align_cfg = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from lib.phenotype.align_channels import align_phenotype_channels
        from lib.shared.align import apply_custom_offsets
        data = read_image(input_path)

        if align_cfg.get("custom_channel_offsets"):
            data = apply_custom_offsets(data, offsets_dict=align_cfg["custom_channel_offsets"])

        if align_cfg.get("align", False):
            if align_cfg.get("multi_step", False):
                for step in align_cfg["steps"]:
                    data = align_phenotype_channels(
                        data,
                        target=step["target"],
                        source=step["source"],
                        riders=step.get("riders", []),
                        remove_channel=step["remove_channel"],
                        upsample_factor=step.get("upsample_factor", align_cfg.get("upsample_factor", 2)),
                        window=step.get("window", align_cfg.get("window", 2)),
                    )
            else:
                data = align_phenotype_channels(
                    data,
                    target=align_cfg["target"],
                    source=align_cfg["source"],
                    riders=align_cfg.get("riders", []),
                    remove_channel=align_cfg.get("remove_channel", False),
                    upsample_factor=align_cfg.get("upsample_factor", 2),
                    window=align_cfg.get("window", 2),
                )

        save_image(data, output_path)
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


def _identify_cytoplasm_one(task):
    nuclei_path, cells_path, output_path, segment_cells = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        nuclei = read_image(nuclei_path)
        cells = read_image(cells_path)
        if segment_cells:
            from lib.phenotype.identify_cytoplasm_cellpose import identify_cytoplasm_cellpose
            cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)
            if cytoplasms is None:
                cytoplasms = np.zeros_like(nuclei, dtype=np.int32)
        else:
            cytoplasms = np.zeros_like(nuclei, dtype=np.int32)
        save_image(cytoplasms.astype(np.uint32), output_path, is_label=True)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _extract_phenotype_info_one(task):
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


def _extract_phenotype_one(task):
    aligned_path, nuclei_path, cells_path, cyto_path, output_path, params = task
    tag = Path(output_path).stem
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = read_image(aligned_path)
        nuclei = read_image(nuclei_path)
        cells_data = read_image(cells_path)
        cytoplasms = read_image(cyto_path)

        segment_cells = params["segment_cells"]
        if not segment_cells:
            cells_data = None
            cytoplasms = None

        cp_method = params["cp_method"]
        wc = params["wildcards"]

        if cp_method == "cp_measure":
            from lib.phenotype.extract_phenotype_cp_measure import extract_phenotype_cp_measure
            phenotype_cp = extract_phenotype_cp_measure(
                data_phenotype=data, nuclei=nuclei, cells=cells_data,
                cytoplasms=cytoplasms, channel_names=params["channel_names"],
            )
        elif cp_method == "cp_emulator":
            from lib.phenotype.extract_phenotype_cp_emulator import extract_phenotype_cp_emulator
            phenotype_cp = extract_phenotype_cp_emulator(
                data_phenotype=data, nuclei=nuclei, cells=cells_data,
                cytoplasms=cytoplasms, foci_channel=params.get("foci_channel_index"),
                channel_names=params["channel_names"], wildcards=wc,
            )
        else:
            raise ValueError(f"Unknown cp_method: {cp_method}")

        phenotype_cp.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_phenotype(config, args):
    phen_cfg = config["phenotype"]
    pp_cfg = config.get("preprocess", {})
    root = config["all"]["root_fp"]
    fmt = config.get("all", {}).get("image_format", "tiff")
    phen_fp = Path(root) / "phenotype"
    pp_fp = Path(root) / "preprocess"

    # Load combos (plate, well, tile)
    combos = pd.read_csv(pp_cfg["phenotype_combo_fp"], sep="\t").astype(str)

    if args.plate_filter:
        combos = combos[combos["plate"] == str(args.plate_filter)]

    if args.max_tiles and "tile" in combos.columns:
        tiles = sorted(combos["tile"].unique(), key=lambda x: int(x))
        if len(tiles) > args.max_tiles:
            combos = combos[combos["tile"].isin(set(tiles[: args.max_tiles]))]

    # Phenotype combos may have plate/well/tile (plus cycle/round from preprocess)
    # We only need unique plate/well/tile
    tile_cols = [c for c in ["plate", "well", "tile"] if c in combos.columns]
    tile_combos = combos[tile_cols].drop_duplicates()

    if tile_combos.empty:
        print("  Phenotype: no combos")
        return 0

    print(f"\n{'=' * 60}")
    print(f"  Phenotype: {len(tile_combos)} tiles")
    print(f"{'=' * 60}")

    errs = 0
    w = args.workers
    seg_params = get_segmentation_params("phenotype", config)
    segment_cells = phen_cfg.get("segment_cells", True)
    channel_names = phen_cfg.get("channel_names", [])
    cp_method = phen_cfg.get("cp_method", "cp_emulator")
    foci_channel_index = phen_cfg.get("foci_channel_index")

    # Pre-compute alignment configs per plate
    align_cfgs = {}
    for plate in tile_combos["plate"].unique():
        wc = SimpleNamespace(plate=plate)
        align_cfgs[plate] = get_alignment_params(wc, config)

    # ponytail: per-phase tile views (step: pre-seg=1-2 CPU, segment=3 GPU, post-seg=4-10 CPU)
    step = getattr(args, "step", "all")
    pre_seg_tc = tile_combos if step in ("pre-seg", "all") else tile_combos.iloc[0:0]
    segment_tc = tile_combos if step in ("segment", "all") else tile_combos.iloc[0:0]
    post_seg_tc = tile_combos if step in ("post-seg", "all") else tile_combos.iloc[0:0]

    # --- Step 1: Apply IC field ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        raw = preprocess_phen_img_path(pp_fp, fmt, p, we, ti)
        ic = preprocess_phen_ic_path(pp_fp, fmt, p, we)
        out = phen_img_path(phen_fp, fmt, p, we, ti, "illumination_corrected")
        tasks.append((raw, ic, out))
    errs += run_parallel(tasks, _apply_ic_one, w, "Apply IC field")

    # --- Step 2: Align ---
    tasks = []
    for _, r in pre_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = phen_img_path(phen_fp, fmt, p, we, ti, "illumination_corrected")
        out = phen_img_path(phen_fp, fmt, p, we, ti, "aligned")
        tasks.append((inp, out, align_cfgs[p]))
    errs += run_parallel(tasks, _align_one, w, "Align phenotype")

    # --- Step 3: Segment ---
    seg_workers = 1 if seg_params.get("gpu", False) else min(w, 16)
    tasks = []
    for _, r in segment_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        inp = phen_img_path(phen_fp, fmt, p, we, ti, "aligned")
        n_out = phen_img_path(phen_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        c_out = phen_img_path(phen_fp, fmt, p, we, ti, "cells", subdirectory="labels")
        s_out = phen_data_path(phen_fp, fmt, p, we, ti, "segmentation_stats", "tsv")
        tasks.append((inp, n_out, c_out, s_out, seg_params))
    errs += run_parallel(tasks, _segment_one, seg_workers, "Segment phenotype")

    # --- Step 4: Identify cytoplasm ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        n_in = phen_img_path(phen_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        c_in = phen_img_path(phen_fp, fmt, p, we, ti, "cells", subdirectory="labels")
        out = phen_img_path(phen_fp, fmt, p, we, ti, "identified_cytoplasms", subdirectory="labels")
        tasks.append((n_in, c_in, out, segment_cells))
    errs += run_parallel(tasks, _identify_cytoplasm_one, w, "Identify cytoplasm")

    # --- Step 5: Extract phenotype info ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        n_in = phen_img_path(phen_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        out = phen_data_path(phen_fp, fmt, p, we, ti, "phenotype_info", "tsv")
        wc = {"plate": p, "well": we, "tile": ti}
        tasks.append((n_in, out, wc))
    errs += run_parallel(tasks, _extract_phenotype_info_one, w, "Extract phenotype info")

    # --- Step 6: Extract phenotype (full features) ---
    tasks = []
    for _, r in post_seg_tc.iterrows():
        p, we, ti = r["plate"], r["well"], r["tile"]
        aligned = phen_img_path(phen_fp, fmt, p, we, ti, "aligned")
        nuclei = phen_img_path(phen_fp, fmt, p, we, ti, "nuclei", subdirectory="labels")
        cells_p = phen_img_path(phen_fp, fmt, p, we, ti, "cells", subdirectory="labels")
        cyto = phen_img_path(phen_fp, fmt, p, we, ti, "identified_cytoplasms", subdirectory="labels")
        out = phen_data_path(phen_fp, fmt, p, we, ti, "phenotype_cp", "tsv")
        params = {
            "cp_method": cp_method,
            "channel_names": channel_names,
            "foci_channel_index": foci_channel_index,
            "segment_cells": segment_cells,
            "wildcards": {"plate": p, "well": we, "tile": ti},
        }
        tasks.append((aligned, nuclei, cells_p, cyto, out, params))
    errs += run_parallel(tasks, _extract_phenotype_one, min(w, 16), "Extract phenotype")

    # --- Step 7: Combine phenotype info (per well) ---
    print(f"\n  Combine phenotype info per well...")
    for (plate, well), gdf in tile_combos.groupby(["plate", "well"]):
        out = phen_well_path(phen_fp, fmt, plate, well, "phenotype_info", "parquet")
        if out_exists(out):
            print(f"    SKIP combine phenotype_info P{plate}/W{well}")
            continue

        input_paths = [
            phen_data_path(phen_fp, fmt, plate, well, str(tr["tile"]), "phenotype_info", "tsv")
            for _, tr in gdf.iterrows()
        ]
        dfs = []
        for f in input_paths:
            try:
                dfs.append(pd.read_csv(f, sep="\t"))
            except Exception:
                pass
        if not dfs:
            print(f"    WARN combine phenotype_info P{plate}/W{well}: no inputs")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined = validate_dtypes(combined)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        write_parquet(combined, out)
        print(f"    OK combine phenotype_info P{plate}/W{well} ({len(combined)} rows)")

    # --- Step 8: Merge phenotype (per well) ---
    print(f"\n  Merge phenotype per well...")
    prefix = "cell" if segment_cells else "nucleus"
    for (plate, well), gdf in tile_combos.groupby(["plate", "well"]):
        out_full = phen_well_path(phen_fp, fmt, plate, well, "phenotype_cp", "parquet")
        out_min = phen_well_path(phen_fp, fmt, plate, well, "phenotype_cp_min", "parquet")
        if out_exists(out_full) and out_exists(out_min):
            print(f"    SKIP merge phenotype P{plate}/W{well}")
            continue

        input_paths = [
            phen_data_path(phen_fp, fmt, plate, well, str(tr["tile"]), "phenotype_cp", "tsv")
            for _, tr in gdf.iterrows()
        ]
        dfs = []
        for f in input_paths:
            try:
                dfs.append(pd.read_csv(f, sep="\t"))
            except Exception:
                pass
        if not dfs:
            print(f"    WARN merge phenotype P{plate}/W{well}: no inputs")
            continue

        phenotype_cp = pd.concat(dfs, ignore_index=True)
        Path(out_full).parent.mkdir(parents=True, exist_ok=True)
        write_parquet(phenotype_cp, out_full)

        bounds_features = [f"{prefix}_bounds_{i}" for i in range(4)]
        channel_min_features = [f"{prefix}_{ch}_min" for ch in channel_names]
        min_cols = ["plate", "well", "tile", "label", f"{prefix}_i", f"{prefix}_j"]
        min_cols.extend(bounds_features + channel_min_features)
        available = [c for c in min_cols if c in phenotype_cp.columns]
        phenotype_cp_min = phenotype_cp[available]
        write_parquet(phenotype_cp_min, out_min)
        print(f"    OK merge phenotype P{plate}/W{well} ({len(phenotype_cp)} rows)")

    # --- Step 9: Eval segmentation (per plate) ---
    print(f"\n  Eval per plate...")
    for plate in sorted(tile_combos["plate"].unique()):
        overview_out = phen_plate_path(phen_fp, fmt, plate, "segmentation_overview", "tsv", "segmentation")
        if not out_exists(overview_out):
            try:
                from lib.shared.eval_segmentation import segmentation_overview, plot_cell_density_heatmap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                stats_paths = [
                    phen_data_path(phen_fp, fmt, plate, r["well"], r["tile"], "segmentation_stats", "tsv")
                    for _, r in tile_combos[tile_combos["plate"] == plate].iterrows()
                ]
                stats_paths = [p for p in stats_paths if Path(p).exists()]

                if stats_paths:
                    overview_df = segmentation_overview(stats_paths)
                    Path(overview_out).parent.mkdir(parents=True, exist_ok=True)
                    overview_df.to_csv(overview_out, sep="\t", index=False)

                    wells = tile_combos[tile_combos["plate"] == plate]["well"].unique()
                    cells_paths = [phen_well_path(phen_fp, fmt, plate, w, "phenotype_info", "parquet") for w in wells]
                    cells_paths = [p for p in cells_paths if Path(p).exists()]

                    md_paths = []
                    for w in wells:
                        mp = str(pp_fp / "metadata" / "phenotype" / get_data_output_path(
                            make_loc(fmt, plate, w), "combined_metadata", "parquet", fmt))
                        if Path(mp).exists():
                            md_paths.append(mp)

                    if cells_paths and md_paths:
                        cells_df = read_parquets(cells_paths)
                        md_df = pd.concat([pd.read_parquet(p) for p in md_paths], ignore_index=True)
                        md_df = md_df.drop_duplicates(subset=["well", "tile"])
                        summary, fig = plot_cell_density_heatmap(cells_df, metadata=md_df)
                        heatmap_tsv = phen_plate_path(phen_fp, fmt, plate, "cell_density_heatmap", "tsv", "segmentation")
                        heatmap_png = phen_plate_path(phen_fp, fmt, plate, "cell_density_heatmap", "png", "segmentation")
                        summary.to_csv(heatmap_tsv, index=False, sep="\t")
                        fig.savefig(heatmap_png, dpi=300, bbox_inches="tight", transparent=True)
                        plt.close(fig)
                    print(f"    OK eval_seg P{plate}")
            except Exception as e:
                print(f"    ERR eval_seg P{plate}: {e}")
                errs += 1

    # --- Step 10: Eval features (per plate) ---
    for plate in sorted(tile_combos["plate"].unique()):
        eval_features = [f"{prefix}_{ch}_min" for ch in channel_names]
        first_out = phen_plate_path(phen_fp, fmt, plate, f"{eval_features[0]}_heatmap", "png", "features") if eval_features else None
        if first_out and out_exists(first_out):
            print(f"    SKIP eval_features P{plate}")
            continue
        try:
            from lib.phenotype.eval_features import plot_feature_heatmap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            wells = tile_combos[tile_combos["plate"] == plate]["well"].unique()
            min_paths = [phen_well_path(phen_fp, fmt, plate, w, "phenotype_cp_min", "parquet") for w in wells]
            min_paths = [p for p in min_paths if Path(p).exists()]
            if not min_paths:
                print(f"    SKIP eval_features P{plate}: no inputs")
                continue

            md_paths = []
            for w in wells:
                mp = str(pp_fp / "metadata" / "phenotype" / get_data_output_path(
                    make_loc(fmt, plate, w), "combined_metadata", "parquet", fmt))
                if Path(mp).exists():
                    md_paths.append(mp)

            phenotype_cp_min = read_parquets(min_paths)
            metadata = pd.concat([pd.read_parquet(p) for p in md_paths], ignore_index=True).drop_duplicates(subset=["well", "tile"])

            min_feature_names = [col for col in phenotype_cp_min.columns if col.endswith("_min")]
            for feature_name in min_feature_names:
                tsv_out = phen_plate_path(phen_fp, fmt, plate, f"{feature_name}_heatmap", "tsv", "features")
                png_out = phen_plate_path(phen_fp, fmt, plate, f"{feature_name}_heatmap", "png", "features")
                Path(tsv_out).parent.mkdir(parents=True, exist_ok=True)
                df_summary, fig = plot_feature_heatmap(
                    phenotype_cp_min, feature=feature_name,
                    metadata=metadata, return_summary=True,
                )
                df_summary.to_csv(tsv_out, index=False, sep="\t")
                fig.savefig(png_out, dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig)
            print(f"    OK eval_features P{plate} ({len(min_feature_names)} features)")
        except Exception as e:
            print(f"    ERR eval_features P{plate}: {e}")
            errs += 1

    return errs


def main():
    p = argparse.ArgumentParser(description="Direct phenotype runner (bypasses Snakemake)")
    p.add_argument("--config", required=True)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--plate-filter", type=int, default=None)
    p.add_argument("--gpu", action="store_true", help="Enable GPU for cellpose segmentation")
    p.add_argument("--step", choices=["pre-seg", "segment", "post-seg", "all"], default="all",
                   help="pre-seg=IC+align (CPU), segment=cellpose (GPU), post-seg=extract+combine (CPU), all=everything")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.gpu:
        config.setdefault("phenotype", {})["gpu"] = True
    fmt = config.get("all", {}).get("image_format", "tiff")

    print(f"{'#' * 60}")
    print(f"  Direct Phenotype Runner | format={fmt}")
    print(f"  config={args.config} workers={args.workers} max_tiles={args.max_tiles or 'all'}")
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
