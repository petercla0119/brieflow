#!/usr/bin/env python3
"""Direct merge runner — bypasses Snakemake DAG build.

Supports both "fast" and "stitch" merge approaches:

Fast approach:
  1. fast_alignment — Delaunay hash + multistep alignment per well
  2. fast_merge     — tile-by-tile cell merging using alignment

Stitch approach:
  1. estimate_stitch — coordinate-based tile position estimation (pheno + sbs)
  2. stitch          — assemble stitched masks, extract cell positions (pheno + sbs)
  3. stitch_alignment — scale + triangle hash + align pheno↔SBS
  4. stitch_merge    — cell-to-cell matching using alignment

Shared steps (both approaches):
  5. format_merge   — FOV distances, gene mappings, channel mins, global coords
  6. deduplicate    — two-step dedup + matching rate stats
  7. final_merge    — join full phenotype features
  8. eval_merge     — plate-level summary and plots

Usage:
    python run_merge_direct.py --config config/config.yml --workers 16
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
# scripts/direct/ -> scripts/ -> brieflow/ -> brieflow/workflow
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "workflow"))

from lib.shared.file_utils import get_data_output_path, get_image_output_path, validate_dtypes
from lib.shared.parquet_io import read_parquet, read_parquets, write_parquet


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _loc(fmt, plate, well=None):
    import re
    loc = {"plate": str(plate)}
    if well is not None:
        w = str(well)
        if fmt == "zarr":
            m = re.match(r"^([A-Za-z]+)(\d+)$", w)
            loc["row"] = m.group(1) if m else w
            loc["col"] = m.group(2) if m else "0"
        else:
            loc["well"] = w
    return loc


def merge_parquet_path(merge_fp, fmt, plate, well, info_type):
    loc = _loc(fmt, plate, well)
    return str(merge_fp / "parquets" / get_data_output_path(loc, info_type, "parquet", fmt))


def merge_eval_path(merge_fp, fmt, plate, info_type, ext):
    loc = _loc(fmt, plate)
    return str(merge_fp / "eval" / get_data_output_path(loc, info_type, ext, fmt))


def merge_eval_well_path(merge_fp, fmt, plate, well, info_type, ext):
    loc = _loc(fmt, plate, well)
    return str(merge_fp / "eval" / get_data_output_path(loc, info_type, ext, fmt))


def preprocess_metadata_path(pp_fp, fmt, plate, well, modality):
    loc = _loc(fmt, plate, well)
    return str(pp_fp / "metadata" / modality / get_data_output_path(loc, "combined_metadata", "parquet", fmt))


def module_parquet_path(module_fp, fmt, plate, well, info_type):
    loc = _loc(fmt, plate, well)
    return str(module_fp / "parquets" / get_data_output_path(loc, info_type, "parquet", fmt))


def out_exists(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def merge_stitch_config_path(merge_fp, fmt, plate, well, data_type):
    loc = _loc(fmt, plate, well)
    return str(merge_fp / "stitch_configs" / get_data_output_path(loc, f"{data_type}_stitch_config", "yml", fmt))


def merge_image_path(merge_fp, fmt, plate, well, info_type):
    loc = _loc(fmt, plate, well)
    return str(merge_fp / "images" / get_data_output_path(loc, info_type, "npy", fmt))


def tile_image_path(module_fp, fmt, plate, well, tile, info_type, subdirectory=None):
    import re
    loc = {"plate": str(plate)}
    w = str(well)
    if fmt == "zarr":
        m = re.match(r"^([A-Za-z]+)(\d+)$", w)
        loc["row"] = m.group(1) if m else w
        loc["col"] = m.group(2) if m else "0"
    else:
        loc["well"] = w
    loc["tile"] = str(tile)
    return str(module_fp / get_image_output_path(loc, info_type, fmt, subdirectory=subdirectory))


# ---------------------------------------------------------------------------
# Step 1: Fast Alignment
# ---------------------------------------------------------------------------

def run_fast_alignment(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well, n_jobs):
    out = merge_parquet_path(merge_fp, fmt, plate, well, "fast_alignment")
    if out_exists(out):
        print(f"    SKIP fast_alignment P{plate}/W{well}")
        return out

    from lib.merge.hash import hash_cell_locations, multistep_alignment, extract_rotation, initial_alignment
    from lib.merge.merge_utils import align_metadata, find_closest_tiles

    ph_meta = validate_dtypes(read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, "phenotype")))
    sbs_meta = validate_dtypes(read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, "sbs")))

    alignment_params = {
        "flip_x": merge_cfg.get("alignment_flip_x"),
        "flip_y": merge_cfg.get("alignment_flip_y"),
        "rotate_90": merge_cfg.get("alignment_rotate_90"),
    }
    if any(alignment_params.values()):
        ph_meta, sbs_meta, _ = align_metadata(ph_meta, sbs_meta, x_col="x_pos", y_col="y_pos", **alignment_params)

    sbs_cycle = merge_cfg.get("sbs_metadata_cycle")
    sbs_channel = merge_cfg.get("sbs_metadata_channel")
    ph_channel = merge_cfg.get("ph_metadata_channel")

    if sbs_cycle is not None:
        sbs_meta = sbs_meta[sbs_meta["cycle"] == sbs_cycle]
    if sbs_channel is not None:
        sbs_meta = sbs_meta[sbs_meta["channel"] == sbs_channel]
    else:
        sbs_meta = sbs_meta.drop_duplicates(subset=["plate", "well", "tile"])

    if ph_channel is not None:
        ph_meta = ph_meta[ph_meta["channel"] == ph_channel]
    else:
        ph_meta = ph_meta.drop_duplicates(subset=["plate", "well", "tile"])

    ph_info = validate_dtypes(read_parquet(module_parquet_path(phenotype_fp, fmt, plate, well, "phenotype_info")))
    sbs_info = validate_dtypes(read_parquet(module_parquet_path(sbs_fp, fmt, plate, well, "sbs_info")))

    ph_xy = ph_meta.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index("tile")[["x", "y"]]
    sbs_xy = sbs_meta.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index("tile")[["x", "y"]]

    ph_hash = validate_dtypes(hash_cell_locations(ph_info))
    sbs_hash = validate_dtypes(hash_cell_locations(sbs_info).rename(columns={"tile": "site"}))

    initial_sbs_tiles = merge_cfg.get("initial_sbs_tiles")
    initial_sites_param = merge_cfg.get("initial_sites")

    if (initial_sbs_tiles is None) == (initial_sites_param is None):
        raise ValueError("Exactly one of 'initial_sbs_tiles' or 'initial_sites' must be set in merge config")

    d0, d1 = merge_cfg["det_range"]
    score_thresh = merge_cfg["score"]

    if initial_sbs_tiles is not None:
        candidate_pairs = []
        for sbs_tile in initial_sbs_tiles:
            closest = find_closest_tiles(sbs_meta, ph_meta, sbs_tile, verbose=False)
            best_ph = int(closest.iloc[0]["tile"])
            candidate_pairs.append([best_ph, sbs_tile])
        print(f"    Discovered {len(candidate_pairs)} candidate pairs")
    else:
        candidate_pairs = initial_sites_param
        print(f"    Using {len(candidate_pairs)} manual initial sites")

    init_df = initial_alignment(ph_hash, sbs_hash, initial_sites=candidate_pairs)
    valid = init_df.query("@d0 <= determinant <= @d1 & score > @score_thresh")

    if len(candidate_pairs) > 5 and len(valid) < 5:
        raise ValueError(f"Only {len(valid)} initial sites passed thresholds (need >= 5)")

    initial_sites = valid[["tile", "site"]].astype(int).values.tolist()
    print(f"    {len(initial_sites)} initial sites passed thresholds")

    well_alignment = multistep_alignment(
        ph_hash, sbs_hash, ph_xy, sbs_xy,
        det_range=merge_cfg["det_range"],
        score=merge_cfg["score"],
        initial_sites=initial_sites,
        n_jobs=n_jobs,
    )

    well_alignment.reset_index(drop=True, inplace=True)
    well_alignment["rotation_1"] = well_alignment["rotation"].apply(lambda r: extract_rotation(r, 1))
    well_alignment["rotation_2"] = well_alignment["rotation"].apply(lambda r: extract_rotation(r, 2))
    well_alignment.drop(columns=["rotation"], inplace=True)
    well_alignment["plate"] = plate
    well_alignment["well"] = well

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(well_alignment, out)
    print(f"    OK fast_alignment P{plate}/W{well} ({len(well_alignment)} alignments)")
    return out


# ---------------------------------------------------------------------------
# Step 2: Fast Merge
# ---------------------------------------------------------------------------

def run_fast_merge(merge_cfg, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well):
    out = merge_parquet_path(merge_fp, fmt, plate, well, "fast_merge")
    if out_exists(out):
        print(f"    SKIP fast_merge P{plate}/W{well}")
        return out

    from lib.merge.fast_merge import merge_triangle_hash

    ph_info = validate_dtypes(read_parquet(module_parquet_path(phenotype_fp, fmt, plate, well, "phenotype_info")))
    sbs_info = validate_dtypes(read_parquet(module_parquet_path(sbs_fp, fmt, plate, well, "sbs_info")))

    alignment_path = merge_parquet_path(merge_fp, fmt, plate, well, "fast_alignment")
    fast_alignment = read_parquet(alignment_path)
    fast_alignment["rotation"] = [
        np.array([r1, r2])
        for r1, r2 in zip(fast_alignment["rotation_1"], fast_alignment["rotation_2"])
    ]
    fast_alignment.drop(columns=["rotation_1", "rotation_2"], inplace=True)

    d0, d1 = merge_cfg["det_range"]
    score = merge_cfg["score"]
    filtered = fast_alignment.query("@d0 <= determinant <= @d1 & score > @score")

    print(f"    fast_merge: {len(filtered)}/{len(fast_alignment)} alignments pass threshold")

    merge_data = []
    for _, row in filtered.iterrows():
        ph_tile = row["tile"]
        sbs_site = row["site"]
        ph_filtered = ph_info[ph_info["tile"] == ph_tile]
        sbs_filtered = sbs_info[sbs_info["tile"] == sbs_site]
        result = merge_triangle_hash(ph_filtered, sbs_filtered, row, threshold=merge_cfg["threshold"])
        merge_data.append(result)

    merge_data = pd.concat(merge_data, ignore_index=True)
    print(f"    OK fast_merge P{plate}/W{well}: {len(merge_data)} cells merged")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merge_data, out)
    return out



# ---------------------------------------------------------------------------
# Stitch Steps 1-2: Estimate Stitch + Stitch (per data type)
# ---------------------------------------------------------------------------

def run_estimate_stitch(merge_cfg, pp_fp, merge_fp, fmt, plate, well, data_type):
    out = merge_stitch_config_path(merge_fp, fmt, plate, well, data_type)
    if out_exists(out):
        print(f"    SKIP estimate_stitch_{data_type} P{plate}/W{well}")
        return out

    from lib.merge.estimate_stitch import estimate_stitch_coordinate_based, convert_numpy_types

    metadata = validate_dtypes(read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, data_type)))

    if data_type == "sbs":
        alignment_params = {
            "flip_x": merge_cfg.get("alignment_flip_x"),
            "flip_y": merge_cfg.get("alignment_flip_y"),
            "rotate_90": merge_cfg.get("alignment_rotate_90"),
        }
        if any(alignment_params.values()):
            from lib.merge.merge_utils import align_metadata
            ph_meta = validate_dtypes(read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, "phenotype")))
            ph_meta, metadata, _ = align_metadata(ph_meta, metadata, x_col="x_pos", y_col="y_pos", reference_df=1, **alignment_params)

        sbs_cycle = merge_cfg.get("sbs_metadata_cycle")
        sbs_channel = merge_cfg.get("sbs_metadata_channel")
        if sbs_cycle is not None:
            metadata = metadata[metadata["cycle"] == sbs_cycle]
        if sbs_channel is not None:
            metadata = metadata[metadata["channel"] == sbs_channel]

        fallback_pixel_size = merge_cfg.get("sbs_pixel_size")
    else:
        fallback_pixel_size = merge_cfg.get("phenotype_pixel_size")

    if not metadata.get("channel", pd.Series()).empty:
        ph_channel = merge_cfg.get("ph_metadata_channel")
        if data_type == "phenotype" and ph_channel is not None:
            metadata = metadata[metadata["channel"] == ph_channel]
        elif data_type == "phenotype":
            metadata = metadata.drop_duplicates(subset=["plate", "well", "tile"])

    stitch_result = estimate_stitch_coordinate_based(
        metadata_df=metadata, well=well, data_type=data_type,
        fallback_pixel_size=fallback_pixel_size,
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(convert_numpy_types(stitch_result), f)
    print(f"    OK estimate_stitch_{data_type} P{plate}/W{well} ({len(stitch_result.get('total_translation', {}))} tiles)")
    return out


def run_stitch(merge_cfg, pp_fp, module_fp, merge_fp, fmt, plate, well, data_type, tile_combos):
    out_positions = merge_parquet_path(merge_fp, fmt, plate, well, f"{data_type}_cell_positions")
    if out_exists(out_positions):
        print(f"    SKIP stitch_{data_type} P{plate}/W{well}")
        return out_positions

    from lib.shared.file_utils import files_to_tile_mapping
    from lib.merge.stitch import assemble_aligned_tiff_well, assemble_stitched_masks, extract_cell_positions_from_stitched_mask
    from lib.merge.eval_stitch import create_tile_arrangement_qc_plot, create_empty_qc_plot

    # ponytail: patch io.imread to handle zarr tile paths
    if fmt == "zarr":
        import skimage.io as _skio
        _orig_imread = _skio.imread
        def _zarr_imread(path, **kw):
            p = Path(path)
            if p.is_dir() or str(path).endswith("zarr.json"):
                import zarr
                d = p.parent if str(path).endswith("zarr.json") else p
                z = zarr.open(str(d), mode="r")
                arr = z["0"][:] if "0" in z else z[:]
                return arr.squeeze()
            return _orig_imread(path, **kw)
        _skio.imread = _zarr_imread

    metadata = validate_dtypes(read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, data_type)))

    filters = {}
    if data_type == "sbs":
        if merge_cfg.get("sbs_metadata_cycle") is not None:
            filters["cycle"] = merge_cfg["sbs_metadata_cycle"]
        if merge_cfg.get("sbs_metadata_channel") is not None:
            filters["channel"] = merge_cfg["sbs_metadata_channel"]
    elif data_type == "phenotype":
        if merge_cfg.get("ph_metadata_channel") is not None:
            filters["channel"] = merge_cfg["ph_metadata_channel"]

    for fk, fv in filters.items():
        metadata = metadata[metadata[fk] == fv]
    if not filters:
        metadata = metadata.drop_duplicates(subset=["plate", "well", "tile"])

    stitch_config_path = merge_stitch_config_path(merge_fp, fmt, plate, well, data_type)
    with open(stitch_config_path, "r") as f:
        stitch_config = yaml.safe_load(f)
    shifts = stitch_config["total_translation"]

    tiles = sorted(tile_combos["tile"].unique())
    tile_files = {}
    mask_files = {}
    for t in tiles:
        tp = tile_image_path(module_fp, fmt, plate, well, t, "aligned")
        mp = tile_image_path(module_fp, fmt, plate, well, t, "nuclei", subdirectory="labels")
        # ponytail: zarr paths point to directories, not files
        if fmt == "zarr":
            tp = str(Path(tp).parent) if tp.endswith("zarr.json") else tp
            mp = str(Path(mp).parent) if mp.endswith("zarr.json") else mp
        if Path(tp).exists():
            tile_files[int(t)] = tp
        if Path(mp).exists():
            mask_files[int(t)] = mp

    flipud = merge_cfg.get("flipud", False)
    fliplr = merge_cfg.get("fliplr", False)
    rot90 = merge_cfg.get("rot90", 0)
    create_stitched_image = merge_cfg.get("stitched_image", True)

    if create_stitched_image:
        stitched_image = assemble_aligned_tiff_well(
            tile_files=tile_files, shifts=shifts, well=well,
            flipud=flipud, fliplr=fliplr, rot90=rot90, channel=0,
        )
    else:
        stitched_image = np.array([[0]], dtype=np.uint16)

    if mask_files:
        stitched_mask, cell_id_mapping = assemble_stitched_masks(
            mask_files=mask_files, shifts=shifts, well=well,
            flipud=flipud, fliplr=fliplr, rot90=rot90, return_cell_mapping=True,
        )
        if stitched_mask.max() > 0:
            cell_positions = extract_cell_positions_from_stitched_mask(
                stitched_mask=stitched_mask, well=well, plate=plate,
                tile_metadata=metadata, shifts=shifts,
                cell_id_mapping=cell_id_mapping, data_type=data_type,
            )
        else:
            cell_positions = pd.DataFrame()
    else:
        stitched_mask = np.zeros(stitched_image.shape, dtype=np.uint16)
        cell_positions = pd.DataFrame()

    Path(out_positions).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(cell_positions, out_positions)

    qc_out = merge_eval_well_path(merge_fp, fmt, plate, well, f"{data_type}_tile_qc", "png")
    Path(qc_out).parent.mkdir(parents=True, exist_ok=True)
    if len(cell_positions) > 0:
        create_tile_arrangement_qc_plot(cell_positions, qc_out, data_type, well, plate)
    else:
        create_empty_qc_plot(qc_out, data_type, well)

    img_out = merge_image_path(merge_fp, fmt, plate, well, f"{data_type}_stitched_image")
    mask_out = merge_image_path(merge_fp, fmt, plate, well, f"{data_type}_stitched_mask")
    Path(img_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(img_out, stitched_image if create_stitched_image else np.array([[0]], dtype=np.uint16))
    np.save(mask_out, stitched_mask if create_stitched_image else np.array([[0]], dtype=np.uint16))

    print(f"    OK stitch_{data_type} P{plate}/W{well} ({len(cell_positions)} cells)")
    return out_positions


# ---------------------------------------------------------------------------
# Stitch Steps 3-4: Stitch Alignment + Stitch Merge
# ---------------------------------------------------------------------------

def run_stitch_alignment(merge_cfg, merge_fp, fmt, plate, well):
    out_params = merge_parquet_path(merge_fp, fmt, plate, well, "alignment_params")
    if out_exists(out_params):
        print(f"    SKIP stitch_alignment P{plate}/W{well}")
        return out_params

    from lib.merge.stitch_alignment import align_well_positions

    ph_positions = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_cell_positions")))
    sbs_positions = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "sbs_cell_positions")))
    score_threshold = merge_cfg["score"]

    result = align_well_positions(
        phenotype_positions=ph_positions, sbs_positions=sbs_positions,
        score_threshold=score_threshold, adaptive_region=True,
        max_cells_for_hash=75000, initial_region_size=7000,
        min_triangles=100, threshold_triangle=0.3, threshold_point=2.0,
    )

    ph_scaled = result["phenotype_scaled"]
    ph_transformed = result["phenotype_transformed"]
    ph_triangles = result["phenotype_triangles"]
    sbs_triangles = result["sbs_triangles"]
    alignment_params = result["alignment_params"]
    summary = result["summary"]

    Path(out_params).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(ph_scaled, merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_scaled"))
    write_parquet(ph_triangles, merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_triangles"))
    write_parquet(sbs_triangles, merge_parquet_path(merge_fp, fmt, plate, well, "sbs_triangles"))
    write_parquet(alignment_params, out_params)
    write_parquet(ph_transformed, merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_transformed"))

    summary_out = merge_eval_well_path(merge_fp, fmt, plate, well, "alignment_summary", "tsv")
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([{"plate": str(plate), "well": str(well), **summary}])
    summary_df.to_csv(summary_out, sep="\t", index=False, float_format="%.6g")

    print(f"    OK stitch_alignment P{plate}/W{well} (status={result['status']})")
    return out_params


def run_stitch_merge(merge_cfg, merge_fp, fmt, plate, well):
    out = merge_parquet_path(merge_fp, fmt, plate, well, "merged_cells")
    if out_exists(out):
        print(f"    SKIP stitch_merge P{plate}/W{well}")
        return out

    from lib.merge.stitch_merge import (
        load_alignment_parameters, find_cell_matches,
        filter_tiles_by_diversity, build_final_matches, create_merge_summary,
    )

    ph_scaled = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_scaled")))
    sbs_positions = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "sbs_cell_positions")))
    alignment_params = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "alignment_params")))
    ph_transformed = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "phenotype_transformed")))

    threshold = merge_cfg["threshold"]

    ph_filtered = filter_tiles_by_diversity(ph_scaled, "Phenotype")
    sbs_filtered = filter_tiles_by_diversity(sbs_positions, "SBS")
    ph_transformed_filtered = ph_transformed[ph_transformed.index.isin(ph_filtered.index)]

    alignment = load_alignment_parameters(alignment_params.iloc[0])

    raw_matches, match_stats = find_cell_matches(
        phenotype_positions=ph_filtered, sbs_positions=sbs_filtered,
        alignment=alignment, threshold=threshold,
        transformed_phenotype_positions=ph_transformed_filtered,
    )

    final_matches = build_final_matches(
        raw_matches=raw_matches, phenotype_filtered=ph_filtered,
        sbs_filtered=sbs_filtered, plate=plate, well=well,
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(final_matches, out)
    write_parquet(final_matches, merge_parquet_path(merge_fp, fmt, plate, well, "raw_matches"))

    summary_df = create_merge_summary(
        final_matches=final_matches, phenotype_scaled=ph_scaled,
        sbs_positions=sbs_positions, phenotype_filtered=ph_filtered,
        sbs_filtered=sbs_filtered, alignment=alignment,
        threshold=threshold, plate=plate, well=well,
    )
    summary_out = merge_eval_well_path(merge_fp, fmt, plate, well, "merge_summary", "tsv")
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, sep="\t", index=False)

    print(f"    OK stitch_merge P{plate}/W{well} ({len(final_matches)} matched cells)")
    return out


# ---------------------------------------------------------------------------
# Stitch Step 5: Summarize Stitch (plate-level)
# ---------------------------------------------------------------------------

def run_summarize_stitch(merge_fp, fmt, plate, wells):
    out = merge_eval_path(merge_fp, fmt, plate, "alignment_summaries", "tsv")
    if out_exists(out):
        print(f"    SKIP summarize_stitch P{plate}")
        return

    alignment_dfs = []
    for w in wells:
        p = merge_eval_well_path(merge_fp, fmt, plate, w, "alignment_summary", "tsv")
        if Path(p).exists():
            df = pd.read_csv(p, sep="\t")
            if not df.empty:
                alignment_dfs.append(df)

    alignment_all = pd.concat(alignment_dfs, ignore_index=True) if alignment_dfs else pd.DataFrame(columns=["plate", "well"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    alignment_all.to_csv(out, sep="\t", index=False)

    merge_dfs = []
    for w in wells:
        p = merge_eval_well_path(merge_fp, fmt, plate, w, "merge_summary", "tsv")
        if Path(p).exists():
            df = pd.read_csv(p, sep="\t")
            if not df.empty:
                merge_dfs.append(df)

    merge_all = pd.concat(merge_dfs, ignore_index=True) if merge_dfs else pd.DataFrame(columns=["plate", "well"])
    merge_out = merge_eval_path(merge_fp, fmt, plate, "cell_merge_summaries", "tsv")
    merge_all.to_csv(merge_out, sep="\t", index=False)

    print(f"    OK summarize_stitch P{plate}")

# ---------------------------------------------------------------------------
# Step 3: Format Merge
# ---------------------------------------------------------------------------

def run_format_merge(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well):
    out = merge_parquet_path(merge_fp, fmt, plate, well, "merge_formatted")
    if out_exists(out):
        print(f"    SKIP format_merge P{plate}/W{well}")
        return out

    from lib.merge.format_merge import (
        fov_distance, identify_single_gene_mappings,
        calculate_channel_mins, attach_global_pixel_coords,
    )

    approach = merge_cfg.get("approach", "fast")
    merge_input = "fast_merge" if approach == "fast" else "merged_cells"
    merge_data = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, merge_input)))
    sbs_cells = validate_dtypes(read_parquet(module_parquet_path(sbs_fp, fmt, plate, well, "cells")))
    ph_min_cp = validate_dtypes(read_parquet(module_parquet_path(phenotype_fp, fmt, plate, well, "phenotype_cp_min")))

    ph_dims = tuple(merge_cfg.get("phenotype_dimensions") or (2960, 2960))
    sbs_dims = tuple(merge_cfg.get("sbs_dimensions") or (1480, 1480))

    merge_fmt = merge_data.pipe(fov_distance, i="i_0", j="j_0", dimensions=ph_dims, suffix="_0")
    merge_fmt = merge_fmt.pipe(fov_distance, i="i_1", j="j_1", dimensions=sbs_dims, suffix="_1")

    sbs_cells["mapped_single_gene"] = sbs_cells.apply(identify_single_gene_mappings, axis=1)
    sbs_merge_cols = ["plate", "well", "tile", "cell", "mapped_single_gene"]
    for col in sbs_cells.columns:
        if any(col.startswith(p) for p in [
            "cell_barcode_", "gene_symbol_", "gene_id_", "no_recomb_",
            "Q_min_", "Q_recomb_", "cell_barcode_peak_", "cell_barcode_count_",
        ]):
            sbs_merge_cols.append(col)
    sbs_merge_cols = [c for c in sbs_merge_cols if c in sbs_cells.columns]

    merge_fmt = merge_fmt.merge(
        sbs_cells[sbs_merge_cols].rename({"tile": "site", "cell": "cell_1"}, axis=1),
        how="left", on=["plate", "well", "site", "cell_1"],
    )

    ph_min_cp = calculate_channel_mins(ph_min_cp)
    merge_fmt = merge_fmt.merge(
        ph_min_cp[["tile", "label", "channels_min"]].rename(columns={"label": "cell_0"}),
        how="left", on=["tile", "cell_0"],
    )

    if approach == "fast":
        ph_metadata = pd.read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, "phenotype"))
        sbs_metadata = pd.read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, well, "sbs"))
        merge_fmt = attach_global_pixel_coords(merge_fmt, ph_metadata, ph_dims, suffix="0")
        merge_fmt = attach_global_pixel_coords(merge_fmt, sbs_metadata, sbs_dims, suffix="1")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merge_fmt, out)
    print(f"    OK format_merge P{plate}/W{well} ({len(merge_fmt)} rows)")
    return out


# ---------------------------------------------------------------------------
# Step 4: Deduplicate Merge
# ---------------------------------------------------------------------------

def run_deduplicate_merge(merge_cfg, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well):
    out_dedup = merge_parquet_path(merge_fp, fmt, plate, well, "merge_deduplicated")
    if out_exists(out_dedup):
        print(f"    SKIP deduplicate P{plate}/W{well}")
        return out_dedup

    from lib.merge.deduplicate_merge import deduplicate_cells, check_matching_rates, analyze_distance_distribution

    merge_formatted = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "merge_formatted")))
    sbs_cells = validate_dtypes(read_parquet(module_parquet_path(sbs_fp, fmt, plate, well, "cells")))
    ph_min_cp = validate_dtypes(read_parquet(module_parquet_path(phenotype_fp, fmt, plate, well, "phenotype_cp_min")))

    approach = merge_cfg.get("approach", "fast")
    merge_dedup, dedup_stats = deduplicate_cells(
        merge_formatted, mapped_single_gene=False, return_stats=True,
        approach=approach,
        sbs_dedup_prior=merge_cfg.get("sbs_dedup_prior"),
        pheno_dedup_prior=merge_cfg.get("pheno_dedup_prior"),
    )

    dist_analysis = analyze_distance_distribution(merge_dedup)
    if dist_analysis:
        ds = dist_analysis["distance_stats"]
        print(f"    Distance: mean={ds['mean']:.2f}px, median={ds['median']:.2f}px")

    stats_out = merge_eval_well_path(merge_fp, fmt, plate, well, "deduplication_stats", "tsv")
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    dedup_stats.to_csv(stats_out, sep="\t", index=False)

    Path(out_dedup).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merge_dedup, out_dedup)

    sbs_rates_out = merge_eval_well_path(merge_fp, fmt, plate, well, "final_sbs_matching_rates", "tsv")
    sbs_rates = check_matching_rates(sbs_cells, merge_dedup, modality="sbs", return_stats=True)
    sbs_rates.to_csv(sbs_rates_out, sep="\t", index=False)

    ph_rates_out = merge_eval_well_path(merge_fp, fmt, plate, well, "final_phenotype_matching_rates", "tsv")
    ph_rates = check_matching_rates(ph_min_cp, merge_dedup, modality="phenotype", return_stats=True)
    ph_rates.to_csv(ph_rates_out, sep="\t", index=False)

    print(f"    OK deduplicate P{plate}/W{well} ({len(merge_dedup)} cells)")
    return out_dedup


# ---------------------------------------------------------------------------
# Step 5: Final Merge
# ---------------------------------------------------------------------------

def run_final_merge(merge_cfg, phenotype_fp, merge_fp, fmt, plate, well):
    out = merge_parquet_path(merge_fp, fmt, plate, well, "merge_final")
    if out_exists(out):
        print(f"    SKIP final_merge P{plate}/W{well}")
        return out

    merge_dedup = validate_dtypes(read_parquet(merge_parquet_path(merge_fp, fmt, plate, well, "merge_deduplicated")))
    cp_phenotype = validate_dtypes(read_parquet(module_parquet_path(phenotype_fp, fmt, plate, well, "phenotype_cp")))

    merged_final = merge_dedup.merge(
        cp_phenotype.rename(columns={"label": "cell_0"}),
        how="left", on=["plate", "well", "tile", "cell_0"],
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merged_final, out)
    print(f"    OK final_merge P{plate}/W{well} ({len(merged_final)} cells)")
    return out


# ---------------------------------------------------------------------------
# Step 6: Eval Merge (plate-level)
# ---------------------------------------------------------------------------

def run_eval_merge(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, wells):
    summary_out = merge_eval_path(merge_fp, fmt, plate, "merge_summary", "tsv")
    if out_exists(summary_out):
        print(f"    SKIP eval_merge P{plate}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    })

    from lib.merge.format_merge import identify_single_gene_mappings
    from lib.merge.eval_merge import plot_sbs_ph_matching_heatmap, plot_cell_positions

    dedup_paths = [merge_parquet_path(merge_fp, fmt, plate, w, "merge_deduplicated") for w in wells]
    dedup_paths = [p for p in dedup_paths if Path(p).exists()]
    merge_dedup = validate_dtypes(read_parquets(dedup_paths))

    fmt_paths = [merge_parquet_path(merge_fp, fmt, plate, w, "merge_formatted") for w in wells]
    fmt_paths = [p for p in fmt_paths if Path(p).exists()]
    merge_formatted = validate_dtypes(read_parquets(fmt_paths))

    sbs_cells_paths = [module_parquet_path(sbs_fp, fmt, plate, w, "cells") for w in wells]
    sbs_cells_paths = [p for p in sbs_cells_paths if Path(p).exists()]
    sbs_cells = validate_dtypes(read_parquets(sbs_cells_paths))

    sbs_info_paths = [module_parquet_path(sbs_fp, fmt, plate, w, "sbs_info") for w in wells]
    sbs_info_paths = [p for p in sbs_info_paths if Path(p).exists()]
    sbs_info = validate_dtypes(read_parquets(sbs_info_paths))

    ph_min_paths = [module_parquet_path(phenotype_fp, fmt, plate, w, "phenotype_cp_min") for w in wells]
    ph_min_paths = [p for p in ph_min_paths if Path(p).exists()]
    ph_min_cp = validate_dtypes(read_parquets(ph_min_paths))

    ph_info_paths = [module_parquet_path(phenotype_fp, fmt, plate, w, "phenotype_info") for w in wells]
    ph_info_paths = [p for p in ph_info_paths if Path(p).exists()]
    ph_info = validate_dtypes(read_parquets(ph_info_paths))

    sbs_meta = pd.concat([
        pd.read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, w, "sbs"))
        for w in wells if Path(preprocess_metadata_path(pp_fp, fmt, plate, w, "sbs")).exists()
    ], ignore_index=True).drop_duplicates(subset=["well", "tile"])

    ph_meta = pd.concat([
        pd.read_parquet(preprocess_metadata_path(pp_fp, fmt, plate, w, "phenotype"))
        for w in wells if Path(preprocess_metadata_path(pp_fp, fmt, plate, w, "phenotype")).exists()
    ], ignore_index=True).drop_duplicates(subset=["well", "tile"])

    # Load dedup stats
    dedup_stats = {}
    for w in wells:
        stats_path = merge_eval_well_path(merge_fp, fmt, plate, w, "deduplication_stats", "tsv")
        if Path(stats_path).exists():
            dedup_stats[w] = pd.read_csv(stats_path, sep="\t")

    merge_dedup["mapped_single_gene"] = merge_dedup.apply(identify_single_gene_mappings, axis=1)

    # Build summary
    rows = []
    for well in sorted(merge_dedup["well"].unique()):
        wm = merge_dedup[merge_dedup["well"] == well]
        w_ph = ph_info[ph_info["well"] == well]
        w_sbs = sbs_info[sbs_info["well"] == well]
        w_dedup = dedup_stats.get(well)

        matched_raw = int(w_dedup[w_dedup["stage"] == "Initial"]["total_cells"].iloc[0]) if w_dedup is not None and "Initial" in w_dedup["stage"].values else len(wm)
        matched_final = len(wm)
        ph_cells = len(w_ph)
        sbs_cells_count = len(w_sbs)
        unique_ph = wm[["plate", "well", "tile", "site", "cell_0"]].drop_duplicates().shape[0]
        unique_sbs = wm[["plate", "well", "tile", "cell_1"]].drop_duplicates().shape[0]

        rows.append({
            "well": well,
            "ph_cells": ph_cells,
            "sbs_cells": sbs_cells_count,
            "matched_raw": matched_raw,
            "total_match_pairs": matched_final,
            "unique_ph_in_merge": unique_ph,
            "unique_sbs_in_merge": unique_sbs,
            "ph_recovery_rate": round(unique_ph / ph_cells, 3) if ph_cells > 0 else 0,
            "sbs_recovery_rate": round(unique_sbs / sbs_cells_count, 3) if sbs_cells_count > 0 else 0,
            "dist_mean": round(wm["distance"].mean(), 2),
            "dist_median": round(wm["distance"].median(), 2),
            "cells_with_barcode": int(wm["gene_symbol_0"].notna().sum()),
            "single_gene_count": int(wm["mapped_single_gene"].sum()),
            "single_gene_rate": round(wm["mapped_single_gene"].mean(), 3),
        })

    summary_df = pd.DataFrame(rows)
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, sep="\t", index=False)

    merge_minimal = merge_formatted[["plate", "well", "tile", "site", "cell_0", "cell_1", "distance"]]

    def _save_fig(fig, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)

    sbs_summary, fig = plot_sbs_ph_matching_heatmap(
        merge_minimal, sbs_info.rename(columns={"cell": "cell_1"}),
        target="sbs", metadata=sbs_meta, return_summary=True,
    )
    sbs_summary.to_csv(merge_eval_path(merge_fp, fmt, plate, "sbs_to_ph_matching_rates", "tsv"), sep="\t", index=False)
    _save_fig(fig, merge_eval_path(merge_fp, fmt, plate, "sbs_to_ph_matching_rates", "png"))

    ph_summary, fig = plot_sbs_ph_matching_heatmap(
        merge_minimal, ph_info.rename(columns={"cell": "cell_0"}),
        target="phenotype", metadata=ph_meta, return_summary=True,
    )
    ph_summary.to_csv(merge_eval_path(merge_fp, fmt, plate, "ph_to_sbs_matching_rates", "tsv"), sep="\t", index=False)
    _save_fig(fig, merge_eval_path(merge_fp, fmt, plate, "ph_to_sbs_matching_rates", "png"))

    fig = plot_cell_positions(merge_dedup, title="All Cells by Channel Min")
    _save_fig(fig, merge_eval_path(merge_fp, fmt, plate, "all_cells_by_channel_min", "png"))

    fig = plot_cell_positions(merge_dedup.query("channels_min==0"), title="Cells with Channel Min = 0", color="red")
    _save_fig(fig, merge_eval_path(merge_fp, fmt, plate, "cells_with_channel_min_0", "png"))

    # Aggregate dedup summaries
    dedup_dfs = []
    for w, df in dedup_stats.items():
        df = df.copy()
        df["well"] = w
        dedup_dfs.append(df)
    if dedup_dfs:
        dedup_all = pd.concat(dedup_dfs, ignore_index=True)
        stage_order = ["Initial", "After SBS dedup", "After phenotype dedup"]
        dedup_all["stage"] = pd.Categorical(dedup_all["stage"], categories=stage_order, ordered=True)
        dedup_all = dedup_all.sort_values(["well", "stage"]).reset_index(drop=True)
    else:
        dedup_all = pd.DataFrame(columns=["well", "stage", "total_cells"])
    dedup_all.to_csv(merge_eval_path(merge_fp, fmt, plate, "dedup_summaries", "tsv"), sep="\t", index=False)

    print(f"    OK eval_merge P{plate}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_merge(config, args):
    merge_cfg = config.get("merge", {})
    root = config["all"]["root_fp"]
    fmt = config.get("all", {}).get("image_format", "tiff")

    merge_fp = Path(root) / "merge"
    pp_fp = Path(root) / "preprocess"
    phenotype_fp = Path(root) / "phenotype"
    sbs_fp = Path(root) / "sbs"

    combo_path = merge_cfg.get("merge_combo_fp")
    if not combo_path:
        raise ValueError("merge.merge_combo_fp must be set in config")

    combos = pd.read_csv(combo_path, sep="\t").astype(str)
    if args.plate_filter:
        combos = combos[combos["plate"] == str(args.plate_filter)]
    if args.well_filter:
        combos = combos[combos["well"] == str(args.well_filter)]

    if combos.empty:
        print("  No merge combos to process")
        return 0

    approach = merge_cfg.get("approach", "fast")
    errs = 0
    n_jobs = args.workers

    run_per_well = args.step in ("per-well", "all")
    run_eval = args.step in ("eval", "all")
    run_stitch_only = args.step == "stitch-only"

    # Load tile combos for stitch approach
    tile_combos = None
    if approach == "stitch" and (run_per_well or run_stitch_only):
        ph_combo_fp = config.get("phenotype", {}).get("phenotype_combo_fp")
        if ph_combo_fp:
            tile_combos = pd.read_csv(ph_combo_fp, sep="\t").astype(str)

    if run_per_well or run_stitch_only:
        for _, row in combos.iterrows():
            plate, well = row["plate"], row["well"]
            print(f"\n{'=' * 60}")
            print(f"  Merge ({approach}): Plate {plate}, Well {well}")
            print(f"{'=' * 60}")

            try:
                if approach == "fast":
                    t0 = time.time()
                    run_fast_alignment(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well, n_jobs)
                    print(f"    fast_alignment: {time.time() - t0:.1f}s")

                    t0 = time.time()
                    run_fast_merge(merge_cfg, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well)
                    print(f"    fast_merge: {time.time() - t0:.1f}s")

                elif approach == "stitch":
                    well_tiles = tile_combos[(tile_combos["plate"] == str(plate)) & (tile_combos["well"] == str(well))] if tile_combos is not None else pd.DataFrame(columns=["tile"])

                    for dt in ("phenotype", "sbs"):
                        t0 = time.time()
                        run_estimate_stitch(merge_cfg, pp_fp, merge_fp, fmt, plate, well, dt)
                        print(f"    estimate_stitch_{dt}: {time.time() - t0:.1f}s")

                    for dt in ("phenotype", "sbs"):
                        module_fp = phenotype_fp if dt == "phenotype" else sbs_fp
                        t0 = time.time()
                        run_stitch(merge_cfg, pp_fp, module_fp, merge_fp, fmt, plate, well, dt, well_tiles)
                        print(f"    stitch_{dt}: {time.time() - t0:.1f}s")

                    t0 = time.time()
                    run_stitch_alignment(merge_cfg, merge_fp, fmt, plate, well)
                    print(f"    stitch_alignment: {time.time() - t0:.1f}s")

                    t0 = time.time()
                    run_stitch_merge(merge_cfg, merge_fp, fmt, plate, well)
                    print(f"    stitch_merge: {time.time() - t0:.1f}s")

                if not run_stitch_only:
                    t0 = time.time()
                    run_format_merge(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well)
                    print(f"    format_merge: {time.time() - t0:.1f}s")

                    t0 = time.time()
                    run_deduplicate_merge(merge_cfg, phenotype_fp, sbs_fp, merge_fp, fmt, plate, well)
                    print(f"    deduplicate: {time.time() - t0:.1f}s")

                    t0 = time.time()
                    run_final_merge(merge_cfg, phenotype_fp, merge_fp, fmt, plate, well)
                    print(f"    final_merge: {time.time() - t0:.1f}s")

            except Exception as e:
                print(f"    ERR merge P{plate}/W{well}: {e}")
                import traceback
                traceback.print_exc()
                errs += 1

    if run_eval and not run_stitch_only:
        for plate in sorted(combos["plate"].unique()):
            wells = sorted(combos[combos["plate"] == plate]["well"].unique())
            print(f"\n  Eval merge: Plate {plate}, wells={wells}")
            try:
                if approach == "stitch":
                    t0 = time.time()
                    run_summarize_stitch(merge_fp, fmt, plate, wells)
                    print(f"    summarize_stitch: {time.time() - t0:.1f}s")

                t0 = time.time()
                run_eval_merge(merge_cfg, pp_fp, phenotype_fp, sbs_fp, merge_fp, fmt, plate, wells)
                print(f"    eval_merge: {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"    ERR eval_merge P{plate}: {e}")
                import traceback
                traceback.print_exc()
                errs += 1

    return errs


def main():
    p = argparse.ArgumentParser(description="Direct merge runner (bypasses Snakemake)")
    p.add_argument("--config", required=True, help="Path to config.yml")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for alignment (joblib n_jobs)")
    p.add_argument("--plate-filter", type=str, default=None, help="Process only this plate")
    p.add_argument("--well-filter", type=str, default=None, help="Process only this well")
    p.add_argument("--step", choices=["per-well", "eval", "all", "stitch-only"], default="all",
                   help="per-well=approach steps+format+dedup+final, eval=plate-level eval, all=everything, stitch-only=stitch steps only (no format/dedup/final)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without executing")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    fmt = config.get("all", {}).get("image_format", "tiff")

    if "merge" not in config:
        print("ERROR: No 'merge' section in config. Run 5.configure_merge_params.ipynb first.")
        sys.exit(1)

    merge_cfg = config["merge"]

    print(f"{'#' * 60}")
    print(f"  Direct Merge Runner | format={fmt}")
    print(f"  approach={merge_cfg.get('approach', 'fast')} workers={args.workers} step={args.step}")
    print(f"  plate_filter={args.plate_filter or 'none'} well_filter={args.well_filter or 'none'}")
    print(f"{'#' * 60}")

    approach = merge_cfg.get("approach", "fast")
    if approach not in ("fast", "stitch"):
        print(f"ERROR: Unsupported approach '{approach}'. Use 'fast' or 'stitch'.")
        sys.exit(1)

    if args.dry_run:
        combos = pd.read_csv(merge_cfg["merge_combo_fp"], sep="\t").astype(str)
        if args.plate_filter:
            combos = combos[combos["plate"] == str(args.plate_filter)]
        if args.well_filter:
            combos = combos[combos["well"] == str(args.well_filter)]
        print(f"\n  DRY RUN ({approach}): would process {len(combos)} plate-well combos:")
        for _, row in combos.iterrows():
            print(f"    Plate {row['plate']}, Well {row['well']}")
        if approach == "fast":
            print(f"\n  Steps: fast_alignment → fast_merge → format → dedup → final_merge → eval")
        else:
            print(f"\n  Steps: estimate_stitch → stitch → stitch_alignment → stitch_merge → format → dedup → final_merge → summarize_stitch → eval")
        sys.exit(0)

    t0 = time.time()
    errs = process_merge(config, args)
    status = "DONE" if errs == 0 else "FAILED"
    print(f"\n{'#' * 60}")
    print(f"  {status}: {time.time() - t0:.1f}s total, {errs} errors")
    print(f"{'#' * 60}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
