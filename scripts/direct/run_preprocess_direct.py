#!/usr/bin/env python3
"""Direct preprocessing runner — bypasses Snakemake DAG build.

Calls the same brieflow library functions as Snakemake, producing identical
outputs in the same directory structure. Uses ProcessPoolExecutor for
per-tile parallelism.

Usage:
    python run_preprocess_direct.py --config config/config.yml --max-tiles 100 --workers 8
    python run_preprocess_direct.py --config config/config.yml --plate-filter 1 --image-type sbs
"""

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

# brieflow library: ../../workflow relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "workflow"))

from lib.preprocess.preprocess import convert_to_array, extract_metadata, get_data_config
from lib.preprocess.file_utils import get_metadata_wildcard_combos, get_sample_fps
from lib.shared.file_utils import get_data_output_path, get_image_output_path, validate_dtypes
from lib.shared.illumination_correction import calculate_ic_field
from lib.shared.image_io import save_image
from lib.shared.parquet_io import write_parquet
from lib.shared.resource_monitor import monitor_step, set_benchmark_context


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def make_loc(img_fmt, plate, well=None, tile=None, cycle=None):
    """Build ordered data_location dict. For zarr, splits well into row + col."""
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


def make_md_loc(img_fmt, row_data, columns):
    """Build data_location from a DataFrame row using specified columns.

    Handles well -> row/col splitting for zarr. Preserves column order
    (important for get_nested_path directory nesting).
    """
    loc = {}
    for c in columns:
        v = str(row_data[c])
        if c == "well" and img_fmt == "zarr":
            m = re.match(r"^([A-Za-z]+)(\d+)$", v)
            if m:
                loc["row"] = m.group(1)
                loc["col"] = m.group(2)
        else:
            loc[c] = v
    return loc


def out_exists(path):
    """Check if output exists. Handles zarr.json sentinels and .zarr dirs."""
    p = Path(path)
    if p.name == "zarr.json":
        return p.exists()
    if p.suffix == ".zarr":
        return p.is_dir() and any(p.iterdir()) if p.exists() else False
    return p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Worker functions (run in child processes)
# ---------------------------------------------------------------------------

def _extract_one(task):
    """Extract metadata for one tile/cycle/round combination."""
    (sample_files, meta_files, plate, well, tile, cycle, rnd,
     data_fmt, data_org, ext_metadata_fp, output_path) = task
    tag = f"P{plate}/W{well}/T{tile}" + (f"/C{cycle}" if cycle else "")
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if meta_files:
            file_input = meta_files[0]
            mfp = meta_files[0]
        else:
            file_input = sample_files
            mfp = ext_metadata_fp
        df = extract_metadata(
            file_input, plate=plate, well=well, tile=tile,
            cycle=cycle, round=rnd,
            data_format=data_fmt, data_organization=data_org,
            metadata_file_path=mfp,
        )
        df.to_csv(output_path, index=False, sep="\t")
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def _convert_one(task):
    """Convert one tile image to the output format."""
    (files, data_fmt, data_org, tile_int, flip, nz, ch_names, output_path) = task
    tag = f"T{tile_int}"
    if out_exists(output_path):
        return "skip", tag
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pos = tile_int if data_org == "well" else None
        arr = convert_to_array(
            files, data_format=data_fmt, data_organization=data_org,
            position=pos, channel_order_flip=flip, n_z_planes=nz,
        )
        save_image(arr, output_path, channel_names=ch_names)
        return "ok", tag
    except Exception as e:
        return "err", f"{tag}: {e}"


def run_parallel(tasks, fn, workers, label, max_tasks_per_child=None):
    """Execute tasks with a process pool, reporting progress."""
    n = len(tasks)
    if n == 0:
        print(f"  {label}: nothing to do")
        return 0
    ok = skip = err = 0
    t0 = time.time()
    print(f"\n  {label}: {n} tasks, {workers} workers")
    pool_kwargs = {"max_workers": workers}
    if max_tasks_per_child is not None:
        import multiprocessing as _mp
        pool_kwargs["mp_context"] = _mp.get_context("spawn")
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
    with monitor_step(label), ProcessPoolExecutor(**pool_kwargs) as pool:
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
                elapsed = time.time() - t0
                print(f"    [{total}/{n}] {elapsed:.0f}s  new={ok} skip={skip} err={err}")
    print(f"  {label}: done in {time.time() - t0:.1f}s")
    return err


# ---------------------------------------------------------------------------
# IC field step (well-group parallelism)
# ---------------------------------------------------------------------------

def _ic_per_group_threads(total_workers, concurrency):
    """Threads per IC group when running `concurrency` groups concurrently.

    concurrency<=0 is treated as 1 (per_group == total_workers).
    """
    return max(1, total_workers // max(1, concurrency))


def _calculate_ic_one(task):
    """Compute one well-group IC field. Module-level so the spawn pool can pickle it.

    Returns (status, msg) with status in {"ok", "skip", "err"} — run_parallel's contract.
    """
    ic_out, tag, inputs = task["ic_out"], task["tag"], task["inputs"]
    if out_exists(ic_out):
        return "skip", tag
    missing = [f for f in inputs if not out_exists(f)]
    if missing:
        return "err", f"IC {tag}: {len(missing)}/{len(inputs)} inputs missing"
    try:
        Path(ic_out).parent.mkdir(parents=True, exist_ok=True)
        field = calculate_ic_field(
            inputs, threading=True, n_jobs=task["n_jobs"],
            sample_fraction=task["sample_fraction"], smooth=task["smooth"],
            random_seed=task["random_seed"],
        )
        save_image(field, ic_out)
        return "ok", tag
    except Exception as e:
        return "err", f"IC {tag}: {e}"


def run_ic_step(image_type, combos, pp_fp, fmt, pp, total_workers, has_cycle):
    """Calculate IC fields per well-group.

    preprocess.ic_group_concurrency (default 1) runs N well-groups at once, each given
    total_workers//N threads. A single well's IC caps at ~8 effective threads, so the
    serial loop leaves most cores idle on high-vCPU boxes; running groups concurrently
    reclaims them. concurrency<=1 is byte-identical to the prior serial loop.
    """
    # ponytail: ~4-10 GB/group; preprocess is disk-read-bound so real speedup is
    # capped by storage bandwidth — largest on high-vCPU VMs / fast disks.
    group_cols = ["plate", "well", "cycle"] if has_cycle else ["plate", "well"]
    sample_frac = pp.get("sample_fraction", 1)
    ic_smooth = pp.get("ic_smooth", None)
    ic_seed = pp.get("ic_random_seed", None)
    concurrency = int(pp.get("ic_group_concurrency", 1))

    # Shared task build: output path + input gathering (identical in both branches)
    tasks = []
    for gk, gdf in combos.groupby(group_cols):
        plate, well = str(gk[0]), str(gk[1])
        cycle = str(gk[2]) if has_cycle else None
        tag = f"P{plate}/W{well}" + (f"/C{cycle}" if cycle else "")

        ic_loc = make_loc(fmt, plate, well, cycle=cycle)
        ic_out = str(
            pp_fp / "ic_fields" / image_type / get_data_output_path(ic_loc, "ic_field", fmt, fmt)
        )
        inputs = []
        for _, tr in gdf.iterrows():
            tl = make_loc(fmt, plate, well, str(tr["tile"]), cycle)
            inputs.append(
                str(pp_fp / get_image_output_path(tl, "image", fmt, image_subdir=image_type))
            )
        tasks.append({
            "tag": tag, "ic_out": ic_out, "inputs": inputs,
            "sample_fraction": sample_frac, "smooth": ic_smooth, "random_seed": ic_seed,
        })

    if concurrency <= 1:
        # Serial in-process loop, each IC using all workers — the backward-compat path.
        print(f"\n  IC fields ({image_type})...")
        errs = 0
        ic_mon = monitor_step(f"Calculate IC field ({image_type})").start()
        for task in tasks:
            task["n_jobs"] = total_workers
            status, msg = _calculate_ic_one(task)
            if status == "skip":
                print(f"    SKIP IC {task['tag']}")
            elif status == "ok":
                print(f"    OK IC {task['tag']}")
            else:
                print(f"    ERR {msg}")
                errs += 1
        ic_mon.stop()
        return errs

    per_group = _ic_per_group_threads(total_workers, concurrency)
    print(f"\n  IC fields ({image_type}): {concurrency} groups × {per_group} threads/group")
    for task in tasks:
        task["n_jobs"] = per_group
    return run_parallel(
        tasks, _calculate_ic_one, workers=concurrency,
        label=f"Calculate IC field ({image_type})", max_tasks_per_child=1,
    )


# ---------------------------------------------------------------------------
# Main processing per image type
# ---------------------------------------------------------------------------

def process(image_type, config, args):
    """Run all preprocessing steps for one image type (sbs or phenotype)."""
    pp = config.get("preprocess", {})
    root = config["all"]["root_fp"]
    fmt = config.get("all", {}).get("image_format", "tiff")
    pp_fp = Path(root) / "preprocess"
    dc = get_data_config(image_type, {"preprocess": pp})

    # Load samples and wildcard combos
    samples_fp = Path(pp[f"{image_type}_samples_fp"])
    if not samples_fp.exists():
        print(f"  {image_type}: samples file not found: {samples_fp}")
        return 0
    samples_df = pd.read_csv(samples_fp, sep="\t")
    combos = pd.read_csv(pp[f"{image_type}_combo_fp"], sep="\t").astype(str)

    # External metadata samples (optional)
    mfp = pp.get(f"{image_type}_metadata_samples_df_fp")
    meta_samples = (
        pd.read_csv(mfp, sep="\t") if mfp and Path(mfp).exists() else pd.DataFrame()
    )

    # Apply plate filter
    if args.plate_filter is not None:
        pf = str(args.plate_filter)
        samples_df = samples_df[samples_df["plate"].astype(str) == pf]
        combos = combos[combos["plate"] == pf]
        if not meta_samples.empty and "plate" in meta_samples.columns:
            meta_samples = meta_samples[meta_samples["plate"].astype(str) == pf]

    # Apply max-tiles filter
    if args.max_tiles and "tile" in combos.columns:
        all_tiles = sorted(combos["tile"].unique(), key=lambda x: int(x))
        if len(all_tiles) > args.max_tiles:
            keep = set(all_tiles[: args.max_tiles])
            combos = combos[combos["tile"].isin(keep)]

    if combos.empty:
        print(f"  {image_type}: no combos to process")
        return 0

    has_cycle = "cycle" in combos.columns

    # Metadata wildcard combos (may differ from image combos)
    md_combos = get_metadata_wildcard_combos(samples_df, meta_samples)
    if args.max_tiles and "tile" in md_combos.columns:
        tile_set = set(combos["tile"].unique())
        md_combos = md_combos[md_combos["tile"].astype(str).isin(tile_set)]

    # Columns used in metadata output paths (exclude sample_fp, row, col)
    md_cols = [c for c in md_combos.columns if c not in ("sample_fp", "row", "col")]

    print(f"\n{'=' * 60}")
    print(f"  {image_type.upper()}: {len(combos)} tile combos")
    print(f"  data_format={dc['data_format']}  org={dc['image_data_organization']}  img_fmt={fmt}")
    print(f"  metadata combos: {len(md_combos)}  cols: {md_cols}")
    print(f"{'=' * 60}")

    errs = 0

    # ------------------------------------------------------------------
    # Step 1: Extract metadata
    # ------------------------------------------------------------------
    tasks = []
    for _, r in md_combos.iterrows():
        plate = str(r["plate"])
        well = str(r["well"]) if "well" in r.index else ""
        tile = str(r["tile"]) if "tile" in r.index else None
        cycle = str(r["cycle"]) if "cycle" in r.index else None
        rnd = str(r["round"]) if "round" in r.index else None

        md_loc = make_md_loc(fmt, r, md_cols)
        out = str(
            pp_fp / "metadata" / image_type / get_data_output_path(md_loc, "metadata", "tsv", fmt)
        )

        # Build filter kwargs for get_sample_fps
        filt = {"plate": plate, "well": well}
        if tile and dc["metadata_data_organization"] == "tile":
            filt["tile"] = tile
        if cycle:
            filt["cycle"] = cycle
        if rnd:
            filt["round_order"] = rnd

        sample_files, meta_file_list = [], []
        if not meta_samples.empty:
            mf_kw = {k: v for k, v in filt.items() if k in meta_samples.columns}
            try:
                fp = get_sample_fps(meta_samples, **mf_kw)
                meta_file_list = [fp] if isinstance(fp, str) else list(fp)
            except Exception:
                continue
        else:
            sf_kw = filt.copy()
            if dc["image_data_organization"] == "well":
                sf_kw.pop("tile", None)
            ch_order = pp.get(f"{image_type}_channel_order")
            if ch_order:
                sf_kw["channel_order"] = ch_order
            try:
                fp = get_sample_fps(samples_df, **sf_kw)
                sample_files = fp if isinstance(fp, list) else [fp]
            except Exception:
                continue

        tasks.append((
            sample_files, meta_file_list, plate, well, tile, cycle, rnd,
            dc["data_format"], dc["image_data_organization"],
            pp.get(f"{image_type}_metadata_samples_df_fp"),
            out,
        ))

    # ponytail: phenotype extract_metadata loads a whole ND2 well (~197GB RSS/well, measured);
    # >1 concurrent worker OOMs the 251GB box (24/8/4/3/2 all OOM'd or climbed to danger).
    # SBS wells are small (~5.6GB) so they keep full parallelism.
    extract_workers = 1 if image_type == "phenotype" else args.workers
    errs += run_parallel(tasks, _extract_one, extract_workers, f"Extract metadata ({image_type})", max_tasks_per_child=1)

    # ------------------------------------------------------------------
    # Step 2: Convert images
    # ------------------------------------------------------------------
    tasks = []
    for _, r in combos.iterrows():
        plate, well, tile = str(r["plate"]), str(r["well"]), str(r["tile"])
        cycle = str(r["cycle"]) if has_cycle else None

        loc = make_loc(fmt, plate, well, tile, cycle)
        out = str(pp_fp / get_image_output_path(loc, "image", fmt, image_subdir=image_type))

        filt = {"plate": plate, "well": well}
        if dc["image_data_organization"] == "tile":
            filt["tile"] = tile
        if cycle:
            filt["cycle"] = cycle
        ch_order = pp.get(f"{image_type}_channel_order")
        rd_order = pp.get(f"{image_type}_round_order")
        if ch_order:
            filt["channel_order"] = ch_order
        if rd_order:
            filt["round_order"] = rd_order

        try:
            fp = get_sample_fps(samples_df, **filt)
            files = fp if isinstance(fp, list) else [fp]
        except Exception as e:
            print(f"    WARN convert P{plate}/W{well}/T{tile}: {e}")
            continue

        tasks.append((
            files, dc["data_format"], dc["image_data_organization"],
            int(tile), dc["channel_order_flip"], dc.get("n_z_planes"),
            dc.get("channel_order"), out,
        ))

    errs += run_parallel(tasks, _convert_one, args.workers, f"Convert images ({image_type})")

    # ------------------------------------------------------------------
    # Step 3: Calculate IC fields (per well-group; see run_ic_step)
    # ------------------------------------------------------------------
    errs += run_ic_step(image_type, combos, pp_fp, fmt, pp, args.workers, has_cycle)

    # ------------------------------------------------------------------
    # Step 4: Combine metadata (sequential)
    # ------------------------------------------------------------------
    print(f"\n  Combine metadata ({image_type})...")
    for (plate, well), gdf in md_combos.groupby(["plate", "well"]):
        plate, well = str(plate), str(well)
        c_loc = make_loc(fmt, plate, well)
        c_out = str(
            pp_fp / "metadata" / image_type
            / get_data_output_path(c_loc, "combined_metadata", "parquet", fmt)
        )

        if out_exists(c_out):
            print(f"    SKIP combine P{plate}/W{well}")
            continue

        # Gather per-tile metadata TSVs
        md_inputs = []
        for _, mr in gdf.iterrows():
            ml = make_md_loc(fmt, mr, md_cols)
            md_inputs.append(
                str(pp_fp / "metadata" / image_type / get_data_output_path(ml, "metadata", "tsv", fmt))
            )

        dfs = []
        for f in md_inputs:
            try:
                dfs.append(pd.read_csv(f, sep="\t"))
            except Exception:
                pass

        if not dfs:
            print(f"    WARN combine P{plate}/W{well}: no input files found")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        if "well" in combined.columns:
            combined = combined[combined["well"].astype(str) == well]
        combined = validate_dtypes(combined)

        Path(c_out).parent.mkdir(parents=True, exist_ok=True)
        write_parquet(combined, c_out)
        print(f"    OK combine P{plate}/W{well} ({len(combined)} rows)")

    # ------------------------------------------------------------------
    # Step 5: Finalize HCS metadata (zarr only)
    # ------------------------------------------------------------------
    if fmt == "zarr":
        print(f"\n  HCS finalize ({image_type})...")
        from lib.shared.hcs import write_hcs_metadata

        plates = sorted(combos["plate"].unique())
        ch_meta = pp.get(f"{image_type}_channels_metadata")
        for p in plates:
            zdir = pp_fp / image_type / f"image_{p}.zarr"
            if zdir.exists():
                try:
                    write_hcs_metadata(zdir, channels_metadata=ch_meta)
                    print(f"    OK HCS: {zdir}")
                except Exception as e:
                    print(f"    ERR HCS: {e}")
                    errs += 1

    return errs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Direct preprocessing (bypasses Snakemake)")
    p.add_argument("--config", required=True, help="Path to config.yml")
    p.add_argument("--max-tiles", type=int, default=None, help="Process only first N tiles")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for per-tile steps")
    p.add_argument("--plate-filter", type=int, default=None, help="Process only this plate")
    p.add_argument("--image-type", choices=["sbs", "phenotype", "both"], default="both",
                   help="Which image type(s) to preprocess")
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    set_benchmark_context("preprocess", config["all"]["root_fp"])
    img_fmt = config.get("all", {}).get("image_format", "tiff")

    print(f"{'#' * 60}")
    print(f"  Direct Preprocessor")
    print(f"  config={args.config}")
    print(f"  root={config['all']['root_fp']}")
    print(f"  format={img_fmt}  workers={args.workers}")
    print(f"  max_tiles={args.max_tiles or 'all'}  plate_filter={args.plate_filter or 'none'}")
    print(f"  image_type={args.image_type}")
    print(f"{'#' * 60}")

    t0 = time.time()
    errs = 0
    pp = config.get("preprocess", {})

    if args.image_type in ("sbs", "both") and pp.get("sbs_samples_fp"):
        errs += process("sbs", config, args)

    if args.image_type in ("phenotype", "both") and pp.get("phenotype_samples_fp"):
        errs += process("phenotype", config, args)

    status = "DONE" if errs == 0 else "FAILED"
    print(f"\n{'#' * 60}")
    print(f"  {status}: {time.time() - t0:.1f}s total, {errs} errors")
    print(f"{'#' * 60}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
