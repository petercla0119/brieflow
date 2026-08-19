"""Direct merge runner — bypasses Snakemake DAG build.

Runs the FAST merge chain in-process for one or more wells, mirroring the
snakemake merge scripts exactly (fast_alignment -> fast_merge -> format_merge
-> deduplicate_merge -> final_merge). Written to match the run_*_direct.py
pattern used for preprocess/sbs/phenotype (submit as a single sbatch job so we
don't flood the scheduler).

Why direct instead of `snakemake --until all_merge`:
  - the stitch approach cannot build the DAG under zarr (WildcardError 'row')
  - snakemake's SLURM executor spawned too many jobs and overloaded the HPC
  - the fast chain is a short per-well pipeline, trivially run in one process

Usage (from analysis/):
    python run_merge_direct.py --config config/config.yml --workers 30 --step all
    python run_merge_direct.py --config config/config.yml --plate-filter 1 \
        --well-filter A1 --step all
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# --- import brieflow workflow lib ---
_ANALYSIS_DIR = Path(__file__).resolve().parent
_WORKFLOW = _ANALYSIS_DIR.parent.parent / "workflow"
sys.path.insert(0, str(_WORKFLOW / "lib"))
sys.path.insert(0, str(_WORKFLOW))

from lib.shared.file_utils import validate_dtypes  # noqa: E402
from lib.shared.parquet_io import read_parquet, write_parquet  # noqa: E402
from lib.merge.hash import (  # noqa: E402
    hash_cell_locations,
    multistep_alignment,
    extract_rotation,
    initial_alignment,
)
from lib.merge.merge_utils import align_metadata, find_closest_tiles  # noqa: E402
from lib.merge.fast_merge import merge_triangle_hash  # noqa: E402
from lib.merge.format_merge import (  # noqa: E402
    fov_distance,
    identify_single_gene_mappings,
    calculate_channel_mins,
    attach_global_pixel_coords,
)
from lib.merge.deduplicate_merge import (  # noqa: E402
    deduplicate_cells,
    check_matching_rates,
    analyze_distance_distribution,
)

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


def _row_col(well):
    return well[0], well[1:]


class Paths:
    """Resolve the exact on-disk paths the merge chain reads/writes.

    Auto-detects zarr-style subdirs ({plate}/{row}/{col}/name.parquet) vs
    flat naming (P-{plate}_W-{well}__name.parquet) from the output root.
    """

    def __init__(self, out_root, plate, well):
        self.root = Path(out_root)
        r, c = _row_col(well)
        zarr_probe = (
            self.root / "preprocess/metadata/phenotype"
            / str(plate) / r / c / "combined_metadata.parquet"
        )
        if zarr_probe.exists():
            stem = f"{plate}/{r}/{c}"
            self._parq = lambda subdir, name: self.root / subdir / stem / f"{name}.parquet"
            self._tsv  = lambda subdir, name: self.root / subdir / stem / f"{name}.tsv"
        else:
            pfx = f"P-{plate}_W-{well}__"
            self._parq = lambda subdir, name: self.root / subdir / f"{pfx}{name}.parquet"
            self._tsv  = lambda subdir, name: self.root / subdir / f"{pfx}{name}.tsv"

    # inputs
    @property
    def ph_metadata(self):
        return self._parq("preprocess/metadata/phenotype", "combined_metadata")

    @property
    def sbs_metadata(self):
        return self._parq("preprocess/metadata/sbs", "combined_metadata")

    @property
    def ph_info(self):
        return self._parq("phenotype/parquets", "phenotype_info")

    @property
    def sbs_info(self):
        return self._parq("sbs/parquets", "sbs_info")

    @property
    def sbs_cells(self):
        return self._parq("sbs/parquets", "cells")

    @property
    def ph_cp_min(self):
        return self._parq("phenotype/parquets", "phenotype_cp_min")

    @property
    def ph_cp_full(self):
        return self._parq("phenotype/parquets", "phenotype_cp")

    # outputs
    def _merge_parq(self, name):
        return self._parq("merge/parquets", name)

    def _merge_eval(self, name):
        return self._tsv("merge/eval", name)

    fast_alignment = property(lambda s: s._merge_parq("fast_alignment"))
    fast_merge = property(lambda s: s._merge_parq("fast_merge"))
    merge_formatted = property(lambda s: s._merge_parq("merge_formatted"))
    merge_deduplicated = property(lambda s: s._merge_parq("merge_deduplicated"))
    merge_final = property(lambda s: s._merge_parq("merge_final"))
    dedup_stats = property(lambda s: s._merge_eval("deduplication_stats"))
    final_sbs_rates = property(lambda s: s._merge_eval("final_sbs_matching_rates"))
    final_ph_rates = property(lambda s: s._merge_eval("final_phenotype_matching_rates"))




def _ensure_dirs(paths):
    for p in [paths.fast_alignment, paths.dedup_stats]:
        p.parent.mkdir(parents=True, exist_ok=True)


def step_fast_alignment(cfg, paths, plate, well, workers, force):
    if paths.fast_alignment.exists() and not force:
        log(f"skip fast_alignment (exists): {paths.fast_alignment}")
        return
    log("fast_alignment: loading metadata + cell info")
    ph_meta = validate_dtypes(read_parquet(paths.ph_metadata))
    sbs_meta = validate_dtypes(read_parquet(paths.sbs_metadata))

    align_params = dict(
        flip_x=cfg.get("alignment_flip_x"),
        flip_y=cfg.get("alignment_flip_y"),
        rotate_90=cfg.get("alignment_rotate_90"),
    )
    if any(align_params.values()):
        log("fast_alignment: applying coordinate alignment")
        ph_meta, sbs_meta, _ = align_metadata(
            ph_meta, sbs_meta, x_col="x_pos", y_col="y_pos", **align_params
        )

    # SBS filters
    sbs_filters = {}
    if cfg.get("sbs_metadata_cycle") is not None:
        sbs_filters["cycle"] = cfg["sbs_metadata_cycle"]
    if cfg.get("sbs_metadata_channel") is not None:
        sbs_filters["channel"] = cfg["sbs_metadata_channel"]
    for k, v in sbs_filters.items():
        sbs_meta = sbs_meta[sbs_meta[k] == v]
    ph_filters = {}
    if cfg.get("ph_metadata_channel") is not None:
        ph_filters["channel"] = cfg["ph_metadata_channel"]
    for k, v in ph_filters.items():
        ph_meta = ph_meta[ph_meta[k] == v]
    if not sbs_filters:
        sbs_meta = sbs_meta.drop_duplicates(subset=["plate", "well", "tile"])
    if not ph_filters:
        ph_meta = ph_meta.drop_duplicates(subset=["plate", "well", "tile"])

    ph_info = validate_dtypes(read_parquet(paths.ph_info))
    sbs_info = validate_dtypes(read_parquet(paths.sbs_info))

    ph_xy = ph_meta.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index("tile")[["x", "y"]]
    sbs_xy = sbs_meta.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index("tile")[["x", "y"]]

    log("fast_alignment: hashing cell locations")
    ph_hash = validate_dtypes(hash_cell_locations(ph_info))
    sbs_hash = validate_dtypes(hash_cell_locations(sbs_info).rename(columns={"tile": "site"}))

    initial_sbs_tiles = cfg.get("initial_sbs_tiles")
    initial_sites_param = cfg.get("initial_sites")
    if (initial_sbs_tiles is None) == (initial_sites_param is None):
        raise ValueError("Exactly one of 'initial_sbs_tiles' or 'initial_sites' must be set in merge config")
    d0, d1 = cfg["det_range"]
    score_thresh = cfg["score"]

    if initial_sbs_tiles is not None:
        candidate_pairs = []
        for sbs_tile in initial_sbs_tiles:
            closest = find_closest_tiles(sbs_meta, ph_meta, sbs_tile, verbose=False)
            candidate_pairs.append([int(closest.iloc[0]["tile"]), sbs_tile])
        log(f"fast_alignment: {len(candidate_pairs)} candidate pairs from {len(initial_sbs_tiles)} SBS tiles")
    else:
        candidate_pairs = initial_sites_param
        log(f"fast_alignment: {len(candidate_pairs)} user-specified initial sites")

    initial_alignment_df = initial_alignment(ph_hash, sbs_hash, initial_sites=candidate_pairs)
    valid = initial_alignment_df.query("@d0 <= determinant <= @d1 & score > @score_thresh")
    if len(candidate_pairs) > 5 and len(valid) < 5:
        raise ValueError(
            f"Only {len(valid)} initial sites passed thresholds (need >=5). "
            f"Check det_range={cfg['det_range']} and score={score_thresh}."
        )
    initial_sites = valid[["tile", "site"]].astype(int).values.tolist()
    log(f"fast_alignment: {len(initial_sites)} initial sites passed thresholds; running multistep_alignment (n_jobs={workers})")

    well_alignment = multistep_alignment(
        ph_hash, sbs_hash, ph_xy, sbs_xy,
        det_range=cfg["det_range"], score=cfg["score"],
        initial_sites=initial_sites, n_jobs=workers,
    )
    well_alignment.reset_index(drop=True, inplace=True)
    well_alignment["rotation_1"] = well_alignment["rotation"].apply(lambda r: extract_rotation(r, 1))
    well_alignment["rotation_2"] = well_alignment["rotation"].apply(lambda r: extract_rotation(r, 2))
    well_alignment.drop(columns=["rotation"], inplace=True)
    well_alignment["plate"] = plate
    well_alignment["well"] = well
    write_parquet(well_alignment, paths.fast_alignment)
    log(f"fast_alignment: wrote {len(well_alignment)} alignments -> {paths.fast_alignment}")


def step_fast_merge(cfg, paths, force):
    if paths.fast_merge.exists() and not force:
        log(f"skip fast_merge (exists): {paths.fast_merge}")
        return
    import numpy as np

    log("fast_merge: merging cells per tile pair")
    ph_info = validate_dtypes(read_parquet(paths.ph_info))
    sbs_info = validate_dtypes(read_parquet(paths.sbs_info))
    fa = read_parquet(paths.fast_alignment)
    fa["rotation"] = [np.array([r1, r2]) for r1, r2 in zip(fa["rotation_1"], fa["rotation_2"])]
    fa.drop(columns=["rotation_1", "rotation_2"], inplace=True)

    d0, d1 = cfg["det_range"]
    fa_f = fa[(fa["determinant"] >= d0) & (fa["determinant"] <= d1) & (fa["score"] > cfg["score"])]
    log(f"fast_merge: {len(fa_f)}/{len(fa)} alignments pass filters")

    merge_data = []
    for _, row in fa_f.iterrows():
        ph_f = ph_info[ph_info["tile"] == row["tile"]]
        sbs_f = sbs_info[sbs_info["tile"] == row["site"]]
        merge_data.append(merge_triangle_hash(ph_f, sbs_f, row, threshold=cfg["threshold"]))
    merge_data = pd.concat(merge_data, ignore_index=True)
    write_parquet(merge_data, paths.fast_merge)
    log(f"fast_merge: wrote {len(merge_data)} merged cells -> {paths.fast_merge}")


def step_format_merge(cfg, paths, force):
    if paths.merge_formatted.exists() and not force:
        log(f"skip format_merge (exists): {paths.merge_formatted}")
        return
    log("format_merge: formatting + attaching SBS genotype + channel mins")
    merge_data = validate_dtypes(read_parquet(paths.fast_merge))
    sbs_cells = validate_dtypes(read_parquet(paths.sbs_cells))
    ph_min = validate_dtypes(read_parquet(paths.ph_cp_min))

    ph_dims = tuple(cfg.get("phenotype_dimensions") or (2960, 2960))
    sbs_dims = tuple(cfg.get("sbs_dimensions") or (1480, 1480))

    mf = merge_data.pipe(fov_distance, i="i_0", j="j_0", dimensions=ph_dims, suffix="_0")
    mf = mf.pipe(fov_distance, i="i_1", j="j_1", dimensions=sbs_dims, suffix="_1")

    sbs_cells["mapped_single_gene"] = sbs_cells.apply(identify_single_gene_mappings, axis=1)
    sbs_cols = ["plate", "well", "tile", "cell", "mapped_single_gene"]
    prefixes = ("cell_barcode_", "gene_symbol_", "gene_id_", "no_recomb_", "Q_min_",
                "Q_recomb_", "cell_barcode_peak_", "cell_barcode_count_")
    sbs_cols += [c for c in sbs_cells.columns if c.startswith(prefixes)]
    sbs_cols = [c for c in sbs_cols if c in sbs_cells.columns]
    mf = mf.merge(
        sbs_cells[sbs_cols].rename({"tile": "site", "cell": "cell_1"}, axis=1),
        how="left", on=["plate", "well", "site", "cell_1"],
    )

    ph_min = calculate_channel_mins(ph_min)
    mf = mf.merge(
        ph_min[["tile", "label", "channels_min"]].rename(columns={"label": "cell_0"}),
        how="left", on=["tile", "cell_0"],
    )

    if cfg.get("approach", "fast") == "fast":
        ph_meta = pd.read_parquet(paths.ph_metadata)
        sbs_meta = pd.read_parquet(paths.sbs_metadata)
        mf = attach_global_pixel_coords(mf, ph_meta, ph_dims, suffix="0")
        mf = attach_global_pixel_coords(mf, sbs_meta, sbs_dims, suffix="1")

    write_parquet(mf, paths.merge_formatted)
    log(f"format_merge: wrote {len(mf)} rows -> {paths.merge_formatted}")


def step_deduplicate_merge(cfg, paths, force):
    if paths.merge_deduplicated.exists() and not force:
        log(f"skip deduplicate_merge (exists): {paths.merge_deduplicated}")
        return
    log("deduplicate_merge: two-step dedup + matching rates")
    mf = validate_dtypes(read_parquet(paths.merge_formatted))
    sbs_cells = validate_dtypes(read_parquet(paths.sbs_cells))
    ph_min = validate_dtypes(read_parquet(paths.ph_cp_min))
    approach = cfg.get("approach", "fast")

    dedup, stats = deduplicate_cells(
        mf, mapped_single_gene=False, return_stats=True, approach=approach,
        sbs_dedup_prior=cfg.get("sbs_dedup_prior"), pheno_dedup_prior=cfg.get("pheno_dedup_prior"),
    )
    da = analyze_distance_distribution(dedup)
    if da:
        s = da["distance_stats"]
        log(f"deduplicate_merge: mean dist {s['mean']:.2f}px, median {s['median']:.2f}px, n={len(dedup)}")

    stats.to_csv(paths.dedup_stats, sep="\t", index=False)
    write_parquet(dedup, paths.merge_deduplicated)
    check_matching_rates(sbs_cells, dedup, modality="sbs", return_stats=True).to_csv(
        paths.final_sbs_rates, sep="\t", index=False)
    check_matching_rates(ph_min, dedup, modality="phenotype", return_stats=True).to_csv(
        paths.final_ph_rates, sep="\t", index=False)
    log(f"deduplicate_merge: wrote {len(dedup)} deduplicated cells -> {paths.merge_deduplicated}")


def step_final_merge(cfg, paths, force):
    from lib.merge.final_merge import final_merge

    if paths.merge_final.exists() and not force:
        log(f"skip final_merge (exists): {paths.merge_final}")
        return

    log(f"final_merge: polars streaming left join of CP -> {paths.merge_final}")
    final_merge(
        deduplicated_path=paths.merge_deduplicated,
        phenotype_cp_path=paths.ph_cp_full,
        output_path=paths.merge_final,
        approach=cfg.get("approach", "fast"),
        exclude_markers=cfg.get("exclude_markers"),
    )
    log(f"final_merge: done -> {paths.merge_final}")


def run_well(cfg, out_root, plate, well, workers, force):
    log(f"===== MERGE well plate={plate} well={well} =====")
    paths = Paths(out_root, plate, well)
    _ensure_dirs(paths)
    step_fast_alignment(cfg, paths, plate, well, workers, force)
    step_fast_merge(cfg, paths, force)
    step_format_merge(cfg, paths, force)
    step_deduplicate_merge(cfg, paths, force)
    step_final_merge(cfg, paths, force)
    log(f"===== DONE well plate={plate} well={well} -> {paths.merge_final} =====")


def main():
    import yaml

    ap = argparse.ArgumentParser(description="Direct merge runner (fast chain, bypasses Snakemake)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8, help="n_jobs for multistep_alignment")
    ap.add_argument("--plate-filter", type=int, default=None)
    ap.add_argument("--well-filter", default=None)
    ap.add_argument("--step", choices=["all"], default="all")
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    cfg = config.get("merge", {})
    if cfg.get("approach", "fast") != "fast":
        raise SystemExit(f"run_merge_direct only implements the fast chain; config approach={cfg.get('approach')}")
    out_root = Path(config["all"]["root_fp"])

    combo_fp = cfg.get("merge_combo_fp", "config/merge_combo.tsv")
    combos = pd.read_csv(combo_fp, sep="\t")
    if args.plate_filter is not None:
        combos = combos[combos["plate"] == args.plate_filter]
    if args.well_filter is not None:
        combos = combos[combos["well"].astype(str) == str(args.well_filter)]
    if combos.empty:
        raise SystemExit("No (plate, well) combos to process after filtering")

    log(f"Processing {len(combos)} well(s) from {combo_fp}")
    for _, row in combos.iterrows():
        run_well(cfg, out_root, row["plate"], str(row["well"]), args.workers, args.force)
    log("ALL WELLS COMPLETE")


if __name__ == "__main__":
    main()
