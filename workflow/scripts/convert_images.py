#!/usr/bin/env python
"""Bidirectional TIFF <-> Zarr converter for already-generated brieflow output.

Operates on finished brieflow_output (tiff) / brieflow_output_zarr (zarr) trees,
outside the Snakemake DAG. Images/labels only -- no tsv/parquet/eval.

Phase 0-1: file discovery + bidirectional path mapping + --dry-run (with an
existence check against a reference tree). Pixel conversion is wired via
save_image(read_image()); the label write-order caveat (Phase 2) is untested.
"""

import argparse
import re
import sys
from pathlib import Path

# workflow/ must be importable so `lib.shared.*` resolves -- same rooting the
# snakemake scripts rely on.
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from lib.shared.file_utils import get_image_output_path, get_filename, parse_filename

WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def split_well(well):
    m = WELL_RE.match(str(well))
    if not m:
        raise ValueError(f"Cannot split well {well!r} into row/col")
    return m.group(1), m.group(2)


def _store_plate(store_path):
    # image_1.zarr -> "1"; illumination_corrected_1.zarr -> "1"
    return Path(store_path).stem.rsplit("_", 1)[-1]


# --- Category registry -------------------------------------------------------
# Builder categories map through get_image_output_path. IC fields are special
# (no path builder exists) and handled separately below.
#
# Fields: module_root, image_subdir, subdirectory, info_type, has_cycle,
#         is_label, tiff_glob, zarr_store_glob, zarr_group_depth
BUILDER = {
    "sbs_pp": dict(
        module_root="preprocess", image_subdir="sbs", subdirectory=None,
        info_type="image", has_cycle=True, is_label=False,
        tiff_glob="preprocess/images/sbs/*__image.tiff",
        zarr_store_glob="preprocess/sbs/image_*.zarr", depth=4),
    "pheno_pp": dict(
        module_root="preprocess", image_subdir="phenotype", subdirectory=None,
        info_type="image", has_cycle=False, is_label=False,
        tiff_glob="preprocess/images/phenotype/*__image.tiff",
        zarr_store_glob="preprocess/phenotype/image_*.zarr", depth=3),
    "aligned": dict(
        module_root="phenotype", image_subdir=None, subdirectory=None,
        info_type="aligned", has_cycle=False, is_label=False,
        tiff_glob="phenotype/images/*__aligned.tiff",
        zarr_store_glob="phenotype/aligned_*.zarr", depth=3),
    "illum": dict(
        module_root="phenotype", image_subdir=None, subdirectory=None,
        info_type="illumination_corrected", has_cycle=False, is_label=False,
        tiff_glob="phenotype/images/*__illumination_corrected.tiff",
        zarr_store_glob="phenotype/illumination_corrected_*.zarr", depth=3),
    "nuclei": dict(
        module_root="phenotype", image_subdir=None, subdirectory="labels",
        info_type="nuclei", has_cycle=False, is_label=True,
        tiff_glob="phenotype/images/*__nuclei.tiff",
        zarr_store_glob="phenotype/aligned_*.zarr", depth=5),
    "cells": dict(
        module_root="phenotype", image_subdir=None, subdirectory="labels",
        info_type="cells", has_cycle=False, is_label=True,
        tiff_glob="phenotype/images/*__cells.tiff",
        zarr_store_glob="phenotype/aligned_*.zarr", depth=5),
    "cytoplasm": dict(
        module_root="phenotype", image_subdir=None, subdirectory="labels",
        info_type="identified_cytoplasms", has_cycle=False, is_label=True,
        tiff_glob="phenotype/images/*__identified_cytoplasms.tiff",
        zarr_store_glob="phenotype/aligned_*.zarr", depth=5),
}

IC = {
    "ic_sbs": dict(root="preprocess/ic_fields/sbs", has_cycle=True,
                   tiff_glob="preprocess/ic_fields/sbs/*__ic_field.tiff",
                   zarr_glob="preprocess/ic_fields/sbs/*/*/*/*/ic_field.zarr"),
    "ic_pheno": dict(root="preprocess/ic_fields/phenotype", has_cycle=False,
                     tiff_glob="preprocess/ic_fields/phenotype/*__ic_field.tiff",
                     zarr_glob="preprocess/ic_fields/phenotype/*/*/*/ic_field.zarr"),
}

ALL_CATEGORIES = list(BUILDER) + list(IC)


# --- Path mapping ------------------------------------------------------------

def tiff_to_zarr_rel(tiff_path, spec):
    """Relative zarr dst path for a builder-category tiff source."""
    meta, _info, _ = parse_filename(Path(tiff_path).name)
    row, col = split_well(meta["well"])
    loc = {"plate": meta["plate"], "row": row, "col": col, "tile": meta["tile"]}
    if spec["has_cycle"]:
        loc["cycle"] = meta["cycle"]
    rel = get_image_output_path(
        loc, spec["info_type"], img_fmt="zarr",
        subdirectory=spec["subdirectory"], image_subdir=spec["image_subdir"])
    return str(Path(spec["module_root"]) / rel)


def zarr_to_tiff_rel(group_path, spec):
    """Relative tiff dst path for a builder-category zarr group source."""
    parts = Path(group_path).parts
    if spec["is_label"]:                       # .../row/col/tile/labels/name
        row, col, tile = parts[-5], parts[-4], parts[-3]
    elif spec["has_cycle"]:                    # .../row/col/tile/cycle
        row, col, tile, cycle = parts[-4], parts[-3], parts[-2], parts[-1]
    else:                                      # .../row/col/tile
        row, col, tile = parts[-3], parts[-2], parts[-1]
    store = next(p for p in Path(group_path).parents if p.suffix == ".zarr")
    loc = {"plate": _store_plate(store), "well": f"{row}{col}", "tile": int(tile)}
    if spec["has_cycle"]:
        loc["cycle"] = int(cycle)
    rel = get_image_output_path(
        loc, spec["info_type"], img_fmt="tiff", image_subdir=spec["image_subdir"])
    return str(Path(spec["module_root"]) / rel)


def ic_tiff_to_zarr_rel(tiff_path, spec):
    meta, _info, _ = parse_filename(Path(tiff_path).name)
    row, col = split_well(meta["well"])
    parts = [spec["root"], str(meta["plate"]), row, col]
    if spec["has_cycle"]:
        parts.append(str(meta["cycle"]))
    parts.append("ic_field.zarr")
    return str(Path(*parts))


def ic_zarr_to_tiff_rel(zarr_path, spec):
    parts = Path(zarr_path).parts            # .../plate/row/col[/cycle]/ic_field.zarr
    if spec["has_cycle"]:
        plate, row, col, cycle = parts[-5], parts[-4], parts[-3], parts[-2]
    else:
        plate, row, col = parts[-4], parts[-3], parts[-2]
    loc = {"plate": plate, "well": f"{row}{col}"}
    if spec["has_cycle"]:
        loc["cycle"] = int(cycle)
    return str(Path(spec["root"]) / get_filename(loc, "ic_field", "tiff"))


# --- Discovery + planning ----------------------------------------------------

def _image_group_dirs(store, depth):
    """Yield image-group dirs at `depth` levels under a zarr store that hold a
    level-0 array ("0" child). Scoped glob on the store only -- no broad find."""
    for grp in store.glob("/".join(["*"] * depth)):
        if grp.is_dir() and (grp / "0").exists():
            yield grp


def plan(direction, src_root, categories):
    """Return list of (src, dst_rel, is_label) mappings."""
    src_root = Path(src_root)
    out = []
    for cat in categories:
        if cat in BUILDER:
            spec = BUILDER[cat]
            if direction == "tiff2zarr":
                for f in sorted(src_root.glob(spec["tiff_glob"])):
                    out.append((f, tiff_to_zarr_rel(f, spec), spec["is_label"]))
            else:
                for store in sorted(src_root.glob(spec["zarr_store_glob"])):
                    if spec["is_label"]:
                        groups = store.glob("*/*/*/labels/" + spec["info_type"])
                    else:
                        groups = _image_group_dirs(store, spec["depth"])
                    for g in sorted(groups):
                        out.append((g, zarr_to_tiff_rel(g, spec), spec["is_label"]))
        else:
            spec = IC[cat]
            if direction == "tiff2zarr":
                for f in sorted(src_root.glob(spec["tiff_glob"])):
                    out.append((f, ic_tiff_to_zarr_rel(f, spec), False))
            else:
                for z in sorted(src_root.glob(spec["zarr_glob"])):
                    out.append((z, ic_zarr_to_tiff_rel(z, spec), False))
    return out


def convert_one(src, dst, is_label):
    from lib.shared.image_io import read_image, save_image
    dst = Path(dst)
    # save_image writes the store at dst.parent when dst ends in zarr.json,
    # otherwise at dst itself; ensure the containing directory exists.
    store_or_file = dst.parent if dst.name == "zarr.json" else dst
    store_or_file.parent.mkdir(parents=True, exist_ok=True)
    save_image(read_image(str(src)), str(dst), is_label=is_label)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", required=True, choices=["tiff2zarr", "zarr2tiff"])
    ap.add_argument("--src", required=True, help="source brieflow_output[_zarr] root")
    ap.add_argument("--dst", required=True, help="destination root (created)")
    ap.add_argument("--categories", default="all",
                    help="comma-separated: " + ",".join(ALL_CATEGORIES) + " (default all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned src->dst; check dst existence under --dst; write nothing")
    args = ap.parse_args()

    cats = ALL_CATEGORIES if args.categories == "all" else args.categories.split(",")
    bad = [c for c in cats if c not in ALL_CATEGORIES]
    if bad:
        ap.error(f"unknown categories: {bad}; valid: {ALL_CATEGORIES}")

    mappings = plan(args.direction, args.src, cats)
    dst_root = Path(args.dst)
    n_ok = n_missing = 0
    for src, rel, is_label in mappings:
        dst = dst_root / rel
        if args.dry_run:
            tag = "OK" if dst.exists() else "MISSING"
            n_ok += tag == "OK"
            n_missing += tag == "MISSING"
            print(f"[{tag}] {Path(src).relative_to(Path(args.src))}  ->  {rel}")
        else:
            convert_one(src, dst, is_label)
            print(f"[done] {rel}")
    print(f"\n{len(mappings)} mappings"
          + (f"  |  {n_ok} OK, {n_missing} MISSING (checked vs {dst_root})"
             if args.dry_run else "  converted"))
    if args.dry_run and n_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
