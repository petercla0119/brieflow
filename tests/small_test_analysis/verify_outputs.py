#!/usr/bin/env python3
"""Assert a small-test run actually wrote artifacts. Exit 0 is not a test.

A driver in this repo is known to print COMPLETE and exit 0 while writing zero
parquets (BrokenProcessPool swallowed by the pool), so CI checks the outputs
rather than the return code: every required glob must match at least one path,
and every match must hold real data -- non-zero bytes, >0 parquet rows, and for
a .zarr store at least one chunk file rather than metadata alone (a hollow tile
left by an OOM keeps zarr.json and no chunks).

Usage:
    python verify_outputs.py tiff | zarr | direct [--root DIR]

ponytail: a flat glob list, not a per-rule manifest. It checks the well-level
outputs each stage must produce; per-tile outputs are deliberately not listed
because the small test ships empty FOVs that legitimately yield 0 rows. Grow
the list if a stage regresses without tripping it.
"""

import argparse
import sys
from pathlib import Path

# Well-level outputs every leg must produce, whatever the layout. The `**/`
# makes one pattern cover both namings: flat TIFF (`P-1_W-A1__cells.parquet`)
# and nested zarr (`1/A/1/cells.parquet`).
COMMON = [
    "preprocess/metadata/sbs/**/*combined_metadata.parquet",
    "preprocess/metadata/phenotype/**/*combined_metadata.parquet",
    "sbs/parquets/**/*sbs_info.parquet",
    "sbs/parquets/**/*cells.parquet",
    "phenotype/parquets/**/*phenotype_info.parquet",
    "phenotype/parquets/**/*phenotype_cp.parquet",
    "merge/parquets/**/*merge_final.parquet",
]

# leg -> (default output root, extra format-specific patterns)
LEGS = {
    "tiff": ("brieflow_output", ["preprocess/images/**/*.tiff"]),
    "zarr": ("brieflow_output_zarr", ["preprocess/**/*.zarr"]),
    # The direct runners read config/config.yml, i.e. the TIFF config and root.
    "direct": ("brieflow_output", ["preprocess/images/**/*.tiff"]),
}

ZARR_METADATA_FILES = {"zarr.json", ".zarray", ".zgroup", ".zattrs"}


def empty_reason(path):
    """Return why `path` holds no real data, or None if it does."""
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and child.name not in ZARR_METADATA_FILES:
                return None
        return f"{path}: zarr store has metadata but no chunk data (hollow)"
    if path.stat().st_size == 0:
        return f"{path}: zero bytes"
    if path.suffix == ".parquet":
        from pyarrow.parquet import ParquetFile

        if ParquetFile(path).metadata.num_rows == 0:
            return f"{path}: parquet has 0 rows"
    return None


def check(root, patterns):
    """Return a list of failure strings; empty means every pattern passed."""
    root = Path(root)
    failures = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            failures.append(f"no output matches {root}/{pattern}")
            continue
        reasons = [r for r in (empty_reason(m) for m in matches) if r]
        failures.extend(reasons)
        if not reasons:
            print(f"ok   {pattern} ({len(matches)} file(s))")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("leg", choices=sorted(LEGS))
    p.add_argument("--root", default=None, help="override the output root")
    args = p.parse_args()

    default_root, extra = LEGS[args.leg]
    root = Path(args.root or default_root)
    print(f"verifying {args.leg} outputs under {root}")
    failures = check(root, COMMON + extra)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    if failures:
        sys.exit(f"{len(failures)} output check(s) failed for leg '{args.leg}'")
    print(f"all output checks passed for leg '{args.leg}'")


if __name__ == "__main__":
    main()
