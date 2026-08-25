#!/usr/bin/env python3
"""Retroactively convert TIFF brieflow pipeline output to OME-Zarr.

Globs known TIFF subdirectories, dispatches conversion jobs in parallel
with ProcessPoolExecutor, and optionally finalizes HCS plate metadata.

Usage:
    python tiff_to_zarr.py \
        --input /path/to/brieflow_output \
        --output /path/to/brieflow_output_zarr \
        [--brieflow-workflow /path/to/brieflow/workflow] \
        [--modules preprocess,sbs,phenotype] \
        [--tile-limit N] \
        [--dry-run] \
        [--workers N] \
        [--skip-finalize] \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# sys.path — auto-detect workflow/lib/shared before lib imports
# ---------------------------------------------------------------------------


def _pre_parse_workflow_arg() -> Optional[str]:
    """Extract --brieflow-workflow value from sys.argv before argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--brieflow-workflow" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--brieflow-workflow="):
            return arg.split("=", 1)[1]
    return None


def _add_workflow_to_path(override: Optional[str] = None) -> None:
    """Insert brieflow workflow dir at front of sys.path."""
    if override:
        p = str(Path(override).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
        return
    # Walk up from script location to find workflow/lib/shared
    cand = Path(__file__).resolve().parent
    for _ in range(8):
        wf = cand / "workflow"
        if (wf / "lib" / "shared").is_dir():
            if str(wf) not in sys.path:
                sys.path.insert(0, str(wf))
            return
        cand = cand.parent


# Run at module load — workers inherit via fork on Linux
_add_workflow_to_path(_pre_parse_workflow_arg())

from lib.shared.file_utils import get_hcs_nested_path, get_nested_path, parse_filename  # noqa: E402
from lib.shared.image_io import save_image  # noqa: E402
from tifffile import imread as tiff_imread  # noqa: E402

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ImageSpec:
    """Describes one image type: output location and zarr write parameters."""

    module_dir: str           # relative dir under output_root (e.g. "sbs")
    zarr_info_type: str       # zarr-side name (e.g. "aligned", "nuclei")
    is_label: bool            # True for segmentation label arrays
    label_parent: Optional[str]  # "aligned" for labels, None otherwise
    channel_names: Optional[list]  # set for preprocess images only
    has_cycle: bool = False   # True for sbs preprocess images


@dataclass
class ConvertJob:
    """One TIFF → zarr conversion task."""

    tiff_path: Path
    out_path: Path
    spec: ImageSpec
    module_key: str   # e.g. "sbs", "preprocess_sbs"
    plate: str
    well: str
    tile: Optional[int]   # None for IC fields (no tile)
    cycle: Optional[int]  # None for non-cycle images


@dataclass
class Result:
    """Outcome of one convert_one() call."""

    status: str   # "ok" | "skipped" | "empty" | "error"
    job: ConvertJob
    msg: str


# ---------------------------------------------------------------------------
# Image type registry
# ---------------------------------------------------------------------------

REGISTRY: dict[tuple[str, str], ImageSpec] = {
    # preprocess images
    ("preprocess_sbs", "image"): ImageSpec(
        "preprocess/sbs", "image", False, None,
        ["DAPI", "G", "T", "A", "C"], has_cycle=True,
    ),
    ("preprocess_pheno", "image"): ImageSpec(
        "preprocess/phenotype", "image", False, None,
        ["MAP2", "STMN2_CE", "STMN2_FL", "TDP43", "DAPI", "TOMM20", "FUS", "DAPI_round2"],
    ),
    # sbs processing images
    ("sbs", "aligned"):               ImageSpec("sbs", "aligned",               False, None, None),
    ("sbs", "log_filtered"):          ImageSpec("sbs", "log_filtered",          False, None, None),
    ("sbs", "max_filtered"):          ImageSpec("sbs", "max_filtered",          False, None, None),
    ("sbs", "illumination_corrected"): ImageSpec("sbs", "illumination_corrected", False, None, None),
    ("sbs", "standard_deviation"):    ImageSpec("sbs", "standard_deviation",   False, None, None),
    ("sbs", "peaks"):                 ImageSpec("sbs", "peaks",                 False, None, None),
    # sbs labels (nested inside aligned plate zarr)
    ("sbs", "nuclei"):                ImageSpec("sbs", "nuclei",  True, "aligned", None),
    ("sbs", "cells"):                 ImageSpec("sbs", "cells",   True, "aligned", None),
    # phenotype images
    ("phenotype", "aligned"):              ImageSpec("phenotype", "aligned",               False, None, None),
    ("phenotype", "illumination_corrected"): ImageSpec("phenotype", "illumination_corrected", False, None, None),
    # phenotype labels
    ("phenotype", "nuclei"):    ImageSpec("phenotype", "nuclei",               True, "aligned", None),
    ("phenotype", "cells"):     ImageSpec("phenotype", "cells",                True, "aligned", None),
    ("phenotype", "cytoplasm"): ImageSpec("phenotype", "identified_cytoplasms", True, "aligned", None),
    # IC fields (flat nested path, no HCS plate zarr)
    ("preprocess_ic_sbs",   "ic_field"): ImageSpec("preprocess/ic_fields/sbs",       "ic_field", False, None, None),
    ("preprocess_ic_pheno", "ic_field"): ImageSpec("preprocess/ic_fields/phenotype", "ic_field", False, None, None),
}

# Glob patterns relative to input_root, keyed by module_key
GLOB_MAP: dict[str, str] = {
    "preprocess_sbs":    "preprocess/images/sbs/*.tiff",
    "preprocess_pheno":  "preprocess/images/phenotype/*.tiff",
    "sbs":               "sbs/images/*.tiff",
    "phenotype":         "phenotype/images/*.tiff",
    "preprocess_ic_sbs": "preprocess/ic_fields/sbs/*.tiff",
    "preprocess_ic_pheno": "preprocess/ic_fields/phenotype/*.tiff",
}

# --modules argument → list of module_keys to activate
MODULES_MAP: dict[str, list[str]] = {
    "preprocess": ["preprocess_sbs", "preprocess_pheno", "preprocess_ic_sbs", "preprocess_ic_pheno"],
    "sbs":        ["sbs"],
    "phenotype":  ["phenotype"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_well(well: str) -> tuple[str, str]:
    """Parse "A1" → ("A", "1")."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(well))
    if not m:
        raise ValueError(f"Cannot parse well: {well!r}")
    return m.group(1), m.group(2)


def zarr_field_exists(out_path: Path) -> bool:
    """Return True if the output zarr store already has data."""
    if out_path.name == "zarr.json":
        return out_path.exists()
    return out_path.is_dir() and (out_path / "zarr.json").exists()


def _build_out_path(
    output_root: Path,
    spec: ImageSpec,
    plate: str,
    well: str,
    tile: Optional[int],
    cycle: Optional[int],
) -> Path:
    """Compute the output zarr path for one image."""
    row, col = _split_well(well)

    if spec.zarr_info_type == "ic_field":
        # IC fields: flat nested path (no HCS plate zarr)
        loc_ic: dict = {"plate": plate, "row": row, "col": col}
        if cycle is not None:
            loc_ic["cycle"] = str(cycle)
        return output_root / spec.module_dir / get_nested_path(loc_ic, "ic_field", "zarr")

    loc: dict = {"plate": plate, "row": row, "col": col, "tile": str(tile)}
    if spec.has_cycle and cycle is not None:
        loc["cycle"] = str(cycle)

    if spec.is_label:
        rel = get_hcs_nested_path(loc, spec.zarr_info_type, subdirectory="labels")
    else:
        rel = get_hcs_nested_path(loc, spec.zarr_info_type)

    return output_root / spec.module_dir / rel


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_jobs(
    input_root: Path,
    active_module_keys: list[str],
    output_root: Path,
) -> list[ConvertJob]:
    """Glob TIFFs and build ConvertJob list (no I/O)."""
    jobs: list[ConvertJob] = []
    for mk in active_module_keys:
        pattern = GLOB_MAP[mk]
        for tiff_path in sorted(input_root.glob(pattern)):
            try:
                meta, info_type, _ = parse_filename(str(tiff_path))
            except Exception as e:
                print(f"  WARN: cannot parse {tiff_path.name}: {e}")
                continue

            spec = REGISTRY.get((mk, info_type))
            if spec is None:
                continue  # unrecognised type, skip silently

            plate = str(meta.get("plate", ""))
            well = str(meta.get("well", ""))
            tile = meta.get("tile")    # int or None
            cycle = meta.get("cycle")  # int or None

            if not plate or not well:
                continue

            try:
                out_path = _build_out_path(output_root, spec, plate, well, tile, cycle)
            except Exception as e:
                print(f"  WARN: cannot build path for {tiff_path.name}: {e}")
                continue

            jobs.append(ConvertJob(
                tiff_path=tiff_path,
                out_path=out_path,
                spec=spec,
                module_key=mk,
                plate=plate,
                well=well,
                tile=tile,
                cycle=cycle,
            ))
    return jobs


def apply_tile_limit(jobs: list[ConvertJob], limit: int) -> list[ConvertJob]:
    """Keep only the first `limit` tiles per (module_key, well); keep all IC fields."""
    if limit <= 0:
        return jobs

    # Collect sorted distinct tiles per (module_key, well)
    tile_sets: dict[tuple[str, str], list] = {}
    for job in jobs:
        if job.tile is None:
            continue
        key = (job.module_key, job.well)
        if key not in tile_sets:
            tile_sets[key] = []
        if job.tile not in tile_sets[key]:
            tile_sets[key].append(job.tile)

    allowed: dict[tuple[str, str], set] = {
        k: set(sorted(v)[:limit]) for k, v in tile_sets.items()
    }

    result = []
    for job in jobs:
        if job.tile is None:
            result.append(job)
        elif job.tile in allowed.get((job.module_key, job.well), set()):
            result.append(job)
    return result


# ---------------------------------------------------------------------------
# Conversion worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------


def convert_one(args_tuple: tuple) -> Result:
    """Convert one TIFF to zarr. Must be a top-level function for pickling."""
    job, overwrite = args_tuple

    if not overwrite and zarr_field_exists(job.out_path):
        return Result("skipped", job, "")

    # Guard against empty files before tifffile raises "not a TIFF file b''"
    try:
        if job.tiff_path.stat().st_size == 0:
            return Result("empty", job, "empty file (0 bytes)")
    except OSError as e:
        return Result("error", job, f"stat failed: {e}")

    try:
        img = tiff_imread(str(job.tiff_path))
        if img.size == 0:
            return Result("empty", job, "empty array")
        save_image(
            img,
            str(job.out_path),
            channel_names=job.spec.channel_names,
            is_label=job.spec.is_label,
        )
        return Result("ok", job, "")
    except Exception as e:
        return Result("error", job, str(e))


# ---------------------------------------------------------------------------
# Finalize pass
# ---------------------------------------------------------------------------


def finalize_plates(output_root: Path, input_root: Path) -> None:
    """Write HCS metadata and compute OMERO display windows for all plate zarrs."""
    from lib.shared.hcs import (
        compute_and_inject_omero_windows,
        patch_store_metadata_with_iohub,
        write_hcs_metadata,
    )

    # Collect top-level plate zarr dirs matching {type}_{plate}.zarr
    plate_zarr_dirs: list[Path] = []
    for d in sorted(output_root.rglob("*.zarr")):
        if not d.is_dir():
            continue
        if not re.match(r".+_\d+\.zarr$", d.name):
            continue
        # Skip if nested inside another .zarr store
        if any(part.endswith(".zarr") for part in d.parent.parts):
            continue
        plate_zarr_dirs.append(d)

    if not plate_zarr_dirs:
        print("  No plate zarr dirs found; skipping finalization.")
        return

    preprocess_root = input_root / "preprocess"
    renderable_plates: list[Path] = []

    for plate_path in plate_zarr_dirs:
        is_preprocess = "preprocess" in plate_path.parts
        print(f"  HCS metadata: {plate_path.relative_to(output_root)}")
        try:
            write_hcs_metadata(plate_path)
        except Exception as e:
            print(f"    WARN write_hcs_metadata: {e}")

        if not is_preprocess:
            try:
                patch_store_metadata_with_iohub(
                    plate_path,
                    preprocess_root,
                    config_channel_names=None,
                    modality_config=None,
                    channels_metadata=None,
                )
            except Exception as e:
                print(f"    WARN patch_store_metadata_with_iohub: {e}")
            renderable_plates.append(plate_path)

    if renderable_plates:
        try:
            compute_and_inject_omero_windows(renderable_plates)
        except Exception as e:
            print(f"  WARN compute_and_inject_omero_windows: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert brieflow TIFF pipeline output to OME-Zarr retroactively"
    )
    p.add_argument("--input", required=True,
                   help="brieflow_output root containing TIFF files")
    p.add_argument("--output", required=True,
                   help="Output zarr root (created if absent)")
    p.add_argument("--brieflow-workflow", default=None,
                   help="Path to brieflow/workflow (auto-detected if omitted)")
    p.add_argument("--modules", default="preprocess,sbs,phenotype",
                   help="Comma-separated modules to convert (default: all)")
    p.add_argument("--tile-limit", type=int, default=None,
                   help="Max tiles per (module, well) to convert")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover jobs and summarise; do not write files")
    p.add_argument("--workers", type=int, default=4,
                   help="ProcessPoolExecutor workers (default: 4)")
    p.add_argument("--skip-finalize", action="store_true",
                   help="Skip HCS metadata finalization pass")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing zarr stores")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure workflow path is set (handles explicit --brieflow-workflow in main)
    _add_workflow_to_path(args.brieflow_workflow)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.is_dir():
        print(f"ERROR: input does not exist: {input_root}")
        sys.exit(1)

    # Resolve active module keys from --modules
    active_keys: list[str] = []
    for mn in [m.strip() for m in args.modules.split(",")]:
        if mn not in MODULES_MAP:
            print(f"WARN: unknown module '{mn}', known: {list(MODULES_MAP)}")
            continue
        active_keys.extend(MODULES_MAP[mn])

    t0 = time.time()
    print(f"{'#' * 60}")
    print(f"  tiff_to_zarr | modules={args.modules} workers={args.workers}")
    print(f"  input:  {input_root}")
    print(f"  output: {output_root}")
    if args.tile_limit:
        print(f"  tile-limit: {args.tile_limit}")
    if args.dry_run:
        print(f"  DRY RUN — no files will be written")
    print(f"{'#' * 60}")

    # --- Discover ---
    print("\nDiscovering TIFFs...")
    jobs = discover_jobs(input_root, active_keys, output_root)
    print(f"  Found {len(jobs)} total jobs")

    if args.tile_limit:
        jobs = apply_tile_limit(jobs, args.tile_limit)
        print(f"  After tile-limit={args.tile_limit}: {len(jobs)} jobs")

    if args.dry_run:
        by_module: dict[str, int] = {}
        for j in jobs:
            by_module[j.spec.module_dir] = by_module.get(j.spec.module_dir, 0) + 1
        for mdir, cnt in sorted(by_module.items()):
            print(f"    {mdir}: {cnt} jobs")
        print(f"\nDry-run complete. {len(jobs)} jobs would run.")
        return

    if not jobs:
        print("  No jobs to run.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    def _run_batch(batch: list, label: str) -> tuple:
        """Submit one batch to the executor; return (ok, skipped, empty, err)."""
        _ok = _sk = _em = _er = 0
        if not batch:
            return _ok, _sk, _em, _er
        print(f"\n  {label}: {len(batch)} jobs, {args.workers} workers")
        task_args = [(j, args.overwrite) for j in batch]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(convert_one, t): t[0] for t in task_args}
            it = as_completed(futs)
            if _has_tqdm:
                it = _tqdm(it, total=len(futs), unit="file", desc=label)
            for fut in it:
                r = fut.result()
                if r.status == "ok":
                    _ok += 1
                elif r.status == "skipped":
                    _sk += 1
                elif r.status == "empty":
                    _em += 1
                else:
                    _er += 1
                    print(f"    ERR [{r.job.tiff_path.name}]: {r.msg}")
        return _ok, _sk, _em, _er

    # Two-phase conversion: image stores first, then labels.
    # Labels nest inside aligned_N.zarr; zarr mode="w" clears the tile dir,
    # so aligned must be written before any labels write into that dir.
    non_label_jobs = [j for j in jobs if not j.spec.is_label]
    label_jobs = [j for j in jobs if j.spec.is_label]

    print(f"\nConverting {len(jobs)} files with {args.workers} workers (2 phases)...")
    ok = skipped = empty = err = 0

    o, s, e, r = _run_batch(non_label_jobs, "Phase 1: images + IC fields")
    ok += o; skipped += s; empty += e; err += r

    o, s, e, r = _run_batch(label_jobs, "Phase 2: labels")
    ok += o; skipped += s; empty += e; err += r

    elapsed = time.time() - t0
    print(f"\nConversion: ok={ok} skipped={skipped} empty={empty} err={err} [{elapsed:.1f}s]")

    # --- Finalize ---
    if not args.skip_finalize:
        print("\nFinalizing HCS plate metadata...")
        finalize_plates(output_root, input_root)

    print(f"\n{'#' * 60}")
    status = "DONE" if err == 0 else "DONE WITH ERRORS"
    print(f"  {status}: {time.time() - t0:.1f}s total")
    print(f"{'#' * 60}")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
