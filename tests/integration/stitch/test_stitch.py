"""Integration test for tile stitching.

Reads ND2 tiles, creates an OME-Zarr store, estimates stitch positions from
metadata, assembles a stitched image, and generates a JPG preview.

Modes:
  basic   -- stitch a subset of tiles (default 3) without IC
  all     -- stitch all R1 tiles without IC
  ic      -- stitch all R1 tiles with illumination correction
  compare -- run both 'all' and 'ic', then generate a side-by-side comparison

Usage:
  python test_stitch.py --mode basic
  python test_stitch.py --mode compare --output-dir /tmp/stitch_output
  python test_stitch.py --mode basic --max-tiles 5
"""

import argparse
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np

WORKFLOW_DIR = Path(__file__).parents[3] / "workflow"
sys.path.insert(0, str(WORKFLOW_DIR))

import nd2
from iohub.ngff import open_ome_zarr
from PIL import Image, ImageDraw

DEFAULT_DATA_DIR = Path(
    "/Users/cspeters/projects/ops/data/new_imgs_copy/phenotype/real_images"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def read_nd2_tiles(input_files):
    """Read ND2 files and extract tile data with metadata."""
    tile_data = []
    metadata_rows = []

    for i, fp in enumerate(input_files):
        fp = Path(fp)
        tile_num = int(fp.name.split("Points-")[1].split("_")[0])

        if i % 10 == 0:
            print(f"  Reading tile {i + 1}/{len(input_files)}: {fp.name[:50]}...")

        with nd2.ND2File(fp) as f:
            img = f.asarray()
            frame_meta = f.frame_metadata(0)

            x_pos = y_pos = None
            if frame_meta.channels and hasattr(frame_meta.channels[0], "position"):
                stage_pos = frame_meta.channels[0].position.stagePositionUm
                x_pos, y_pos = stage_pos.x, stage_pos.y

            pixel_size = 0.325
            if frame_meta.channels and hasattr(frame_meta.channels[0], "volume"):
                vol = frame_meta.channels[0].volume
                if vol and hasattr(vol, "axesCalibration"):
                    pixel_size = vol.axesCalibration[0]

            tile_data.append(
                {
                    "tile": tile_num,
                    "data": img,
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "pixel_size": pixel_size,
                }
            )
            metadata_rows.append({"tile": tile_num, "x_pos": x_pos, "y_pos": y_pos})

    return tile_data, metadata_rows


def normalize_tile_dimensions(tile_data):
    """Max-project Z if needed; return (n_channels, tile_h, tile_w)."""
    tile_shape = tile_data[0]["data"].shape
    print(f"  Tile shape: {tile_shape}")

    if len(tile_shape) == 3:
        n_channels, tile_h, tile_w = tile_shape
    elif len(tile_shape) == 4:
        n_channels, _, tile_h, tile_w = tile_shape
        print("  Max projecting Z dimension...")
        for td in tile_data:
            td["data"] = td["data"].max(axis=1)
    else:
        tile_h, tile_w = tile_shape[-2:]
        n_channels = 1

    print(f"  Tile size: {tile_h} x {tile_w}, Channels: {n_channels}")
    return n_channels, tile_h, tile_w


def create_ome_zarr_store(tile_data, zarr_path, n_channels, tile_h, tile_w, data_key="data"):
    """Create an HCS OME-Zarr store from tile data."""
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    channel_names = ["Blue", "Red", "Green", "Far Red"][:n_channels]
    store = open_ome_zarr(str(zarr_path), layout="hcs", mode="w-", channel_names=channel_names)

    for i, td in enumerate(tile_data):
        tile_name = f"{td['tile']:03d}000"
        pos = store.create_position("A", "06", tile_name)

        data = td[data_key]
        if data.ndim == 3:
            data = data[np.newaxis, :, np.newaxis, :, :]
        elif data.ndim == 2:
            data = data[np.newaxis, np.newaxis, np.newaxis, :, :]

        pos.create_zeros(
            "0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 1, tile_h, tile_w)
        )
        pos["0"][:] = data

        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{len(tile_data)} positions...")

    print(f"  OME-Zarr store created: {zarr_path}")
    return store


def calculate_ic_field(tiles):
    """Compute a simple IC field as the per-channel median across all tiles."""
    stacked = np.stack(tiles, axis=0)
    ic_field = np.median(stacked, axis=0).astype(np.float32)
    ic_field = ic_field / ic_field.mean(axis=(-2, -1), keepdims=True)
    ic_field = np.clip(ic_field, 0.1, 10.0)
    return ic_field


def apply_ic_correction(tile_data, ic_field):
    """Apply IC correction, storing result in tile_data[*]['data_corrected']."""
    for i, td in enumerate(tile_data):
        corrected = td["data"].astype(np.float32) / ic_field
        td["data_corrected"] = np.clip(corrected, 0, 65535).astype(np.uint16)
        if (i + 1) % 10 == 0:
            print(f"  Corrected {i + 1}/{len(tile_data)} tiles...")


def estimate_and_assemble(tile_data, metadata_rows, zarr_path, output_dir, label):
    """Estimate stitch positions, assemble tiles, return stitched zarr path."""
    import pandas as pd
    from lib.preprocess.stitch import (
        detect_overlap_from_metadata,
        estimate_stitch_from_metadata,
        stitch_tiles_to_well,
    )

    metadata_df = pd.DataFrame(metadata_rows)
    pixel_size = tile_data[0]["pixel_size"]
    n_channels, tile_h, tile_w = tile_data[0]["data"].shape[-3], *tile_data[0]["data"].shape[-2:]

    print(f"  Metadata: {len(metadata_df)} tiles")
    print(
        f"  X range: {metadata_df['x_pos'].min():.1f} - {metadata_df['x_pos'].max():.1f} um"
    )
    print(
        f"  Y range: {metadata_df['y_pos'].min():.1f} - {metadata_df['y_pos'].max():.1f} um"
    )

    stitch_config_path = output_dir / f"stitch_config_{label}.yml"
    detected_overlap = detect_overlap_from_metadata(
        metadata_df, (tile_h, tile_w), pixel_size
    )
    print(f"  Detected overlap: {detected_overlap} pixels")

    shifts = estimate_stitch_from_metadata(
        metadata_df=metadata_df,
        tile_size=(tile_h, tile_w),
        pixel_size=pixel_size,
        well="A/06",
        output_path=stitch_config_path,
    )
    print(f"  Estimated shifts for {len(shifts)} tiles")

    stitched_path = output_dir / f"stitched_{label}.zarr"
    if stitched_path.exists():
        shutil.rmtree(stitched_path)

    stitch_tiles_to_well(
        input_store_path=zarr_path,
        stitch_config_path=stitch_config_path,
        output_store_path=stitched_path,
        blending_method="edt",
    )
    print(f"  Stitched image saved: {stitched_path}")
    return stitched_path


def create_preview(stitched_path, output_path, max_size=4000):
    """Create a JPG preview from a stitched OME-Zarr store."""
    store = open_ome_zarr(str(stitched_path), mode="r")
    for _, pos in store.positions():
        data = pos["0"][:]
        print(f"  Stitched shape: {data.shape}")
        break

    img_data = data[0, :, 0, :, :]

    def _normalize(arr):
        arr = arr.astype(np.float32)
        if arr.max() > 0:
            p_lo, p_hi = np.percentile(arr[arr > 0], [1, 99])
        else:
            p_lo, p_hi = 0, 1
        return (np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1) * 255).astype(np.uint8)

    if img_data.shape[0] >= 3:
        rgb = np.stack(
            [_normalize(img_data[1]), _normalize(img_data[2]), _normalize(img_data[0])],
            axis=-1,
        )
    else:
        gray = _normalize(img_data[0])
        rgb = np.stack([gray, gray, gray], axis=-1)

    img_pil = Image.fromarray(rgb)
    if max(img_pil.size) > max_size:
        ratio = max_size / max(img_pil.size)
        img_pil = img_pil.resize(
            (int(img_pil.size[0] * ratio), int(img_pil.size[1] * ratio)),
            Image.Resampling.LANCZOS,
        )

    img_pil.save(str(output_path), quality=95)
    print(f"  Preview saved: {output_path} ({img_pil.size})")
    return output_path


def create_comparison(ic_preview_path, no_ic_preview_path, output_path):
    """Generate a stacked comparison image: IC on top, no-IC on bottom."""
    img_ic = Image.open(ic_preview_path)
    img_no_ic = Image.open(no_ic_preview_path)

    target_w = max(img_ic.width, img_no_ic.width)
    if img_ic.width != target_w:
        ratio = target_w / img_ic.width
        img_ic = img_ic.resize(
            (target_w, int(img_ic.height * ratio)), Image.Resampling.LANCZOS
        )
    if img_no_ic.width != target_w:
        ratio = target_w / img_no_ic.width
        img_no_ic = img_no_ic.resize(
            (target_w, int(img_no_ic.height * ratio)), Image.Resampling.LANCZOS
        )

    label_h, gap = 40, 10
    total_h = label_h + img_ic.height + gap + label_h + img_no_ic.height
    canvas = Image.new("RGB", (target_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 8), "Row 1: With Illumination Correction", fill=(255, 255, 255))
    canvas.paste(img_ic, (0, label_h))

    y2 = label_h + img_ic.height + gap
    draw.text((10, y2 + 8), "Row 2: Without Illumination Correction", fill=(255, 255, 255))
    canvas.paste(img_no_ic, (0, y2 + label_h))

    canvas.save(str(output_path), quality=95)
    print(f"  Comparison saved: {output_path} ({canvas.size})")


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_stitch(data_dir, output_dir, max_tiles=None, apply_ic=False, preview_size=4000):
    """Core stitch pipeline: read tiles, optionally apply IC, stitch, preview.

    Returns the path to the preview JPG.
    """
    input_files = sorted(glob(str(data_dir / "*R1*.nd2")))
    if not input_files:
        raise FileNotFoundError(f"No R1 ND2 files found in {data_dir}")

    if max_tiles is not None:
        input_files = input_files[:max_tiles]

    label = "ic" if apply_ic else "raw"
    ic_tag = " (with IC)" if apply_ic else ""
    print("=" * 60)
    print(f"STITCHING {len(input_files)} TILES{ic_tag}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read tiles
    print(f"\n[1] Reading {len(input_files)} ND2 files...")
    tile_data, metadata_rows = read_nd2_tiles(input_files)
    n_channels, tile_h, tile_w = normalize_tile_dimensions(tile_data)

    # IC
    data_key = "data"
    if apply_ic:
        print("\n[2] Computing illumination correction field...")
        ic_field = calculate_ic_field([td["data"] for td in tile_data])
        print(f"  IC field shape: {ic_field.shape}")

        print("\n[3] Applying IC correction...")
        apply_ic_correction(tile_data, ic_field)
        data_key = "data_corrected"

    # Create OME-Zarr store
    step = 4 if apply_ic else 2
    print(f"\n[{step}] Creating OME-Zarr store...")
    zarr_path = output_dir / f"tiles_{label}.zarr"
    create_ome_zarr_store(tile_data, zarr_path, n_channels, tile_h, tile_w, data_key=data_key)

    # Estimate + assemble
    step += 1
    print(f"\n[{step}] Estimating positions and assembling...")
    stitched_path = estimate_and_assemble(
        tile_data, metadata_rows, zarr_path, output_dir, label
    )

    # Preview
    step += 1
    print(f"\n[{step}] Creating JPG preview...")
    preview_path = output_dir / f"stitched_{label}_preview.jpg"
    create_preview(stitched_path, preview_path, max_size=preview_size)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    return preview_path


def run_compare(data_dir, output_dir, preview_size=4000):
    """Run both raw and IC stitching, then generate a comparison image."""
    no_ic_dir = output_dir / "raw"
    ic_dir = output_dir / "ic"

    no_ic_preview = run_stitch(data_dir, no_ic_dir, apply_ic=False, preview_size=preview_size)
    print()
    ic_preview = run_stitch(data_dir, ic_dir, apply_ic=True, preview_size=preview_size)

    print("\nGenerating comparison image...")
    comparison_path = output_dir / "stitch_comparison.jpg"
    create_comparison(ic_preview, no_ic_preview, comparison_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for tile stitching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "all", "ic", "compare"],
        default="all",
        help="Test mode (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Input directory containing ND2 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Limit number of tiles to process (default: 3 for basic, all for other modes)",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=4000,
        help="Max preview image dimension in pixels (default: 4000)",
    )
    args = parser.parse_args()

    if args.mode == "basic":
        max_tiles = args.max_tiles if args.max_tiles is not None else 3
        run_stitch(args.data_dir, args.output_dir, max_tiles=max_tiles, preview_size=args.preview_size)
    elif args.mode == "all":
        run_stitch(args.data_dir, args.output_dir, max_tiles=args.max_tiles, preview_size=args.preview_size)
    elif args.mode == "ic":
        run_stitch(args.data_dir, args.output_dir, max_tiles=args.max_tiles, apply_ic=True, preview_size=args.preview_size)
    elif args.mode == "compare":
        run_compare(args.data_dir, args.output_dir, preview_size=args.preview_size)


if __name__ == "__main__":
    main()
