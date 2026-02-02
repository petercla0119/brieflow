"""Extract tile position and overlap metadata from ND2 files.

This script analyzes ND2 files in a directory to extract stage positions
and calculate tile overlap for stitching purposes.
"""

import argparse
import json
import os
from pathlib import Path

import nd2


def extract_tile_positions(nd2_dir: str | Path) -> list[dict]:
    """Extract stage positions from all ND2 files in a directory.

    Args:
        nd2_dir: Path to directory containing ND2 files.

    Returns:
        List of dictionaries containing tile metadata.
    """
    nd2_dir = Path(nd2_dir)
    nd2_files = sorted(nd2_dir.glob("*.nd2"))

    tiles = []
    for nd2_path in nd2_files:
        with nd2.ND2File(nd2_path) as f:
            # Get basic image info
            shape = f.shape
            sizes = f.sizes
            voxel_size = f.voxel_size()

            # Get stage position
            frame_meta = f.frame_metadata(0)
            stage_pos = None
            if frame_meta and frame_meta.channels:
                pos = frame_meta.channels[0].position
                if pos and pos.stagePositionUm:
                    stage_pos = {
                        "x": pos.stagePositionUm.x,
                        "y": pos.stagePositionUm.y,
                        "z": pos.stagePositionUm.z,
                    }

            tiles.append(
                {
                    "filename": nd2_path.name,
                    "path": str(nd2_path),
                    "shape": shape,
                    "sizes": sizes,
                    "pixel_size_um": {
                        "x": voxel_size.x,
                        "y": voxel_size.y,
                        "z": voxel_size.z,
                    },
                    "stage_position_um": stage_pos,
                }
            )

    return tiles


def calculate_overlap(tiles: list[dict]) -> dict:
    """Calculate tile overlap from stage positions.

    Args:
        tiles: List of tile metadata dictionaries.

    Returns:
        Dictionary containing overlap calculations.
    """
    if not tiles or len(tiles) < 2:
        return {"error": "Need at least 2 tiles to calculate overlap"}

    # Get positions and image dimensions
    positions = [
        (t["stage_position_um"]["x"], t["stage_position_um"]["y"])
        for t in tiles
        if t["stage_position_um"]
    ]

    if len(positions) < 2:
        return {"error": "Insufficient stage position data"}

    # Get image size in microns
    first_tile = tiles[0]
    pixel_size_x = first_tile["pixel_size_um"]["x"]
    pixel_size_y = first_tile["pixel_size_um"]["y"]

    # Handle different dimension orderings
    sizes = first_tile["sizes"]
    width_px = sizes.get("X", first_tile["shape"][-1])
    height_px = sizes.get("Y", first_tile["shape"][-2])

    image_width_um = width_px * pixel_size_x
    image_height_um = height_px * pixel_size_y

    # Sort positions to find unique X and Y values
    xs = sorted(set(round(p[0], 1) for p in positions))
    ys = sorted(set(round(p[1], 1) for p in positions))

    # Determine grid layout by clustering positions
    # Use a threshold based on image size to detect true grid vs strip
    x_range = max(xs) - min(xs) if xs else 0
    y_range = max(ys) - min(ys) if ys else 0

    # If range is less than half an image, it's effectively a single row/column
    is_single_row = y_range < (image_height_um * 0.5)
    is_single_col = x_range < (image_width_um * 0.5)

    # Recalculate grid dimensions
    n_cols = len(xs) if not is_single_col else 1
    n_rows = len(ys) if not is_single_row else 1

    # For strips, count tiles in the strip direction
    if is_single_row:
        n_cols = len(tiles)
        n_rows = 1
    elif is_single_col:
        n_rows = len(tiles)
        n_cols = 1

    result = {
        "image_size_px": {"width": width_px, "height": height_px},
        "image_size_um": {"width": image_width_um, "height": image_height_um},
        "pixel_size_um": {"x": pixel_size_x, "y": pixel_size_y},
        "tile_count": len(tiles),
        "grid_size": {"cols": n_cols, "rows": n_rows},
        "layout": "horizontal_strip" if is_single_row else ("vertical_strip" if is_single_col else "grid"),
    }

    # Calculate X overlap (only if not a single column)
    if len(xs) > 1 and not is_single_col:
        x_spacings = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        avg_x_spacing = sum(x_spacings) / len(x_spacings)
        x_overlap_um = image_width_um - avg_x_spacing
        x_overlap_pct = (x_overlap_um / image_width_um) * 100
        x_overlap_px = x_overlap_um / pixel_size_x

        result["x_spacing_um"] = avg_x_spacing
        result["x_overlap_um"] = x_overlap_um
        result["x_overlap_percent"] = x_overlap_pct
        result["x_overlap_px"] = x_overlap_px

    # Calculate Y overlap (only if not a single row)
    if len(ys) > 1 and not is_single_row:
        y_spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        avg_y_spacing = sum(y_spacings) / len(y_spacings)
        y_overlap_um = image_height_um - avg_y_spacing
        y_overlap_pct = (y_overlap_um / image_height_um) * 100
        y_overlap_px = y_overlap_um / pixel_size_y

        result["y_spacing_um"] = avg_y_spacing
        result["y_overlap_um"] = y_overlap_um
        result["y_overlap_percent"] = y_overlap_pct
        result["y_overlap_px"] = y_overlap_px

    return result


def extract_tile_metadata(nd2_dir: str | Path, output_path: str | Path = None) -> dict:
    """Extract complete tile metadata including positions and overlap.

    Args:
        nd2_dir: Path to directory containing ND2 files.
        output_path: Optional path to save JSON output.

    Returns:
        Dictionary containing all tile metadata and overlap calculations.
    """
    nd2_dir = Path(nd2_dir)

    tiles = extract_tile_positions(nd2_dir)
    overlap = calculate_overlap(tiles)

    result = {
        "source_directory": str(nd2_dir),
        "tiles": tiles,
        "overlap": overlap,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def print_summary(metadata: dict) -> None:
    """Print a human-readable summary of tile metadata."""
    overlap = metadata["overlap"]

    print("=" * 50)
    print("TILE METADATA SUMMARY")
    print("=" * 50)
    print(f"Source: {metadata['source_directory']}")
    print(f"Total tiles: {overlap.get('tile_count', 'N/A')}")
    layout = overlap.get('layout', 'unknown')
    grid = overlap.get('grid_size', {})
    print(f"Layout: {layout} ({grid.get('cols', '?')} cols x {grid.get('rows', '?')} rows)")
    print()
    print("Image dimensions:")
    print(f"  Size: {overlap.get('image_size_px', {}).get('width', '?')} x {overlap.get('image_size_px', {}).get('height', '?')} px")
    print(f"  Size: {overlap.get('image_size_um', {}).get('width', '?'):.1f} x {overlap.get('image_size_um', {}).get('height', '?'):.1f} um")
    print(f"  Pixel size: {overlap.get('pixel_size_um', {}).get('x', '?')} um/px")
    print()
    print("Overlap:")
    if "x_overlap_um" in overlap:
        print(f"  X: {overlap['x_overlap_um']:.1f} um ({overlap['x_overlap_percent']:.1f}%) = {overlap['x_overlap_px']:.0f} px")
    if "y_overlap_um" in overlap:
        print(f"  Y: {overlap['y_overlap_um']:.1f} um ({overlap['y_overlap_percent']:.1f}%) = {overlap['y_overlap_px']:.0f} px")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Extract tile position and overlap metadata from ND2 files."
    )
    parser.add_argument(
        "nd2_dir",
        type=str,
        help="Directory containing ND2 files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress summary output",
    )

    args = parser.parse_args()

    metadata = extract_tile_metadata(args.nd2_dir, args.output)

    if not args.quiet:
        print_summary(metadata)

    if args.output:
        print(f"\nMetadata saved to: {args.output}")


if __name__ == "__main__":
    main()
