"""Estimate tile positions for stitching.

This Snakemake script estimates tile positions using either:
- Phase correlation (requires OME-Zarr input store)
- Stage coordinates from metadata (requires combined metadata parquet)

The output is a YAML configuration file with tile shifts that can be
used by the stitch_tiles.py script.
"""
# TODO: test run for all rounds in /Users/cspeters/projects/ops/data/new_imgs_copy/phenotype/real_images. Then look further into the illumination mismatch on the right hand side of the stiched images
from pathlib import Path
import pandas as pd

from lib.preprocess.stitch import (
    estimate_stitch_from_metadata,
    estimate_stitch_from_tiles,
    get_stitch_config,
)

# Get stitch configuration
stitch_config = get_stitch_config(snakemake.config)
method = stitch_config.get("method", "phase_correlation")

# Determine image type from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

# Get type-specific configuration
type_config = stitch_config.get(image_type, {})
reference_channel = type_config.get("reference_channel", 0)

# Extract wildcards
plate = snakemake.wildcards.plate
well = snakemake.wildcards.well
cycle = getattr(snakemake.wildcards, "cycle", None)

# Format well for HCS layout (e.g., "A/01" from "A01")
if len(well) >= 2:
    well_hcs = f"{well[0]}/{well[1:]}"
else:
    well_hcs = well

print(f"Estimating stitch positions for {plate} {well} using {method}")
print(f"Image type: {image_type}, Reference channel: {reference_channel}")

if method == "coordinate_based":
    # Use stage coordinates from metadata
    metadata_path = snakemake.input.metadata
    print(f"Loading metadata from: {metadata_path}")

    metadata_df = pd.read_parquet(metadata_path)

    # Filter to this well if needed
    if "well" in metadata_df.columns:
        metadata_df = metadata_df[metadata_df["well"] == well]

    # Get pixel size from metadata or config
    pixel_size = snakemake.params.get("pixel_size", None)
    if pixel_size is None:
        # Try to get from metadata
        if "pixel_size_x" in metadata_df.columns:
            pixel_size = metadata_df["pixel_size_x"].dropna().iloc[0]
        else:
            raise ValueError(
                "pixel_size not provided and not found in metadata. "
                "Set preprocess.stitch.pixel_size in config."
            )

    # Get tile size from params or config
    tile_size = snakemake.params.get("tile_size", (2048, 2048))
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    shifts = estimate_stitch_from_metadata(
        metadata_df=metadata_df,
        tile_size=tile_size,
        pixel_size=pixel_size,
        well=well_hcs,
        output_path=snakemake.output[0],
    )

else:  # phase_correlation
    # Use phase correlation on tile images
    input_store = snakemake.input.tiles
    print(f"Input OME-Zarr store: {input_store}")

    # Get parameters from config
    tile_size = snakemake.params.get("tile_size", (2048, 2048))
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    overlap_pixels = stitch_config.get("overlap_pixels", 150)
    flipud = stitch_config.get("flipud", False)
    fliplr = stitch_config.get("fliplr", False)
    rot90 = stitch_config.get("rot90", 0)

    # Limit positions for faster testing (optional)
    limit_positions = snakemake.params.get("limit_positions", None)

    shifts = estimate_stitch_from_tiles(
        input_store_path=input_store,
        output_path=snakemake.output[0],
        tile_size=tile_size,
        overlap_pixels=overlap_pixels,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        reference_channel=reference_channel,
        limit_positions=limit_positions,
    )

print(f"Estimated positions for {len(shifts)} tiles")
print(f"Output config written to: {snakemake.output[0]}")
