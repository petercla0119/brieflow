"""Assemble tiles into a stitched well image.

This Snakemake script uses the stitch configuration from estimate_stitch.py
to assemble tiles into a complete stitched well image in OME-Zarr format.

Uses GPU-accelerated streaming assembly when available, with CPU fallback.
"""

from pathlib import Path

from lib.preprocess.stitch import (
    stitch_tiles_to_well,
    get_stitch_config,
    get_compute_backend,
)

# Get stitch configuration
stitch_config = get_stitch_config(snakemake.config)

# Determine image type from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

# Extract wildcards
plate = snakemake.wildcards.plate
well = snakemake.wildcards.well
cycle = getattr(snakemake.wildcards, "cycle", None)

print(f"Stitching tiles for {plate} {well}")
print(f"Image type: {image_type}")
print(f"Compute backend: {get_compute_backend()}")

# Get input paths
input_store = snakemake.input.tiles
stitch_config_path = snakemake.input.config
output_path = snakemake.output[0]

print(f"Input OME-Zarr store: {input_store}")
print(f"Stitch config: {stitch_config_path}")
print(f"Output path: {output_path}")

# Get stitching parameters from config
flipud = stitch_config.get("flipud", False)
fliplr = stitch_config.get("fliplr", False)
rot90 = stitch_config.get("rot90", 0)
blending_method = stitch_config.get("blending_method", "edt")

# Get channel names if available
channel_names = snakemake.params.get("channel_names", None)

print(f"Blending method: {blending_method}")
print(f"Augmentation: flipud={flipud}, fliplr={fliplr}, rot90={rot90}")

# Perform stitching
stitch_tiles_to_well(
    input_store_path=input_store,
    stitch_config_path=stitch_config_path,
    output_store_path=output_path,
    flipud=flipud,
    fliplr=fliplr,
    rot90=rot90,
    blending_method=blending_method,
    channel_names=channel_names,
)

print(f"Stitched image written to: {output_path}")
