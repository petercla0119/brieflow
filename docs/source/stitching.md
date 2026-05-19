# Stitching

## Overview

Stitching assembles individually acquired microscopy tiles into complete well images. This is useful for:

- **Quality control**: Visually inspect tile alignment, illumination uniformity, and acquisition artifacts across an entire well.
- **Visualization**: View stitched wells in Napari or other OME-Zarr compatible viewers.
- **Downstream analysis**: Merge SBS and phenotype data using whole-well spatial coordinates (merge module stitching).

Brieflow provides stitching in two pipeline contexts:

| Context | Module | Input Format | Purpose |
|---------|--------|-------------|---------|
| **Preprocessing** | `preprocess` | OME-Zarr tiles | Early QC and visualization |
| **Merge** | `merge` | Aligned TIFFs | Spatial merge of SBS + phenotype data |

Preprocessing stitching runs early in the pipeline (after illumination correction) and produces stitched OME-Zarr images for visual inspection. Merge stitching runs later and is used to spatially align SBS barcodes with phenotype measurements across the well.

## Prerequisites

### Dependencies

Preprocessing stitching requires two additional packages:

- **`stitch`** -- GPU-accelerated stitching library (local package)
- **`iohub`** -- OME-Zarr I/O (>= 0.2.0)

These are declared in `pyproject.toml`. Install with:

```sh
pip install -e /path/to/brieflow
```

Merge stitching uses `dexp` (included in the base Brieflow environment) and does not require the `stitch` package.

### OME-Zarr Format Required

Preprocessing stitching operates on OME-Zarr tiles. Your preprocessing config must include `"zarr"` in `output_formats`:

```yaml
preprocess:
  output_formats: ["zarr"]
```

This ensures that OME-Zarr tile images are created during preprocessing, which the stitching rules consume as input.

### GPU (Optional)

GPU acceleration via CuPy is supported but not required. When a GPU is not available, stitching falls back to NumPy (CPU). The fallback is automatic -- no configuration change is needed.

## Quick Start

Add the following to your `config.yml` to enable preprocessing stitching with default settings:

```yaml
preprocess:
  output_formats: ["zarr"]
  stitch:
    enabled: true
```

Then run preprocessing as usual:

```sh
# Dry run to verify DAG
snakemake --configfile config/config.yml --until all_preprocess -n

# Full run
snakemake --configfile config/config.yml --until all_preprocess
```

Stitching adds four rules to the DAG (two per image type when both phenotype and SBS are configured):

1. `estimate_stitch_phenotype` / `estimate_stitch_sbs` -- compute tile positions
2. `stitch_phenotype` / `stitch_sbs` -- assemble tiles into stitched wells

## Configuration Reference

### Preprocessing Stitching

All preprocessing stitching parameters live under `preprocess.stitch` in `config.yml`:

```yaml
preprocess:
  stitch:
    enabled: true                    # Enable/disable stitching (default: false)
    method: "phase_correlation"      # Position estimation method (default: "phase_correlation")
    use_gpu: true                    # Use GPU if available (default: true)
    overlap_pixels: "auto"           # Tile overlap in pixels, or "auto" (default: "auto")
    flipud: false                    # Flip tiles vertically (default: false)
    fliplr: false                    # Flip tiles horizontally (default: false)
    rot90: 0                         # 90-degree rotations: 0, 1, 2, or 3 (default: 0)
    output_format: "omezarr"         # Output format (default: "omezarr")
    blending_method: "edt"           # Blending: "edt" or "average" (default: "edt")
    tile_size: [2048, 2048]          # Tile dimensions [height, width] in pixels
    pixel_size: 0.325                # Physical pixel size in um/pixel (used by coordinate_based)
    phenotype:
      enabled: true                  # Enable phenotype stitching (default: true)
      reference_channel: 0           # Channel for registration (default: 0)
    sbs:
      enabled: true                  # Enable SBS stitching (default: true)
      reference_cycle: 1             # Cycle for registration (default: 1)
      reference_channel: 0           # Channel for registration (default: 0)
```

#### Parameter Details

**`method`**

| Value | Description | Speed | Accuracy |
|-------|------------|-------|----------|
| `"phase_correlation"` | Registers overlapping tile regions using phase correlation. Requires OME-Zarr input store. | Slower | Higher |
| `"coordinate_based"` | Converts microscope stage coordinates (um) to pixel positions. Uses metadata only. | Faster | Depends on stage accuracy |

**`overlap_pixels`**

The number of pixels that adjacent tiles overlap. When set to `"auto"` (the default), the overlap is detected from stage coordinate metadata:

```
overlap = tile_size - (distance_between_adjacent_tiles / pixel_size)
```

If auto-detection fails (e.g., fewer than 2 tiles or missing coordinates), it falls back to 150 pixels.

**`blending_method`**

- `"edt"` (Euclidean distance transform): Weights each pixel by its distance from the tile edge, producing smooth transitions in overlap regions. Recommended for most cases.
- `"average"`: Simple averaging in overlap regions. Faster but may show visible seams.

**`flipud` / `fliplr` / `rot90`**

Geometric transformations applied to tiles before stitching. Use these to correct for microscope-specific coordinate conventions (e.g., inverted Y axis, rotated stage). These are applied consistently to all tiles.

**`phenotype` and `sbs` sub-configs**

Each image type can be independently enabled/disabled. The `reference_channel` (and `reference_cycle` for SBS) determines which channel is used for phase correlation registration. The stitching itself includes all channels.

### Merge Stitching

Merge stitching is configured under `merge` in `config.yml` by setting the merge approach to `"stitch"`:

```yaml
merge:
  approach: "stitch"                 # Use stitch-based merge (default: "fast")
  flipud: false                      # Flip tiles vertically
  fliplr: false                      # Flip tiles horizontally
  rot90: 0                           # 90-degree rotations
  phenotype_pixel_size: 0.325        # Phenotype pixel size (um/pixel), fallback if not in metadata
  sbs_pixel_size: 0.325              # SBS pixel size (um/pixel), fallback if not in metadata
  sbs_metadata_cycle: 1              # SBS cycle to use for metadata
  sbs_metadata_channel: null         # SBS channel filter (null = all)
  alignment_flip_x: false            # Flip SBS X coordinates to match phenotype
  alignment_flip_y: false            # Flip SBS Y coordinates to match phenotype
  alignment_rotate_90: false         # Rotate SBS coordinates 90 degrees
```

## Pipeline Order

Preprocessing stitching runs after illumination correction (IC) calculation. The IC field is an explicit input dependency to ensure proper pipeline ordering:

```
1. Extract metadata
2. Combine metadata
3. Convert images (TIFF / Zarr / OME-Zarr)
4. Calculate illumination correction
5. Estimate stitch positions          <-- uses metadata + IC dependency
6. Assemble stitched images           <-- uses OME-Zarr tiles + stitch config + IC dependency
```

The stitched images use raw (not IC-corrected) tile data. The IC dependency ensures ordering only -- it does not apply IC to the stitched output.

Merge stitching runs later, after SBS and phenotype processing are complete:

```
1. Preprocess (including IC)
2. SBS processing (align, call reads, call cells)
3. Phenotype processing (align, segment, extract features)
4. Estimate stitch (phenotype + SBS)   <-- merge module
5. Stitch alignment                    <-- merge module
6. Stitch merge                        <-- merge module
```

## Output Files

### Preprocessing Stitching

```
brieflow_output/
└── preprocess/
    ├── stitch_configs/
    │   ├── phenotype/
    │   │   └── P-{plate}_W-{well}__stitch_config.yml     # Tile positions (YAML)
    │   └── sbs/
    │       └── P-{plate}_W-{well}_C-{cycle}__stitch_config.yml
    └── stitched/
        ├── phenotype/
        │   └── P-{plate}_W-{well}__stitched.zarr/         # Stitched OME-Zarr
        └── sbs/
            └── P-{plate}_W-{well}_C-{cycle}__stitched.zarr/
```

**Stitch config YAML** contains tile positions used for assembly:

```yaml
total_translation:
  A/01/000000: [0, 0]
  A/01/001000: [0, 2015]
  A/01/002000: [0, 4030]
  # ...
method: coordinate_based
pixel_size_um: 0.325
tile_size: [2400, 2400]
```

**Stitched OME-Zarr** is a standard OME-Zarr store viewable in Napari:

```sh
python workflow/scripts/shared/load_omezarr_in_napari.py \
  brieflow_output/preprocess/stitched/phenotype/P-1_W-A01__stitched.zarr
```

### Merge Stitching

Merge stitching outputs are under `brieflow_output/merge/` and include:
- Stitch config YAMLs (phenotype + SBS tile positions)
- Stitched TIFF images and segmentation masks
- Cell position DataFrames with tile-to-well coordinate mapping

## Estimation Methods

### Coordinate-Based Estimation

Uses microscope stage coordinates from tile metadata to calculate pixel positions:

```
pixel_position = (stage_position_um - origin_um) / pixel_size_um
```

**When to use:**
- Stage coordinates are reliable (modern motorized microscopes)
- Fast turnaround is needed
- You want to avoid loading image data for estimation

**Requirements:**
- Combined metadata parquet with `x_pos`, `y_pos` columns (in micrometers)
- Known `pixel_size` (from metadata or config)
- At least 2 tiles with valid coordinates for overlap auto-detection

### Phase Correlation Estimation

Registers overlapping tile regions using cross-correlation in the frequency domain to find sub-pixel-accurate alignment:

**When to use:**
- Stage coordinates are imprecise or unavailable
- Maximum stitching accuracy is needed
- GPU is available (significantly faster)

**Requirements:**
- OME-Zarr tile store as input
- Known `overlap_pixels` (or auto-detected)
- The `stitch` library installed

Both methods produce the same output format (YAML config with `total_translation` dict), so the assembly step is identical regardless of estimation method.

## Preprocessing vs. Merge Stitching

| Feature | Preprocessing | Merge |
|---------|--------------|-------|
| Input format | OME-Zarr tiles | Aligned TIFFs |
| Registration library | `stitch` (GPU-accelerated) | `dexp` (CPU) |
| Blending | EDT or average | Weighted average |
| Mask stitching | No | Yes (preserves cell IDs) |
| Cell position extraction | No | Yes (tile-to-well mapping) |
| Output format | OME-Zarr | TIFF + DataFrame |
| GPU support | Yes (CuPy) | No |
| When it runs | After IC calculation | After SBS + phenotype processing |

Preprocessing stitching is for early visual QC. Merge stitching is for spatial data analysis.

## Troubleshooting

### GPU Not Available

If GPU is requested but not available, a warning is emitted and processing continues on CPU:

```
GPU requested but not available. Falling back to CPU processing.
```

To suppress the warning, set `use_gpu: false` in the stitch config. CPU stitching is slower but produces identical results.

### Missing Pixel Size

Coordinate-based estimation requires a pixel size. The script checks for it in this order:

1. `pixel_size_x` column in the combined metadata parquet
2. `preprocess.stitch.pixel_size` in `config.yml`

If neither is available, the estimation fails with:

```
pixel_size not provided and not found in metadata.
Set preprocess.stitch.pixel_size in config.
```

Fix: Add `pixel_size` to your stitch config (typically 0.325 um/pixel for 20x objectives).

### Overlap Auto-Detection Fails

If fewer than 2 tiles have valid stage coordinates, overlap detection falls back to 150 pixels:

```
[stitch] Could not detect overlap, using default: 150
```

Fix: Either ensure metadata has valid `x_pos`/`y_pos` values, or set `overlap_pixels` explicitly.

### Illumination Mismatch at Tile Boundaries

If tile boundaries are visible in the stitched image (brightness differences), this typically indicates uneven illumination across the field of view. Consider:

- Verifying the IC field is computed correctly
- Using `blending_method: "edt"` (smoother transitions than `"average"`)
- Checking whether the issue appears on specific sides of the image (may indicate optical path issues)

### Stitching Rules Not in DAG

If `estimate_stitch_*` / `stitch_*` rules don't appear in the dry run, check:

1. `preprocess.stitch.enabled` is `true`
2. `preprocess.output_formats` includes `"zarr"` (OME-Zarr tiles must be created)
3. The image type is enabled (`phenotype.enabled` / `sbs.enabled` under `preprocess.stitch`)
