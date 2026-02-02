# Stitching Integration Changelog

This document tracks all changes made during the integration of GPU-accelerated stitching into Brieflow's preprocessing module.

## Overview

**Goal**: Integrate the GPU-accelerated stitching library from `/Users/cspeters/projects/ops/stitching/stitch/` into Brieflow's preprocessing module.

**Branch**: `feat/preprocess/stitching`

**Key Design Decisions**:
1. Import stitching package as dependency (not copy)
2. Pipeline position: After illumination correction calculation
3. Complement existing merge stitching (preprocessing stitching for early visualization/QC)
4. Optional via config (disabled by default)
5. Auto-detect tile overlap from stage coordinate metadata

---

## [2026-02-02] - Step 1.1: Create Changelog

### Changes
- File: `docs/CHANGELOG_STITCHING.md`
  - Created: This changelog file to track integration progress

### Rationale
Following the plan requirement to maintain a detailed log of all changes during integration.

### Tests Added
None (documentation only)

### Issues/Notes
None

---

## [2026-02-02] - Step 1.2: Add Stitch Package Dependency

### Changes
- File: `pyproject.toml`
  - Added: `iohub>=0.2.0` - Required by stitching library for OME-Zarr I/O
  - Added: `stitch @ file:///Users/cspeters/projects/ops/stitching` - Local stitching library

### Rationale
Using local file reference for development allows rapid iteration on stitching library
while maintaining proper dependency management.

### Tests Added
None (dependency addition)

### Issues/Notes
None

---

## [2026-02-02] - Step 1.3 & 1.4: Create Stitch Adapter Module

### Changes
- File: `workflow/lib/preprocess/stitch.py`
  - Created: New adapter module bridging external stitching library to Brieflow
  - Added: `is_gpu_available()` - Cached GPU detection with CuPy fallback
  - Added: `get_compute_backend()` - Returns 'gpu' or 'cpu' string
  - Added: `validate_stitch_config()` - Validates and normalizes stitch configuration
  - Added: `get_stitch_config()` - Extracts stitch config from main config
  - Added: `is_stitching_enabled()` - Checks if stitching is enabled globally/per-type

- File: `tests/unit/preprocess/__init__.py`
  - Created: Test package for preprocessing unit tests

- File: `tests/unit/preprocess/test_stitch.py`
  - Created: 21 unit tests covering:
    - GPU availability detection and caching
    - Compute backend selection
    - Config validation (valid values, defaults, error cases)
    - Config extraction from main config
    - Stitching enablement checks

### Rationale
Combined config validation and GPU detection into single module since they're closely related.
GPU detection uses lazy evaluation with caching to avoid repeated CuPy initialization checks.
Validation applies sensible defaults while allowing override of all parameters.

### Tests Added
- `TestIsGpuAvailable` (3 tests): GPU detection with/without CuPy, caching
- `TestGetComputeBackend` (2 tests): Backend string selection
- `TestValidateStitchConfig` (11 tests): All validation scenarios
- `TestGetStitchConfig` (2 tests): Config extraction
- `TestIsStitchingEnabled` (3 tests): Global and per-type enablement

### Issues/Notes
All 21 tests pass. GPU fallback warning is emitted when GPU requested but unavailable.

---

## [2026-02-02] - Step 2.1 & 2.2: Add Stitch Estimation Functions

### Changes
- File: `workflow/lib/preprocess/stitch.py`
  - Added: `estimate_stitch_from_metadata()` - Convert stage coordinates to pixel shifts
  - Added: `estimate_stitch_from_tiles()` - Phase correlation-based position estimation
  - Added: `_format_tile_name()` - Helper to format tile IDs for stitching library

- File: `tests/unit/preprocess/test_stitch.py`
  - Added: 9 new tests for estimation functions:
    - `TestFormatTileName` (3 tests): Tile name formatting
    - `TestEstimateStitchFromMetadata` (4 tests): Coordinate-based estimation
    - `TestEstimateStitchFromTiles` (2 tests): Phase correlation estimation

### Rationale
Two estimation methods provide flexibility:
- `coordinate_based`: Fast, uses stage metadata, good when coordinates are accurate
- `phase_correlation`: Slower but more accurate, uses GPU when available

Both methods output the same YAML config format for consumption by tile assembly.

### Tests Added
- `TestFormatTileName` (3 tests): Integer, formatted string, numeric string inputs
- `TestEstimateStitchFromMetadata` (4 tests): Valid data, missing columns, invalid input, NaN handling
- `TestEstimateStitchFromTiles` (2 tests): Missing store, library call verification

### Issues/Notes
All 30 tests pass. Phase correlation function wraps the external stitch library.

---

## [2026-02-02] - Step 2.3 & 3.1-3.2: Add Snakemake Scripts and Assembly Function

### Changes
- File: `workflow/lib/preprocess/stitch.py`
  - Added: `stitch_tiles_to_well()` - GPU-accelerated streaming tile assembly
  - Added: `load_stitch_config()` - Load stitch configuration from YAML

- File: `workflow/scripts/preprocess/estimate_stitch.py`
  - Created: Snakemake script for tile position estimation
  - Supports both coordinate_based and phase_correlation methods
  - Extracts parameters from config and wildcards

- File: `workflow/scripts/preprocess/stitch_tiles.py`
  - Created: Snakemake script for tile assembly
  - Uses streaming assembly with GPU acceleration when available
  - Outputs OME-Zarr format

- File: `tests/unit/preprocess/test_stitch.py`
  - Added: 5 new tests for assembly and config loading functions

### Rationale
Scripts follow existing Brieflow pattern of accessing snakemake context object.
Assembly uses streaming to avoid loading full canvas into memory, enabling
stitching of large datasets that wouldn't fit in GPU/CPU memory.

### Tests Added
- `TestStitchTilesToWell` (3 tests): Missing store, missing config, library call
- `TestLoadStitchConfig` (2 tests): Valid config, missing file

### Issues/Notes
All 35 tests pass.

---

## [2026-02-02] - Step 2.4-2.6 & 3.3-3.5: Add Snakemake Targets and Rules

### Changes
- File: `workflow/targets/preprocess.smk`
  - Added: Stitch enablement checks using `is_stitching_enabled()`
  - Added: Output definitions for estimate_stitch and stitch rules
  - Added: Output mappings (YAML files as regular, Zarr as directory)
  - Added: Conditional filtering based on stitch config enablement

- File: `workflow/rules/preprocess.smk`
  - Added: `estimate_stitch_phenotype` rule - Position estimation for phenotype
  - Added: `stitch_phenotype` rule - Tile assembly for phenotype (GPU resource)
  - Added: `estimate_stitch_sbs` rule - Position estimation for SBS
  - Added: `stitch_sbs` rule - Tile assembly for SBS (GPU resource)
  - All rules are conditional on config enablement

### Rationale
Rules follow existing Brieflow pattern:
- Input functions use `output_to_input()` to expand tile wildcards
- Stitching takes OME-Zarr tiles (from convert_*_omezarr rules) as input
- GPU resources declared for assembly rules to enable cluster scheduling
- Stitch rules are after tile conversion, before downstream analysis

### Tests Added
None (Snakemake rules - tested via dry-run)

### Issues/Notes
Stitching requires OME-Zarr output format to be enabled (zarr in output_formats).
Rules will be skipped if stitch.enabled is false in config.

---

## [2026-02-02] - Auto-Detect Overlap from Metadata

### Changes
- File: `workflow/lib/preprocess/stitch.py`
  - Added: `detect_overlap_from_metadata()` - Auto-detect tile overlap from stage coordinates
  - Modified: `validate_stitch_config()` - Default overlap_pixels to "auto" instead of 150

- File: `tests/unit/preprocess/test_stitch.py`
  - Added: `TestDetectOverlapFromMetadata` (2 tests) - Overlap detection tests
  - Added: `test_overlap_auto_accepted()` - Test "auto" value accepted
  - Updated: `test_enabled_config_applies_defaults()` - Default is now "auto"

### Rationale
Auto-detection eliminates the need to manually specify overlap_pixels. The overlap is
calculated from stage coordinates: overlap = tile_size - distance_between_tiles.
Default reference_channel is already 0 (first channel) for both phenotype and SBS.

### Tests Added
- `TestDetectOverlapFromMetadata` (2 tests): Horizontal overlap, single tile fallback
- Updated config validation test for "auto" default

### Issues/Notes
All 38 tests pass. Auto-detection requires valid stage coordinates in metadata.

---

## [2026-02-02] - Run Stitching After Illumination Correction

### Changes
- File: `workflow/rules/preprocess.smk`
  - Modified: `estimate_stitch_phenotype` - Added `ic_field` input dependency
  - Modified: `estimate_stitch_sbs` - Added `ic_field` input dependency
  - Modified: `stitch_phenotype` - Added `ic_field` input, renamed `config` to `stitch_config`
  - Modified: `stitch_sbs` - Added `ic_field` input, renamed `config` to `stitch_config`

- File: `workflow/scripts/preprocess/stitch_tiles.py`
  - Modified: Use `snakemake.input.stitch_config` instead of `snakemake.input.config`

### Rationale
Stitching should run after illumination correction to ensure the complete
preprocessing pipeline is finished before generating stitched images. The IC
field is added as an explicit dependency to all stitch rules, creating the
following pipeline order:

1. Extract metadata
2. Combine metadata
3. Convert images (TIFF/Zarr/OME-Zarr)
4. Calculate illumination correction
5. **Estimate stitch positions** (now depends on IC)
6. **Assemble stitched images** (now depends on IC)

### Tests Added
None (Snakemake dependency change)

### Issues/Notes
The stitched images themselves are not IC-corrected (they use raw tiles).
This change ensures stitching runs after IC for pipeline ordering purposes.
Future enhancement: optionally apply IC before stitching.

---
