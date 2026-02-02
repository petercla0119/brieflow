# Stitching Integration Changelog

This document tracks all changes made during the integration of GPU-accelerated stitching into Brieflow's preprocessing module.

## Overview

**Goal**: Integrate the GPU-accelerated stitching library from `/Users/cspeters/projects/ops/stitching/stitch/` into Brieflow's preprocessing module.

**Branch**: `feat/preprocess/stitching`

**Key Design Decisions**:
1. Import stitching package as dependency (not copy)
2. Pipeline position: After tile conversion, before illumination correction
3. Complement existing merge stitching (preprocessing stitching for early visualization/QC)
4. Optional via config (disabled by default)

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
