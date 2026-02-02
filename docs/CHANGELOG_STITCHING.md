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
