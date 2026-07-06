# Branch Overview: feat/preprocess/stitching

This document covers everything on the `feat/preprocess/stitching` branch -- what features are included, how to set up a new machine, configure each feature, and run the pipeline.

**Branch**: `feat/preprocess/stitching`
**Base**: `main` (commit `91b2458`)
**Scope**: 167 files changed, +29,153 / -3,676 lines vs main

---

## Setup on a New Machine

### 1. Clone and checkout

```bash
git clone https://github.com/petercla0119/Brieflow.git
cd Brieflow
git checkout feat/preprocess/stitching
```

### 2. Create the conda environment

Follow the standard installation in [docs/source/2.installation_analysis_setup.md](source/2.installation_analysis_setup.md), then:

```bash
conda activate brieflow_SCREEN_NAME
pip install -e .
```

The `stitch` dependency (GPU-accelerated stitching library) is fetched automatically from GitHub during `pip install`. No local clone needed.

### 3. GPU setup (optional, recommended)

GPU acceleration is supported for preprocessing stitching via CuPy. If a CUDA-capable GPU is available:

```bash
pip install cupy-cuda12x   # adjust for your CUDA version
```

If no GPU is available, stitching falls back to NumPy (CPU) automatically. No config change required.

### 4. Verify installation

```bash
python -c "from workflow.lib.preprocess.stitch import is_gpu_available; print('GPU:', is_gpu_available())"
python -c "from stitch.stitch.assemble import estimate_stitch; print('stitch library OK')"
```

---

## What's New on This Branch

| Feature | Module | Key Files | Opt-in? | Docs |
|---------|--------|-----------|---------|------|
| [Preprocessing stitching](#preprocessing-stitching) | preprocess | `lib/preprocess/stitch.py`, `rules/preprocess.smk` | Yes (`preprocess.stitch.enabled`) | [stitching.md](source/stitching.md) |
| [Merge stitching](#merge-stitching) | merge | `lib/merge/stitch.py`, `stitch_alignment.py`, `stitch_merge.py` | Yes (`merge.approach: stitch`) | [stitching.md](source/stitching.md) |
| [OME-Zarr exports](#ome-zarr-exports) | shared | `lib/shared/omezarr_writer.py`, `omezarr_io.py` | Yes (`output.omezarr.enabled`) | [zarr_support.md](source/zarr_support.md) |
| [Configurable TIFF/Zarr output](#configurable-tiffzarr-output) | preprocess | `targets/preprocess.smk`, `rules/preprocess.smk` | Default on | [config_glossary.md](source/config_glossary.md) |
| [Classifier module](#classifier-module) | classify | `lib/classify/` (6 files) | Yes (`classify:` config) | -- |
| [Bootstrap testing](#bootstrap-statistical-testing) | aggregate | `lib/aggregate/bootstrap.py` | Yes (`aggregate.num_sims`) | -- |
| [Perturbation scoring](#perturbation-scoring) | aggregate | `lib/aggregate/perturbation_score.py` | Yes (`aggregate.skip_perturbation_score`) | -- |

### Preprocessing Stitching

GPU-accelerated tile stitching that runs early in the pipeline (after illumination correction) to produce whole-well OME-Zarr images for QC and visualization.

**Pipeline position**: extract metadata -> combine metadata -> convert images -> calculate IC -> **estimate stitch** -> **assemble stitched images**

**Two estimation methods**:
- `phase_correlation` -- registers overlapping tile regions in frequency domain (slower, more accurate, GPU-accelerated)
- `coordinate_based` -- converts stage coordinates to pixel positions using metadata (fast, no image loading)

**Key components**:
- `workflow/lib/preprocess/stitch.py` -- adapter module (557 lines): GPU detection, config validation, overlap auto-detection, estimation functions, tile assembly
- `workflow/scripts/preprocess/estimate_stitch.py` -- Snakemake script for position estimation
- `workflow/scripts/preprocess/stitch_tiles.py` -- Snakemake script for tile assembly
- `workflow/rules/preprocess.smk` -- 4 conditional rules: `estimate_stitch_phenotype`, `estimate_stitch_sbs`, `stitch_phenotype`, `stitch_sbs`
- `workflow/targets/preprocess.smk` -- conditional target definitions

**Tests**: 38 unit tests in `tests/unit/preprocess/test_stitch.py`, integration tests in `tests/integration/stitch/test_stitch.py`

### Merge Stitching

Alternative merge approach that stitches tiles into whole-well mosaic images and uses Delaunay triangulation for spatial alignment between SBS and phenotype modalities.

**Pipeline**: estimate positions -> stitch tiles + extract cell positions -> align coordinate systems (triangle hashing) -> match cells across modalities -> format + deduplicate + finalize

**Key components**:
- `workflow/lib/merge/stitch.py` (1137 lines) -- TIFF-based well stitching: tile assembly with weighted blending, mask stitching with cell ID relabeling, cell position extraction, grid position estimation, connectivity graphs, least-squares tile position optimization via `dexp`
- `workflow/lib/merge/stitch_alignment.py` (902 lines) -- well-level alignment: auto-scale factor calculation, Delaunay triangle hashing (9-edge hash), RANSAC evaluation, adaptive regional alignment
- `workflow/lib/merge/stitch_merge.py` (680 lines) -- cell matching: distance-based matching with direct/chunked strategies, match validation, merge summary generation
- `workflow/rules/merge.smk` -- 7 stitch-specific rules: `estimate_stitch_phenotype`, `estimate_stitch_sbs`, `stitch_phenotype`, `stitch_sbs`, `stitch_alignment`, `stitch_merge`, `summarize_stitch`

### OME-Zarr Exports

Exports pipeline outputs to OME-Zarr v2 (NGFF v0.4) with multiscale pyramids for visualization in napari.

**Key components**:
- `workflow/lib/shared/omezarr_writer.py` -- `write_image_omezarr()`: writes arrays to OME-Zarr with pyramids, channel names, pixel size metadata
- `workflow/lib/shared/omezarr_io.py` -- I/O utilities for reading OME-Zarr stores
- Export scripts in `workflow/scripts/shared/` and `workflow/scripts/phenotype/`
- Visualization scripts: `load_omezarr_in_napari.py`, `load_omezarr_notebook.py`

**Available export points**: preprocess, sbs, phenotype, aggregate, cluster

**Tests**: `tests/unit/test_omezarr_writer.py`, `tests/unit/test_omezarr_writer_ngff_validation.py`, `tests/unit/test_omezarr_writer_scales.py`, `tests/integration/test_omezarr_exports.py`

### Configurable TIFF/Zarr Output

Preprocessing can output TIFF, Zarr, or both. Downstream modules can read from either format.

**Key parameters**:
- `preprocess.output_formats` -- list of `"tiff"`, `"zarr"`, or both (default: `["zarr"]`)
- `preprocess.downstream_input_format` -- which format SBS/phenotype modules read from (default: `"tiff"` if TIFF enabled, else `"zarr"`)

### Classifier Module

End-to-end cell classification system with ML model training, application, and calibration.

**Key components**:
- `workflow/lib/classify/train.py` (1964 lines) -- `SciKitCellClassifier` class supporting SVC, Random Forest, XGBoost, LightGBM, Logistic Regression; feature selection (variance, correlation, ANOVA); grid search; evaluation plots
- `workflow/lib/classify/apply.py` (1475 lines) -- `launch_rankline_ui()` interactive Jupyter UI for browsing classified cells; confidence thresholds with exclude/reassign modes; binary search for threshold discovery
- `workflow/lib/classify/shared.py` (567 lines) -- imaging/IO utilities for loading aligned TIFFs and masks
- `workflow/lib/classify/calibration.py` -- post-hoc confidence calibration (isotonic/sigmoid)
- `workflow/lib/classify/labeling.py` -- manual labeling tools and training data preparation
- `workflow/lib/classify/path_utils.py` -- parquet path resolution

Classification is applied during the `split_datasets` step of the aggregate module. Training and manual labeling are notebook-driven workflows.

### Bootstrap Statistical Testing

Bootstrap analysis for validating perturbation effects with FDR-corrected p-values.

**Key components**:
- `workflow/lib/aggregate/bootstrap.py` (443 lines) -- `run_construct_bootstrap()`: generates null distributions from control data; `create_pseudogene_groups()`: groups constructs into pseudo-genes; `apply_multiple_hypothesis_correction()`: Benjamini-Hochberg FDR correction

**Snakemake rules**: `prepare_bootstrap_data` (checkpoint), `bootstrap_construct` (parallelized per construct), `construct_bootstrap_complete`, `bootstrap_gene`, `combine_bootstrap`

### Perturbation Scoring

Per-cell perturbation effect scoring using out-of-fold logistic regression.

**Key components**:
- `workflow/lib/aggregate/perturbation_score.py` -- `perturbation_score()`: processes all perturbations in parallel with joblib; `calculate_perturbation_scores()`: 5/10-fold cross-validation with top-k feature selection

Integrated into the `align` rule. Disabled by default (`aggregate.skip_perturbation_score: true`).

---

## Configuration Quick Reference

### Minimal config to enable preprocessing stitching

```yaml
preprocess:
  output_formats: ["zarr"]
  stitch:
    enabled: true
```

All other stitch parameters have sensible defaults (method: phase_correlation, use_gpu: true, overlap_pixels: auto).

### Minimal config to enable merge stitching

```yaml
merge:
  approach: "stitch"
```

### Minimal config to enable OME-Zarr exports

```yaml
output:
  omezarr:
    enabled: true
    after_steps: ["sbs", "phenotype"]
```

### Full preprocessing stitch config

```yaml
preprocess:
  output_formats: ["zarr"]
  stitch:
    enabled: true
    method: "phase_correlation"      # or "coordinate_based"
    use_gpu: true
    overlap_pixels: "auto"           # or integer (e.g., 150)
    flipud: false
    fliplr: false
    rot90: 0
    blending_method: "edt"           # or "average"
    tile_size: [2048, 2048]
    pixel_size: 0.325
    phenotype:
      enabled: true
      reference_channel: 0
    sbs:
      enabled: true
      reference_cycle: 1
      reference_channel: 0
```

### Classifier config

```yaml
classify:
  classifier_path: config/multiclass_xgb_none_model.dill
  confidence_thresholds:
    1: { threshold: 0.5, mode: exclude }
    2: { threshold: 0.5, mode: exclude }
  class_mapping:
    label_to_class:
      1: Mitotic
      2: Interphase
```

### Bootstrap config

```yaml
aggregate:
  num_sims: 100
  control_key: nontargeting
  bootstrap_combinations:
    - cell_class: Interphase
      channel_combo: DAPI_COXIV_CENPA_WGA
```

For complete parameter references, see [config_glossary.md](source/config_glossary.md).

---

## Running the Pipeline

### Dry run (verify DAG)

```bash
snakemake --configfile config/config.yml --until all_preprocess -n
snakemake --configfile config/config.yml --until all_merge -n
snakemake --configfile config/config.yml --until all_aggregate -n
```

### Full run

```bash
# Preprocessing (includes stitching if enabled)
snakemake --configfile config/config.yml --until all_preprocess

# SBS + Phenotype
snakemake --configfile config/config.yml --until all_sbs
snakemake --configfile config/config.yml --until all_phenotype

# Merge (uses stitch approach if merge.approach: "stitch")
snakemake --configfile config/config.yml --until all_merge

# Aggregate (includes classifier, bootstrap, perturbation scoring)
snakemake --configfile config/config.yml --until all_aggregate

# Cluster
snakemake --configfile config/config.yml --until all_cluster
```

### GPU resources

Stitch assembly rules declare `gpu=1` as a Snakemake resource. On a Slurm cluster:

```bash
snakemake --configfile config/config.yml --until all_preprocess \
  --slurm --default-resources gpu=0 \
  --set-resources stitch_phenotype:gpu=1 stitch_sbs:gpu=1
```

### Running tests

```bash
# Unit tests
pytest tests/unit/ -m unit -v

# Stitch-specific unit tests
pytest tests/unit/preprocess/test_stitch.py -v

# OME-Zarr writer tests
pytest tests/unit/test_omezarr_writer.py tests/unit/test_omezarr_writer_ngff_validation.py -v

# Integration tests
pytest tests/integration/ -m integration -v
```

---

## Pipeline Architecture

```
                     +-----------+
                     | Raw ND2/  |
                     | TIFF data |
                     +-----+-----+
                           |
                    +------v-------+
                    |  Preprocess  |
                    |  - metadata  |
                    |  - convert   |
                    |  - IC calc   |
                    +------+-------+
                           |
              +------------+------------+
              |   (if stitch enabled)   |
              v                         v
     +-----------------+        +-------+-------+
     | Estimate Stitch |        |               |
     | Assemble Stitch |        |               |
     | (QC / viz)      |        |               |
     +-----------------+        |               |
                                v               v
                         +------+---+    +------+------+
                         |   SBS    |    |  Phenotype  |
                         | - align  |    | - align     |
                         | - reads  |    | - segment   |
                         | - cells  |    | - features  |
                         +------+---+    +------+------+
                                |               |
                         +------v---------------v------+
                         |          Merge              |
                         |  fast: hash-based matching  |
                         |  stitch: spatial stitching   |
                         +-------------+---------------+
                                       |
                         +-------------v---------------+
                         |        Aggregate            |
                         |  - classify (split)         |
                         |  - filter + impute          |
                         |  - align (batch correct)    |
                         |  - perturbation scoring     |
                         |  - aggregate to gene level  |
                         |  - bootstrap statistics     |
                         +-------------+---------------+
                                       |
                         +-------------v---------------+
                         |         Cluster             |
                         |  - Leiden clustering         |
                         |  - PHATE embedding          |
                         +-----------------------------+

  OME-Zarr exports can be enabled at: preprocess, sbs, phenotype,
  aggregate, and cluster stages via output.omezarr config.
```

---

## Output Directory Structure

```
brieflow_output/
├── preprocess/
│   ├── images/                          # TIFF or Zarr tiles
│   ├── omezarr/                         # OME-Zarr tiles (if zarr in output_formats)
│   ├── ic_fields/                       # Illumination correction fields
│   ├── metadata/                        # Extracted + combined metadata
│   ├── stitch_configs/                  # Tile position YAMLs (if stitch enabled)
│   │   ├── phenotype/
│   │   └── sbs/
│   └── stitched/                        # Stitched well images (if stitch enabled)
│       ├── phenotype/
│       └── sbs/
├── sbs/
│   ├── aligned/                         # Aligned SBS images
│   ├── reads/                           # Called reads
│   ├── cells/                           # Cell assignments
│   └── omezarr/                         # OME-Zarr exports (if enabled)
├── phenotype/
│   ├── aligned/                         # Aligned phenotype images
│   ├── masks/                           # Segmentation masks
│   ├── parquets/                        # Extracted features
│   └── omezarr/                         # OME-Zarr exports (if enabled)
├── merge/
│   ├── parquets/                        # Merge results
│   ├── stitch_configs/                  # Stitch configs (stitch approach)
│   ├── images/                          # Stitched images (stitch approach)
│   └── eval/                            # Merge evaluation summaries
├── aggregate/
│   ├── parquets/                        # Aggregated features
│   ├── bootstrap/                       # Bootstrap results
│   ├── montage/                         # Gene montages
│   └── zarr/                            # Zarr exports (if enabled)
└── cluster/
    ├── parquets/                        # Cluster assignments
    └── zarr/                            # Zarr exports (if enabled)
```

---

## Existing Documentation

| Document | Location | Contents |
|----------|----------|----------|
| Stitching User Guide | [docs/source/stitching.md](source/stitching.md) | Both preprocess and merge stitching: config reference, pipeline order, output files, estimation methods, troubleshooting |
| Zarr Support Guide | [docs/source/zarr_support.md](source/zarr_support.md) | Zarr/OME-Zarr config, output structure, napari visualization |
| Config Glossary | [docs/source/config_glossary.md](source/config_glossary.md) | All config parameters with examples |
| Stitching Changelog | [docs/CHANGELOG_STITCHING.md](CHANGELOG_STITCHING.md) | Step-by-step integration log with rationale and test counts |
| Installation Guide | [docs/source/2.installation_analysis_setup.md](source/2.installation_analysis_setup.md) | Environment setup, dependencies, test data |
| Running Modules | [docs/source/3.running_modules.md](source/3.running_modules.md) | Per-module execution guide with Slurm integration |

---

## Dependencies Added on This Branch

| Package | Version | Purpose |
|---------|---------|---------|
| `stitch` | `bd8500e` ([ahillsley/stitching](https://github.com/ahillsley/stitching)) | GPU-accelerated tile stitching (preprocess) |
| `iohub` | >= 0.2.0 | OME-Zarr I/O for stitching library |
| `ome-zarr` | >= 0.8.0 | OME-Zarr read/write |
| `zarr` | >= 2.16.0 | Zarr array storage |
| `dask[array]` | -- | Lazy array operations for large datasets |
| `dexp` | 2023.4.10.686 | Tile registration for merge stitching |

All dependencies are declared in `pyproject.toml` and installed automatically via `pip install -e .`.
