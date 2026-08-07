# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch: optimize-pheno (off zarr3)

This worktree is on `optimize-pheno`, branched from `zarr3`.
The zarr3 branch introduced polars-backed parquet I/O and zarr image support.
The optimize-pheno branch is where performance work on the phenotype/merge/aggregate pipeline is done.

## zarr3 Branch State

### What was done on zarr3 (already present here)

- **`workflow/lib/shared/parquet_io.py`** — new module with polars-backed fast I/O.
  Three public helpers: `read_parquet(path, columns=None)`, `write_parquet(df, path)`, `read_parquets(paths, columns=None)`.
  All accept/return pandas DataFrames; polars is used under the hood, with silent fallback to pandas if polars is absent.
  `polars[rtcompat]==1.39.3` is in `pyproject.toml`.

- **`workflow/scripts/phenotype/merge_phenotype.py`** — already uses joblib `Parallel(n_jobs=snakemake.threads)` to read per-tile TSVs concurrently, then `write_parquet` for the two well-level parquet outputs.

- **`workflow/scripts/merge/fast_merge.py`** — uses `read_parquet` / `write_parquet` from `parquet_io`.

- **zarr image mode** — `workflow/rules/aggregate.smk` has zarr-specific `generate_montage` / `initiate_montage` branches; `workflow/rules/merge.smk` uses `_merge_well_expand = ["row", "col"] if IMG_FMT == "zarr" else []`.

- **Dependencies added**: `zarr==3.1.6`, `ome-zarr==0.13.0`, `dask[array]==2026.1.1`, `polars[rtcompat]==1.39.3`, `iohub==0.3.0`.

### Current I/O patterns (as of this branch)

| Step | Script | Format |
|------|--------|--------|
| `extract_phenotype` (per tile) | `scripts/phenotype/extract_phenotype.py` | **TSV** (`.to_csv(sep="\t")`) |
| `merge_phenotype` (per well) | `scripts/phenotype/merge_phenotype.py` | reads TSV tiles → writes **parquet** (x2 outputs) |
| `fast_merge` | `scripts/merge/fast_merge.py` | reads parquet → writes **parquet** |
| `aggregate` (output) | `scripts/aggregate/aggregate.py` | reads parquet (pyarrow dataset) → writes **TSV** |
| most other aggregate scripts | various | mix; many still TSV outputs |

### Missing performance work (targets for optimize-pheno)

- **No Snakemake resource directives anywhere in phenotype.smk or merge.smk**: no `threads:`, `resources:`, `group:`, or `benchmark:` on any rule.
  `aggregate.smk` has `priority:` only (100 for main rules, 50 for montage).

- **`merge_phenotype` rule has no `threads:` directive**: the script calls `Parallel(n_jobs=snakemake.threads)` but the rule never sets `threads:`, so Snakemake passes 1 → joblib is single-threaded despite the parallel call.
  Fix: add `threads: <N>` to `rule merge_phenotype` in `workflow/rules/phenotype.smk`.

- **`extract_phenotype` still writes TSV**: per-tile phenotype files are the largest intermediate outputs; converting to parquet would cut I/O in `merge_phenotype` significantly.
  Key change: replace `phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")` with `write_parquet(phenotype_cp, snakemake.output[0])` in `scripts/phenotype/extract_phenotype.py`, and update `merge_phenotype.py` to use `read_parquets` instead of the TSV `get_file` path.

- **Feature extraction itself is not parallelized**: `workflow/lib/shared/feature_table_utils.py` loops over regionprops serially; `workflow/lib/phenotype/extract_phenotype_cp_emulator.py` (and `_cp_measure.py`) are the bottleneck callers. joblib parallelism at the tile level (one Snakemake job per tile) is the existing model; intra-tile parallelism could be added with joblib over channels/compartments.

- **No benchmark directives**: adding `benchmark:` to `extract_phenotype`, `merge_phenotype`, `fast_merge` would expose per-rule wall-time data to guide further optimization.

### Key files to edit for optimization

```
workflow/rules/phenotype.smk          # add threads/resources/benchmark to extract_phenotype, merge_phenotype
workflow/scripts/phenotype/extract_phenotype.py   # TSV → parquet output
workflow/scripts/phenotype/merge_phenotype.py     # switch from TSV read to read_parquets
workflow/lib/shared/parquet_io.py     # already done; extend if needed
workflow/lib/phenotype/extract_phenotype_cp_emulator.py  # intra-tile parallelism if needed
```

## What is Brieflow

Brieflow is a Snakemake-based computational pipeline for high-throughput analysis of optical pooled screening (OPS) data. It processes microscopy images through six sequential modules: preprocess, sbs (sequencing by synthesis), phenotype, merge, aggregate, and cluster. Python 3.11.

This repo contains the pipeline source code. A companion repo ([brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis)) holds user-facing configuration notebooks and execution scripts.

## Build & Install

Always use the `brieflow-150` conda env for all shell commands (build, test, lint, snakemake, etc.) and tmux sessions.

When launching tmux sessions, always source conda first so child processes (including Snakemake's `--use-conda`) can find it:

```bash
tmux new-session -d -s <name> -c <working-dir> \
  "source /Users/pmihack/miniforge3/etc/profile.d/conda.sh && conda activate brieflow-150 && <command>"

# If deps need reinstalling:
uv pip install -r pyproject.toml
uv pip install -e .
```

## Lint & Format

Ruff enforces Google-style docstrings (`select = ["D"]`) and formats code. CI runs `ruff check .` and `ruff format --check` on PRs.

```bash
ruff check workflow/           # lint
ruff check --fix workflow/     # auto-fix
ruff format workflow/          # format
```

Docstring rules are suppressed for `workflow/scripts/`, `tests/`, `visualization/`, notebooks, and `workflow/lib/external/`.

## Tests

Tests use pytest with `unit` and `integration` markers.

```bash
# Full integration test (downloads test data, runs full pipeline, then pytest)
cd tests/small_test_analysis/
python small_test_analysis_setup.py   # downloads ~100MB test data from Zenodo
sh run_brieflow.sh                    # runs Snakemake pipeline on test data
cd ../..
pytest                                # runs all tests (unit + integration)

# Run with zarr output instead of tiff
cd tests/small_test_analysis/
sh run_brieflow.sh --zarr

# Run a single test file or specific test
pytest tests/test_omezarr.py
pytest tests/test_omezarr.py::TestIORoundtrip::test_save_and_read_tiff_3d
pytest -m unit        # unit tests only
pytest -m integration # integration tests only (need prior pipeline run)
```

Integration tests (`@pytest.mark.integration`) require a prior `run_brieflow.sh` run to produce output artifacts. They skip gracefully if outputs are missing.

## Running

When re-running after a code-only edit that does not change outputs, pass
`--rerun-triggers mtime` to avoid Snakemake re-running rules purely because
params/code hashes changed. Use with care — it will NOT pick up logic changes.

## Architecture

### Pipeline structure (workflow/)

Each of the six modules follows a four-layer pattern:

```
workflow/
├── Snakefile           # Top-level: loads config, includes modules conditionally
├── targets/{module}.smk  # Output path definitions and wildcard expansions
├── rules/{module}.smk    # Snakemake rules (input/output/params/script)
├── scripts/{module}/     # Thin script wrappers called by rules
└── lib/{module}/         # Core library functions (business logic lives here)
```

**Edit order matters**: lib/ first, then scripts/, then rules/, then targets/. The Snakefile only needs updating when adding a new module.

### Modules (processing order)

1. **preprocess** - Image conversion (nd2/tiff/zarr), metadata extraction, illumination correction
2. **sbs** - Cycle alignment, base extraction, peak detection, barcode calling, cell assignment
3. **phenotype** - Cell segmentation (cellpose/stardist/watershed), feature extraction, cytoplasm identification
4. **merge** - Spatial alignment of SBS and phenotype data, stitching across tiles, deduplication
5. **aggregate** - Cell filtering, feature normalization, batch alignment, bootstrap analysis, perturbation scoring
6. **cluster** - PHATE dimensionality reduction, Leiden clustering, benchmark evaluation

### Module enable/disable

Each module can be disabled in config.yml without removing its section:

```yaml
sbs:
  enabled: false
```

Targets and rules are included separately -- targets load when the config section exists, rules load only when `{module}_rules_enabled` is true (default) and `enabled` is not false.

### Image format duality

The pipeline supports two output modes controlled by `all.image_format` in config:
- **tiff** (default): Traditional TIFF files, one per tile
- **zarr**: OME-Zarr with HCS (High Content Screening) plate layout, multiscale pyramids

In zarr mode, well identifiers (e.g., "A1") are split into row/col wildcards for HCS directory structure. The `image_io.py` module provides unified `read_image`/`save_image` functions that dispatch based on file extension.

### Shared library (workflow/lib/shared/)

Cross-module utilities: `file_utils.py` (path/filename generation), `target_utils.py` (wildcard expansion), `image_io.py` (TIFF/Zarr I/O), `segmentation_utils.py`, `feature_extraction.py`, `rule_utils.py` (parameter extraction for rules).

### Visualization (visualization/)

Streamlit app with pages for pipeline stats, QC, screen overview, and analysis overview. Entry point: `visualization/Cluster_Analysis.py`.

### Configuration

Pipeline config is a YAML file with sections per module. The test config lives at `tests/small_test_analysis/config/config.yml`. Combo files (TSV) define wildcard combinations (plate/well/tile/cycle) for Snakemake expansion.

### File naming convention

Data files follow: `P-{plate}_W-{well}_T-{tile}[_C-{cycle}]__{description}.{ext}` (generated by `get_filename` in `file_utils.py`).

## Test Data — Plate big1

### Experimental Setup

- Constitutive dCas9; d8 fixation; iNeurons
- **SBS**: 4-color chemistry (MiSeq), 8 cycles, 10x imaging on "Simone" setup in Blainey lab
- **Phenotype**: 5 rounds, 20x imaging on dual-camera setup in Ward lab at NIH

### Phenotype Rounds and Markers

| Round | DAPI | 488 | 568 | 647 |
|-------|------|-----|-----|-----|
| 1 | DAPI | pTDP43 | PGRN | HDGFL2-CE |
| 2 | DAPI | TGN46 | ~~TOMM20~~ (remove) | TDP43 |
| 3 | DAPI | Tuj1 | CytoC | LAMP1 |
| 4 | DAPI | PEX14 | NCAM1 | FUS |
| 5 | DAPI | G3BP1 | TOMM20 | — |

**Channel exclusions:**
- Round 2: remove 568 channel (TOMM20)
- Round 5: remove first blue channel, last red channel, and far red channel (568 and DAPI imaged separately on dual-camera)

### SBS Image Paths

```
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs01_260522
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs02_260524
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs03_260524
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs04_260525
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs05_260526
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs06_260526
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs07_260527
/Users/pmihack/claire/ops/data/260516_Blainey/big1/sbs08_260528
```

### Phenotype Image Paths

```
/Users/pmihack/claire/ops/data/260516_Blainey/big1/pheno1_260501_Omni1_d8_iNs_20X
/Users/pmihack/claire/ops/data/260516_Blainey/big1/pheno2_260503_Omni1_d8_iNs_20X
/Users/pmihack/claire/ops/data/260516_Blainey/big1/pheno3_260506_Omni1_d8_iNs_20X
/Users/pmihack/claire/ops/data/260516_Blainey/big1/pheno4_260508_Omni1_d8_iNs_20X
/Users/pmihack/claire/ops/data/260516_Blainey/big1/pheno5_260517_Omni1_d8_iNs_20X
```

## Conventions

- Google-style docstrings on library functions
- TSV for small readable dataframes, Parquet for large ones
- Semantic versioning -- version tracked in `pyproject.toml`
- One sentence per line in markdown files
