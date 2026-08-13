# optimize-pheno — Implementation Plan

Performance work on the phenotype / merge / aggregate pipeline.
Branch `optimize-pheno` off `zarr3`.
All line numbers below are from the current worktree state (read 2026-08-07).

Scope: Tier 1 and Tier 2 only. Tier 3 (feature-set profiling) is planned separately and not implemented here.

Ground rules for the implementing agent:
- `parquet_io.read_parquet / read_parquets / write_parquet` already exist and take/return pandas DataFrames. Use them; do not re-implement.
- `write_parquet` always writes a valid parquet with schema even for a 0-row DataFrame, so zero-cell tiles round-trip cleanly (no `EmptyDataError` guard needed, unlike TSV).
- Resource numbers (`threads`, `mem_mb`, `runtime`) below are starting estimates. They are calibration knobs — tune against real benchmark output on plate big1. Keep them as config-overridable comments where noted.
- Edit order per brieflow convention: lib/ → scripts/ → rules/ → targets/.

---

## TIER 1

### 1. Per-tile phenotype output: TSV → parquet  (PAIRED CHANGE — 3 files, do together)

This is a paired change across an output producer, its target path, and its sole consumer.
Consumer audit (grep `phenotype_cp` across `workflow/`): the **per-tile** `extract_phenotype` output is consumed **only** by `merge_phenotype`. The well-level `phenotype_cp.parquet` / `phenotype_cp_min.parquet` (written by `merge_phenotype`) are already parquet and consumed by `classify/`, `metrics.py`, `eval_merge.py`, `eval_features.py` — those are unaffected.

#### 1a. `workflow/scripts/phenotype/extract_phenotype.py`

Line 1 (add import alongside existing):
```python
from lib.shared.image_io import read_image
```
→
```python
from lib.shared.image_io import read_image
from lib.shared.parquet_io import write_parquet
```

Line 61 (last line):
```python
# Save phenotype cp
phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")
```
→
```python
# Save phenotype cp
write_parquet(phenotype_cp, snakemake.output[0])
```

#### 1b. `workflow/targets/phenotype.smk`

Line 47-49, `extract_phenotype` output — change extension `"tsv"` → `"parquet"` AND move it out of the `tsvs/` subdir into `parquets/` for consistency with `merge_phenotype`:
```python
    "extract_phenotype": [
        PHENOTYPE_FP / "tsvs" / get_data_output_path(_tile, "phenotype_cp", "tsv", PHENOTYPE_IMG_FMT),
    ],
```
→
```python
    "extract_phenotype": [
        PHENOTYPE_FP / "parquets" / get_data_output_path(_tile, "phenotype_cp", "parquet", PHENOTYPE_IMG_FMT),
    ],
```
Note: `get_data_output_path` only uses `file_type` for the extension (TIFF branch → `get_filename`; zarr branch → `get_nested_path`), so changing the 3rd arg is sufficient and works in both image formats. The subdir string (`"tsvs"` vs `"parquets"`) is purely the on-disk folder; changing it is optional but recommended. If you keep it in `tsvs/`, the file will still be a `.parquet` inside a `tsvs/` folder — harmless but misleading. Prefer `parquets/`.

#### 1c. `workflow/scripts/phenotype/merge_phenotype.py`

Current (lines 1-25) uses a joblib `Parallel` loop over `pd.read_csv(sep="\t")` via `get_file`. Replace the TSV read path with `read_parquets` (polars scans the whole file list in parallel internally; the joblib layer becomes redundant).

Lines 1-25:
```python
import pandas as pd
from joblib import Parallel, delayed

from lib.shared.parquet_io import write_parquet


# Validate required params
if getattr(snakemake.params, "channel_names", None) is None:
    raise ValueError("Required config parameter 'channel_names' is not set")


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Load, concatenate, and save the phenotype CellProfiler data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(get_file)(file) for file in snakemake.input
)
phenotype_cp = pd.concat(arr_reads)
write_parquet(phenotype_cp, snakemake.output[0])
```
→
```python
from lib.shared.parquet_io import read_parquets, write_parquet


# Validate required params
if getattr(snakemake.params, "channel_names", None) is None:
    raise ValueError("Required config parameter 'channel_names' is not set")


# Load and concatenate the per-tile phenotype CellProfiler parquets.
# read_parquets uses polars scan+concat with a pandas fallback on schema
# mismatch across tiles; it returns an empty DataFrame for an empty input list.
phenotype_cp = read_parquets(list(snakemake.input))
write_parquet(phenotype_cp, snakemake.output[0])
```
The rest of the file (lines 27-53: min-feature subset + second `write_parquet`) is unchanged. The `pd` import is no longer needed in this file; remove it (the subset selection uses only DataFrame indexing, no top-level `pd` call). Verify with `ruff check` — it will flag an unused import if any remains.

**Edge cases handled:** empty input list → empty DataFrame (then the `phenotype_cp[...]` subset on line 52 will raise `KeyError` on missing columns — but this path only occurs if a well has zero tiles, which cannot happen given wildcard expansion, so no guard needed). Zero-cell tiles → valid empty-but-schema'd parquet, concatenated fine. Schema mismatch across tiles → `read_parquets` falls back to per-file pandas reads + `pd.concat` (see `parquet_io.py` line 74-77).

**Tests for change 1:**

- Unit — `tests/unit/test_phenotype_parquet_roundtrip.py`:
  - `test_extract_merge_roundtrip_preserves_data`: build 2 synthetic per-tile DataFrames (e.g. columns `plate, well, tile, label, cell_i, cell_j, cell_DAPI_min, cell_bounds_0..3` + a few float feature cols), `write_parquet` each to `tmp_path`, then `read_parquets([p1, p2])`; assert row count == sum of inputs, column set preserved, and float dtypes preserved (`df["cell_DAPI_min"].dtype == float`).
  - `test_zero_cell_tile_roundtrips`: `write_parquet(pd.DataFrame(columns=[...]), path)`; `read_parquets([path])` returns empty DataFrame with same columns (asserts no `EmptyDataError`).
  - `test_single_cell_tile`: one-row DataFrame round-trips with identical values.
  - `test_schema_mismatch_falls_back`: two parquets where the second has an extra column; `read_parquets` returns a concatenation (union of columns, NaN-filled) rather than raising — asserts the fallback path works.
- Integration — `tests/integration/test_phenotype_io.py::test_extract_phenotype_output_is_parquet`:
  - Resolve the brieflow output dir (copy `_resolve_output_dir` pattern from `tests/test_omezarr.py` lines 359-380). `pytest.skip` if absent.
  - Glob `phenotype/parquets/**/*__phenotype_cp.parquet` (per-tile files). Assert ≥1 exists and that **no** `phenotype/tsvs/**/*__phenotype_cp.tsv` remain (confirms the migration; guards against a stale target path).
  - `read_parquet` one tile file; assert it has a `label` column and at least one `*_min` column. Mark `@pytest.mark.integration`.

**Verification without full pipeline:**
`cd tests/small_test_analysis && snakemake -s ../../workflow/Snakefile -n <extract_phenotype target>` (dry run) and confirm the printed output path ends in `.parquet` under `parquets/`. Then run just the phenotype module on the small test data and confirm `merge_phenotype` consumes the parquets (no "MissingInputException").

---

### 2. `group: "phenotype_tile"` across the per-tile chain

File: `workflow/rules/phenotype.smk`. Add `group: "phenotype_tile"` to the five rules that form the per-tile dependency chain so Snakemake schedules them as a single group job per tile (cuts scheduler/submit overhead on SLURM; keeps intermediate images local to one job).

Rules to annotate (add the directive after `output:`/`params:`, before `script:`):
- `apply_ic_field_phenotype` (line 6)
- `align_phenotype` (line 17)
- `segment_phenotype` (line 29)
- `identify_cytoplasm` (line 41)
- `extract_phenotype` (line 82)
- `extract_phenotype_info` (line 56) — same per-tile granularity; include it.

Example for `extract_phenotype` (lines 82-100), insert one line before `script:`:
```python
    params:
        ...
        segment_cells=config.get("phenotype", {}).get("segment_cells", True),
    group:
        "phenotype_tile"
    script:
        "../scripts/phenotype/extract_phenotype.py"
```

Do **not** group `merge_phenotype` (line 104) — it is a per-well fan-in and must stay a separate job.

**Caveat / risk:** grouping only helps when running with a cluster/group-jobs executor (`--groups` / `--group-components`). On a single local run it is a no-op-to-slightly-helpful. In tiff mode the intermediate `apply_ic_field_phenotype` output is `temp()` (targets line 69-73); grouping keeps temp files within the job and is compatible. In zarr mode outputs are `directory()` — grouping is still valid.

**Test:** no unit test (pure scheduler directive). Verification: `snakemake -n --groups phenotype_tile=1 ...` dry run parses without error and reports grouped jobs. Add a note to CLAUDE.md rather than a pytest test.

---

### 3. `threads:` + `resources:` on heavy rules

File: `workflow/rules/phenotype.smk` (and `merge.smk` for `fast_merge`). Add directives. Values are starting points — leave a `# tune:` comment.

`segment_phenotype` (line 29, cellpose/stardist — heaviest):
```python
    threads: 4
    resources:
        mem_mb=16000,   # tune: cellpose model + image
        runtime=60,     # minutes
```

`extract_phenotype` (line 82, feature extraction — heavy, CPU-bound regionprops):
```python
    threads: 4
    resources:
        mem_mb=8000,    # tune: widest table in memory
        runtime=30,
```

`merge_phenotype` (line 104, per-well concat of all tile parquets):
```python
    threads: 4
    resources:
        mem_mb=8000,    # tune: holds full well phenotype_cp
        runtime=20,
```

`fast_merge` in `workflow/rules/merge.smk` (rule at line 54):
```python
    threads: 1
    resources:
        mem_mb=8000,
        runtime=20,
```

**Test:** none (declarative). Verification: `snakemake -n` still builds the DAG; on SLURM, confirm `sacct` shows the reserved memory. No pytest.

---

### 4. joblib `n_jobs` alignment with `snakemake.threads`

After change 1c, `merge_phenotype` no longer uses joblib, so its `n_jobs` concern is resolved. The remaining in-scope `n_jobs=-1` that ignores Snakemake's core budget is in `workflow/lib/aggregate/filter.py` line 235 (called from the `filter` rule).

`workflow/lib/aggregate/filter.py` line 235:
```python
        n_jobs=-1,
```
This is inside a library function. Two options, pick the lazy one that fits the call site:
- If the `filter` rule sets `threads:`, thread that value through: add a `n_jobs` param to the enclosing function (check the function signature around line 235) defaulting to `-1`, pass `snakemake.threads` from `workflow/scripts/aggregate/filter.py`, and add `threads: 4` to the `filter` rule (`workflow/rules/aggregate.smk` line 30).
- If that is too invasive for the payoff, leave `filter.py` as-is and only add `threads:` where joblib already reads `snakemake.threads`. Ponytail: prefer this unless filter oversubscription is actually observed.

**Test:** if you thread `n_jobs` through, add `tests/unit/test_filter_njobs.py::test_njobs_defaults_to_all` asserting the function default is unchanged (`-1`) and that an explicit value is respected (call with `n_jobs=2`, assert no crash / correct shape on a tiny synthetic frame). Otherwise no test.

---

### 5. `--rerun-triggers mtime` note (docs only)

File: `CLAUDE.md`. Add under the "Tests" or a new "Running" section:
```
When re-running after a code-only edit that does not change outputs, pass
`--rerun-triggers mtime` to avoid Snakemake re-running rules purely because
params/code hashes changed. Use with care — it will NOT pick up logic changes.
```
No code, no test. Pure documentation.

---

## TIER 2

### 6. polars `group_by().median()` in the gene/construct aggregation  (combine with #8)

The heavy Python-level group loops are in `workflow/scripts/aggregate/generate_feature_table.py` (construct-level accumulation, lines 73-197) and `workflow/lib/aggregate/aggregate.py` (`aggregate()` loop, lines 70-96). Item 6 and item 8 target the same file (`generate_feature_table.py`) and are best done as one rewrite — see #8. The `lib/aggregate/aggregate.py` loop (used by the `aggregate` rule) is a smaller, separate win; treat it as optional:

Optional `aggregate.py` lib vectorization (lines 70-96): for the common path where `ps_probability_threshold is None and ps_percentile_threshold is None`, the loop reduces to a group-wise mean/median + cell_count + first(`perturbation_auc`). Replace with a single pandas `groupby(pert_col).agg(...)` (already available, no new dep) or polars. Ponytail: pandas `groupby` here is one call and keeps the numpy `embeddings` alignment logic simpler than converting to polars — prefer pandas `groupby` unless profiling says otherwise. Keep the existing loop for the thresholded path.
- **Test:** `tests/unit/test_aggregate_median.py::test_median_matches_loop` — synthetic embeddings (10×3) + metadata with 3 perturbations, call `aggregate(..., method="median")`, assert output equals a hand-computed `np.median` per group and `cell_count` correct. Run before AND after the change (it is a characterization test that must stay green).

### 7. Vectorize the `fast_merge.py` inner loop

File: `workflow/scripts/merge/fast_merge.py`, lines 36-54. The loop re-filters the full `phenotype_info` and `sbs_info` DataFrames with a boolean mask **on every alignment row** → O(n_alignments × n_cells). Pre-group once into dicts.

Lines 36-54:
```python
# Merge cells across well
merge_data = []
for index, alignment_row in fast_alignment_filtered.iterrows():
    # Determine tiles and sites for merging
    phenotype_tile = alignment_row["tile"]
    sbs_site = alignment_row["site"]

    # Filter phenotype and sbs info to the relevant well and tile for merging
    phenotype_info_filtered = phenotype_info[phenotype_info["tile"] == phenotype_tile]
    sbs_info_filtered = sbs_info[sbs_info["tile"] == sbs_site]

    # Merge cells for row of alignment data
    alignment_row_merge = merge_triangle_hash(
        phenotype_info_filtered,
        sbs_info_filtered,
        alignment_row,
        threshold=snakemake.params.threshold,
    )
    merge_data.append(alignment_row_merge)
```
→
```python
# Pre-group cell tables by tile once (O(n) instead of O(n_alignments * n_cells)).
phenotype_by_tile = dict(tuple(phenotype_info.groupby("tile")))
sbs_by_tile = dict(tuple(sbs_info.groupby("tile")))
_empty_ph = phenotype_info.iloc[0:0]
_empty_sbs = sbs_info.iloc[0:0]

# Merge cells across well
merge_data = []
for _index, alignment_row in fast_alignment_filtered.iterrows():
    phenotype_tile = alignment_row["tile"]
    sbs_site = alignment_row["site"]

    phenotype_info_filtered = phenotype_by_tile.get(phenotype_tile, _empty_ph)
    sbs_info_filtered = sbs_by_tile.get(sbs_site, _empty_sbs)

    alignment_row_merge = merge_triangle_hash(
        phenotype_info_filtered,
        sbs_info_filtered,
        alignment_row,
        threshold=snakemake.params.threshold,
    )
    merge_data.append(alignment_row_merge)
```
`merge_triangle_hash` → `merge_sbs_phenotype` already returns a correctly-columned empty DataFrame for empty input (see `workflow/lib/merge/fast_merge.py` lines 84-92), so a missing tile key is safe.

**Behavioral equivalence note:** `groupby(...).get_group`-style dict lookup returns the same rows as the boolean mask, in the same order, so `merge_data` is byte-identical to the old path. This is a pure speedup, not a science change.

**Tests:**
- Unit — `tests/unit/test_fast_merge_grouping.py::test_groupby_dict_matches_boolean_mask`: build a synthetic `phenotype_info` with `tile` in {0,1,2} and cells per tile; assert `dict(tuple(df.groupby("tile"))).get(1)` equals `df[df["tile"]==1]` (use `assert_frame_equal`). Add `test_missing_tile_returns_empty`: `.get(99, df.iloc[0:0])` is empty with correct columns.
- Integration — `tests/integration/test_fast_merge.py::test_merge_output_stable`: skip unless `merge/**/*__merged.parquet` (or the actual `fast_merge` output name) exists; `read_parquet` it and assert expected columns present (`cell_0, cell_1, distance, tile, site`). This guards the refactor against column drift end-to-end.

**Verification:** run `fast_merge` on the small test data before and after; diff the output parquet (`read_parquet(...).equals(...)`) — must be identical.

### 8. Fix aggregate memory hoard in `generate_feature_table.py`

File: `workflow/scripts/aggregate/generate_feature_table.py`.

Current design accumulates **every cell's feature vector** in Python dicts across all batches (`construct_feature_values`, lines 79, 152-157) then `np.vstack` per construct (line 174) to compute a median. This holds the entire aligned dataset in RAM twice (once written to parquet, once in the dict). For median you need all values, but they are already being written to `aligned_output` — read them back and let polars compute the grouped median lazily instead of hoarding in Python.

Change plan:
1. Remove the accumulator init (lines 74-80) and the per-batch accumulation block (lines 144-161: the `for construct_id in metadata[pert_id_col].unique(): ...` loop). Keep the parquet writer loop (lines 128-142) and `del metadata, features` cleanup.
2. After `writer.close()` (line 165), build the construct table from the just-written parquet with polars:
```python
import polars as pl

# Construct-level medians computed lazily from the aligned parquet
# (avoids holding every cell's feature vector in memory).
lf = pl.scan_parquet(aligned_output)
construct_agg = (
    lf.group_by(pert_id_col)
    .agg(
        [pl.first(pert_col).alias(pert_col), pl.len().alias("cell_count")]
        + [pl.median(c).alias(c) for c in feature_cols]
    )
    .collect()
    .to_pandas()
)
construct_table = construct_agg[[pert_id_col, pert_col, "cell_count"] + feature_cols]
```
   This replaces lines 168-197 (the `construct_rows` build + `del construct_feature_values` + reorder). Everything downstream (gene-level table lines 199-237, pseudogene logic lines 239-344, output writes 346-354) consumes `construct_table` and is unchanged.
3. `cell_count` from `pl.len()` is an int matching the old `mask.sum()` sum-across-batches (each cell counted once). Verify dtype: cast if a downstream `.sum()`/comparison needs int (`construct_table["cell_count"] = construct_table["cell_count"].astype(int)`).

**Numeric caveat:** the old code medians the **float32** center-scaled `features` array directly; the new code medians the float32 values as written to parquet (same values). Median of the same numbers is identical. But confirm `feature_cols` order matches (it does — parquet columns were written from `pd.DataFrame(features, columns=feature_cols)`, line 131).

**Risk:** this is the single riskiest change — it rewrites the core construct aggregation and its output feeds bootstrap + montage. Guard with a characterization test comparing old vs new on the small test dataset.

**Tests:**
- Unit — `tests/unit/test_construct_median.py::test_polars_median_matches_numpy`: write a synthetic aligned parquet (columns: `sgRNA_id`, `gene`, 5 feature cols, ~30 rows across 4 constructs) to `tmp_path`; compute construct medians two ways — the new polars `group_by(pert_id_col).median()` and a reference `pandas groupby(pert_id_col)[feature_cols].median()` + gene `first` + `len` count; assert equal with `assert_frame_equal(check_dtype=False)` after aligning column order. Covers the median correctness in isolation.
  - Edge: `test_single_cell_construct` — a construct with one cell → median == that cell's values, count == 1.
  - Edge: `test_all_constructs_present` — no construct dropped vs input unique ids.
- Integration — `tests/integration/test_generate_feature_table.py::test_construct_table_stable`: skip unless a prior aggregate run produced `aggregate/**/*construct*.tsv`; read it and assert schema (`sgRNA` col, `gene` col, `cell_count`, feature cols) and that `cell_count.sum()` is positive. If you can preserve a pre-change copy, add an exact-value diff; otherwise schema + row-count check.

**Verification:** run the `generate_feature_table` rule on small test data before the change (save the two output TSVs), apply the change, re-run, and `pandas`-diff the construct and gene tables — feature medians must match to float32 precision, row counts identical.

### 9. `benchmark:` directives on heavy rules

Files: `workflow/rules/phenotype.smk`, `workflow/rules/merge.smk`. Add `benchmark:` to expose per-rule wall time / peak RSS. Path pattern mirrors existing output naming.

`extract_phenotype` (line 82) — add before `script:`:
```python
    benchmark:
        PHENOTYPE_FP / "benchmarks" / get_data_output_path(_tile, "extract_phenotype", "tsv", PHENOTYPE_IMG_FMT)
```
`segment_phenotype` (line 29):
```python
    benchmark:
        PHENOTYPE_FP / "benchmarks" / get_data_output_path(_tile, "segment_phenotype", "tsv", PHENOTYPE_IMG_FMT)
```
`merge_phenotype` (line 104):
```python
    benchmark:
        PHENOTYPE_FP / "benchmarks" / get_data_output_path(_well, "merge_phenotype", "tsv", PHENOTYPE_IMG_FMT)
```
`fast_merge` (`merge.smk` line 54) — use `MERGE_OUTPUTS` FP root:
```python
    benchmark:
        MERGE_FP / "benchmarks" / get_data_output_path({"plate": "{plate}", "well": "{well}"}, "fast_merge", "tsv")
```
(Confirm the FP variable name in `merge.smk` / `targets/merge.smk`; use whatever the module's root path constant is. If `get_data_output_path` is not imported in `merge.smk`, import it or build the path with an f-string.)

Snakemake `benchmark:` requires a single non-wildcard-conflicting file per rule instance; the per-tile/well wildcards above satisfy that. Benchmark files are plain TSV and are not tracked as rule outputs by consumers.

**Test:** none (declarative). Verification: `snakemake -n` builds; after a real run, `phenotype/benchmarks/*.tsv` contain the `s`/`max_rss` columns.

---

## Test strategy

**Layout (create these dirs — currently all tests live flat in `tests/`):**
- `tests/unit/` — pure-logic tests, no pipeline artifacts, run anywhere. Fast.
- `tests/integration/` — read artifacts from a prior `run_brieflow.sh` run; `pytest.skip` when absent (reuse the `_resolve_output_dir()` helper pattern from `tests/test_omezarr.py` lines 359-380).

**Markers:** `@pytest.mark.unit` / `@pytest.mark.integration` (declared in `pyproject.toml` lines 88-90). Unit tests may be left unmarked or marked `unit`; integration tests MUST be marked `integration` so `pytest -m "not integration"` skips them in CI without pipeline data.

**Fixtures/data:**
- Unit: build tiny synthetic DataFrames inline (no pytest fixtures for the simple cases — just `def test_*` with a few lines constructing data, per project preference). Use `tmp_path` (built-in) for parquet round-trip tests.
- Integration: real outputs under `tests/small_test_analysis/brieflow_output/` produced by `sh run_brieflow.sh`. No new fixtures needed beyond the `_resolve_output_dir` helper.

**Which run without pipeline data (unit):** 1 (roundtrip), 6 (aggregate median), 7 (groupby equivalence), 8 (construct median). These are the correctness-critical ones and gate the risky changes.

**Which need artifacts (integration):** 1 (parquet-on-disk migration check), 7 (merge output schema), 8 (construct table schema/stability). All skip gracefully.

**Edge cases to cover explicitly (spread across the unit tests above):**
- Empty DataFrame / zero-cell tile → `write_parquet`+`read_parquets` round-trip, no `EmptyDataError`.
- Single-cell / single-construct tile → median == the value, count == 1.
- Schema mismatch across wells/tiles → `read_parquets` pandas fallback returns union-of-columns concat.
- Missing tile key in `fast_merge` grouping → empty, correctly-columned frame.

**Running:**
```bash
conda activate brieflow-150
pytest tests/unit -m unit           # fast, no data
pytest tests/integration -m integration   # after run_brieflow.sh
ruff check workflow/ && ruff format --check workflow/
```

---

## Ordering

**Sequential (must be done as a unit, in this order):**
1. Change **1a + 1b + 1c together** (extract_phenotype write → target path ext → merge_phenotype read). Partial application breaks the DAG (`MissingInputException`) or feeds TSV bytes into a parquet reader. Write the change-1 unit tests first (they need no pipeline), then apply, then run the phenotype module on small test data to confirm end-to-end.

**Independent of each other and of #1 (any order, additive):**
2. `group:` directives (#2) — pure scheduler metadata.
3. `threads:` / `resources:` (#3) — declarative.
4. `n_jobs`/threads alignment (#4) — only if you choose the threading option; otherwise skip.
5. benchmark directives (#9) — declarative.
7. `fast_merge` vectorization (#7) — self-contained in one script; write unit + integration tests, verify byte-identical output.

**Independent but risk-gated (do after its characterization test is green):**
6. `aggregate.py` lib vectorization (#6, optional) — land the characterization test first.
8. `generate_feature_table.py` rewrite (#8) — riskiest. Land the unit median test AND capture a pre-change output snapshot from small test data before editing; diff after.

Recommended sequence: tests-for-#1 → #1 → #3 → #2 → #9 (get Tier-1 wins + benchmarks visible) → #7 → #8 → #6 → #4.

---

## Top 3 riskiest changes

1. **#8 `generate_feature_table.py` construct-median rewrite** — replaces the core per-construct aggregation; output feeds bootstrap and montage. Median must match old float32 values exactly; `cell_count` dtype and construct completeness must be preserved. Gate with the unit median test + a before/after diff on small test data.
2. **#1 TSV→parquet paired change** — three files must land together or the DAG breaks; polars vs pandas type coercion on the widest table risks silent dtype drift (e.g. `fix_uint16` int columns) that propagates into `classify/` and `metrics.py`. Verify column dtypes in the round-trip unit test and confirm no stale `tsvs/*.tsv` remain.
3. **#2 `group: "phenotype_tile"`** — grouping semantics interact with `temp()` (tiff mode) and `directory()` (zarr mode) outputs and only pay off under a group-aware executor; misapplied grouping can serialize memory-heavy tile jobs into one process. Validate with `snakemake -n --groups` dry runs in both image formats before trusting it on the cluster.
