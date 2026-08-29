# Implementation Plan — Bidirectional TIFF ↔ Zarr Converter for Brieflow Output

Branch: `feat/bidirectional-tiff-zarr-converter` (off `tdp-gws-integration`)
Worktree: `/mnt/work/broad-analysis/brieflow-tiff-zarr-converter`

## 1. Overview

Brieflow can emit image arrays as either flat TIFFs (`image_format: tiff`) or
OME-NGFF v0.5 / Zarr v3 HCS plate stores (`image_format: zarr`). This tool
converts **already-generated** output between the two layouts, **in both
directions**, without re-running the pipeline.

Scope is **image arrays only**: SBS images, phenotype images, IC fields,
aligned, illumination_corrected, and label masks (nuclei / cells /
identified_cytoplasms). Tabular (`tsv`/`parquet`), `eval`, and `png` outputs are
left untouched — they are not copied and not converted.

Core insight: the pixel conversion is already solved by the shared I/O layer.
The tool is essentially:

```python
save_image(read_image(src), dst, is_label=..., channel_names=..., pixel_size=...)
```

The real engineering is (a) enumerating image files/stores in a source tree,
(b) mapping each source path to its destination path in the *other* layout, and
(c) supplying `is_label` / `channel_names` / `pixel_size` so the written store is
faithful. Path mapping is **also mostly already solved** — `get_image_output_path`
in `file_utils.py` produces the exact zarr and tiff relative paths — so the tool
reuses it for everything except IC fields, which have no builder and get an
explicit rule.

### Verified facts (checked on `broad-cpu`, not assumed)

- `read_image(path)` / `save_image(image, output_path, *, pixel_size, channel_names, coarsening_factor=2, max_levels=1, is_label=False)` / `write_image_omezarr(...)` signatures are as documented in the brief. `save_image` dispatches on suffix; `.zarr` anywhere in the path (or a `zarr.json` sentinel name) routes to zarr; `.tif/.tiff` writes a raw `tifffile.imwrite` with no metadata/compression.
- `get_image_output_path(data_location, info_type, img_fmt, subdirectory=None, image_subdir=None)` and `get_hcs_nested_path(...)` build the HCS nested paths, **including the cycle level** when `data_location` contains a `cycle` key, and the `labels/{name}` nesting when `subdirectory="labels"`.
- `parse_filename(path)` returns `(metadata_dict, info_type, file_type)` where metadata keys are `plate`/`well`/`tile`/`cycle` as present in the name.
- SBS zarr store: `preprocess/sbs/image_1.zarr/{row}/{col}/{tile}/{cycle}/0`. Confirmed the tile group (e.g. `A/1/0/`) has **no `zarr.json`**; each cycle (`A/1/0/1/`) is its own image group with a `zarr.json` + level-0 array `0`. One SBS TIFF = one (tile, cycle) 5-channel array.
- Phenotype zarr store: `preprocess/phenotype/image_1.zarr/{row}/{col}/{tile}/0` — image directly at the tile group, no cycle level.
- Aligned + labels: `phenotype/aligned_1.zarr/{row}/{col}/{tile}/0` for the aligned image, labels nested at `phenotype/aligned_1.zarr/{row}/{col}/{tile}/labels/{nuclei|cells|identified_cytoplasms}`. `illumination_corrected` is a **separate** store `phenotype/illumination_corrected_1.zarr/...`.
- A real level-0 `zarr.json` is OME v0.5, `multiscales[0].datasets[0].path == "0"`, single level (no pyramid), axes TCZYX, `omero.channels` labeled generically `c0..c4` — i.e. the **native zarr does not carry real channel names**; generic `c{i}` labels are faithful.
- IC fields have **no path builder** in `file_utils.py` (grep found none). Layouts observed directly:
  - TIFF sbs: `preprocess/ic_fields/sbs/P-{plate}_W-{well}_C-{cycle}__ic_field.tiff` (C,Y,X) float64
  - TIFF phenotype: `preprocess/ic_fields/phenotype/P-{plate}_W-{well}__ic_field.tiff` (C,Y,X) float64
  - ZARR sbs: `preprocess/ic_fields/sbs/{plate}/{row}/{col}/{cycle}/ic_field.zarr`
  - ZARR phenotype: `preprocess/ic_fields/phenotype/{plate}/{row}/{col}/ic_field.zarr`

## 2. Path / layout mapping rules

`well` (e.g. `A1`) splits to `row=A`, `col=1` (regex `^([A-Za-z]+)(\d+)$`, same as `split_well_to_cols`). `data_location` dicts below carry `plate`, `well` (or `row`/`col`), `tile`, and `cycle` where applicable.

### 2.1 Reusable via `get_image_output_path` (everything except IC fields)

For a source TIFF, `parse_filename` yields `(meta, info_type, _)`. Combine with the category → (`image_subdir`, `subdirectory`, module-root prefix) table below, then call `get_image_output_path(meta, info_type, img_fmt="zarr", image_subdir=..., subdirectory=...)` to get the zarr destination (and `img_fmt="tiff"` for the reverse). The returned path is relative; prepend the module root.

| Category | TIFF path | ZARR path | `image_subdir` | `subdirectory` | `info_type` |
|---|---|---|---|---|---|
| SBS preprocess image | `preprocess/images/sbs/P-{p}_W-{w}_T-{t}_C-{c}__image.tiff` | `preprocess/sbs/image_{p}.zarr/{row}/{col}/{t}/{c}/0` | `preprocess/sbs` | — | `image` |
| Phenotype preprocess image | `preprocess/images/phenotype/P-{p}_W-{w}_T-{t}__image.tiff` | `preprocess/phenotype/image_{p}.zarr/{row}/{col}/{t}/0` | `preprocess/phenotype` | — | `image` |
| Phenotype aligned | `phenotype/images/P-{p}_W-{w}_T-{t}__aligned.tiff` | `phenotype/aligned_{p}.zarr/{row}/{col}/{t}/0` | `phenotype` | — | `aligned` |
| Phenotype illum. corrected | `phenotype/images/P-{p}_W-{w}_T-{t}__illumination_corrected.tiff` | `phenotype/illumination_corrected_{p}.zarr/{row}/{col}/{t}/0` | `phenotype` | — | `illumination_corrected` |
| Label: nuclei | `phenotype/images/P-{p}_W-{w}_T-{t}__nuclei.tiff` | `phenotype/aligned_{p}.zarr/{row}/{col}/{t}/labels/nuclei` | `phenotype` | `labels` | `nuclei` |
| Label: cells | `...__cells.tiff` | `.../labels/cells` | `phenotype` | `labels` | `cells` |
| Label: cytoplasm | `...__identified_cytoplasms.tiff` | `.../labels/identified_cytoplasms` | `phenotype` | `labels` | `identified_cytoplasms` |

Notes:
- The SBS `cycle` key MUST be present in `data_location` for the cycle level to appear (`get_hcs_nested_path` appends it). Phenotype images have no cycle.
- `image_subdir` for zarr becomes the **top-level directory**; for TIFF the builder returns `images/{image_subdir}/...`, so the module-root prefix differs by direction:
  - preprocess TIFFs live under `preprocess/` (builder already returns `images/sbs/...`, so prefix `preprocess`).
  - phenotype TIFFs live under `phenotype/` (builder returns `images/...`; note the phenotype `image_subdir` on the *tiff* side is empty — verify by round-trip; if the builder emits `images/phenotype/...` for phenotype, strip to match the observed `phenotype/images/P-...` layout). **This is the one path-mapping detail to confirm empirically in Phase 1 by running the mapping on one file each way and diffing against the real tree.**
- Zarr destination uses the `zarr.json` sentinel that `get_image_output_path` returns for image stores; `save_image` strips `zarr.json` and writes the store at the parent. Labels return a directory path (no sentinel) — `save_image` writes the nested label group.

### 2.2 Explicit rule for IC fields (no builder)

```
sbs:        P-{p}_W-{w}_C-{c}__ic_field.tiff  <->  ic_fields/sbs/{p}/{row}/{col}/{c}/ic_field.zarr
phenotype:  P-{p}_W-{w}__ic_field.tiff        <->  ic_fields/phenotype/{p}/{row}/{col}/ic_field.zarr
```

Both roots are under `preprocess/`. This is a small dedicated function in the tool (≈10 lines each direction); do **not** try to force it through `get_image_output_path`.

### 2.3 SBS cycle-nesting and label-nesting (the special cases)

- **SBS cycle-nesting**: enumerate per (tile, cycle). Each SBS TIFF maps to a distinct cycle sub-group array. When going zarr→tiff, walk `image_{p}.zarr/{row}/{col}/{tile}/{cycle}/` (one array group per cycle), not the tile group.
- **Label-nesting**: labels live *inside* the aligned store, not as standalone `.zarr`. `read_image` on `.../labels/nuclei` opens it as a group and reads level-0. When writing (tiff→zarr) the aligned store must exist first OR the label write must create the nested group; sequence aligned before its labels within a well/tile (or let `save_image`/`write_image_omezarr` create the group — verify write order in Phase 2, since `write_image_omezarr` opens the group with `mode="w"` which would clobber a sibling; label writes must target the nested subpath, not the store root).

## 3. Metadata fidelity limitations (state honestly)

- **TIFFs carry no metadata.** `save_image` writes raw `tifffile.imwrite(path, image)` — no channel names, no pixel size, no axes. So **zarr→tiff loses all OME metadata** (expected; the native tiff tree has none either).
- **tiff→zarr cannot recover real channel names or pixel size** from the source, because they were never in the tiff. The native zarr itself uses generic `c0..c4` labels and unit-scale pixel sizes (verified in the real `zarr.json`), so writing generic channel names + `pixel_size=None` is **faithful to the native zarr**. Do not fabricate biological channel names.
- If channel names/pixel size are ever wanted, they must come from the run `config.yml` (`config["all"]`), not the tiff — out of scope; leave a `ponytail:` note.
- `is_label=True` MUST be set for nuclei/cells/identified_cytoplasms so the label group is written under `labels/` with integer-preserving settings; label dtype is uint32 (verify preserved through round-trip).

## 4. Converter location + CLI

**Location**: `workflow/scripts/convert_images.py` — a standalone CLI script (not a Snakemake rule; this operates on finished output, outside the DAG). It imports the shared helpers:

```python
from lib.shared.image_io import read_image, save_image
from lib.shared.file_utils import get_image_output_path, parse_filename, split_well_to_cols
```

(match the import style already used by scripts under `workflow/scripts/`; verify the `sys.path`/package rooting they use.)

**CLI** (argparse, stdlib):

```
python workflow/scripts/convert_images.py \
    --direction {tiff2zarr,zarr2tiff} \
    --src  <source brieflow_output[_zarr] root> \
    --dst  <destination root (created)> \
    [--categories sbs_pp,pheno_pp,aligned,illum,labels,ic_sbs,ic_pheno]  # default: all \
    [--dry-run]   # print planned src->dst mappings, convert nothing \
    [--jobs N]    # optional parallelism; default 1
```

`--dry-run` prints the mapping table so path logic can be eyeballed before writing. Keep it single-process by default (lazy); add `--jobs` only if the small-test run is too slow.

**File discovery without broad finds**: never `find /mnt`. Discovery is scoped to the known category subdirectories under `--src`:
- tiff→zarr: iterate `preprocess/images/{sbs,phenotype}/*.tiff`, `phenotype/images/*.tiff`, `preprocess/ic_fields/{sbs,phenotype}/*.tiff` via `pathlib.Path.glob` on those specific dirs.
- zarr→tiff: iterate the specific stores (`preprocess/{sbs,phenotype}/image_*.zarr`, `phenotype/aligned_*.zarr`, `phenotype/illumination_corrected_*.zarr`, `preprocess/ic_fields/**/ic_field.zarr`); walk `{row}/{col}/{tile}[/{cycle}]` groups by reading directory entries (integer-named dirs), and the `labels/` subgroup for masks. Use `Path.iterdir()` / scoped `glob`, never a recursive `find` from a broad root.

## 5. Phased implementation

1. **Phase 0 — scaffold**: CLI skeleton, category registry (the table in §2.1 + IC rule in §2.2), `--dry-run` mapping only. No pixel writes.
2. **Phase 1 — path mapping, both directions**: implement `tiff_to_zarr_path` (reuse `get_image_output_path`) and `zarr_to_tiff_path` (inverse). Validate with `--dry-run` against the real small-test trees: for a handful of files, assert the computed dst path actually exists in the native other-format tree. **This nails down the phenotype `image_subdir` ambiguity in §2.1.**
3. **Phase 2 — pixel conversion**: wire `save_image(read_image(src), dst, is_label=...)`. Handle write ordering for labels (aligned store before its nested labels) and confirm the nested label write does not clobber the aligned array (inspect `write_image_omezarr` group `mode`). Confirm SBS per-cycle arrays and IC float64 dtype survive round-trip.
4. **Phase 3 — validation test** (§6).
5. **Phase 4 — full small-test conversion** both directions end to end; run the validation test; document runtime.

Each phase leaves the tool runnable. Non-trivial path logic gets a `__main__` self-check or the pytest in §6 — no extra scaffolding.

## 6. Validation policy + how to run

Native reference trees (do not modify):
- TIFF: `/mnt/work/broad-analysis/brieflow-small-test-zarr3/tests/small_test_analysis/brieflow_output`
- ZARR: `/mnt/work/broad-analysis/brieflow-small-test-zarr3/tests/small_test_analysis/brieflow_output_zarr`

Validation converts one tree and compares to the **native** other tree, per the IC-nondeterminism caveat: `brieflow_output` and `brieflow_output_zarr` came from two separate runs, and `calculate_ic_field` uses **unseeded** 5% tile sampling, so IC fields and everything derived from them differ 2–5% per channel between the two native trees. Comparison policy by image type:

| Image type | Comparison vs native other-tree | Rationale |
|---|---|---|
| SBS preprocess `image` (raw) | **Exact** `np.array_equal` | Pre-IC raw acquisition; identical across runs. |
| Phenotype preprocess `image` (raw) | **Exact** `np.array_equal` | Same — raw, pre-IC. |
| `ic_field` (sbs + phenotype) | **Round-trip only** (see below); cross-tree = shape/dtype/axes only | Unseeded sampling → values differ across the two native runs. |
| `illumination_corrected` | Round-trip; cross-tree structural only | IC-derived. |
| `aligned` | Round-trip; cross-tree structural only | IC-derived. |
| Labels (nuclei/cells/cytoplasm) | Round-trip; cross-tree structural only | Segmentation of aligned → IC-derived. |

- **Exact cross-tree**: convert native TIFF→zarr (and zarr→TIFF), then for raw images assert `np.array_equal(converted_level0, native_other_tree_level0)` after axis-squeezing (zarr TCZYX singleton T,Z removed to natural shape, matching `read_image`).
- **Structural cross-tree** (IC-derived): assert equal `shape`, `dtype`, and — for zarr — axes order / v0.5 metadata presence (`multiscales`, single level, `datasets[0].path == "0"`). Do **not** assert value equality.
- **Round-trip self-consistency** (the real correctness check for IC-derived): `read_image(src) -> save_image(tmp) -> read_image(tmp)`; assert `np.array_equal` (lossless codec: zstd level 0, no quantization; float64 and uint32 must survive). Do both tiff→zarr→tiff and zarr→tiff→zarr round trips on a couple of IC-derived files. Round-trip is what proves the converter is faithful; cross-tree exact is only meaningful where the two native trees agree (raw images).

**Test file**: add `tests/test_convert_images.py` **beside** `test_omezarr.py` (do not extend it — its unit fixtures are Zarr v2 / NGFF v0.4 and its integration paths don't match real output). Parametrize over a small sample of files per category (1–2 each) discovered from the small-test trees; skip cleanly if the trees are absent.

**Run**:
```bash
ssh broad-cpu 'cd /mnt/work/broad-analysis/brieflow-tiff-zarr-converter && \
  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate brieflow-150 && \
  python -m pytest tests/test_convert_images.py -v'
```
(Env: `source /opt/miniforge3/etc/profile.d/conda.sh && conda activate brieflow-150` on broad-cpu. Note: broad-cpu has BOTH `brieflow-150` (/opt/miniforge3, use this) and `brieflow-150-gpu` (/mnt/work/miniforge3, GPU VMs only).)

## 7. Open questions / risks

1. **Phenotype `image_subdir` on the TIFF side** (§2.1) — whether `get_image_output_path(img_fmt="tiff", image_subdir="phenotype")` emits `images/phenotype/...` vs the observed `phenotype/images/P-...`. Resolve empirically in Phase 1 `--dry-run`; may need a small per-category root prefix instead of relying on the builder for the tiff side.
2. **Label write ordering / clobbering** — `write_image_omezarr` opens the store root with `zarr.open_group(mode="w")`. Writing a nested `labels/nuclei` group must target the subpath and must not reopen/clobber the aligned array at the store root. Verify the actual write path a label takes through `save_image` in Phase 2 before trusting it.
4. **`read_image` on nested label groups** — confirmed `read_image` resolves level-0 via `multiscales[0].datasets[0].path` with `"0"` fallback; verify it works when pointed at `.../labels/nuclei` (a nested group, not a `.zarr`-suffixed dir) rather than only at top-level stores.
5. **IC field zarr axes** — IC fields are (C,Y,X) with no tile; confirm `save_image` expands them to TCZYX correctly and that the native IC zarr uses the same axes, so structural comparison is apples-to-apples.
6. **Multi-plate / multi-well generality** — rules derived from a single-plate small test (plate 1, well A1/A2). Path builders are plate/well-parametric so this should generalize, but only the small test is validated here.

<!-- ponytail: converter is ~save_image(read_image(src), dst) + a path table; no new deps, no abstraction layer. Channel-name/pixel-size recovery from config skipped — native zarr uses generic labels anyway; add only if a real consumer needs named channels. -->

---

## Status (Phases 0-4 complete, 2026-08-28)

Implemented `workflow/scripts/convert_images.py` + `tests/test_convert_images.py`. All work + testing on **broad-cpu** (pure conversion, no GPU needed), env **`brieflow-150`** at `/opt/miniforge3`.

**Verified on the small-test trees (all bit-exact unless noted):**
- Path mapping both directions: 126 <-> 126, 100% resolve to existing native paths.
- Full conversion both directions (126 files each); tiff2zarr ~48s.
- Raw preprocess images (sbs/phenotype): converted == source AND == native other-format tree, bit-exact.
- Labels (nuclei/cells/identified_cytoplasms) uint32 and IC fields float64: lossless round-trip, bit-exact.
- pytest `tests/test_convert_images.py`: 8 passed.

**Open questions resolved:**
1. Phenotype path layout -- resolved: per-category `(module_root, image_subdir, subdirectory)` through the real `get_image_output_path` reproduces `phenotype/images/...` and `preprocess/images/phenotype/...` exactly; no prefix-stripping needed.
2. Label write ordering / clobbering -- resolved: writing nested `labels/{name}` does NOT clobber the aligned array at the store root; both readable and bit-exact after. Aligned is converted before its labels (category order).

**New gotcha found & handled:** `_well_to_rowcol` in `file_utils.py` hardcodes literal `{row}`/`{col}` (it builds Snakemake *templates*, not values). The converter splits the well itself and passes real `row`/`col` keys to `get_image_output_path`.

**Deferred (not needed for lossless conversion):** channel-name / pixel-size recovery on tiff->zarr (never present in tiff; native zarr uses generic c0..c4 anyway). A benign tifffile "RGB planes" DeprecationWarning on multi-channel writes mirrors brieflow's own raw imwrite and does not affect bit-exactness.
