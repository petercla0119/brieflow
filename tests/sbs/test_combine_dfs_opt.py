"""Phase-1 SBS combine/eval refactor: prove output is byte-identical to the old logic.

Covers commit 361f0f5:
  * ``combine_tile_dfs`` (workflow/lib/shared/combine_dfs.py) vs a self-contained
    reference implementation of the ORIGINAL per-tile TSV concat + dtype-normalization.
  * eval read-projection identity: the projected ``read_parquets(..., columns=[...])``
    frames feed the summary-producing eval functions with results identical to the
    full frames.

POLARS FAST PATH: combine_tile_dfs reads via pl.scan_csv per file + a
diagonal_relaxed concat (pl.read_csv rejects a list of paths in polars 1.39.x).
This path is LIVE -- test_polars_fast_path_runs asserts it actually executes.
Because polars is correctly-rounded and the old pandas-DEFAULT CSV parser is
not, output matches the old logic within rtol=1e-9 on float columns and byte-
exactly elsewhere (Option A; see _assert_identical and combine_dfs.py).
Validated both ways on real plate-4 data 2026-08-28.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from lib.shared.combine_dfs import combine_tile_dfs, _HAS_POLARS
from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import write_parquet
from lib.sbs.eval_mapping import (
    mapping_overview,
    plot_cell_mapping_heatmap,
    plot_mapping_vs_threshold,
)
from lib.shared.eval_segmentation import plot_cell_density_heatmap

SBS_BASE = Path(
    "/mnt/work/broad-analysis/broad-tdp-gws/analysis/brieflow_output/sbs"
)
TSV_WELL = SBS_BASE / "tsvs" / "4" / "A" / "1"  # plate 4, well A1
GOLDEN_WELL = SBS_BASE / "parquets" / "4" / "A" / "1"

# tile 0 is header-only (7-col, no `plate`); non-empty tiles are 8-col (with
# `plate`). Mixing them exercises the pandas fallback + column-union path, which
# is what the real plate-4 combine actually does.
INTEG_TILES = [0, 2, 32, 33, 34]


# ─── reference (ORIGINAL, pre-refactor) combine logic ───────────────────────
def _reference_combine(paths):
    """Byte-for-byte reproduction of the pre-refactor combine logic.

    Matches the old ``--step combine`` block of run_sbs_direct.py:
      per-file ``pd.read_csv`` (skip empty/unreadable) ->
      ``pd.concat(ignore_index=True)`` -> ``validate_dtypes`` ->
      95%-threshold numeric coercion of any leftover object columns.
    """
    dfs = []
    for f in paths:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception:
            pass
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined = validate_dtypes(combined)
    for col in combined.select_dtypes(include="object").columns:
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
            combined[col] = converted
    return combined


def _assert_identical(new, ref, float_rtol=1e-9):
    """new ~= ref under the Phase-1 Option-A contract: same columns and dtypes,
    non-float columns byte-identical, float columns equal within rtol=1e-9.

    polars parses CSV floats correctly-rounded (== pandas float_precision=
    "round_trip"); the old pandas-DEFAULT parser is imprecise, so they differ by
    <~2e-12 on high-precision float columns. Validated both ways on real plate-4
    data 2026-08-28 (round_trip==polars EXACT; default-vs-polars max |diff|
    1.8e-12 reads / 9.1e-13 cells / 2.3e-13 sbs_info).
    """
    assert list(new.columns) == list(ref.columns), (
        f"column mismatch: {list(new.columns)} vs {list(ref.columns)}"
    )
    assert new.dtypes.equals(ref.dtypes), (
        f"dtype mismatch:\nnew:\n{new.dtypes}\nref:\n{ref.dtypes}"
    )
    fcols = [c for c in new.columns if pd.api.types.is_float_dtype(new[c])]
    nfcols = [c for c in new.columns if c not in fcols]
    assert new[nfcols].reset_index(drop=True).equals(
        ref[nfcols].reset_index(drop=True)
    ), "non-float value mismatch between combine_tile_dfs and reference"
    for c in fcols:
        x = new[c].astype("float64").to_numpy()
        y = ref[c].astype("float64").to_numpy()
        assert np.allclose(x, y, rtol=float_rtol, atol=0.0, equal_nan=True), (
            f"float column {c} differs beyond rtol={float_rtol}"
        )


def _write_tsv(path, header, rows):
    lines = ["\t".join(str(h) for h in header)]
    for r in rows:
        lines.append("\t".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n")


# ─── unit tests: synthetic fixtures ─────────────────────────────────────────
@pytest.mark.unit
def test_unit_consistent_schema(tmp_path):
    """Multiple tiles, identical schema -> combine == reference."""
    hdr = ["well", "tile", "cell", "area", "i", "bounds"]
    p1 = tmp_path / "t1.tsv"
    _write_tsv(
        p1,
        hdr,
        [
            ["A1", 2, 1, 69.0, 10.5, "(1,2,3,4)"],
            ["A1", 2, 2, 70.0, 11.5, "(5,6,7,8)"],
        ],
    )
    p2 = tmp_path / "t2.tsv"
    _write_tsv(p2, hdr, [["A1", 3, 1, 80.0, 12.5, "(9,1,2,3)"]])
    paths = [str(p1), str(p2)]
    _assert_identical(combine_tile_dfs(paths), _reference_combine(paths))


@pytest.mark.unit
def test_unit_empty_and_schema_mismatch(tmp_path):
    """Header-only + 0-byte + a schema-mismatch (extra col) mix.

    The 0-byte file is skipped (EmptyDataError); the header-only 0-row file is
    concatenated (contributes columns, no rows); the extra `plate` column forces
    the column-union path. combine must equal the reference exactly.
    """
    hdr7 = ["well", "tile", "cell", "area", "i", "bounds"]
    hdr8 = hdr7 + ["plate"]
    header_only = tmp_path / "empty_hdr.tsv"
    _write_tsv(header_only, hdr7, [])
    zero = tmp_path / "zero.tsv"
    zero.write_text("")  # 0-byte -> EmptyDataError -> skipped
    d1 = tmp_path / "d1.tsv"
    _write_tsv(d1, hdr8, [["A1", 2, 1, 69.0, 10.5, "(1,2,3,4)", 4]])
    d2 = tmp_path / "d2.tsv"
    _write_tsv(d2, hdr8, [["A1", 3, 1, 80.0, 12.5, "(9,1,2,3)", 4]])
    paths = [str(zero), str(header_only), str(d1), str(d2)]
    new = combine_tile_dfs(paths)
    _assert_identical(new, _reference_combine(paths))
    # column union kept `plate`; header-only-first ordering appends it last.
    assert "plate" in new.columns
    assert list(new.columns)[-1] == "plate"
    assert len(new) == 2


@pytest.mark.unit
def test_unit_numeric_coercion_threshold(tmp_path):
    """A >=95%-numeric object col and a <95%-numeric one normalize identically.

    (In this env validate_dtypes converts object -> StringDtype, so the explicit
    95% loop is effectively a no-op -- but both new and reference share that loop,
    so their results are identical either way, which is what we assert.)
    """
    n = 20
    hdr = ["well", "tile", "no_recomb_0", "note"]
    rows = []
    for i in range(n - 1):
        rows.append(["A1", 2, i, i if i % 2 else f"s{i}"])
    # one non-numeric value -> no_recomb_0 is 19/20 = 95% numeric-coercible
    rows.append(["A1", 2, "missing", "s99"])
    p = tmp_path / "a.tsv"
    _write_tsv(p, hdr, rows)
    paths = [str(p)]
    _assert_identical(combine_tile_dfs(paths), _reference_combine(paths))


@pytest.mark.unit
def test_unit_all_empty_returns_none(tmp_path):
    """All-empty (0-byte) input, and an empty path list, both -> None."""
    z1 = tmp_path / "z1.tsv"
    z1.write_text("")
    z2 = tmp_path / "z2.tsv"
    z2.write_text("")
    assert combine_tile_dfs([str(z1), str(z2)]) is None
    assert _reference_combine([str(z1), str(z2)]) is None
    assert combine_tile_dfs([]) is None


@pytest.mark.unit
def test_polars_list_read_unsupported(tmp_path):
    """pl.read_csv rejects a *list* of paths in this polars version.

    This is WHY combine_tile_dfs uses pl.scan_csv per file + diagonal_relaxed
    concat instead of read_csv(list). The fast path is live and separately
    asserted by test_polars_fast_path_runs.
    """
    if not _HAS_POLARS:
        pytest.skip("polars not installed")
    import polars as pl

    hdr = ["well", "tile", "cell"]
    p1 = tmp_path / "t1.tsv"
    _write_tsv(p1, hdr, [["A1", 2, 1]])
    p2 = tmp_path / "t2.tsv"
    _write_tsv(p2, hdr, [["A1", 3, 1]])
    # single path works ...
    assert pl.read_csv(str(p1), separator="\t").height == 1
    # ... but a list raises, so combine_tile_dfs never takes the fast path.
    with pytest.raises(Exception):
        pl.read_csv([str(p1), str(p2)], separator="\t")


# ─── integration: real plate-4 subset (gentle) ─────────────────────────────
@pytest.mark.unit
def test_polars_fast_path_runs(tmp_path):
    """The polars fast path (scan_csv + diagonal_relaxed) actually executes.

    Guards against silent regression to the pandas fallback: on a consistent-
    schema multi-file read, combine_tile_dfs must increment _READ_STATS["polars"]
    and take no fallback.
    """
    if not _HAS_POLARS:
        pytest.skip("polars not installed")
    from lib.shared import combine_dfs as _cd

    hdr = ["well", "tile", "cell", "val"]
    p1 = tmp_path / "t1.tsv"
    _write_tsv(p1, hdr, [["A1", 2, 1, 10.533333333333333]])
    p2 = tmp_path / "t2.tsv"
    _write_tsv(p2, hdr, [["A1", 3, 1, 20.700000000000003]])
    _cd._READ_STATS["polars"] = 0
    _cd._READ_STATS["pandas_fallback"] = 0
    out = _cd.combine_tile_dfs([str(p1), str(p2)])
    assert out is not None
    assert _cd._READ_STATS["polars"] == 1, _cd._READ_STATS
    assert _cd._READ_STATS["pandas_fallback"] == 0, _cd._READ_STATS


@pytest.mark.unit
def test_fallback_corrupt_tile_raises(tmp_path, monkeypatch):
    """Pandas fallback must FAIL LOUD on a non-EmptyDataError read error.

    Regression guard: the fallback previously swallowed all exceptions,
    silently dropping corrupt/truncated tiles and writing an incomplete
    combined parquet. Only EmptyDataError (empty tiles) may be skipped; any
    other read error (e.g. ParserError) must propagate and fail the rule.
    """
    from lib.shared import combine_dfs as _cd
    monkeypatch.setattr(_cd, "_HAS_POLARS", False)  # force pandas fallback
    good = tmp_path / "good.tsv"
    _write_tsv(good, ["well", "tile", "cell"], [["A1", 0, 1]])
    bad = tmp_path / "bad.tsv"
    _write_tsv(bad, ["well", "tile", "cell"], [["A1", 1, 1]])
    real_read = _cd.pd.read_csv
    def _fake(path, *a, **k):
        if str(path) == str(bad):
            raise pd.errors.ParserError("Error tokenizing data")
        return real_read(path, *a, **k)
    monkeypatch.setattr(_cd.pd, "read_csv", _fake)
    with pytest.raises(pd.errors.ParserError):
        _cd.combine_tile_dfs([str(good), str(bad)])


@pytest.mark.unit
def test_fallback_skips_empty_tile(tmp_path, monkeypatch):
    """Pandas fallback still skips genuinely empty tiles (EmptyDataError)."""
    from lib.shared import combine_dfs as _cd
    monkeypatch.setattr(_cd, "_HAS_POLARS", False)
    good = tmp_path / "good.tsv"
    _write_tsv(good, ["well", "tile", "cell"], [["A1", 0, 1]])
    empty = tmp_path / "empty.tsv"
    empty.write_text("")  # 0-byte -> EmptyDataError -> skipped
    out = _cd.combine_tile_dfs([str(good), str(empty)])
    assert out is not None and len(out) == 1


def _subset_paths(info):
    return [
        str(TSV_WELL / str(t) / f"{info}.tsv")
        for t in INTEG_TILES
        if (TSV_WELL / str(t) / f"{info}.tsv").exists()
    ]


@pytest.mark.integration
@pytest.mark.skipif(not TSV_WELL.exists(), reason="plate-4 tsvs unavailable")
@pytest.mark.parametrize("info", ["reads", "cells", "sbs_info"])
def test_integration_subset_matches_reference(info):
    """Real plate-4 well A1, ~5 tiles: combine == reference for every info type."""
    paths = _subset_paths(info)
    assert len(paths) >= 3, f"expected >=3 tiles for {info}, got {len(paths)}"
    _assert_identical(combine_tile_dfs(paths), _reference_combine(paths))


# ─── golden on-disk comparison ─────────────────────────────────────────────
@pytest.mark.integration
@pytest.mark.skipif(
    not (GOLDEN_WELL / "sbs_info.parquet").exists() or not TSV_WELL.exists(),
    reason="plate-4 sbs_info golden / tsvs unavailable",
)
def test_golden_sbs_info_full_well(tmp_path):
    """Full-well (all 333 tiles) sbs_info: match self + on-disk golden parquet.

    sbs_info is the smallest info type (~25MB / 1.37M rows) so combining the
    whole well is gentle. reads (~656MB) and cells (~271MB) full-well golden are
    deferred (see test_golden_reads_cells_full_well_deferred).
    """
    tiles = sorted(int(d.name) for d in TSV_WELL.iterdir() if d.name.isdigit())
    paths = [str(TSV_WELL / str(t) / "sbs_info.tsv") for t in tiles]
    combined = combine_tile_dfs(paths)
    # (a) new == old reference on the full real well (provenance-independent).
    _assert_identical(combined, _reference_combine(paths))
    # (b) corroborate against the on-disk golden via an identical write+read
    #     round-trip (write_parquet is the pipeline's final step).
    out = tmp_path / "sbs_info.parquet"
    write_parquet(combined, str(out))
    got = pd.read_parquet(out)
    golden = pd.read_parquet(GOLDEN_WELL / "sbs_info.parquet")
    assert list(got.columns) == list(golden.columns), (
        f"golden column mismatch: {list(got.columns)} vs {list(golden.columns)}"
    )
    _assert_identical(got, golden)


@pytest.mark.integration
@pytest.mark.skip(
    reason="DEFERRED: reads.parquet ~656MB / cells.parquet ~271MB -- combining all "
    "333 tiles is not gentle while sharing the box with production combine jobs. "
    "test_golden_sbs_info_full_well provides the full-well on-disk golden match; "
    "test_integration_subset_matches_reference covers reads+cells on a subset. "
    "Run this once the box is idle to complete the reads/cells full-well golden."
)
def test_golden_reads_cells_full_well_deferred():
    pass


# ─── eval column-projection identity ───────────────────────────────────────
@pytest.fixture(scope="module")
def small_frames():
    if not TSV_WELL.exists():
        pytest.skip("plate-4 tsvs unavailable")
    frames = {
        info: combine_tile_dfs(_subset_paths(info))
        for info in ("reads", "cells", "sbs_info")
    }
    # barcode list only needs to be identical on both (full vs projected) sides.
    frames["barcodes"] = list(frames["reads"]["barcode"].dropna().unique())[:50]
    return frames


@pytest.mark.integration
def test_projection_segmentation_cells(small_frames):
    """eval_segmentation cells -> [well, tile]: density summary unchanged."""
    cells = small_frames["cells"]
    full, _ = plot_cell_density_heatmap(cells.copy(), metadata=None)
    proj, _ = plot_cell_density_heatmap(cells[["well", "tile"]].copy(), metadata=None)
    assert full.equals(proj)


@pytest.mark.integration
def test_projection_mapping_sbs_info(small_frames):
    """eval_mapping sbs_info -> [well, tile, cell]: summaries unchanged."""
    cells = small_frames["cells"]
    sbs = small_frames["sbs_info"]
    barcodes = small_frames["barcodes"]
    cols = ["well", "tile", "cell"]
    for mt in ("one", "any"):
        full = plot_cell_mapping_heatmap(
            cells.copy(), sbs.copy(), barcodes,
            mapping_to=mt, mapping_strategy="gene symbols",
            return_plot=False, return_summary=True,
        )
        proj = plot_cell_mapping_heatmap(
            cells.copy(), sbs[cols].copy(), barcodes,
            mapping_to=mt, mapping_strategy="gene symbols",
            return_plot=False, return_summary=True,
        )
        assert full.equals(proj), f"cell mapping summary differs for mapping_to={mt}"
    full_ov = mapping_overview(sbs.copy(), cells.copy(), sort_by="peak")
    proj_ov = mapping_overview(sbs[cols].copy(), cells.copy(), sort_by="peak")
    assert full_ov.equals(proj_ov), "mapping_overview summary differs under sbs_info projection"


@pytest.mark.integration
def test_projection_mapping_reads(small_frames):
    """eval_mapping reads -> [cell, well, tile, barcode, Q_min, peak]: summary unchanged."""
    reads = small_frames["reads"]
    cols = ["cell", "well", "tile", "barcode", "Q_min", "peak"]
    proj = reads[cols]
    for var in ("peak", "Q_min"):
        full_s, _ = plot_mapping_vs_threshold(reads.copy(), small_frames["barcodes"], var, num_thresholds=5)
        proj_s, _ = plot_mapping_vs_threshold(proj.copy(), small_frames["barcodes"], var, num_thresholds=5)
        assert full_s.equals(proj_s), f"reads projection changes plot_mapping_vs_threshold summary for {var}"
