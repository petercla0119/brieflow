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


def _norm_nullable(df):
    """Downcast pandas nullable Int64/Float64 to numpy int64/float64 (no nulls).

    A 0-row header-only TSV tile is read as object columns, which
    ``validate_dtypes`` upcasts to nullable ``Int64``/``Float64``. Parquet tiles
    are typed, so the parquet fast path yields plain ``int64``/``float64``. The
    VALUES are identical and the difference vanishes on parquet write (the
    shipped artifact is int64 either way -- see test_golden_sbs_info_full_well,
    where combine->write->read matches the on-disk golden exactly). Normalize so
    the dtype comparison reflects real output rather than the transient artifact.
    """
    out = df.copy()
    for c in out.columns:
        dt = str(out[c].dtype)
        if dt == "Int64" and out[c].notna().all():
            out[c] = out[c].astype("int64")
        elif dt == "Float64":
            out[c] = out[c].astype("float64")
    return out


def _parse_bounds(v):
    """Bounds as a list of ints from either the new parquet list/array or the
    old garbled TSV string ``"(a, b, c, d)"``."""
    if isinstance(v, str):
        inner = v.strip().lstrip("([").rstrip(")]")
        return [int(x) for x in inner.replace(" ", "").split(",") if x != ""]
    return [int(x) for x in v]


def _assert_parquet_matches_tsv(new, ref):
    """Parquet-mode combine vs the original TSV reference, under the migrated
    contract: same columns/order, same numbers, dtypes equal after nullable
    normalization, and ``bounds`` compared by value (parquet keeps a native int
    list; the old TSV form is a garbled string -- user-approved migration, the
    numbers are unchanged)."""
    assert list(new.columns) == list(ref.columns), (
        f"column mismatch: {list(new.columns)} vs {list(ref.columns)}"
    )
    if "bounds" in new.columns:
        nb = [_parse_bounds(v) for v in new["bounds"]]
        rb = [_parse_bounds(v) for v in ref["bounds"]]
        assert nb == rb, "bounds numbers differ between parquet list and tsv string"
        new = new.drop(columns=["bounds"])
        ref = ref.drop(columns=["bounds"])
    _assert_identical(_norm_nullable(new), _norm_nullable(ref))


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: TSV -> parquet migration for reads / cells / sbs_info.
# resolve_table_path + read_table prefer-parquet, format-agnostic combine,
# writer round-trip identity, shared-writer ext branch, fail-loud on corrupt
# parquet, and the reads->cells chain link over parquet reads.
# ─────────────────────────────────────────────────────────────────────────────
from lib.shared.parquet_io import read_table, resolve_table_path, read_parquet
from lib.sbs.call_cells import call_cells as _call_cells

GOLDEN_DIR = Path(__file__).parent / "golden"
BARCODE_LIB_FP = "/mnt/work/broad-analysis/broad-tdp-gws/analysis/config/barcode_library.tsv"
_CHAIN_TILES = [
    t for t in ("P-4_W-A1_T-0", "P-4_W-A1_T-50", "P-4_W-A1_T-100")
    if (GOLDEN_DIR / f"{t}__reads.tsv").exists()
    and (GOLDEN_DIR / f"{t}__cells.tsv").exists()
]
MULTI_CC_KWARGS = dict(
    q_min=0.0, map_start=1, map_end=12, prefix_map="prefix_map",
    recomb_start=13, recomb_end=15, prefix_recomb="prefix_recomb",
    recomb_filter_col="Q_recomb", recomb_q_thresh=0.1, error_correct=True,
    sort_calls="peak", max_distance=1, n_barcodes=2, barcode_info_cols=None,
)


def _sample_df():
    """A frame exercising str / int / high-precision float columns."""
    return pd.DataFrame(
        {
            "well": ["A1", "A1", "A1"],
            "tile": [2, 2, 3],
            "cell": [1, 2, 1],
            "area": [69, 70, 80],
            "i": [10.533333333333333, 11.700000000000001, 12.5],
            "barcode": ["ACGT", "TGCA", "AAAA"],
        }
    )


# ─── 1. writer round-trip identity ──────────────────────────────────────────
@pytest.mark.unit
def test_parquet_writer_roundtrip_identity(tmp_path):
    """write_parquet -> read_parquet is EXACT to the source frame (contrast the
    ~1e-12 tsv text drift), and matches a tsv write/read under Option A.

    Covers the reads/cells/sbs_info writer swap: parquet preserves dtypes and
    floats bit-for-bit, so the combine-over-parquet path stays within Option A.
    """
    df = _sample_df()
    pq = tmp_path / "x.parquet"
    write_parquet(df, str(pq))
    back = read_parquet(str(pq))
    # parquet is EXACT (same dtypes, floats bit-identical).
    pd.testing.assert_frame_equal(back, df, check_exact=True)

    # ... and equals a tsv write+read under the Option-A float contract.
    tsv = tmp_path / "x.tsv"
    df.to_csv(tsv, index=False, sep="\t")
    tsv_back = pd.read_csv(tsv, sep="\t")
    _assert_identical(back, tsv_back)


# ─── 2. resolve_table_path prefer-parquet ───────────────────────────────────
@pytest.mark.unit
def test_resolve_table_path_prefers_parquet(tmp_path):
    """parquet+tsv -> parquet; tsv-only -> tsv; parquet-only -> parquet;
    0-byte gated; neither -> (None, None). Candidate suffix is irrelevant."""
    d = tmp_path
    df = _sample_df()

    # both present -> parquet wins, regardless of candidate suffix
    write_parquet(df, str(d / "both.parquet"))
    (d / "both.tsv").write_text("well\ttile\nA1\t2\n")
    assert resolve_table_path(str(d / "both.tsv")) == (str(d / "both.parquet"), "parquet")
    assert resolve_table_path(str(d / "both.parquet")) == (str(d / "both.parquet"), "parquet")

    # tsv only -> tsv (even when asked with a .parquet candidate)
    (d / "tsvonly.tsv").write_text("well\ttile\nA1\t2\n")
    assert resolve_table_path(str(d / "tsvonly.parquet")) == (str(d / "tsvonly.tsv"), "tsv")

    # parquet only -> parquet
    write_parquet(df, str(d / "pqonly.parquet"))
    assert resolve_table_path(str(d / "pqonly.tsv")) == (str(d / "pqonly.parquet"), "parquet")

    # 0-byte parquet with a real tsv sibling -> falls through to tsv
    (d / "zpq.parquet").write_bytes(b"")
    (d / "zpq.tsv").write_text("well\ttile\nA1\t2\n")
    assert resolve_table_path(str(d / "zpq.parquet")) == (str(d / "zpq.tsv"), "tsv")

    # neither present -> (None, None)
    assert resolve_table_path(str(d / "missing.parquet")) == (None, None)
    # both 0-byte -> (None, None)
    (d / "z2.parquet").write_bytes(b"")
    (d / "z2.tsv").write_bytes(b"")
    assert resolve_table_path(str(d / "z2.parquet")) == (None, None)


# ─── 3. read_table format-agnostic ──────────────────────────────────────────
@pytest.mark.unit
def test_read_table_format_agnostic(tmp_path):
    """parquet-present and tsv-only both read to equal frames; missing raises."""
    df = _sample_df()

    # parquet present -> exact frame back
    write_parquet(df, str(tmp_path / "a.parquet"))
    pq_frame = read_table(str(tmp_path / "a.tsv"))  # candidate suffix ignored
    pd.testing.assert_frame_equal(pq_frame, df, check_exact=True)

    # tsv only -> equals the same frame under Option A
    df.to_csv(tmp_path / "b.tsv", index=False, sep="\t")
    tsv_frame = read_table(str(tmp_path / "b.parquet"))
    _assert_identical(tsv_frame, df)

    # neither sibling -> FileNotFoundError (call_cells must not silently skip)
    with pytest.raises(FileNotFoundError):
        read_table(str(tmp_path / "nope.parquet"))
    # 0-byte only -> also FileNotFoundError
    (tmp_path / "empty.tsv").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        read_table(str(tmp_path / "empty.parquet"))


# ─── 4. format-agnostic combine: parquet-only fast path ─────────────────────
@pytest.mark.unit
def test_combine_parquet_only_uses_polars_fast_path(tmp_path):
    """>=2 parquet tiles combine via the polars fast path (zero fallback)."""
    if not _HAS_POLARS:
        pytest.skip("polars not installed")
    from lib.shared import combine_dfs as _cd

    hdr_df1 = pd.DataFrame({"well": ["A1"], "tile": [2], "cell": [1], "i": [10.5333333333]})
    hdr_df2 = pd.DataFrame({"well": ["A1"], "tile": [3], "cell": [1], "i": [20.7000000001]})
    t1 = tmp_path / "1"; t1.mkdir(); write_parquet(hdr_df1, str(t1 / "reads.parquet"))
    t2 = tmp_path / "2"; t2.mkdir(); write_parquet(hdr_df2, str(t2 / "reads.parquet"))
    cands = [str(t1 / "reads.parquet"), str(t2 / "reads.parquet")]

    _cd._READ_STATS["polars"] = 0
    _cd._READ_STATS["pandas_fallback"] = 0
    out = _cd.combine_tile_dfs(cands)
    assert out is not None and len(out) == 2
    assert _cd._READ_STATS["polars"] == 1, _cd._READ_STATS
    assert _cd._READ_STATS["pandas_fallback"] == 0, _cd._READ_STATS


@pytest.mark.unit
def test_combine_tsv_only_via_parquet_candidate(tmp_path):
    """Passing .parquet candidates for a TSV-only well still combines via tsv
    (the production backward-compat path: no parquet sibling exists)."""
    hdr = ["well", "tile", "cell", "i"]
    t1 = tmp_path / "1"; t1.mkdir()
    _write_tsv(t1 / "reads.tsv", hdr, [["A1", 2, 1, 10.5]])
    t2 = tmp_path / "2"; t2.mkdir()
    _write_tsv(t2 / "reads.tsv", hdr, [["A1", 3, 1, 20.7]])
    cands = [str(t1 / "reads.parquet"), str(t2 / "reads.parquet")]  # no parquet on disk
    tsvs = [str(t1 / "reads.tsv"), str(t2 / "reads.tsv")]
    _assert_identical(combine_tile_dfs(cands), _reference_combine(tsvs))


# ─── mixed well + union-missing cast ────────────────────────────────────────
@pytest.mark.unit
def test_combine_mixed_well_union_and_cast(tmp_path):
    """Mixed well: tile_a parquet (no `plate`), tile_b tsv (int `plate`),
    tile_c 0-byte. Union keeps `plate`, drops the empty tile, and the
    integer union-missing column is upcast to float (matching pandas concat)."""
    hdr7 = ["well", "tile", "cell", "area", "i"]
    hdr8 = hdr7 + ["plate"]
    a_df = pd.DataFrame(
        {"well": ["A1"], "tile": [2], "cell": [1], "area": [69], "i": [10.5333333333]}
    )
    ta = tmp_path / "a"; ta.mkdir()
    write_parquet(a_df, str(ta / "reads.parquet"))     # parquet for combine
    _write_tsv(ta / "reads.tsv", hdr7, [["A1", 2, 1, 69, 10.5333333333]])  # tsv for reference

    tb = tmp_path / "b"; tb.mkdir()
    _write_tsv(tb / "reads.tsv", hdr8, [["A1", 3, 1, 80, 12.5, 4]])  # tsv only

    tc = tmp_path / "c"; tc.mkdir()
    (tc / "reads.tsv").write_bytes(b"")  # 0-byte -> dropped

    cands = [str(ta / "reads.parquet"), str(tb / "reads.parquet"), str(tc / "reads.parquet")]
    tsvs = [str(ta / "reads.tsv"), str(tb / "reads.tsv"), str(tc / "reads.tsv")]

    new = combine_tile_dfs(cands)
    _assert_identical(new, _reference_combine(tsvs))
    assert "plate" in new.columns
    assert len(new) == 2  # 0-byte tile dropped
    # union-missing cast fires (Float64 intermediate); validate_dtypes then
    # lands the integer-valued column on nullable Int64 -- same as the pandas
    # concat path (dtype parity is already asserted vs reference above).
    assert isinstance(new["plate"].dtype, pd.Int64Dtype), new.dtypes["plate"]
    assert new["plate"].isna().sum() == 1  # the parquet tile row (no `plate`)


# ─── 7. fail-loud on corrupt parquet ────────────────────────────────────────
@pytest.mark.unit
def test_combine_corrupt_parquet_raises(tmp_path):
    """A truncated/garbage .parquet tile must make combine RAISE (not silently
    drop it), mirroring the corrupt-tsv fail-loud guard. resolve_table_path sees
    a non-empty file; both the polars fast path and the pandas fallback error out
    on the bad bytes and the error propagates."""
    good_df = pd.DataFrame({"well": ["A1"], "tile": [2], "cell": [1]})
    tg = tmp_path / "good"; tg.mkdir()
    write_parquet(good_df, str(tg / "reads.parquet"))
    tb = tmp_path / "bad"; tb.mkdir()
    (tb / "reads.parquet").write_bytes(b"PAR1 not a real parquet file \x00\x01\x02")
    cands = [str(tg / "reads.parquet"), str(tb / "reads.parquet")]
    with pytest.raises(Exception):
        combine_tile_dfs(cands)


# ─── 6. phenotype safety: shared writer keeps .tsv (byte-identical) ─────────
@pytest.mark.unit
def test_shared_writer_ext_branch_keeps_phenotype_tsv(tmp_path, monkeypatch):
    """Run the SHARED extract_phenotype_minimal.py script: a `.tsv` output stays
    byte-identical to df.to_csv (phenotype path); a `.parquet` output becomes
    parquet (SBS sbs_info path). Guards Assumption-1 in the spec."""
    import lib.shared.extract_phenotype_minimal as epm_mod
    import lib.shared.image_io as img_mod

    df = pd.DataFrame(
        {"well": ["A1", "A1"], "tile": [0, 0], "cell": [1, 2],
         "i": [1.5, 2.25], "j": [3.5, 4.75]}
    )
    monkeypatch.setattr(epm_mod, "extract_phenotype_minimal", lambda **k: df.copy())
    monkeypatch.setattr(img_mod, "read_image", lambda p: np.zeros((2, 2)))

    script = WORKFLOW / "scripts" / "shared" / "extract_phenotype_minimal.py"
    src = compile(script.read_text(), str(script), "exec")

    class _SM:
        pass

    # phenotype: .tsv output -> byte-identical to plain to_csv, and NOT parquet
    sm = _SM()
    sm.input = [str(tmp_path / "nuc.tiff")]
    sm.output = [str(tmp_path / "phenotype_info.tsv")]
    sm.wildcards = {"row": "A", "col": "1", "tile": "0"}
    exec(src, {"snakemake": sm})
    tsv_bytes = Path(sm.output[0]).read_text()
    assert tsv_bytes == df.to_csv(index=False, sep="\t")
    with pytest.raises(Exception):
        pd.read_parquet(sm.output[0])  # it's text, not parquet

    # SBS: .parquet output -> real parquet
    sm2 = _SM()
    sm2.input = [str(tmp_path / "nuc.tiff")]
    sm2.output = [str(tmp_path / "sbs_info.parquet")]
    sm2.wildcards = {"row": "A", "col": "1", "tile": "0"}
    exec(src, {"snakemake": sm2})
    back = pd.read_parquet(sm2.output[0])
    pd.testing.assert_frame_equal(back, df, check_exact=True)


# ─── 5 + integration: real plate-4 write->combine in parquet mode ───────────
@pytest.mark.integration
@pytest.mark.skipif(not TSV_WELL.exists(), reason="plate-4 tsvs unavailable")
@pytest.mark.parametrize("info", ["reads", "cells", "sbs_info"])
def test_integration_parquet_mode_matches_tsv_reference(info, tmp_path):
    """Small-FOV e2e: materialize real plate-4 subset tiles as parquet, combine
    over the parquet copies, and assert the result matches _reference_combine
    over the original on-disk .tsv tiles under Option A."""
    cands = []
    tsv_paths = []
    n_written = 0
    for t in INTEG_TILES:
        src_tsv = TSV_WELL / str(t) / f"{info}.tsv"
        if not src_tsv.exists():
            continue
        tsv_paths.append(str(src_tsv))
        td = tmp_path / str(t)
        td.mkdir(parents=True, exist_ok=True)
        cands.append(str(td / f"{info}.parquet"))
        try:
            frame = pd.read_csv(src_tsv, sep="\t")
        except pd.errors.EmptyDataError:
            continue  # 0-byte tile: leave no parquet -> resolver drops it (matches ref)
        write_parquet(frame, str(td / f"{info}.parquet"))
        n_written += 1

    assert n_written >= 3, f"expected >=3 non-empty {info} tiles, wrote {n_written}"
    parquet_combined = combine_tile_dfs(cands)
    reference = _reference_combine(tsv_paths)
    # Migrated contract: parquet tiles are typed, so the fast path yields clean
    # int64 (vs the header-only-tile Int64 artifact of the TSV pandas path) and
    # preserves `bounds` as a native int list (vs the old garbled string). Same
    # columns, order, and numbers; the shipped parquet is identical either way.
    _assert_parquet_matches_tsv(parquet_combined, reference)
    # cross-check: parquet-mode == tsv-mode of the SAME tiles, same contract.
    _assert_parquet_matches_tsv(parquet_combined, combine_tile_dfs(tsv_paths))


# ─── 4. call_cells chain integrity over parquet reads ───────────────────────
@pytest.mark.integration
@pytest.mark.skipif(
    not _CHAIN_TILES or not Path(BARCODE_LIB_FP).exists(),
    reason="golden reads/cells tiles or barcode library unavailable",
)
def test_call_cells_chain_over_parquet_reads(tmp_path):
    """The reads->cells chain link: reads written as parquet, read back via
    read_table, produces cells identical to the TSV-fed pipeline (golden).

    Gentle: one tile. First proves read_table(parquet) == pd.read_csv(tsv)
    exactly (the only changed line in the chain), then runs call_cells once on
    the parquet-fed reads and matches the golden cells under Option A.
    """
    tile = _CHAIN_TILES[0]
    reads_tsv = pd.read_csv(GOLDEN_DIR / f"{tile}__reads.tsv", sep="\t")
    pq = tmp_path / "reads.parquet"
    write_parquet(reads_tsv, str(pq))

    # chain link: read_table must reconstruct the exact reads frame from parquet
    reads_via_read_table = read_table(str(tmp_path / "reads.tsv"))  # candidate suffix ignored
    pd.testing.assert_frame_equal(reads_via_read_table, reads_tsv, check_exact=True)

    barcode_lib = pd.read_csv(BARCODE_LIB_FP, sep="\t")
    cells = _call_cells(
        reads_data=reads_via_read_table, df_barcode_library=barcode_lib.copy(),
        **MULTI_CC_KWARGS,
    )
    golden = pd.read_csv(GOLDEN_DIR / f"{tile}__cells.tsv", sep="\t")
    key = ["well", "tile", "cell"]
    got = cells.sort_values(key).reset_index(drop=True)
    exp = golden.sort_values(key).reset_index(drop=True)
    assert got.shape == exp.shape, f"shape {got.shape} vs golden {exp.shape}"
    assert set(got.columns) == set(exp.columns)
    for c in exp.columns:
        g, e = got[c], exp[c]
        assert (g.isna() == e.isna()).all(), f"NA positions differ in {c}"
        if pd.api.types.is_float_dtype(e):
            np.testing.assert_allclose(
                g.astype(float).to_numpy(), e.astype(float).to_numpy(),
                rtol=1e-6, atol=0.0, equal_nan=True,
            )
        else:
            m = ~e.isna()
            assert (g[m].astype(str).to_numpy() == e[m].astype(str).to_numpy()).all(), (
                f"non-float column {c} differs"
            )


# ─── regression: bounds migration + empty-parquet-tile handling ─────────────
@pytest.mark.unit
def test_parquet_bounds_preserved_as_list(tmp_path):
    """sbs_info combined from parquet tiles keeps ``bounds`` as a native int
    list, and write->read stores it as a parquet List (not a string).

    Regression for the old glitch: the ndarray bbox came out of the parquet
    fast path and was flattened to ``str(ndarray)`` by validate_dtypes, so the
    combined file wrote a garbled string. User-approved migration: bounds is a
    clean int list; the numbers are unchanged.
    """
    if not _HAS_POLARS:
        pytest.skip("polars not installed")
    import pyarrow.parquet as pq

    tiles = []
    for t, base in [(1, 10), (2, 20)]:
        df = pd.DataFrame(
            {
                "area": [base, base + 1],
                "cell": [1, 2],
                "bounds": [
                    np.array([base, base + 1, base + 2, base + 3]),
                    np.array([base + 4, base + 5, base + 6, base + 7]),
                ],
                "tile": [t, t],
                "well": ["A1", "A1"],
                "plate": [1, 1],
            }
        )
        p = tmp_path / f"{t}.parquet"
        write_parquet(df, str(p))
        tiles.append(str(p))

    combined = combine_tile_dfs(tiles)
    # in-memory: bounds cells are int sequences, never strings
    assert not any(isinstance(b, str) for b in combined["bounds"]), (
        f"bounds regressed to string: {combined['bounds'].tolist()}"
    )
    assert [int(x) for x in combined["bounds"].iloc[0]] == [10, 11, 12, 13]
    # on-disk: parquet stores bounds as a List column, not a string
    out = tmp_path / "sbs_info.parquet"
    write_parquet(combined, str(out))
    btype = str(pq.read_schema(out).field("bounds").type)
    assert "list" in btype, f"bounds should be a parquet list, got {btype}"


@pytest.mark.unit
def test_fast_path_skips_empty_parquet_tile_and_completes_union(tmp_path):
    """A 0-row parquet tile with a stale, fuller schema (an extra column):

    * the fast path SKIPS it, so its String-inferred columns can't poison the
      List-typed ``bounds`` concat (the bug that silently forced the pandas
      fallback for sbs_info), and
    * its unique column still surfaces as an all-null object column, matching a
      pandas union concat.
    """
    if not _HAS_POLARS:
        pytest.skip("polars not installed")
    from lib.shared import combine_dfs as _cd

    for t, v in [(1, 5), (2, 7)]:
        df = pd.DataFrame(
            {
                "cell": [1, 2],
                "tile": [t, t],
                "well": ["A1", "A1"],
                "bounds": [
                    np.array([v, v + 1, v + 2, v + 3]),
                    np.array([v + 4, v + 5, v + 6, v + 7]),
                ],
                "plate": [1, 1],
            }
        )
        write_parquet(df, str(tmp_path / f"{t}.parquet"))
    empty = pd.DataFrame(
        {"cell": [], "tile": [], "well": [], "bounds": [], "plate": [], "gene_id": []}
    )
    write_parquet(empty, str(tmp_path / "0.parquet"))
    paths = [str(tmp_path / f"{t}.parquet") for t in (0, 1, 2)]

    _cd._READ_STATS["polars"] = 0
    _cd._READ_STATS["pandas_fallback"] = 0
    combined = combine_tile_dfs(paths)
    assert _cd._READ_STATS["polars"] == 1, _cd._READ_STATS
    assert _cd._READ_STATS["pandas_fallback"] == 0, _cd._READ_STATS
    assert len(combined) == 4
    # unique column from the empty tile is preserved, all-null
    assert "gene_id" in combined.columns
    assert combined["gene_id"].isna().all()
    # bounds survived as int lists despite the empty-tile skip
    assert not any(isinstance(b, str) for b in combined["bounds"])
    assert [int(x) for x in combined["bounds"].iloc[0]] == [5, 6, 7, 8]
