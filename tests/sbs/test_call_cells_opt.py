"""Tests for Phase 1 (barcode library cache) and Phase 2 (error correction dedup)."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from lib.sbs.call_cells import (
    load_barcode_library,
    _read_barcode_library_cached,
    error_correct_reads,
    _barcode_distance_matrix,
    _build_hamming1_index,
    call_cells,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
BARCODE_LIB_FP = "/mnt/work/broad-analysis/broad-tdp-gws/analysis/config/barcode_library.tsv"

_all_tiles = ["P-4_W-A1_T-0", "P-4_W-A1_T-50", "P-4_W-A1_T-100"]
INTEGRATION_TILES = [
    t for t in _all_tiles
    if (GOLDEN_DIR / f"{t}__reads.tsv").exists()
    and (GOLDEN_DIR / f"{t}__cells.tsv").exists()
]

MULTI_CC_KWARGS = dict(
    q_min=0.0,
    map_start=1,
    map_end=12,
    prefix_map="prefix_map",
    recomb_start=13,
    recomb_end=15,
    prefix_recomb="prefix_recomb",
    recomb_filter_col="Q_recomb",
    recomb_q_thresh=0.1,
    error_correct=True,
    sort_calls="peak",
    max_distance=1,
    n_barcodes=2,
    barcode_info_cols=None,
)


# ─── Phase 1: barcode library cache ─────────────────────────────────────────

def test_library_cache_hit_no_reread(tmp_path):
    """Cache should parse the file exactly once; second call hits lru_cache."""
    tsv = tmp_path / "lib.tsv"
    tsv.write_text("prefix_map\tgene_symbol\nAAAAAAAAAAAA\tGENE1\n")

    _read_barcode_library_cached.cache_clear()

    call_count = 0
    original_read_csv = pd.read_csv

    def counting_read_csv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_read_csv(*args, **kwargs)

    with patch("lib.sbs.call_cells.pd.read_csv", side_effect=counting_read_csv):
        _read_barcode_library_cached.cache_clear()
        load_barcode_library(str(tsv))
        load_barcode_library(str(tsv))

    assert call_count == 1, f"Expected 1 read_csv call, got {call_count}"
    _read_barcode_library_cached.cache_clear()


def test_library_cache_returns_independent_copies(tmp_path):
    """Mutating one returned frame must not corrupt the cache."""
    tsv = tmp_path / "lib2.tsv"
    tsv.write_text("prefix_map\tgene_symbol\nAAAAAAAAAAAA\tGENE1\n")

    _read_barcode_library_cached.cache_clear()
    df1 = load_barcode_library(str(tsv))
    df1["prefix_map"] = "XXXXXXXXXXXX"  # mutate in-place

    df2 = load_barcode_library(str(tsv))
    assert df2["prefix_map"].iloc[0] == "AAAAAAAAAAAA", \
        "Cache was corrupted by mutation of returned copy"
    _read_barcode_library_cached.cache_clear()


# ─── Phase 2: error correction dedup ────────────────────────────────────────

def _make_ref(*barcodes):
    return pd.Series(list(barcodes))


def test_dedup_runs_levenshtein_once_per_unique():
    """50 identical unmapped reads → distance matrix built with exactly 1 row (slow path, max_distance=2)."""
    ref = _make_ref("AAAAAAAAAAAA")
    reads = pd.Series(["GGGGGGGGGGGG"] * 50)

    matrix_shapes = []
    original = _barcode_distance_matrix

    def spy_matrix(bc1, bc2, **kw):
        matrix_shapes.append(len(bc1))
        return original(bc1, bc2, **kw)

    with patch("lib.sbs.call_cells._barcode_distance_matrix", side_effect=spy_matrix):
        error_correct_reads(reads, ref, max_distance=2)

    assert len(matrix_shapes) == 1
    assert matrix_shapes[0] == 1, f"Expected 1 unique unmapped row, got {matrix_shapes[0]}"


def test_exact_matches_bypass_matrix():
    """All reads exactly in library → _barcode_distance_matrix never called."""
    ref = _make_ref("AAAAAAAAAAAA", "CCCCCCCCCCCC")
    reads = pd.Series(["AAAAAAAAAAAA", "CCCCCCCCCCCC", "AAAAAAAAAAAA"])

    with patch("lib.sbs.call_cells._barcode_distance_matrix") as mock_mat:
        result = error_correct_reads(reads, ref, max_distance=1)

    mock_mat.assert_not_called()
    pd.testing.assert_series_equal(result, reads, check_names=False)


def test_correction_within_and_beyond_max_distance():
    """Dist-1 read corrected; dist-3 read left unchanged (max_distance=1)."""
    ref = _make_ref("AAAAAAAAAA")   # 10-char ref
    close = "AAAAAAAAAC"            # hamming-1
    far = "AAAAAACCCA"              # hamming-3

    reads = pd.Series([close, far])
    result = error_correct_reads(reads, ref, max_distance=1, distance_metric="hamming")

    assert result.iloc[0] == "AAAAAAAAAA", f"dist-1 should be corrected, got {result.iloc[0]}"
    assert result.iloc[1] == far, f"dist-3 should be unchanged, got {result.iloc[1]}"


def test_ambiguous_correction_left_unchanged():
    """Read equidistant to two refs is left unchanged."""
    # "AAAC" is hamming-1 from "AAAA" and hamming-1 from "AAAG"
    ref = _make_ref("AAAA", "AAAG")
    reads = pd.Series(["AAAC"])
    result = error_correct_reads(reads, ref, max_distance=1, distance_metric="hamming")

    assert result.iloc[0] == "AAAC", \
        f"Ambiguous read should be unchanged, got {result.iloc[0]}"


def test_zero_reads_produces_empty_output():
    """Empty reads Series → empty output, no error."""
    ref = _make_ref("AAAAAAAAAAAA")
    reads = pd.Series([], dtype=str)
    result = error_correct_reads(reads, ref, max_distance=1)
    assert len(result) == 0


# ─── Integration: output matches golden ─────────────────────────────────────

@pytest.fixture(scope="module")
def barcode_lib_df():
    return pd.read_csv(BARCODE_LIB_FP, sep="\t")


def _assert_series_match(r, g, col):
    """Compare two series, tolerating pd.NA vs np.nan as equivalent nulls."""
    r_null = r.isna()
    g_null = g.isna()
    assert (r_null == g_null).all(), \
        f"NA positions differ in column '{col}'"

    if pd.api.types.is_float_dtype(g):
        pd.testing.assert_series_equal(
            r.rename(col), g.rename(col),
            check_exact=False, rtol=1e-6, check_names=False, check_dtype=False,
        )
    elif (~g_null).any():
        # For non-float columns: compare non-NA values with dtype coercion
        pd.testing.assert_series_equal(
            r[~r_null].reset_index(drop=True).astype(str).rename(col),
            g[~g_null].reset_index(drop=True).astype(str).rename(col),
            check_exact=True, check_names=False,
        )


@pytest.mark.parametrize("tile", INTEGRATION_TILES)
def test_cells_output_matches_golden(tile, barcode_lib_df):
    """call_cells output must match the golden TSV within float tolerance."""
    reads_fp = GOLDEN_DIR / f"{tile}__reads.tsv"
    golden_fp = GOLDEN_DIR / f"{tile}__cells.tsv"

    reads_data = pd.read_csv(reads_fp, sep="\t")
    golden = pd.read_csv(golden_fp, sep="\t")

    result = call_cells(
        reads_data=reads_data,
        df_barcode_library=barcode_lib_df.copy(),
        **MULTI_CC_KWARGS,
    )

    key_cols = ["well", "tile", "cell"]
    result_s = result.sort_values(key_cols).reset_index(drop=True)
    golden_s = golden.sort_values(key_cols).reset_index(drop=True)

    assert result_s.shape == golden_s.shape, (
        f"Shape mismatch: got {result_s.shape}, expected {golden_s.shape}"
    )

    assert set(result_s.columns) == set(golden_s.columns), (
        f"Column mismatch:\n"
        f"  extra={set(result_s.columns)-set(golden_s.columns)}\n"
        f"  missing={set(golden_s.columns)-set(result_s.columns)}"
    )

    for col in golden_s.columns:
        _assert_series_match(result_s[col], golden_s[col], col)

def test_hamming1_index_used_for_production_config(monkeypatch):
    """Fast path (precomputed index) fires for max_distance=1, metric=hamming."""
    called = {"n": 0}
    import lib.sbs.call_cells as ccmod
    real_build = ccmod._build_hamming1_index

    def spy(*a, **k):
        called["n"] += 1
        return real_build(*a, **k)

    monkeypatch.setattr(ccmod, "_build_hamming1_index", spy)

    reference = pd.Series(["AAAAAAAAAAAA", "CCCCCCCCCCCC"])
    reads = pd.Series(["AAAAAAAAAAAC", "AAAAAAAAAAAA"])
    out = error_correct_reads(reads, reference, max_distance=1, distance_metric="hamming")
    assert called["n"] >= 1, "index builder must be called"
    assert out.iloc[0] == "AAAAAAAAAAAA"  # corrected
    assert out.iloc[1] == "AAAAAAAAAAAA"  # exact match
