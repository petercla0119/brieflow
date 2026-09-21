"""Tests for Phase 1 (barcode library cache) and Phase 2 (error correction dedup)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from lib.sbs.call_cells import (
    load_barcode_library,
    _read_barcode_library_cached,
    error_correct_reads,
    _barcode_distance_matrix,
    _build_hamming1_index,
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
    assert df2["prefix_map"].iloc[0] == "AAAAAAAAAAAA", (
        "Cache was corrupted by mutation of returned copy"
    )
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
    assert matrix_shapes[0] == 1, (
        f"Expected 1 unique unmapped row, got {matrix_shapes[0]}"
    )


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
    ref = _make_ref("AAAAAAAAAA")  # 10-char ref
    close = "AAAAAAAAAC"  # hamming-1
    far = "AAAAAACCCA"  # hamming-3

    reads = pd.Series([close, far])
    result = error_correct_reads(reads, ref, max_distance=1, distance_metric="hamming")

    assert result.iloc[0] == "AAAAAAAAAA", (
        f"dist-1 should be corrected, got {result.iloc[0]}"
    )
    assert result.iloc[1] == far, f"dist-3 should be unchanged, got {result.iloc[1]}"


def test_ambiguous_correction_left_unchanged():
    """Read equidistant to two refs is left unchanged."""
    # "AAAC" is hamming-1 from "AAAA" and hamming-1 from "AAAG"
    ref = _make_ref("AAAA", "AAAG")
    reads = pd.Series(["AAAC"])
    result = error_correct_reads(reads, ref, max_distance=1, distance_metric="hamming")

    assert result.iloc[0] == "AAAC", (
        f"Ambiguous read should be unchanged, got {result.iloc[0]}"
    )


def test_zero_reads_produces_empty_output():
    """Empty reads Series → empty output, no error."""
    ref = _make_ref("AAAAAAAAAAAA")
    reads = pd.Series([], dtype=str)
    result = error_correct_reads(reads, ref, max_distance=1)
    assert len(result) == 0


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
    out = error_correct_reads(
        reads, reference, max_distance=1, distance_metric="hamming"
    )
    assert called["n"] >= 1, "index builder must be called"
    assert out.iloc[0] == "AAAAAAAAAAAA"  # corrected
    assert out.iloc[1] == "AAAAAAAAAAAA"  # exact match
