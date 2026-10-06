"""Tests for annotate_reads() and the read-level recomb / correction state it exposes.

Seven behaviours from the 2026-09-24 plan, plus a golden-file check that the
call_cells refactor is output-identical on every pre-existing column.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.sbs.call_cells import annotate_reads, call_cells  # noqa: E402
from lib.merge.format_merge import select_sbs_merge_cols  # noqa: E402
from recomb_fixture import (  # noqa: E402
    LIBRARY,
    MULTI_PARAMS,
    SIMPLE_PARAMS,
    READ_CLEAN,
    READ_LOW_Q_MIN,
    READ_LOW_Q_RECOMB,
    READ_MAP_MISCALL,
    READ_RECOMB_OFF,
    READ_UNMAPPED,
    make_reads,
    make_simple_inputs,
)

DATA = Path(__file__).resolve().parent / "data"
ANNOT_PARAMS = {
    k: v for k, v in MULTI_PARAMS.items() if k not in ("sort_calls", "n_barcodes")
}


def _annotated(**overrides):
    params = {**ANNOT_PARAMS, **overrides}
    return annotate_reads(make_reads(), LIBRARY.copy(), **params).set_index("read")


# 1. one row per input read, even when q_min would drop some
def test_row_count_preserved_with_q_min():
    reads = make_reads()
    out = annotate_reads(reads, LIBRARY.copy(), **{**ANNOT_PARAMS, "q_min": 0.5})
    assert len(out) == len(reads)
    assert set(reads.columns) <= set(out.columns)
    assert out["passed_q_min"].dtype == bool
    # two low-quality reads fail the threshold, everything else passes
    assert (~out["passed_q_min"]).sum() == 2
    assert set(out.loc[~out["passed_q_min"], "read"]) == {
        READ_LOW_Q_RECOMB,
        READ_LOW_Q_MIN,
    }


# 2. indeterminant is exactly the NA leg of the ternary
def test_indeterminant_complements_no_recomb():
    out = _annotated()
    nr = out["no_recomb"]
    assert str(nr.dtype) == "boolean"
    assert str(out["indeterminant"].dtype) == "boolean"
    assert (out["indeterminant"] == nr.isna()).all()
    n_true = (nr == True).sum()  # noqa: E712
    n_false = (nr == False).sum()  # noqa: E712
    assert n_true + n_false + nr.isna().sum() == len(out)
    assert n_true > 0 and n_false > 0 and nr.isna().sum() > 0


# 3. a quality abstention survives as NA + indeterminant, not as a dropped row
def test_low_q_recomb_is_indeterminant():
    out = _annotated()
    row = out.loc[READ_LOW_Q_RECOMB]
    assert row["mapped"]
    assert row["Q_recomb"] < MULTI_PARAMS["recomb_q_thresh"]
    assert pd.isna(row["no_recomb"])
    assert row["indeterminant"]
    # and the unmapped read is indeterminant for the other reason
    assert not out.loc[READ_UNMAPPED, "mapped"]
    assert out.loc[READ_UNMAPPED, "indeterminant"]
    # the clean read is a real call
    assert out.loc[READ_CLEAN, "no_recomb"] is not pd.NA
    assert out.loc[READ_CLEAN, "no_recomb"]
    assert not out.loc[READ_CLEAN, "indeterminant"]


# 4. corrected / correction_cycle name the substituted base in absolute cycles
def test_corrected_and_correction_cycle():
    out = _annotated()
    assert str(out["corrected"].dtype) == "boolean"
    assert str(out["correction_cycle"].dtype) == "Int64"
    fixed = out.loc[READ_MAP_MISCALL]
    assert fixed["corrected"]
    assert fixed["correction_cycle"] == 5  # 0-idx position 4 + map_start 1
    assert fixed["prefix_map"] == "AAAAAAAAAAAA"
    assert fixed["pre_correction_prefix_map"] == "AAAAGAAAAAAA"
    assert fixed["no_recomb"]  # recomb intact once the map barcode is fixed
    clean = out.loc[READ_CLEAN]
    assert not clean["corrected"]
    assert pd.isna(clean["correction_cycle"])
    # correction_cycle is NA exactly where corrected is False
    assert (out["correction_cycle"].isna() == ~out["corrected"]).all()


# 5. the RECOMB barcode is never error-corrected (regression for F5)
def test_prefix_recomb_never_corrected():
    out = _annotated()
    off = out.loc[READ_RECOMB_OFF]
    assert off["mapped"]
    assert not off["corrected"]
    assert off["prefix_recomb"] == "CGA"  # observed, one base off expected CGT
    assert off["no_recomb"] is not pd.NA and not off["no_recomb"]
    assert not off["indeterminant"]
    # across every read the recomb column is the raw slice of cycles 13-15
    raw = make_reads().set_index("read")["barcode"].str[12:15]
    pd.testing.assert_series_equal(
        out["prefix_recomb"].sort_index(), raw.sort_index(), check_names=False
    )


# 6. refactor equivalence: call_cells output unchanged on every pre-existing column
@pytest.mark.parametrize("mode", ["multi", "simple"])
def test_call_cells_refactor_equivalence_golden(mode):
    golden = pd.read_parquet(DATA / f"call_cells_golden_{mode}.parquet")
    if mode == "multi":
        after = call_cells(make_reads(), LIBRARY.copy(), **MULTI_PARAMS)
    else:
        reads, lib = make_simple_inputs()
        after = call_cells(reads, lib, **SIMPLE_PARAMS)
    after = after.reset_index(drop=True)
    assert set(golden.columns) <= set(after.columns)
    # parquet round-trip turns None into nan in object columns; compare on values
    after = after[golden.columns].astype(object).where(lambda d: d.notna(), None)
    golden = golden.astype(object).where(lambda d: d.notna(), None)
    pd.testing.assert_frame_equal(after, golden)


# 7. new columns reach cells.parquet (rank-suffixed) and the merge column selection
def test_new_columns_reach_cells_and_merge():
    cells = call_cells(make_reads(), LIBRARY.copy(), **MULTI_PARAMS).set_index("cell")
    for col in (
        "indeterminant_0",
        "indeterminant_1",
        "corrected_0",
        "corrected_1",
        "correction_cycle_0",
        "correction_cycle_1",
        "prefix_recomb",
    ):
        assert col in cells.columns, col
    assert "passed_q_min" not in cells.columns
    # cell 2's rank-0 read is the corrected one (peak 120)
    assert cells.loc[2, "corrected_0"]
    assert cells.loc[2, "correction_cycle_0"] == 5
    assert cells.loc[2, "pre_correction_cell_barcode_0"] == "AAAAGAAAAAAA"
    # cell 5: rank 0 mapped T, rank 1 the unmapped read -> indeterminant_1
    assert cells.loc[5, "cell_barcode_0"] == "TTTTTTTTTTTT"
    assert cells.loc[5, "indeterminant_1"]
    assert cells.loc[4, "indeterminant_0"]
    assert not cells.loc[1, "indeterminant_0"]

    merge_cols = select_sbs_merge_cols(cells.reset_index())
    for col in (
        "prefix_recomb",
        "no_recomb_0",
        "indeterminant_0",
        "corrected_0",
        "correction_cycle_0",
        "Q_recomb_0",
    ):
        assert col in merge_cols, col
    assert "Q_0_0" not in merge_cols  # per-cycle Q columns still excluded
    assert "pre_correction_cell_barcode_0" not in merge_cols
