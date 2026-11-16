"""The Snakemake call_cells script must forward sbs.recomb_call_mode to lib.sbs.call_cells.call_cells.

PR 81 plumbed the mode through annotate_reads.py and the direct runner; this pins the
rule script too, so cells.parquet and processed_reads.parquet always use the same rule.
Runs the real script under a stub `snakemake` object with recording stubs for the lib calls.
Set CALL_CELLS_SCRIPT to test a different copy of the script (used once to watch it fail).
"""

import os
import runpy
import types
from pathlib import Path

import pandas as pd

import lib.sbs.call_cells as cc
import lib.shared.parquet_io as pio

SCRIPT = os.environ.get(
    "CALL_CELLS_SCRIPT",
    str(
        Path(__file__).resolve().parents[2]
        / "workflow"
        / "scripts"
        / "sbs"
        / "call_cells.py"
    ),
)


def _run_script(monkeypatch, tmp_path, params):
    seen = {}
    monkeypatch.setattr(
        cc, "call_cells", lambda **kw: seen.update(kw) or pd.DataFrame()
    )
    monkeypatch.setattr(cc, "load_barcode_library", lambda fp: pd.DataFrame())
    monkeypatch.setattr(pio, "read_table", lambda p: pd.DataFrame())
    monkeypatch.setattr(pio, "write_parquet", lambda df, p: None)
    snakemake = types.SimpleNamespace(
        params=types.SimpleNamespace(config=params),
        input=[str(tmp_path / "reads.parquet")],
        output=[str(tmp_path / "cells.parquet")],
    )
    runpy.run_path(SCRIPT, init_globals={"snakemake": snakemake})
    return seen


def test_multi_mode_forwards_recomb_call_mode(monkeypatch, tmp_path):
    params = dict(
        df_barcode_library_fp="x",
        barcode_type="multi",
        q_min=0,
        map_start=1,
        map_end=12,
        prefix_map="prefix_map",
        recomb_start=13,
        recomb_end=15,
        prefix_recomb="prefix_recomb",
        recomb_filter_col="Q_recomb",
        recomb_q_thresh=0.1,
        recomb_call_mode="mismatch_low_q",
        error_correct=False,
        sort_calls="count",
        max_distance=1,
        n_barcodes=2,
        barcode_info_cols=[],
    )
    seen = _run_script(monkeypatch, tmp_path, params)
    assert seen.get("recomb_call_mode") == "mismatch_low_q"


def test_multi_mode_defaults_to_low_q_when_unset(monkeypatch, tmp_path):
    params = dict(
        df_barcode_library_fp="x",
        barcode_type="multi",
        q_min=0,
        map_start=1,
        map_end=12,
        prefix_map="prefix_map",
        recomb_start=13,
        recomb_end=15,
        prefix_recomb="prefix_recomb",
        recomb_filter_col="Q_recomb",
        recomb_q_thresh=0.1,
        error_correct=False,
        sort_calls="count",
        max_distance=1,
        n_barcodes=2,
        barcode_info_cols=[],
    )
    seen = _run_script(monkeypatch, tmp_path, params)
    assert seen.get("recomb_call_mode") == "low_q"
