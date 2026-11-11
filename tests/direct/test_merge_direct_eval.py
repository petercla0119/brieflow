"""run_merge_direct writes the per-plate eval_merge outputs (#92).

The Snakemake rule `eval_merge` emits 8 per-plate files; the direct runner never
ported it, so no production plate had them. These tests feed a two-well synthetic
plate through `step_eval_merge` in both layouts the runner auto-detects (flat
`P-1_W-A1__` and nested `1/A/1/`), using the real lib, and assert the files land
with the Snakemake names, that the summary counts are right, and that a missing
input raises instead of writing a partial plate.

Run: python -m pytest -q tests/direct/test_merge_direct_eval.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(ROOT / "workflow"))

RUNNER = ROOT / "scripts" / "direct" / "run_merge_direct.py"
WELLS = ("A1", "A2")
N_INFO = 30  # segmented cells per well, per modality
N_MATCH = 20  # matched pairs per well before dedup
N_DEDUP = 18  # matched pairs per well after dedup


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_merge_direct", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _well_frames(well, rng):
    tiles = [0, 1, 2]
    info = pd.DataFrame(
        {
            "plate": 1,
            "well": well,
            "tile": np.repeat(tiles, N_INFO // 3),
            "cell": np.tile(np.arange(1, N_INFO // 3 + 1), 3),
            "i": rng.uniform(0, 1000, N_INFO),
            "j": rng.uniform(0, 1000, N_INFO),
        }
    )
    match = pd.DataFrame(
        {
            "plate": 1,
            "well": well,
            "tile": info["tile"].values[:N_MATCH],
            "site": info["tile"].values[:N_MATCH],
            "cell_0": info["cell"].values[:N_MATCH],
            "cell_1": info["cell"].values[:N_MATCH],
            "i_0": info["i"].values[:N_MATCH],
            "j_0": info["j"].values[:N_MATCH],
            "distance": rng.uniform(0, 3, N_MATCH),
        }
    )
    dedup = match.iloc[:N_DEDUP].copy()
    dedup["gene_symbol_0"] = ["GENE" if k % 2 else None for k in range(N_DEDUP)]
    dedup["channels_min"] = [0 if k % 3 == 0 else 1 for k in range(N_DEDUP)]
    metadata = pd.DataFrame(
        {
            "well": well,
            "tile": tiles,
            "x_pos": [0.0, 1.0, 2.0],
            "y_pos": [0.0, 0.0, 1.0],
        }
    )
    stats = pd.DataFrame(
        {
            "stage": ["Initial", "After SBS dedup", "After phenotype dedup"],
            "total_cells": [N_MATCH, N_DEDUP + 1, N_DEDUP],
        }
    )
    return info, match, dedup, metadata, stats


def _write_plate(root, nested):
    """Write the eval_merge inputs for plate 1 in the requested layout."""
    rng = np.random.default_rng(0)
    for well in WELLS:
        info, match, dedup, metadata, stats = _well_frames(well, rng)
        r, c = well[0], well[1:]

        def path(subdir, name, ext):
            if nested:
                return root / subdir / "1" / r / c / f"{name}.{ext}"
            return root / subdir / f"P-1_W-{well}__{name}.{ext}"

        files = [
            ("preprocess/metadata/phenotype", "combined_metadata", metadata),
            ("preprocess/metadata/sbs", "combined_metadata", metadata),
            ("phenotype/parquets", "phenotype_info", info),
            ("phenotype/parquets", "phenotype_cp_min", info),
            ("sbs/parquets", "sbs_info", info),
            ("sbs/parquets", "cells", info),
            ("merge/parquets", "merge_formatted", match),
            ("merge/parquets", "merge_deduplicated", dedup),
        ]
        for subdir, name, df in files:
            p = path(subdir, name, "parquet")
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
        p = path("merge/eval", "deduplication_stats", "tsv")
        p.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(p, sep="\t", index=False)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_eval_merge_writes_the_snakemake_outputs(tmp_path, nested):
    rmd = _load_runner()
    _write_plate(tmp_path, nested)

    rmd.step_eval_merge(tmp_path, 1, list(WELLS), force=False)

    eval_dir = tmp_path / "merge" / "eval" / ("1" if nested else "")
    prefix = "" if nested else "P-1__"
    for name, ext in rmd.EVAL_MERGE_OUTPUTS:
        out = eval_dir / f"{prefix}{name}.{ext}"
        assert out.is_file() and out.stat().st_size > 0, out

    summary = pd.read_csv(eval_dir / f"{prefix}merge_summary.tsv", sep="\t")
    assert list(summary["well"]) == list(WELLS)
    assert (summary["ph_cells"] == N_INFO).all()
    assert (summary["sbs_cells"] == N_INFO).all()
    assert (summary["matched_raw"] == N_MATCH).all()  # from dedup stats, not len()
    assert (summary["total_match_pairs"] == N_DEDUP).all()
    assert (summary["single_gene_count"] == N_DEDUP // 2).all()

    dedup = pd.read_csv(eval_dir / f"{prefix}dedup_summaries.tsv", sep="\t")
    assert len(dedup) == 3 * len(WELLS)
    assert set(dedup["well"]) == set(WELLS)

    # skip-safety: a second call is a no-op (does not raise, does not rewrite)
    before = {p: p.stat().st_mtime_ns for p in eval_dir.glob("*")}
    rmd.step_eval_merge(tmp_path, 1, list(WELLS), force=False)
    assert {p: p.stat().st_mtime_ns for p in eval_dir.glob("*")} == before


def test_eval_merge_missing_input_raises(tmp_path):
    rmd = _load_runner()
    _write_plate(tmp_path, nested=False)
    (tmp_path / "sbs/parquets/P-1_W-A2__sbs_info.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="A2__sbs_info"):
        rmd.step_eval_merge(tmp_path, 1, list(WELLS), force=False)
    assert not (tmp_path / "merge/eval/P-1__merge_summary.tsv").exists()
