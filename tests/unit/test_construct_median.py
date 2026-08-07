import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.parquet_io import write_parquet

_FEATURE_COLS = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
_PERT_ID_COL = "sgRNA_id"
_PERT_COL = "gene"


def _make_aligned_df(seed=0):
    rng = np.random.default_rng(seed)
    constructs = ["sg1", "sg2", "sg3", "sg4"]
    genes = {"sg1": "GENE_A", "sg2": "GENE_A", "sg3": "GENE_B", "sg4": "GENE_C"}
    rows = []
    for sg in constructs:
        n = rng.integers(5, 12)
        for _ in range(n):
            row = {_PERT_ID_COL: sg, _PERT_COL: genes[sg]}
            for f in _FEATURE_COLS:
                row[f] = float(rng.standard_normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _polars_construct_median(parquet_path, pert_id_col, pert_col, feature_cols):
    lf = pl.scan_parquet(str(parquet_path))
    agg = (
        lf.group_by(pert_id_col)
        .agg(
            [pl.first(pert_col).alias(pert_col), pl.len().alias("cell_count")]
            + [pl.median(c).alias(c) for c in feature_cols]
        )
        .collect()
        .to_pandas()
    )
    result = agg[[pert_id_col, pert_col, "cell_count"] + feature_cols].copy()
    result["cell_count"] = result["cell_count"].astype(int)
    return result.sort_values(pert_id_col).reset_index(drop=True)


def _pandas_construct_median(df, pert_id_col, pert_col, feature_cols):
    med = df.groupby(pert_id_col, sort=True)[feature_cols].median().reset_index()
    gene = df.groupby(pert_id_col, sort=True)[[pert_col]].first().reset_index()
    counts = df.groupby(pert_id_col, sort=True).size().reset_index(name="cell_count")
    result = med.merge(gene, on=pert_id_col).merge(counts, on=pert_id_col)
    return (
        result[[pert_id_col, pert_col, "cell_count"] + feature_cols]
        .sort_values(pert_id_col)
        .reset_index(drop=True)
    )


@pytest.mark.unit
def test_polars_median_matches_numpy(tmp_path):
    df = _make_aligned_df(seed=42)
    p = tmp_path / "aligned.parquet"
    write_parquet(df, p)

    polars_result = _polars_construct_median(p, _PERT_ID_COL, _PERT_COL, _FEATURE_COLS)
    pandas_ref = _pandas_construct_median(df, _PERT_ID_COL, _PERT_COL, _FEATURE_COLS)

    assert_frame_equal(
        polars_result[_FEATURE_COLS].reset_index(drop=True),
        pandas_ref[_FEATURE_COLS].reset_index(drop=True),
        check_dtype=False,
        atol=1e-5,
    )
    assert (polars_result["cell_count"] == pandas_ref["cell_count"]).all()


@pytest.mark.unit
def test_single_cell_construct(tmp_path):
    df = pd.DataFrame(
        [
            {
                _PERT_ID_COL: "sg_solo",
                _PERT_COL: "GENE_X",
                "feat_a": 3.14,
                "feat_b": -1.0,
                "feat_c": 0.0,
                "feat_d": 2.718,
                "feat_e": 100.0,
            }
        ]
    )
    p = tmp_path / "single.parquet"
    write_parquet(df, p)

    result = _polars_construct_median(p, _PERT_ID_COL, _PERT_COL, _FEATURE_COLS)
    assert len(result) == 1
    assert result["cell_count"].iloc[0] == 1
    assert result["feat_a"].iloc[0] == pytest.approx(3.14, abs=1e-5)


@pytest.mark.unit
def test_all_constructs_present(tmp_path):
    df = _make_aligned_df(seed=7)
    expected_ids = set(df[_PERT_ID_COL].unique())
    p = tmp_path / "aligned2.parquet"
    write_parquet(df, p)

    result = _polars_construct_median(p, _PERT_ID_COL, _PERT_COL, _FEATURE_COLS)
    assert set(result[_PERT_ID_COL]) == expected_ids
