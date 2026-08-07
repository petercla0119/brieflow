import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.parquet_io import read_parquets, write_parquet

# Minimal realistic columns matching extract_phenotype output
_COLS = [
    "plate",
    "well",
    "tile",
    "label",
    "cell_i",
    "cell_j",
    "cell_DAPI_min",
    "cell_bounds_0",
    "cell_bounds_1",
    "cell_bounds_2",
    "cell_bounds_3",
    "cell_feat_a",
    "cell_feat_b",
]


def _make_df(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "plate": [1] * n_rows,
            "well": ["A1"] * n_rows,
            "tile": list(range(n_rows)),
            "label": list(range(1, n_rows + 1)),
            "cell_i": rng.integers(0, 100, n_rows).astype(float),
            "cell_j": rng.integers(0, 100, n_rows).astype(float),
            "cell_DAPI_min": rng.random(n_rows).astype(np.float32),
            "cell_bounds_0": rng.integers(0, 50, n_rows).astype(float),
            "cell_bounds_1": rng.integers(0, 50, n_rows).astype(float),
            "cell_bounds_2": rng.integers(50, 100, n_rows).astype(float),
            "cell_bounds_3": rng.integers(50, 100, n_rows).astype(float),
            "cell_feat_a": rng.random(n_rows).astype(np.float32),
            "cell_feat_b": rng.random(n_rows).astype(np.float32),
        }
    )


@pytest.mark.unit
def test_extract_merge_roundtrip_preserves_data(tmp_path):
    df1 = _make_df(5, seed=1)
    df2 = _make_df(7, seed=2)
    p1 = tmp_path / "tile0.parquet"
    p2 = tmp_path / "tile1.parquet"
    write_parquet(df1, p1)
    write_parquet(df2, p2)

    result = read_parquets([p1, p2])
    assert len(result) == 12
    assert set(result.columns) == set(df1.columns)
    assert result["cell_DAPI_min"].dtype == np.float32


@pytest.mark.unit
def test_zero_cell_tile_roundtrips(tmp_path):
    empty = pd.DataFrame(columns=_COLS)
    p = tmp_path / "empty.parquet"
    write_parquet(empty, p)

    result = read_parquets([p])
    assert len(result) == 0
    assert set(result.columns) == set(_COLS)


@pytest.mark.unit
def test_single_cell_tile(tmp_path):
    df = _make_df(1, seed=3)
    p = tmp_path / "single.parquet"
    write_parquet(df, p)

    result = read_parquets([p])
    assert len(result) == 1
    assert float(result["cell_DAPI_min"].iloc[0]) == pytest.approx(
        float(df["cell_DAPI_min"].iloc[0])
    )


@pytest.mark.unit
def test_schema_mismatch_falls_back(tmp_path):
    df1 = _make_df(3, seed=4)
    df2 = _make_df(4, seed=5)
    df2["extra_col"] = 99.0  # extra column breaks polars unified schema
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    write_parquet(df1, p1)
    write_parquet(df2, p2)

    result = read_parquets([p1, p2])
    assert len(result) == 7
    assert "extra_col" in result.columns
    # NaN-filled for df1 rows that lack extra_col
    assert result["extra_col"].isna().sum() == 3
