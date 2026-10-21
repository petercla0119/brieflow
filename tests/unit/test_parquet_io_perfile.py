"""Per-file parquet readers must match whole-dataset semantics without loading everything."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.parquet_io import (  # noqa: E402
    columns_without_nulls,
    read_parquet_frames,
    sample_parquet_rows,
    write_parquet,
)


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    paths, frames = [], []
    for i, n in enumerate((120, 0, 350, 40)):
        df = pd.DataFrame(
            {
                "plate": np.full(n, i + 1, dtype=np.int64),
                "gene": [f"g{j % 7}" for j in range(n)],
                "f_clean": rng.normal(size=n),
                "f_null_in_file_3": rng.normal(size=n),
                "f_float32": rng.normal(size=n).astype(np.float32),
            }
        )
        if i == 2:
            df.loc[df.index[:5], "f_null_in_file_3"] = np.nan
        path = tmp_path / f"part_{i}.parquet"
        write_parquet(df, path)
        paths.append(path)
        frames.append(df)
    return paths, pd.concat(frames, ignore_index=True)


@pytest.mark.unit
def test_columns_without_nulls_matches_dropna(files):
    paths, full = files
    assert columns_without_nulls(paths) == list(full.dropna(axis=1).columns)


@pytest.mark.unit
def test_read_parquet_frames_yields_nonempty_files_as_float32(files):
    paths, full = files
    out = list(read_parquet_frames(paths))
    assert [p for p, _ in out] == [paths[0], paths[2], paths[3]]
    cat = pd.concat([df for _, df in out], ignore_index=True)
    assert len(cat) == len(full)
    assert cat["f_clean"].dtype == np.float32 and cat["plate"].dtype == np.int64
    np.testing.assert_allclose(
        cat["f_clean"].to_numpy(), full["f_clean"].to_numpy(), rtol=1e-6
    )


@pytest.mark.unit
def test_sample_parquet_rows_is_a_uniform_subset(files):
    paths, full = files
    sample = sample_parquet_rows(paths, 200, seed=1, columns=["plate", "f_clean"])
    assert len(sample) == 200 and list(sample.columns) == ["plate", "f_clean"]
    assert sample["f_clean"].dtype == np.float32
    # every sampled row exists in the full table, and no row is drawn twice
    key = set(np.round(full["f_clean"].to_numpy(), 5))
    drawn = np.round(sample["f_clean"].to_numpy().astype(np.float64), 5)
    assert set(drawn) <= key and len(set(drawn)) == 200
    # proportional allocation: file 2 holds 350/510 of the rows
    share = (sample["plate"] == 3).mean()
    assert abs(share - 350 / 510) < 0.1
    # asking for more rows than exist returns everything
    assert len(sample_parquet_rows(paths, 10_000, seed=1)) == len(full)
    # same seed → same rows
    again = sample_parquet_rows(paths, 200, seed=1, columns=["plate", "f_clean"])
    pd.testing.assert_frame_equal(sample, again)
