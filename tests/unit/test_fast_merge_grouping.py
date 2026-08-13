import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_cell_info(n_per_tile=5, tiles=(0, 1, 2)):
    rows = []
    for t in tiles:
        for k in range(n_per_tile):
            rows.append(
                {"tile": t, "i": float(t * 100 + k), "j": float(k * 10), "label": k + 1}
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_groupby_dict_matches_boolean_mask():
    df = _make_cell_info()
    by_tile = dict(tuple(df.groupby("tile")))

    for t in (0, 1, 2):
        expected = df[df["tile"] == t].reset_index(drop=True)
        actual = by_tile[t].reset_index(drop=True)
        assert_frame_equal(actual, expected)


@pytest.mark.unit
def test_missing_tile_returns_empty():
    df = _make_cell_info()
    by_tile = dict(tuple(df.groupby("tile")))
    _empty = df.iloc[0:0]

    result = by_tile.get(99, _empty)
    assert len(result) == 0
    assert set(result.columns) == set(df.columns)


@pytest.mark.unit
def test_empty_alignment_produces_no_merge_calls():
    # Smoke-test: with an empty fast_alignment_filtered, the loop body never runs
    df = _make_cell_info()
    by_tile = dict(tuple(df.groupby("tile")))
    _empty = df.iloc[0:0]

    merge_data = []
    empty_alignment = pd.DataFrame(columns=["tile", "site"])
    for _idx, row in empty_alignment.iterrows():
        ph_filtered = by_tile.get(row["tile"], _empty)
        sbs_filtered = by_tile.get(row["site"], _empty)
        merge_data.append((ph_filtered, sbs_filtered))

    assert merge_data == []
