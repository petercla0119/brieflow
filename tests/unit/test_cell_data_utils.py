import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.aggregate.cell_data_utils import channel_combo_subset


def _make_features():
    return pd.DataFrame({
        'nucleus_DAPI_mean': [1.0],
        'nucleus_DAPI_round2_mean': [2.0],
        'cell_GFP_mean': [3.0],
        'nucleus_area': [4.0],      # channel-agnostic shape feature
        'cell_solidity': [5.0],     # channel-agnostic shape feature
    })


@pytest.mark.unit
def test_all_sentinel_returns_unchanged():
    df = _make_features()
    result = channel_combo_subset(df, 'all', ['DAPI', 'DAPI_round2', 'GFP'])
    assert list(result.columns) == list(df.columns)


@pytest.mark.unit
def test_all_list_sentinel_returns_unchanged():
    df = _make_features()
    result = channel_combo_subset(df, ['all'], ['DAPI', 'DAPI_round2', 'GFP'])
    assert list(result.columns) == list(df.columns)


@pytest.mark.unit
def test_channel_agnostic_cols_always_kept():
    df = _make_features()
    result = channel_combo_subset(df, ['GFP'], ['DAPI', 'DAPI_round2', 'GFP'])
    assert 'nucleus_area' in result.columns
    assert 'cell_solidity' in result.columns


@pytest.mark.unit
def test_overlapping_channel_names_no_false_drop():
    # 'DAPI' is a substring of 'DAPI_round2'; keeping only DAPI_round2
    # must NOT accidentally drop nucleus_DAPI_round2_mean via 'DAPI' match
    df = _make_features()
    result = channel_combo_subset(df, ['DAPI_round2'], ['DAPI', 'DAPI_round2', 'GFP'])
    assert 'nucleus_DAPI_round2_mean' in result.columns
    assert 'nucleus_DAPI_mean' not in result.columns
    assert 'cell_GFP_mean' not in result.columns


@pytest.mark.unit
def test_subset_drops_unwanted_channels():
    df = _make_features()
    result = channel_combo_subset(df, ['DAPI'], ['DAPI', 'DAPI_round2', 'GFP'])
    assert 'nucleus_DAPI_mean' in result.columns
    assert 'nucleus_DAPI_round2_mean' not in result.columns
    assert 'cell_GFP_mean' not in result.columns
