import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.aggregate.cell_data_utils import (
    channel_combo_subset,
    channels_of,
    parse_channel_combo,
)


def _make_features():
    return pd.DataFrame(
        {
            "nucleus_DAPI_mean": [1.0],
            "nucleus_DAPI_round2_mean": [2.0],
            "cell_GFP_mean": [3.0],
            "nucleus_area": [4.0],  # channel-agnostic shape feature
            "cell_solidity": [5.0],  # channel-agnostic shape feature
            "nucleus_correlation_DAPI_GFP": [6.0],  # two-channel feature
            "cell_lstsq_slope_GFP_DAPI_round2": [7.0],  # two-channel feature
        }
    )


@pytest.mark.unit
def test_all_sentinel_returns_unchanged():
    df = _make_features()
    result = channel_combo_subset(df, "all", ["DAPI", "DAPI_round2", "GFP"])
    assert list(result.columns) == list(df.columns)


@pytest.mark.unit
def test_all_list_sentinel_returns_unchanged():
    df = _make_features()
    result = channel_combo_subset(df, ["all"], ["DAPI", "DAPI_round2", "GFP"])
    assert list(result.columns) == list(df.columns)


@pytest.mark.unit
def test_channel_agnostic_cols_always_kept():
    df = _make_features()
    result = channel_combo_subset(df, ["GFP"], ["DAPI", "DAPI_round2", "GFP"])
    assert "nucleus_area" in result.columns
    assert "cell_solidity" in result.columns


@pytest.mark.unit
def test_overlapping_channel_names_no_false_drop():
    # 'DAPI' is a substring of 'DAPI_round2'; keeping only DAPI_round2
    # must NOT accidentally drop nucleus_DAPI_round2_mean via 'DAPI' match
    df = _make_features()
    result = channel_combo_subset(df, ["DAPI_round2"], ["DAPI", "DAPI_round2", "GFP"])
    assert "nucleus_DAPI_round2_mean" in result.columns
    assert "nucleus_DAPI_mean" not in result.columns
    assert "cell_GFP_mean" not in result.columns


@pytest.mark.unit
def test_subset_drops_unwanted_channels():
    df = _make_features()
    result = channel_combo_subset(df, ["DAPI"], ["DAPI", "DAPI_round2", "GFP"])
    assert "nucleus_DAPI_mean" in result.columns
    assert "nucleus_DAPI_round2_mean" not in result.columns
    assert "cell_GFP_mean" not in result.columns


@pytest.mark.unit
def test_cross_channel_feature_needs_every_channel():
    # nucleus_correlation_DAPI_GFP mentions DAPI and GFP: keep it only when both are in
    # the combo, otherwise the other channel leaks into a single-channel partition
    df = _make_features()
    chans = ["DAPI", "DAPI_round2", "GFP"]
    assert "nucleus_correlation_DAPI_GFP" not in channel_combo_subset(df, ["DAPI"], chans).columns
    assert "nucleus_correlation_DAPI_GFP" not in channel_combo_subset(df, ["GFP"], chans).columns
    assert "nucleus_correlation_DAPI_GFP" in channel_combo_subset(df, ["DAPI", "GFP"], chans).columns
    # DAPI_round2 inside a two-channel name is not DAPI
    assert "cell_lstsq_slope_GFP_DAPI_round2" not in channel_combo_subset(df, ["DAPI", "GFP"], chans).columns
    assert "cell_lstsq_slope_GFP_DAPI_round2" in channel_combo_subset(df, ["GFP", "DAPI_round2"], chans).columns


@pytest.mark.unit
def test_channels_of():
    chans = ["DAPI", "DAPI_round2", "GFP"]
    assert channels_of("nucleus_DAPI_mean", chans) == {"DAPI"}
    assert channels_of("nucleus_DAPI_round2_mean", chans) == {"DAPI_round2"}
    assert channels_of("cell_correlation_DAPI_DAPI_round2", chans) == {"DAPI", "DAPI_round2"}
    assert channels_of("cell_area", chans) == set()


@pytest.mark.unit
def test_parse_channel_combo_handles_underscored_channel_names():
    chans = ["MAP2", "STMN2_CE", "STMN2_FL", "TDP43", "DAPI", "DAPI_round2"]
    assert parse_channel_combo("TDP43_STMN2_CE_STMN2_FL", chans) == ["TDP43", "STMN2_CE", "STMN2_FL"]
    assert parse_channel_combo("DAPI_DAPI_round2", chans) == ["DAPI", "DAPI_round2"]
    assert parse_channel_combo("DAPI_round2_DAPI", chans) == ["DAPI_round2", "DAPI"]
    assert parse_channel_combo("all", chans) == ["all"]
    with pytest.raises(ValueError):
        parse_channel_combo("TDP43_STMN2", chans)
