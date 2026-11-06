"""attach_global_pixel_coords: geometry on clear-cut cases, and loud failure on bad metadata.

The TDP-GWS recomb rerun (2026-10-08) ran format_merge against 0-row
combined_metadata probes; the function returned the frame untouched and 55 wells
of merge tables shipped without x_pos/y_pos/pixel_size/global_i/global_j. These
tests pin the geometry and make that silent path a ValueError.

Run: python -m pytest -q tests/unit/test_format_merge_global_coords.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(ROOT / "workflow"))

from lib.merge.format_merge import (  # noqa: E402
    METADATA_COORD_COLS,
    attach_global_pixel_coords,
)

TILE = (100, 200)  # (height, width) px
PX = 0.5  # um / px
COORD_COLS = ["x_pos", "y_pos", "pixel_size_x", "pixel_size_y", "global_j", "global_i"]


def metadata(tiles, well="A1", plate=1, origin=(1000.0, 2000.0), rows_per_tile=1):
    """Tile centers on a 2x2 grid; tile t sits at column t % 2, row t // 2."""
    rows = []
    for t in tiles:
        cx = origin[0] + (t % 2) * TILE[1] * PX
        cy = origin[1] + (t // 2) * TILE[0] * PX
        for ch in range(rows_per_tile):
            rows.append(
                dict(
                    plate=plate,
                    well=well,
                    tile=t,
                    x_pos=cx,
                    y_pos=cy,
                    pixel_size_x=PX,
                    pixel_size_y=PX,
                    channel=ch,
                )
            )
    return pd.DataFrame(rows)


def merged(tiles, well="A1", plate=1, suffix="0"):
    """One cell per tile at pixel (i=10, j=20); suffix 1 keys on `site`."""
    key = "tile" if suffix == "0" else "site"
    return pd.DataFrame(
        {
            "plate": plate,
            "well": well,
            key: list(tiles),
            f"i_{suffix}": 10.0,
            f"j_{suffix}": 20.0,
        }
    )


def cols(suffix):
    return [f"{c}_{suffix}" for c in COORD_COLS]


def test_adds_twelve_columns_both_suffixes():
    ph = attach_global_pixel_coords(
        merged([0, 1, 2, 3]), metadata([0, 1, 2, 3]), TILE, "0"
    )
    out = attach_global_pixel_coords(
        ph.assign(site=ph["tile"], i_1=10.0, j_1=20.0),
        metadata([0, 1, 2, 3]),
        TILE,
        "1",
    )
    for s in ("0", "1"):
        assert all(c in out.columns for c in cols(s))
        assert (
            out[f"global_i_{s}"].dtype == "Int32"
            and out[f"global_j_{s}"].dtype == "Int32"
        )
        assert (
            out[f"x_pos_{s}"].dtype == float and out[f"pixel_size_y_{s}"].dtype == float
        )
    # tile t at grid (t % 2, t // 2): global = tile offset in px + pixel index
    exp_j = [(t % 2) * TILE[1] + 20 for t in range(4)]
    exp_i = [(t // 2) * TILE[0] + 10 for t in range(4)]
    assert out["global_j_0"].tolist() == exp_j and out["global_i_0"].tolist() == exp_i
    assert out["global_j_1"].tolist() == exp_j and out["global_i_1"].tolist() == exp_i


def test_anchor_is_per_plate_well_min_corner():
    m = pd.concat(
        [merged([0, 1], well="A1"), merged([0, 1], well="B2")], ignore_index=True
    )
    md = pd.concat(
        [metadata([0, 1], well="A1"), metadata([0, 1], well="B2", origin=(9e4, 9e4))]
    )
    out = attach_global_pixel_coords(m, md, TILE, "0")
    per_well = out.groupby("well")[["global_i_0", "global_j_0"]].min()
    assert (per_well["global_i_0"] == 10).all() and (per_well["global_j_0"] == 20).all()


def test_replicated_metadata_rows_do_not_multiply_cells():
    m = merged([0, 1, 2, 3])
    out = attach_global_pixel_coords(
        m, metadata([0, 1, 2, 3], rows_per_tile=4), TILE, "0"
    )
    assert len(out) == len(m)
    assert out["global_j_0"].tolist() == [20, 220, 20, 220]


def test_suffix_1_joins_on_site_not_tile():
    m = merged([0, 1], suffix="1").assign(
        tile=[7, 8]
    )  # tile ids that do not exist in metadata
    out = attach_global_pixel_coords(m, metadata([0, 1]), TILE, "1")
    assert out["global_j_1"].tolist() == [20, 220]
    assert not any(c.startswith("x_pos_0") for c in out.columns)


@pytest.mark.parametrize("bad", [None, "empty"])
def test_empty_metadata_raises(bad):
    md = (
        None if bad is None else pd.DataFrame(columns=METADATA_COORD_COLS)
    )  # the 0-row probe
    m = merged([0, 1])
    with pytest.raises(ValueError, match="phenotype metadata is empty"):
        attach_global_pixel_coords(m, md, TILE, "0")
    with pytest.raises(ValueError, match="sbs metadata is empty"):
        attach_global_pixel_coords(m, md, TILE, "1")
    assert list(m.columns) == ["plate", "well", "tile", "i_0", "j_0"]  # untouched


def test_missing_metadata_column_raises():
    with pytest.raises(ValueError, match="pixel_size_y"):
        attach_global_pixel_coords(
            merged([0]), metadata([0]).drop(columns="pixel_size_y"), TILE, "0"
        )


def test_partial_tile_coverage_warns_and_leaves_na():
    with pytest.warns(
        UserWarning, match="1 of 3 merged cells have no phenotype metadata"
    ):
        out = attach_global_pixel_coords(merged([0, 1, 2]), metadata([0, 1]), TILE, "0")
    assert out["global_j_0"].tolist()[:2] == [20, 220]
    assert pd.isna(out.loc[2, "global_i_0"]) and pd.isna(out.loc[2, "x_pos_0"])
    assert out["global_i_0"].dtype == "Int32"


def test_zero_row_merged_frame():
    out = attach_global_pixel_coords(merged([]), metadata([0, 1]), TILE, "0")
    assert len(out) == 0
    assert all(c in out.columns for c in cols("0"))
