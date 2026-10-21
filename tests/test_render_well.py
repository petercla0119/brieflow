"""Positions-driven whole-well mosaic (#88): geometry, blending, placement round-trip, zarr.

Synthetic only -- tile "images" are generated in memory from the simulated screen in
tests/test_positions_merge.py, whose placement includes a flip, a rotation and a radial
term, so the inverse mapping is exercised end to end without any files.

Run: python -m pytest -q tests/test_render_well.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from test_positions_merge import TILE_PX, make_screen, run_full  # noqa: E402
from lib.merge.positions_merge import tile_pixel_to_um  # noqa: E402
from lib.merge.render_well import (  # noqa: E402
    canvas_geometry,
    edt_weights,
    render_chunk,
    render_well,
    um_to_canvas,
    write_well_zarr,
)


@pytest.fixture(scope="module")
def placement():
    ph, sbs = make_screen(seed=4, orientation=(True, False, 1), radial_k1=0.02)
    _merged, _qc, placement = run_full(ph, sbs, (True, False, 1))
    assert placement is not None
    return placement["phenotype"]


def _constant_tiles(placement, value=None):
    """read_tile callable: every tile is a constant plane (its own id unless `value`)."""
    shape = placement["dimensions"]

    def read_tile(tile):
        return np.full(shape, float(tile) if value is None else value, dtype=np.float32)

    return read_tile


def test_canvas_geometry_matches_tile_corner_extremes(placement):
    geo = canvas_geometry(placement)
    assert geo["um_per_px"] == placement["pixel_size"]
    h, w = placement["dimensions"]
    corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], float)
    xy = np.concatenate(
        [tile_pixel_to_um(placement, t, corners) for t in placement["tiles"].index]
    )
    assert np.all(geo["origin_um"] <= xy.min(axis=0))
    rows, cols = geo["shape"]
    rc = um_to_canvas(geo, xy)
    assert rc[:, 0].max() < rows and rc[:, 1].max() < cols
    assert rc.min() >= -1.0
    assert set(geo["tile_boxes"]) == set(placement["tiles"].index)


def test_edt_weights_zero_at_border_peak_inside():
    w = edt_weights((21, 31))
    assert w.shape == (21, 31)
    assert w[0].max() < 1e-5 and w[:, -1].max() < 1e-5
    assert w[10, 15] == w.max()


def test_single_tile_pixels_keep_their_id_and_overlaps_blend(placement):
    geo = canvas_geometry(placement)
    canvas = render_chunk(
        placement, geo, _constant_tiles(placement), 0, 0, geo["shape"]
    )
    h, w = placement["dimensions"]
    centre = np.array([[(h - 1) / 2, (w - 1) / 2]])
    ids = {t: float(t) for t in placement["tiles"].index}
    hit = 0
    for t in placement["tiles"].index:
        r, c = np.rint(
            um_to_canvas(geo, tile_pixel_to_um(placement, t, centre))[0]
        ).astype(int)
        if 0 <= r < canvas.shape[0] and 0 <= c < canvas.shape[1]:
            np.testing.assert_allclose(canvas[r, c], ids[t], atol=1e-3)
            hit += 1
    assert hit == len(ids)
    # every rendered pixel is a convex combination of tile ids: within [min, max] of ids
    inside = canvas[canvas > 0]
    assert (
        inside.min() >= min(ids.values()) - 1e-3
        and inside.max() <= max(ids.values()) + 1e-3
    )


def test_weights_normalise_to_one_inside_the_union(placement):
    geo = canvas_geometry(placement)
    ones = render_chunk(
        placement, geo, _constant_tiles(placement, 1.0), 0, 0, geo["shape"]
    )
    covered = ones > 0
    assert covered.mean() > 0.5
    np.testing.assert_allclose(ones[covered], 1.0, atol=1e-5)


def test_bright_pixel_lands_where_the_placement_says(placement):
    geo = canvas_geometry(placement)
    tile = int(placement["tiles"].index[len(placement["tiles"]) // 2])
    src = np.array([[123.0, 77.0]])
    h, w = placement["dimensions"]

    def read_tile(t):
        img = np.zeros((h, w), dtype=np.float32)
        if t == tile:
            img[123, 77] = 1000.0
        return img

    r_exp, c_exp = um_to_canvas(geo, tile_pixel_to_um(placement, tile, src))[0]
    r0, c0 = int(r_exp) - 8, int(c_exp) - 8
    chunk = render_chunk(placement, geo, read_tile, r0, c0, (17, 17))
    assert chunk.max() > 0
    rr, cc = np.nonzero(chunk > 0)
    weights = chunk[rr, cc]
    r_got = r0 + (rr * weights).sum() / weights.sum()
    c_got = c0 + (cc * weights).sum() / weights.sum()
    assert abs(r_got - r_exp) < 0.6 and abs(c_got - c_exp) < 0.6


def test_render_well_writes_multiscale_zarr_with_canvas_attrs(placement, tmp_path):
    out = tmp_path / "phenotype_well.zarr"
    geo = write_well_zarr(
        placement,
        {"DAPI": _constant_tiles(placement, 7.0)},
        out,
        max_levels=3,
        chunk=512,
        dtype=np.uint16,
    )
    g = zarr.open_group(str(out), mode="r")
    lvl0 = g["0"]
    assert lvl0.shape == (1,) + tuple(geo["shape"])
    assert g["1"].shape[1:] == tuple(s // 2 for s in geo["shape"])
    attrs = g.attrs.asdict()
    assert attrs["positions_canvas"]["um_per_px"] == pytest.approx(geo["um_per_px"])
    assert list(attrs["positions_canvas"]["origin_um"]) == pytest.approx(
        list(geo["origin_um"])
    )
    data = lvl0[0]
    assert data.dtype == np.uint16 and data.max() == 7
    # dask path: render_well returns a lazy array of the canvas shape
    arr = render_well(placement, _constant_tiles(placement, 1.0), chunk=512)
    assert arr.shape == tuple(geo["shape"]) and arr.dtype == np.float32
