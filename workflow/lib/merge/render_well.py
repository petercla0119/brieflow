"""Positions-driven whole-well mosaic: render aligned tiles into one well canvas.

Uses the placement fitted by `positions_merge` (camera matrix, radial terms, per-tile
shifts) to inverse-map every canvas pixel to source tile pixels, so the mosaic is placed
exactly where the merge placed the cells. Overlapping tiles are blended with an
Euclidean-distance-to-edge weight, which hides seams without registering images again.
Rendering is chunk-wise (one canvas block + one tile plane in memory) and streams into
`write_image_omezarr` as a dask array, so a full-resolution well never exists in RAM.

Canvas convention: pixel (row, col) covers
``origin_um + (col, row) * um_per_px .. + um_per_px``; `um_to_canvas` / `canvas_to_um`
convert between micrometres and fractional pixel indices (pixel centres at integers).
"""

from functools import lru_cache

import dask.array as da
import numpy as np
from scipy import ndimage

from lib.merge.positions_merge import tile_pixel_to_um, um_to_tile_pixel

BOX_MARGIN_PX = 8  # radial distortion can push a tile edge past its mapped corners


def canvas_geometry(placement, pad_px=0):
    """Canvas frame for one modality: origin, pixel size, shape and per-tile pixel boxes.

    Args:
        placement (dict): One modality's placement (see `positions_merge`).
        pad_px (int): Extra border around the tile footprints.

    Returns:
        dict: ``origin_um`` (x, y), ``um_per_px``, ``shape`` (rows, cols),
            ``tile_boxes`` {tile: (row0, row1, col0, col1)} inclusive canvas-pixel bounds.
    """
    h, w = placement["dimensions"]
    um_per_px = float(placement["pixel_size"])
    corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], float)
    tiles = list(placement["tiles"].index)
    xy = {t: tile_pixel_to_um(placement, t, corners) for t in tiles}
    allxy = np.concatenate(list(xy.values()))
    origin = allxy.min(axis=0) - (BOX_MARGIN_PX + pad_px) * um_per_px
    extent = allxy.max(axis=0) + (BOX_MARGIN_PX + pad_px) * um_per_px - origin
    shape = (
        int(np.ceil(extent[1] / um_per_px)) + 1,
        int(np.ceil(extent[0] / um_per_px)) + 1,
    )
    geo = {
        "origin_um": origin,
        "um_per_px": um_per_px,
        "shape": shape,
        "tile_boxes": {},
    }
    for t in tiles:
        rc = um_to_canvas(geo, xy[t])
        lo = np.floor(rc.min(axis=0)).astype(int) - BOX_MARGIN_PX
        hi = np.ceil(rc.max(axis=0)).astype(int) + BOX_MARGIN_PX
        geo["tile_boxes"][int(t)] = (
            max(lo[0], 0),
            min(hi[0], shape[0] - 1),
            max(lo[1], 0),
            min(hi[1], shape[1] - 1),
        )
    return geo


def um_to_canvas(geometry, xy_um):
    """(n, 2) x/y in um -> (n, 2) fractional (row, col) canvas pixels."""
    xy = np.atleast_2d(np.asarray(xy_um, dtype=float))
    px = (xy - geometry["origin_um"]) / geometry["um_per_px"] - 0.5
    return px[:, ::-1]


def canvas_to_um(geometry, rc):
    """(n, 2) fractional (row, col) canvas pixels -> (n, 2) x/y in um."""
    rc = np.atleast_2d(np.asarray(rc, dtype=float))
    return (rc[:, ::-1] + 0.5) * geometry["um_per_px"] + geometry["origin_um"]


@lru_cache(maxsize=4)
def edt_weights(shape, exponent=1.0):
    """Blend weight per tile pixel: distance to the tile border (0 on the border)."""
    mask = np.zeros(shape, dtype=bool)
    if shape[0] > 2 and shape[1] > 2:
        mask[1:-1, 1:-1] = True
    dist = ndimage.distance_transform_edt(mask).astype(np.float32)
    return dist**exponent


def render_chunk(placement, geometry, read_tile, row0, col0, shape, order=1):
    """Render one canvas block by inverse-mapping it through every tile that touches it.

    Args:
        placement (dict): One modality's placement.
        geometry (dict): From `canvas_geometry`.
        read_tile (callable): tile id -> 2D plane (any numeric dtype).
        row0, col0 (int): Block origin in canvas pixels.
        shape (tuple): Block (rows, cols).
        order (int): Interpolation order for `map_coordinates` (1 = bilinear).

    Returns:
        numpy.ndarray: float32 block; 0 where no tile covers the pixel.
    """
    rows, cols = shape
    acc = np.zeros(shape, dtype=np.float32)
    wacc = np.zeros(shape, dtype=np.float32)
    for tile, (tr0, tr1, tc0, tc1) in geometry["tile_boxes"].items():
        if tr1 < row0 or tr0 >= row0 + rows or tc1 < col0 or tc0 >= col0 + cols:
            continue
        rr0, rr1 = max(row0, tr0), min(row0 + rows, tr1 + 1)
        cc0, cc1 = max(col0, tc0), min(col0 + cols, tc1 + 1)
        ii, jj = np.mgrid[rr0:rr1, cc0:cc1]
        src = um_to_tile_pixel(
            placement,
            tile,
            canvas_to_um(geometry, np.column_stack([ii.ravel(), jj.ravel()])),
        )
        image = np.asarray(read_tile(tile), dtype=np.float32)
        h, w = image.shape
        valid = (
            (src[:, 0] >= 0)
            & (src[:, 0] <= h - 1)
            & (src[:, 1] >= 0)
            & (src[:, 1] <= w - 1)
        )
        if not valid.any():
            continue
        coords = src.T
        values = ndimage.map_coordinates(
            image, coords, order=order, mode="constant", cval=0.0
        )
        weights = ndimage.map_coordinates(
            edt_weights((h, w)), coords, order=1, mode="constant", cval=0.0
        )
        weights = np.where(valid, weights, 0.0).astype(np.float32)
        window = (slice(rr0 - row0, rr1 - row0), slice(cc0 - col0, cc1 - col0))
        acc[window] += (values * weights).reshape(ii.shape)
        wacc[window] += weights.reshape(ii.shape)
    return np.where(wacc > 0, acc / np.maximum(wacc, 1e-12), 0.0).astype(np.float32)


def render_well(placement, read_tile, chunk=2048, geometry=None):
    """Lazy dask canvas (rows, cols) of one plane of one modality."""
    geo = geometry or canvas_geometry(placement)
    rows, cols = geo["shape"]

    def _block(block, block_info=None):
        (r0, _), (c0, _) = block_info[None]["array-location"]
        return render_chunk(placement, geo, read_tile, r0, c0, block.shape)

    template = da.zeros((rows, cols), chunks=(chunk, chunk), dtype=np.float32)
    return da.map_blocks(_block, template, dtype=np.float32)


def write_well_zarr(
    placement,
    channels,
    out_path,
    max_levels=5,
    chunk=2048,
    dtype=np.uint16,
    compression=None,
    extra_attrs=None,
):
    """Render named channels of one modality into a multiscale OME-Zarr (axes CYX).

    Args:
        placement (dict): One modality's placement.
        channels (dict): {channel name: read_tile callable} in output order.
        out_path (str | Path): Output .zarr group.
        max_levels (int): Pyramid levels.
        chunk (int): Canvas block size in pixels.
        dtype: Output dtype; floats are rounded and clipped for integer dtypes.
        compression (str | None): Codec spec for `write_image_omezarr`.
        extra_attrs (dict | None): Merged into the ``positions_canvas`` attribute.

    Returns:
        dict: The canvas geometry used.
    """
    import zarr

    from lib.shared.image_io import write_image_omezarr

    geo = canvas_geometry(placement)
    planes = [
        render_well(placement, reader, chunk=chunk, geometry=geo)
        for reader in channels.values()
    ]
    stack = da.stack(planes, axis=0)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        stack = da.clip(da.rint(stack), info.min, info.max)
    stack = stack.astype(dtype)
    write_image_omezarr(
        stack,
        str(out_path),
        channel_names=list(channels),
        axes="CYX",
        pixel_size_um=geo["um_per_px"],
        max_levels=max_levels,
        chunk_size=(1, chunk, chunk),
        compression=compression,
    )
    attrs = {
        "origin_um": [float(v) for v in geo["origin_um"]],
        "um_per_px": float(geo["um_per_px"]),
        "shape": [int(v) for v in geo["shape"]],
        "orientation": [
            bool(placement["orientation"][0]),
            bool(placement["orientation"][1]),
            int(placement["orientation"][2]),
        ],
    }
    attrs.update(extra_attrs or {})
    group = zarr.open_group(str(out_path), mode="a")
    group.attrs["positions_canvas"] = attrs
    return geo
