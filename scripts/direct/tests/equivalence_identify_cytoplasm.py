"""Integration/equivalence check on REAL plate-4 masks.

For a sample of real (well, tile) label pairs it runs:
  - _loop_reference    : verbatim old per-label Python loop (the oracle)
  - identify_cytoplasm : the vectorized production fn (imported from lib)
  - _simple_correct    : np.where(nuclei>0,0,cells) (the 'more correct' variant)

and reports, per tile: cell count, wall time each, pixel diffs (vec vs loop),
and pixel diffs (simple vs loop). A clean run = 0 vec-vs-loop diffs on every
tile (bit-identity confirmed on real data) plus the speedup factor.

Run on broad-cpu inside the perf worktree:
  cd <worktree> && PYTHONPATH=workflow python scripts/direct/tests/equivalence_identify_cytoplasm.py
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# --- resolve worktree + config ------------------------------------------------
WT = os.environ.get("WT", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(WT) / "workflow"))

from lib.shared.file_utils import get_image_output_path  # noqa: E402
from lib.shared.image_io import read_image  # noqa: E402
from lib.phenotype.identify_cytoplasm_cellpose import identify_cytoplasm_cellpose  # noqa: E402

ANALYSIS = os.environ.get(
    "ANALYSIS", "/mnt/work/broad-analysis/broad-tdp-gws/analysis"
)
PHEN_FP = Path(ANALYSIS) / "brieflow_output" / "phenotype"
PLATE = 4
N_SAMPLE = int(os.environ.get("N_SAMPLE", "8"))


def make_loc(img_fmt, plate, well=None, tile=None):
    loc = {"plate": str(plate)}
    if well is not None:
        w = str(well)
        if img_fmt == "zarr":
            m = re.match(r"^([A-Za-z]+)(\d+)$", w)
            loc["row"] = m.group(1) if m else w
            loc["col"] = m.group(2) if m else "0"
        else:
            loc["well"] = w
    if tile is not None:
        loc["tile"] = str(tile)
    return loc


def img_path(well, tile, info_type):
    loc = make_loc("zarr", PLATE, well, tile)
    return str(PHEN_FP / get_image_output_path(loc, info_type, "zarr", subdirectory="labels"))


def _loop_reference(nuclei, cells):
    """Verbatim old implementation (oracle)."""
    if len(np.unique(nuclei)) != len(np.unique(cells)):
        return None
    cytoplasms = np.zeros(cells.shape)
    for cell_label in np.unique(cells):
        if cell_label == 0:
            continue
        nucleus_coords = np.argwhere(nuclei == cell_label)
        cell_coords = np.argwhere(cells == cell_label)
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label
        cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0
    return cytoplasms.astype(int)


def _simple_correct(nuclei, cells):
    return np.where(nuclei > 0, 0, cells).astype(int)


def discover_tiles():
    """Find real (well, tile) pairs by scanning the zarr HCS tree.

    Real layout: <...>.zarr/<row>/<col>/<tile>/labels/nuclei — so the zarr root
    is 4 parents above a built nuclei path (labels, tile, col, row).
    """
    sample_path = Path(img_path("B1", 0, "nuclei"))
    zarr_root = sample_path.parents[4]  # .../<info>.zarr
    print(f"[discover] sample nuclei path: {sample_path}  exists={sample_path.exists()}")
    print(f"[discover] zarr root: {zarr_root}  exists={zarr_root.exists()}")
    found = []
    if not zarr_root.exists():
        return found
    for row_dir in sorted(p for p in zarr_root.iterdir() if p.is_dir() and re.fullmatch(r"[A-Za-z]+", p.name)):
        for c_dir in sorted((p for p in row_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            for t_dir in sorted((p for p in c_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
                if (t_dir / "labels" / "nuclei").exists():
                    found.append((f"{row_dir.name}{c_dir.name}", t_dir.name))
    return found


def main():
    tiles = discover_tiles()
    if not tiles:
        print("NO TILES FOUND — check label store path/layout above.")
        sys.exit(2)
    print(f"[discover] {len(tiles)} total plate-{PLATE} label tiles found")
    # spread the sample across the tile list (varied density), deterministic
    step = max(1, len(tiles) // N_SAMPLE)
    sample = tiles[::step][:N_SAMPLE]
    print(f"[sample] checking {len(sample)} tiles: {sample}\n")

    hdr = f"{'well/tile':>12} {'cells':>6} {'loop_s':>9} {'vec_s':>9} {'speedup':>8} {'vec!=loop':>10} {'simple!=loop':>13}"
    print(hdr)
    print("-" * len(hdr))

    tot_loop = tot_vec = 0.0
    max_vec_diff = 0
    tot_simple_diff = 0
    none_count = 0
    for well, tile in sample:
        try:
            nuclei = read_image(img_path(well, tile, "nuclei"))
            cells = read_image(img_path(well, tile, "cells"))
        except Exception as e:  # noqa: BLE001
            print(f"{well}/{tile:>6}  READ ERROR: {e}")
            continue
        n_cells = len(np.unique(cells)) - 1

        t0 = time.perf_counter()
        ref = _loop_reference(nuclei, cells)
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        vec = identify_cytoplasm_cellpose(nuclei, cells)
        t_vec = time.perf_counter() - t0

        if ref is None or vec is None:
            none_count += 1
            ok = (ref is None) and (vec is None)
            print(f"{well}/{str(tile):>6} {n_cells:>6} {'None-gate':>9} {'':>9} {'':>8} {('OK' if ok else 'MISMATCH'):>10}")
            continue

        vec_diff = int(np.count_nonzero(vec != ref))
        simple = _simple_correct(nuclei, cells)
        simple_diff = int(np.count_nonzero(simple != ref))

        tot_loop += t_loop
        tot_vec += t_vec
        max_vec_diff = max(max_vec_diff, vec_diff)
        tot_simple_diff += simple_diff
        speed = (t_loop / t_vec) if t_vec > 0 else float("inf")
        print(f"{well}/{str(tile):>6} {n_cells:>6} {t_loop:>9.3f} {t_vec:>9.4f} {speed:>7.0f}x {vec_diff:>10} {simple_diff:>13}")

    print("-" * len(hdr))
    print(f"\nTOTALS: loop={tot_loop:.1f}s  vec={tot_vec:.3f}s  "
          f"overall_speedup={ (tot_loop/tot_vec) if tot_vec>0 else float('inf'):.0f}x")
    print(f"max (vec != loop) pixels on any tile: {max_vec_diff}   <-- MUST be 0 for bit-identity")
    print(f"total (simple != loop) pixels across sample: {tot_simple_diff}   "
          f"(characterizes the N<C label-order quirk; informational)")
    print(f"None-gate tiles: {none_count}")
    verdict = "BIT-IDENTICAL on real data" if max_vec_diff == 0 else "DIVERGENCE — investigate"
    print(f"\nVERDICT: {verdict}")
    sys.exit(0 if max_vec_diff == 0 else 1)


if __name__ == "__main__":
    main()
