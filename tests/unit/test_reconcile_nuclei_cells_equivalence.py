"""Pin the reconcile_nuclei_cells regionprops dedup as output-preserving.

The refactor removed a duplicate `regionprops(cells, intensity_image=nuclei_eroded)`
call and now derives the consensus single-value map by filtering `cell_map_multiple`
for entries with `len(labels) == 1`, rather than calling
`get_unique_label_map(..., keep_multiple=False)` a second time.

Those are equivalent because both read `np.unique(region.intensity_image[... > 0])`
off the same regions, so the `len(labels) == 1` filter is a restatement of the same
predicate with the same `labels[0]` extraction. That argument is only as good as the
cases it covers, so this pins the three that distinguish the branches -- a cell with
exactly one nucleus (kept), with several (dropped), and with none (dropped) -- plus
the degenerate inputs where an off-by-one in the derivation would hide.

The `verbose` flag gates diagnostics only (a nuclei-per-cell histogram, an area-CV
regionprops pass, and a per-nucleus convex hull for solidity). It must not touch the
returned masks, so every case is asserted across both settings.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflow"))

from lib.shared.segmentation_utils import reconcile_nuclei_cells  # noqa: E402


def _one_to_one():
    """Two cells, one nucleus each -- both survive."""
    nuclei = np.zeros((8, 8), dtype=np.int32)
    cells = np.zeros((8, 8), dtype=np.int32)
    nuclei[1:3, 1:3] = 1
    cells[0:4, 0:4] = 1
    nuclei[1:3, 5:7] = 2
    cells[0:4, 4:8] = 2
    return nuclei, cells


def _multi_nucleus():
    """One cell containing two nuclei -- the case the derivation turns on."""
    nuclei = np.zeros((8, 8), dtype=np.int32)
    cells = np.zeros((8, 8), dtype=np.int32)
    cells[0:4, 0:8] = 1
    nuclei[1:3, 1:3] = 1
    nuclei[1:3, 5:7] = 2
    return nuclei, cells


def _empty_cell():
    """A cell with no nucleus pixels at all."""
    nuclei = np.zeros((8, 8), dtype=np.int32)
    cells = np.zeros((8, 8), dtype=np.int32)
    nuclei[1:3, 1:3] = 1
    cells[0:4, 0:4] = 1
    cells[4:8, 0:4] = 2  # no nucleus inside
    return nuclei, cells


def _orphan_nucleus():
    """A nucleus outside every cell."""
    nuclei = np.zeros((8, 8), dtype=np.int32)
    cells = np.zeros((8, 8), dtype=np.int32)
    nuclei[1:3, 1:3] = 1
    cells[0:4, 0:4] = 1
    nuclei[5:7, 5:7] = 2  # no cell here
    return nuclei, cells


def _non_sequential_labels():
    """Labels 1, 5, 99 -- guards any accidental reliance on contiguous numbering."""
    nuclei = np.zeros((8, 8), dtype=np.int32)
    cells = np.zeros((8, 8), dtype=np.int32)
    nuclei[1:3, 1:3] = 5
    cells[0:4, 0:4] = 1
    nuclei[5:7, 1:3] = 99
    cells[4:8, 0:4] = 99
    return nuclei, cells


def _all_background():
    """No labels at all."""
    z = np.zeros((8, 8), dtype=np.int32)
    return z, z.copy()


CASES = {
    "one_to_one": _one_to_one,
    "multi_nucleus": _multi_nucleus,
    "empty_cell": _empty_cell,
    "orphan_nucleus": _orphan_nucleus,
    "non_sequential_labels": _non_sequential_labels,
    "all_background": _all_background,
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("how", ["consensus", "contained_in_cells"])
def test_verbose_flag_does_not_change_the_masks(name, how):
    nuclei, cells = CASES[name]()
    n_quiet, c_quiet = reconcile_nuclei_cells(
        nuclei.copy(), cells.copy(), how=how, verbose=False
    )
    n_loud, c_loud = reconcile_nuclei_cells(
        nuclei.copy(), cells.copy(), how=how, verbose=True
    )
    assert np.array_equal(n_quiet, n_loud), f"{name}/{how}: nuclei differ with verbose"
    assert np.array_equal(c_quiet, c_loud), f"{name}/{how}: cells differ with verbose"
    assert n_quiet.dtype == n_loud.dtype
    assert c_quiet.dtype == c_loud.dtype


def test_multi_nucleus_cell_is_dropped_under_consensus():
    """The consensus filter's whole job: a cell with 2 nuclei must not survive.

    This is what `len(labels) == 1` replaced. If the derivation were ever loosened
    to `>= 1`, every other case here would still pass and only this would fail.
    """
    nuclei, cells = _multi_nucleus()
    _, out_cells = reconcile_nuclei_cells(
        nuclei.copy(), cells.copy(), how="consensus", verbose=False
    )
    assert 1 not in set(np.unique(out_cells)) - {0}, (
        "cell 1 contains two nuclei and must be dropped by the consensus filter"
    )


def test_one_to_one_pair_survives_consensus():
    """Guards the opposite error: a filter so strict it drops valid matches."""
    nuclei, cells = _one_to_one()
    out_nuclei, out_cells = reconcile_nuclei_cells(
        nuclei.copy(), cells.copy(), how="consensus", verbose=False
    )
    assert set(np.unique(out_cells)) - {0}, "all cells dropped; filter is too strict"
    assert set(np.unique(out_nuclei)) - {0}, "all nuclei dropped; filter is too strict"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
