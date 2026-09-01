"""Unit tests for the vectorized identify_cytoplasm_cellpose.

The original implementation was a per-cell-label Python loop. It is reproduced
verbatim below as `_loop_reference` and used as the correctness oracle: the
vectorized production function must be BIT-IDENTICAL to it for every mask whose
nuclei/cells share the same label set (the realistic, matched-cellpose case).

Run: pytest -q test_identify_cytoplasm_cellpose.py
"""

import numpy as np
import pytest

from lib.phenotype.identify_cytoplasm_cellpose import identify_cytoplasm_cellpose


def _loop_reference(nuclei, cells):
    """Verbatim copy of the original loop implementation (the oracle)."""
    if len(np.unique(nuclei)) != len(np.unique(cells)):
        return None
    cytoplasms = np.zeros(cells.shape)
    for cell_label in np.unique(cells):
        if cell_label == 0:
            continue
        nucleus_label = cell_label
        nucleus_coords = np.argwhere(nuclei == nucleus_label)
        cell_coords = np.argwhere(cells == cell_label)
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label
        cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0
    return cytoplasms.astype(int)


def _simple_correct(nuclei, cells):
    """The 'definitionally correct' cytoplasm: cell region minus ANY nucleus.

    Not what the old loop computed (it has a label-order quirk); kept here only
    to characterize the divergence, not as the production behavior.
    """
    return np.where(nuclei > 0, 0, cells).astype(int)


# --- bit-identity: vectorized production fn == old loop, matched label sets ----


def _assert_identical(nuclei, cells):
    got = identify_cytoplasm_cellpose(nuclei, cells)
    ref = _loop_reference(nuclei, cells)
    if ref is None:
        assert got is None
        return
    assert got is not None
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"mismatch at {np.argwhere(got != ref).tolist()[:10]}"
    )


def test_clean_nested_nucleus_inside_own_cell():
    # cell 1 fills a block; its nucleus (label 1) sits fully inside it.
    cells = np.zeros((8, 8), dtype=np.uint32)
    cells[1:7, 1:7] = 1
    nuclei = np.zeros((8, 8), dtype=np.uint32)
    nuclei[3:5, 3:5] = 1
    _assert_identical(nuclei, cells)
    # sanity: cytoplasm = cell minus its nucleus
    out = identify_cytoplasm_cellpose(nuclei, cells)
    assert (out[3:5, 3:5] == 0).all()
    assert (out[1:3, 1:7] == 1).all()


def test_two_cells_matched_nuclei():
    cells = np.zeros((10, 10), dtype=np.uint32)
    cells[0:5, 0:5] = 1
    cells[5:10, 5:10] = 2
    nuclei = np.zeros((10, 10), dtype=np.uint32)
    nuclei[1:3, 1:3] = 1
    nuclei[6:8, 6:8] = 2
    _assert_identical(nuclei, cells)


def test_overlap_smaller_nucleus_label_in_larger_cell_NltC():
    # A nucleus labelled 1 overlaps into cell 2's territory (N=1 < C=2).
    # The old loop leaves those pixels as 2 (cell-2 pass runs after nucleus-1
    # zeroing). The vectorized fn must match that quirk.
    cells = np.zeros((6, 6), dtype=np.uint32)
    cells[:, 0:3] = 1
    cells[:, 3:6] = 2
    nuclei = np.zeros((6, 6), dtype=np.uint32)
    nuclei[2:4, 1:2] = 1  # inside cell 1
    nuclei[2:4, 3:5] = 2  # inside cell 2 (its own nucleus)
    nuclei[2:4, 2:3] = 1  # nucleus 1 spills into a pixel that is cell 2? col2 is cell1
    _assert_identical(nuclei, cells)


def test_overlap_larger_nucleus_label_NgtC():
    # nucleus label 2 overlaps a pixel belonging to cell 1 (N=2 > C=1) -> zeroed.
    cells = np.zeros((6, 6), dtype=np.uint32)
    cells[:, 0:3] = 1
    cells[:, 3:6] = 2
    nuclei = np.zeros((6, 6), dtype=np.uint32)
    nuclei[2:4, 1:2] = 1
    nuclei[2:4, 4:5] = 2
    # force a boundary pixel: nucleus 2 lands on a cell-1 pixel
    cells[0, 5] = 1
    nuclei[0, 5] = 2
    # keep unique-count gate satisfiable
    _assert_identical(nuclei, cells)


def test_all_background():
    z = np.zeros((5, 5), dtype=np.uint32)
    _assert_identical(z, z)
    assert (identify_cytoplasm_cellpose(z, z) == 0).all()


def test_incompatible_unique_counts_returns_none():
    cells = np.zeros((5, 5), dtype=np.uint32)
    cells[0:2, 0:2] = 1
    cells[3:5, 3:5] = 2
    nuclei = np.zeros((5, 5), dtype=np.uint32)
    nuclei[0:2, 0:2] = 1  # only 1 nucleus label -> unique counts differ (2 vs 3)
    assert identify_cytoplasm_cellpose(nuclei, cells) is None
    assert _loop_reference(nuclei, cells) is None


def test_dtype_and_labels_preserved():
    cells = np.zeros((8, 8), dtype=np.uint32)
    cells[1:7, 1:7] = 5  # non-contiguous label value
    nuclei = np.zeros((8, 8), dtype=np.uint32)
    nuclei[3:5, 3:5] = 5
    out = identify_cytoplasm_cellpose(nuclei, cells)
    ref = _loop_reference(nuclei, cells)
    assert np.array_equal(out, ref)
    assert set(np.unique(out).tolist()) == {0, 5}


@pytest.mark.parametrize("seed", range(8))
def test_randomized_matched_masks_bit_identical(seed):
    # Random non-overlapping rectangular cells on a 0 background, each with a
    # strictly-interior nucleus of the same label. Both masks share background 0
    # and label set {0,1..k} by construction -> the compatibility gate passes and
    # the vectorized fn must equal the loop on every layout.
    rng = np.random.default_rng(seed)
    h = w = 40
    cells = np.zeros((h, w), dtype=np.uint32)
    nuclei = np.zeros((h, w), dtype=np.uint32)
    label = 0
    target = int(rng.integers(2, 6))
    for _ in range(60):
        if label >= target:
            break
        rh, rw = int(rng.integers(5, 11)), int(rng.integers(5, 11))
        r0, c0 = int(rng.integers(0, h - rh)), int(rng.integers(0, w - rw))
        if np.any(cells[r0 : r0 + rh, c0 : c0 + rw] != 0):
            continue  # keep cells non-overlapping
        label += 1
        cells[r0 : r0 + rh, c0 : c0 + rw] = label
        nuclei[r0 + 1 : r0 + rh - 1, c0 + 1 : c0 + rw - 1] = label  # interior
    if label < 2:
        pytest.skip("degenerate random layout; too few placements")
    assert len(np.unique(nuclei)) == len(np.unique(cells))  # matched by design
    _assert_identical(nuclei, cells)


def test_divergence_from_simple_correct_is_characterized():
    # Documents WHY we did not just use np.where(nuclei>0,0,cells): it differs
    # from the shipped (loop-identical) behavior exactly at N<C overlap pixels.
    cells = np.zeros((4, 5), dtype=np.uint32)  # col 4 stays background 0
    cells[:, 0:2] = 1
    cells[:, 2:4] = 2
    nuclei = np.zeros((4, 5), dtype=np.uint32)
    nuclei[1:3, 2:3] = 1  # nucleus-1 pixel sitting on a cell-2 pixel (N=1<C=2)
    nuclei[1:3, 0:1] = 1  # keep label 1 present as its own
    # make label sets equal (need a nucleus 2 somewhere inside cell 2)
    nuclei[0, 3] = 2
    shipped = identify_cytoplasm_cellpose(nuclei, cells)
    simple = _simple_correct(nuclei, cells)
    ref = _loop_reference(nuclei, cells)
    assert np.array_equal(shipped, ref)  # shipped matches old behavior
    diff = np.argwhere(shipped != simple)
    # the only divergence pixels are the N<C overlap ones (shipped=2, simple=0)
    for r, c in diff:
        assert nuclei[r, c] > 0 and nuclei[r, c] < cells[r, c]
        assert shipped[r, c] == cells[r, c] and simple[r, c] == 0


def test_gate_passes_but_label_sets_differ_documents_ceiling():
    # The compatibility gate only compares unique-label COUNTS, not the label
    # SETS (the 'weak proxy' called out in the ponytail comment). Here counts
    # match (both have 3 uniques) but the sets differ: cells={0,1,2},
    # nuclei={0,1,3}. A nucleus labelled 3 sits on a cell-2 pixel.
    #
    # Old loop: iterates CELL labels {1,2} only -> nucleus label 3 is never
    # zeroed, so that pixel keeps the cell value 2.
    # Vectorized: (nuclei>0)&(nuclei>=cells) -> 3>=2 is True -> the pixel is
    # zeroed. This is the ONE documented divergence; both fns pass the gate.
    #
    # This test pins that divergence so it is a conscious, visible behavior
    # change rather than a silent surprise. On real matched cellpose masks the
    # label sets are equal and this case does not arise (guarded by the
    # real-tile equivalence harness).
    cells = np.zeros((4, 6), dtype=np.uint32)
    cells[:, 0:3] = 1
    cells[:, 3:5] = 2  # col 5 stays background 0
    nuclei = np.zeros((4, 6), dtype=np.uint32)
    nuclei[1:3, 1:2] = 1  # nucleus 1, interior to cell 1
    nuclei[1:3, 3:4] = 3  # nucleus label 3, sitting on cell-2 pixels

    assert len(np.unique(nuclei)) == len(np.unique(cells))  # gate passes
    assert set(np.unique(nuclei).tolist()) != set(np.unique(cells).tolist())

    vec = identify_cytoplasm_cellpose(nuclei, cells)
    ref = _loop_reference(nuclei, cells)
    assert vec is not None and ref is not None

    diff = np.argwhere(vec != ref)
    assert diff.size > 0, "expected the documented divergence to occur"
    # every divergent pixel is a nucleus label absent from cells, with N >= C:
    # loop keeps the cell value, vectorized zeros it.
    cell_labels = set(np.unique(cells).tolist())
    for r, c in diff:
        assert nuclei[r, c] not in cell_labels
        assert nuclei[r, c] >= cells[r, c]
        assert ref[r, c] == cells[r, c] and vec[r, c] == 0
