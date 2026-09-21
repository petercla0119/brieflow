import numpy as np

from lib.phenotype.extract_phenotype_cp_multichannel import remove_border
from lib.shared.segment_watershed import filter_by_region, find_cells


def test_remove_border_clears_selected_labels():
    labels = np.array([[1, 1, 0], [2, 2, 3], [0, 3, 3]], dtype=int)
    mask = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)

    result = remove_border(labels, mask, dilate=1)

    expected = np.array([[0, 0, 0], [2, 2, 0], [0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_filter_by_region_removes_regions_failing_boolean_score():
    labeled = np.array([[1, 1, 0], [0, 2, 2]], dtype=int)

    result = filter_by_region(
        labeled,
        score=lambda region: region.label == 1,
        threshold=0,
        relabel=False,
    )

    expected = np.array([[1, 1, 0], [0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_find_cells_removes_boundary_labels(monkeypatch):
    watershed_labels = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 3, 3, 3, 2],
            [1, 3, 3, 3, 2],
            [1, 3, 3, 3, 2],
            [1, 2, 2, 2, 2],
        ],
        dtype=int,
    )

    monkeypatch.setattr(
        "lib.shared.segment_watershed.watershed",
        lambda distance, nuclei, mask: watershed_labels.copy(),
    )

    result = find_cells(
        np.zeros_like(watershed_labels), np.ones_like(watershed_labels, dtype=bool)
    )

    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(result, expected)
