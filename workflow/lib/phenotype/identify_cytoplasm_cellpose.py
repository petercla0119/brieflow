"""Function for identifying and isolating the cytoplasm region."""

import numpy as np


def identify_cytoplasm_cellpose(nuclei, cells):
    """Identifies and isolates the cytoplasm region in an image based on the provided nuclei and cells masks.

    Args:
        nuclei (ndarray): A 2D array representing the nuclei regions.
        cells (ndarray): A 2D array representing the cells regions.

    Returns:
        ndarray: A 2D array representing the cytoplasm regions.
    """
    # Check if the number of unique labels in nuclei and cells are the same
    if len(np.unique(nuclei)) != len(np.unique(cells)):
        return None  # Break out of the function if the masks are not compatible

    # Vectorized replacement for the former per-label Python loop.
    #
    # The old loop iterated every cell label L in ascending order and, for each,
    # wrote L over `cells == L` then zeroed `nuclei == L`. Because iterations ran
    # in ascending label order and each overwrote the previous, the net per-pixel
    # result was: keep the cell label C, UNLESS a nucleus label N (>0) with N >= C
    # covers the pixel. (Its own nucleus, N == C, zeros it within the same pass; a
    # larger-labelled nucleus, N > C, is processed later and also zeros it. A
    # smaller-labelled nucleus, N < C, is overwritten by the later cell-C pass and
    # therefore left as C — a label-order quirk, preserved here for output parity.)
    #
    # `(nuclei > 0) & (nuclei >= cells)` reproduces that mask exactly in a single
    # pass, eliminating the O(N_cells * N_pixels) `np.argwhere` loop (~119 s/tile
    # -> milliseconds). Bit-identity is proven on synthetic masks in
    # test_identify_cytoplasm_cellpose.py and verified on real plate-4 tiles in
    # equivalence_identify_cytoplasm.py.
    #
    # ponytail: assumes nuclei/cells share the same label set (true for matched
    # cellpose masks; the unique-count gate above is a weak proxy). If a nucleus
    # label ever exists that is absent from `cells`, the old loop never zeroed it
    # while this expression may — the real-tile equivalence check guards that.
    cytoplasms = np.where((nuclei > 0) & (nuclei >= cells), 0, cells)

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)
