"""Tests for the parallelized ``feature_table_multichannel``.

Verifies that the joblib-threaded per-region computation produces results that are
identical to the original sequential implementation, across a range of ``n_jobs``
values and edge cases, and reports the parallel speedup on a realistic workload.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk

# Ensure repo root is importable (mirrors tests/test_omezarr.py convention)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.feature_table_utils import feature_table_multichannel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_labels(n_regions, shape=(200, 200), radius=15, vary_size=False, seed=0):
    """Build a labeled mask with ``n_regions`` non-overlapping disks on a grid."""
    labels = np.zeros(shape, dtype=np.int32)
    if n_regions == 0:
        return labels

    # grid layout large enough to hold n_regions disks
    per_row = int(np.ceil(np.sqrt(n_regions)))
    step_r = shape[0] // (per_row + 1)
    step_c = shape[1] // (per_row + 1)

    placed = 0
    for i in range(per_row):
        for j in range(per_row):
            if placed >= n_regions:
                break
            rr = radius + (placed % 5) if vary_size else radius
            cy = step_r * (i + 1)
            cx = step_c * (j + 1)
            row, col = disk((cy, cx), rr, shape=shape)
            labels[row, col] = placed + 1
            placed += 1
    return labels


# Custom feature dict exercising scalar, len-1-iterable, and len-N-iterable returns.
CUSTOM_FEATURES = {
    "area": lambda r: r.area,  # scalar
    "single_iter": lambda r: np.array([r.area]),  # iterable, len 1 -> col "single_iter"
    "centroid": lambda r: r.local_centroid,  # iterable, len 2 -> centroid_0, centroid_1
    "moments_hu": lambda r: r.moments_hu,  # iterable, len 7 -> moments_hu_0..6
}


def run(labels, n_jobs, features=CUSTOM_FEATURES):
    """Build a matching intensity image and run feature extraction."""
    rng = np.random.default_rng(42)
    intensity = rng.integers(0, 2**16 - 1, labels.shape, dtype=np.uint16)
    return feature_table_multichannel(intensity, labels, features, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("n_jobs", [2, 4, -1])
def test_parallel_matches_sequential(n_jobs):
    """Parallel output must be bit-identical to the n_jobs=1 sequential output."""
    labels = make_labels(12, vary_size=True)
    seq = run(labels, n_jobs=1)
    par = run(labels, n_jobs=n_jobs)
    pd.testing.assert_frame_equal(seq, par, check_exact=True)


@pytest.mark.unit
def test_scalar_and_iterable_columns():
    """Scalar, single-element and multi-element feature returns map to the right columns."""
    labels = make_labels(5)
    df = run(labels, n_jobs=2)
    assert "area" in df.columns  # scalar
    assert "single_iter" in df.columns  # len-1 iterable collapses to single column
    assert {"centroid_0", "centroid_1"} <= set(df.columns)  # len-2 iterable
    assert {f"moments_hu_{i}" for i in range(7)} <= set(df.columns)  # len-7 iterable
    assert len(df) == 5


@pytest.mark.unit
def test_one_region():
    labels = make_labels(1)
    seq = run(labels, n_jobs=1)
    par = run(labels, n_jobs=4)
    assert len(seq) == 1
    pd.testing.assert_frame_equal(seq, par, check_exact=True)


@pytest.mark.unit
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_zero_regions(n_jobs):
    """An empty mask must not crash and returns an empty frame."""
    labels = make_labels(0)
    df = run(labels, n_jobs=n_jobs)
    assert len(df) == 0


@pytest.mark.unit
def test_varying_sizes_identical():
    labels = make_labels(20, shape=(300, 300), vary_size=True)
    seq = run(labels, n_jobs=1)
    par = run(labels, n_jobs=-1)
    pd.testing.assert_frame_equal(seq, par, check_exact=True)


# ---------------------------------------------------------------------------
# Integration test (real brieflow feature dictionaries)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_real_feature_dicts_parallel_equivalence_and_speedup():
    from workflow.lib.external.cp_emulator import (
        grayscale_features_multichannel,
        correlation_features_multichannel,
        shape_features,
    )

    n_channels = 8
    shape = (500, 500)
    rng = np.random.default_rng(7)

    # Synthetic 8-channel image with ~36 labeled regions.
    data = rng.integers(0, 2**16 - 1, (n_channels, *shape), dtype=np.uint16)
    labels = make_labels(36, shape=shape, radius=22, vary_size=True)
    n_regions = int(labels.max())
    assert 20 <= n_regions <= 50

    features = {
        **grayscale_features_multichannel,
        **correlation_features_multichannel,
        **shape_features,
    }

    t0 = time.perf_counter()
    seq = feature_table_multichannel(data, labels, features, n_jobs=1)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    par = feature_table_multichannel(data, labels, features, n_jobs=4)
    t_par4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    par16 = feature_table_multichannel(data, labels, features, n_jobs=16)
    t_par16 = time.perf_counter() - t0

    pd.testing.assert_frame_equal(seq, par, check_exact=False, atol=1e-6)
    pd.testing.assert_frame_equal(seq, par16, check_exact=False, atol=1e-6)

    print(
        f"\n[integration] regions={n_regions} cols={seq.shape[1]}\n"
        f"  n_jobs=1  : {t_seq:6.2f}s\n"
        f"  n_jobs=4  : {t_par4:6.2f}s  (speedup {t_seq / t_par4:.2f}x)\n"
        f"  n_jobs=16 : {t_par16:6.2f}s  (speedup {t_seq / t_par16:.2f}x)"
    )
