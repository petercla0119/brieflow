"""Pin the np.in1d -> np.isin replacement (NumPy 2 compatibility).

`np.in1d` was deprecated in NumPy 1.25 and removed in 2.2. The repo pins
numpy==2.0.2, where it still works, so these call sites were live landmines
rather than current failures -- they break on any numpy bump, which is exactly
what killed the brieflow-150-gpu env (numpy 2.5).

The replacement is NOT a pure rename, and that is what these tests pin:

    np.in1d(a, b)   always flattens -> 1-D mask, len == a.size
    np.isin(a, b)   preserves shape -> mask.shape == a.shape

so `x.flat[in1d(x, y)]` becomes `x[isin(x, y)]` -- the `.flat` must be dropped
with it. Keeping `.flat` alongside `isin` on a 2-D array is either an
IndexError or, worse, a silent partial write.
"""

import numpy as np
import pytest


def test_isin_preserves_shape_while_in1d_flattens():
    """The property the whole migration turns on."""
    x = np.arange(12).reshape(3, 4)
    mask = np.isin(x, [2, 5, 11])
    assert mask.shape == x.shape, "np.isin must preserve shape; the fix relies on it"
    assert mask.ravel().shape == (x.size,)


def test_flat_in1d_and_plain_isin_agree():
    """`x.flat[in1d(x, y)] = 0` and `x[isin(x, y)] = 0` must do the same thing."""
    base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    remove = [2, 5, 9]

    # the new form
    new = base.copy()
    new[np.isin(new, remove)] = 0

    # the old form, written without np.in1d so this still runs on numpy >= 2.2
    old = base.copy()
    old.flat[np.isin(old, remove).ravel()] = 0

    np.testing.assert_array_equal(new, old)
    assert new.tolist() == [[1, 0, 3], [4, 0, 6], [7, 8, 0]]


def test_naive_swap_keeping_flat_is_wrong():
    """Guard the mistake: leaving `.flat` in place when swapping to isin.

    On a non-square 2-D array the shape-preserving mask cannot index `.flat`,
    so this is a hard error rather than a silent miswrite. Pinning it means a
    future 'cleanup' that reintroduces `.flat` fails loudly here.
    """
    x = np.arange(12).reshape(3, 4)
    with pytest.raises(IndexError):
        x.flat[np.isin(x, [2, 5])] = 0


def test_flat_isin_on_already_flat_input_is_fine():
    """segment_watershed.py:341 keeps `.flat` legitimately -- its input is 1-D.

    `labeled.flat[np.isin(labeled.flat[:], cut)] = 0` is correct because
    `labeled.flat[:]` is already 1-D, so the mask is 1-D too.
    """
    labeled = np.arange(12).reshape(3, 4)
    cut = [3, 7]
    got = labeled.copy()
    got.flat[np.isin(got.flat[:], cut)] = 0
    want = labeled.copy()
    want[np.isin(want, cut)] = 0
    np.testing.assert_array_equal(got, want)


def test_no_np_in1d_remains_in_workflow():
    """np.in1d is gone in numpy >= 2.2; keep it from creeping back."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "workflow"
    offenders = [
        f"{p.relative_to(root.parent)}:{i}"
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "np.in1d" in line
    ]
    assert not offenders, (
        f"np.in1d was removed in numpy 2.2; use np.isin (and drop `.flat`): {offenders}"
    )
