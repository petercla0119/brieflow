"""Tests for Phase 3: align_cycles compute_qc flag."""

import sys
import inspect
from pathlib import Path

import numpy as np
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

from lib.sbs.align_cycles import align_cycles


def _synthetic_cycles(n_cycles=3, n_ch=5, size=64, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randint(100, 4000, size=(n_ch, size, size)).astype(np.uint16)
    return [base.copy() for _ in range(n_cycles)]


def test_qc_flag_does_not_change_output():
    """compute_qc=True/False must produce bit-identical aligned arrays."""
    imgs = _synthetic_cycles()
    kw = dict(
        channel_order=["DAPI", "G", "T", "A", "C"],
        method="DAPI",
        upsample_factor=2,
        window=2,
    )
    a_false = align_cycles([x.copy() for x in imgs], compute_qc=False, **kw)
    a_true = align_cycles([x.copy() for x in imgs], compute_qc=True, **kw)
    assert np.array_equal(a_false, a_true), \
        "compute_qc flag changed aligned array output"


def test_qc_false_is_default():
    """compute_qc should default to False."""
    sig = inspect.signature(align_cycles)
    assert "compute_qc" in sig.parameters, "compute_qc not in align_cycles signature"
    assert sig.parameters["compute_qc"].default is False, \
        f"compute_qc default should be False, got {sig.parameters['compute_qc'].default}"
