"""Unit tests for the IC well-group parallelism helpers in run_preprocess_direct.py.

_ic_per_group_threads is pure arithmetic; the default-branch test confirms that with
no `ic_group_concurrency` set the runner takes the serial path and hands each IC the
full worker count (the backward-compat guarantee).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_preprocess_direct as rpd  # noqa: E402


@pytest.mark.parametrize(
    "total,conc,expected",
    [
        (48, 6, 8),
        (48, 1, 48),
        (16, 5, 3),
        (1, 4, 1),
        (8, 16, 1),
        (48, 0, 48),  # concurrency<=0 treated as 1
        (48, -3, 48),
    ],
)
def test_ic_per_group_threads(total, conc, expected):
    assert rpd._ic_per_group_threads(total, conc) == expected


def test_default_is_sequential_full_threads(monkeypatch, tmp_path):
    """No ic_group_concurrency -> serial branch, each IC gets all workers."""
    recorded = {}

    def fake_calc(inputs, **kw):
        recorded["n_jobs"] = kw["n_jobs"]
        return np.zeros((1, 2, 2), dtype="uint16")

    monkeypatch.setattr(rpd, "calculate_ic_field", fake_calc)
    monkeypatch.setattr(rpd, "save_image", lambda *a, **k: None)
    # ic_out contains "ic_fields" -> not yet written; input paths don't -> present
    monkeypatch.setattr(rpd, "out_exists", lambda p: "ic_fields" not in str(p))

    combos = pd.DataFrame(
        {"plate": ["1", "1"], "well": ["A1", "A1"], "tile": ["0", "1"]}
    )
    errs = rpd.run_ic_step(
        "phenotype", combos, tmp_path, "tiff", {}, total_workers=7, has_cycle=False
    )

    assert errs == 0
    assert recorded["n_jobs"] == 7
