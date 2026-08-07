import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.aggregate.aggregate import aggregate

# Characterization test — must stay green before and after any aggregate.py changes.


def _make_data(seed=0):
    rng = np.random.default_rng(seed)
    perts = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
    embeddings = rng.standard_normal((len(perts), 3)).astype(np.float32)
    metadata = pd.DataFrame({"perturbation": perts})
    return embeddings, metadata


@pytest.mark.unit
def test_median_matches_loop():
    embeddings, metadata = _make_data(seed=99)
    agg_emb, agg_meta = aggregate(
        embeddings, metadata, pert_col="perturbation", method="median"
    )

    # Verify output shape
    assert agg_emb.shape == (3, 3)  # 3 perturbations, 3 features
    assert len(agg_meta) == 3

    # Verify cell_count sums to total cells
    assert agg_meta["cell_count"].sum() == len(embeddings)

    # Verify median correctness for perturbation "A" (first 3 rows)
    expected_median_A = np.median(embeddings[:3], axis=0)
    pert_A_idx = agg_meta[agg_meta["perturbation"] == "A"].index[0]
    np.testing.assert_allclose(agg_emb[pert_A_idx], expected_median_A, atol=1e-5)


@pytest.mark.unit
def test_aggregate_cell_count_correct():
    embeddings, metadata = _make_data(seed=0)
    _, agg_meta = aggregate(
        embeddings, metadata, pert_col="perturbation", method="median"
    )

    counts = agg_meta.set_index("perturbation")["cell_count"]
    assert counts["A"] == 3
    assert counts["B"] == 3
    assert counts["C"] == 4
