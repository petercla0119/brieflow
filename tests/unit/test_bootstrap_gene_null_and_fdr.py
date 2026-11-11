"""Gene null independence (the aligned-draw bug) and per-feature Benjamini-Hochberg.

Run: python -m pytest -q tests/unit/test_bootstrap_gene_null_and_fdr.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.aggregate.bootstrap import (  # noqa: E402
    apply_multiple_hypothesis_correction,
    beta_quantile_null,
    bootstrap_gene,
    gene_null,
    prepare_control_pools,
)

NUM_SIMS = 20_000


def three_pools(rng, centers=(-10.0, 0.0, 10.0), n=200):
    """Three nontargeting pools far apart, so the pool pick dominates each null median."""
    rows = [
        np.column_stack(
            [np.full(n, f"ctrl_{i}", dtype=object), rng.normal(c, 0.1, size=(n, 1))]
        )
        for i, c in enumerate(centers)
    ]
    return prepare_control_pools(np.vstack(rows))


@pytest.mark.unit
def test_gene_null_does_not_pair_the_same_control_pool_across_constructs():
    pools, sizes = three_pools(np.random.default_rng(0))
    k, n = 4, 31
    # Aligned draws (the bug): every construct re-uses the same seed, so draw i of each
    # construct comes from the same pool and the gene median inherits the pool spread.
    aligned = [
        beta_quantile_null(pools, sizes, n, NUM_SIMS, np.random.default_rng(1))
        for _ in range(k)
    ]
    aligned_var = np.var(np.median(np.stack(aligned), axis=0))
    # Independent draws: the true null of a median over k independent construct medians.
    rng = np.random.default_rng(2)
    independent = [beta_quantile_null(pools, sizes, n, NUM_SIMS, rng) for _ in range(k)]
    independent_var = np.var(np.median(np.stack(independent), axis=0))
    # Three equally weighted pools at -10/0/+10: an aligned gene median is one pool draw
    # (var ~ 66.7); the median of k = 4 independent picks is about half that.
    assert aligned_var > 1.5 * independent_var  # the effect being guarded against

    # gene_null() permutes each construct, so even aligned inputs give the independent null.
    fixed_var = np.var(gene_null(aligned, np.random.default_rng(3)))
    assert abs(fixed_var - independent_var) < 0.15 * independent_var
    # ...and leaves already-independent inputs where they are.
    same_var = np.var(gene_null(independent, np.random.default_rng(4)))
    assert abs(same_var - independent_var) < 0.15 * independent_var


@pytest.mark.unit
def test_bootstrap_gene_pvalue_matches_independent_null():
    pools, sizes = three_pools(np.random.default_rng(5))
    n, k = 31, 3
    observed = np.array([3.0])  # inside the independent gene null, outside nothing
    constructs = [(f"sg{i}", n, observed) for i in range(k)]
    _, gene_p, used = bootstrap_gene(
        pools, sizes, constructs, observed, NUM_SIMS, np.random.default_rng(6)
    )
    assert used == k
    rng = np.random.default_rng(7)
    ref = np.median(
        np.stack(
            [beta_quantile_null(pools, sizes, n, NUM_SIMS, rng) for _ in range(k)]
        ),
        axis=0,
    )
    one_tail = (ref > observed).mean(axis=0)
    ref_p = 2 * np.where(one_tail > 0.5, 1 - one_tail, one_tail)
    assert abs(gene_p[0] - ref_p[0]) < 0.02


@pytest.mark.unit
def test_bh_runs_within_each_feature():
    rng = np.random.default_rng(8)
    n = 2000
    df = pd.DataFrame(
        {
            "gene": [f"g{i}" for i in range(n)],
            "num_sims": 100_000,
            "A": np.concatenate(
                [np.zeros(10), rng.uniform(size=n - 10)]
            ),  # 10 hits at the floor
            "B": rng.uniform(size=n),
            "C": rng.uniform(size=n),
        }
    )
    out = apply_multiple_hypothesis_correction(df, ["A", "B", "C"])

    # column-wise BH is the reference, with p = 0 entering as 1/num_sims
    for f in ("A", "B", "C"):
        ref = false_discovery_control(np.maximum(df[f].to_numpy(), 1e-5), method="bh")
        np.testing.assert_allclose(out[f"{f}_fdr"].to_numpy(), ref, rtol=1e-12)
    assert (
        out["A_fdr"].iloc[:10] < 0.05
    ).all()  # the 10 hits survive within their feature
    # B is uniform: its smallest adjusted p is not dragged down or up by A's hits
    b_alone = false_discovery_control(df["B"].to_numpy(), method="bh")
    assert np.isclose(out["B_fdr"].min(), b_alone.min())
    # schema: metadata first, then the four columns per feature, originals kept as written
    assert list(out.columns[:2]) == ["gene", "num_sims"]
    assert list(out.columns[2:6]) == [
        "A_pval",
        "A_neg_log10_pval",
        "A_fdr",
        "A_neg_log10_fdr",
    ]
    assert out["A_pval"].iloc[0] == 0.0 and out["A_neg_log10_pval"].iloc[0] == 5.0


@pytest.mark.unit
def test_bh_handles_nan_rows_and_no_num_sims():
    df = pd.DataFrame({"gene": list("abcd"), "A": [0.01, np.nan, 0.5, 0.0]})
    out = apply_multiple_hypothesis_correction(df, ["A"])
    assert np.isnan(out["A_fdr"].iloc[1]) and np.isnan(out["A_neg_log10_pval"].iloc[1])
    ref = false_discovery_control(np.maximum([0.01, 0.5, 0.0], 1e-5), method="bh")
    np.testing.assert_allclose(out["A_fdr"].to_numpy()[[0, 2, 3]], ref)
