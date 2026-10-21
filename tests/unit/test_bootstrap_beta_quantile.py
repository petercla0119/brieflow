"""beta_quantile_null must reproduce the resampling bootstrap's null median distribution.

The resampler picks a control pool (weighted by size), draws ``sample_size`` cells with
replacement and takes per-feature medians. The beta-quantile path replaces that with
F^-1(U), U ~ Beta((n+1)/2, (n+1)/2) on the pre-sorted pool. The identity is exact for odd
n; for even n numpy's median is the mean of the two middle values, which lands between
pool atoms, so the match is approximate (and tighter the larger the pool).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, ks_2samp, spearmanr

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.aggregate.bootstrap import (  # noqa: E402
    beta_quantile_null,
    calculate_pvals,
    prepare_control_pools,
)

NUM_SIMS = 20_000


def make_controls(rng, pool_sizes=(60, 150, 400)):
    rows = []
    for cid, n in enumerate(pool_sizes):
        feats = np.column_stack(
            [
                rng.lognormal(mean=0.3 * cid, sigma=0.8, size=n),
                rng.normal(loc=cid, scale=1.5, size=n),
                rng.exponential(scale=1 + cid, size=n),
            ]
        )
        rows.append(np.column_stack([np.full(n, f"ctrl_{cid}", dtype=object), feats]))
    return np.vstack(rows)


def resample_null(controls_arr, sample_size, num_sims, rng):
    """Brute-force reference: the resampler's semantics in plain numpy."""
    ids = controls_arr[:, 0]
    feats = controls_arr[:, 1:].astype(np.float64)
    pools = [feats[ids == cid] for cid in np.unique(ids)]
    sizes = np.array([len(p) for p in pools])
    pick = rng.choice(len(pools), size=num_sims, p=sizes / sizes.sum())
    out = np.empty((num_sims, feats.shape[1]))
    for i, p_i in enumerate(pick):
        pool = pools[p_i]
        out[i] = np.median(pool[rng.integers(0, len(pool), size=sample_size)], axis=0)
    return out


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sample_size", "ks_max"),
    [(5, 0.02), (31, 0.02), (201, 0.02), (200, 0.03)],
)
def test_null_distribution_matches_resampler(sample_size, ks_max):
    rng = np.random.default_rng(0)
    controls = make_controls(rng)
    pools, sizes = prepare_control_pools(controls)

    bq = beta_quantile_null(
        pools, sizes, sample_size, NUM_SIMS, np.random.default_rng(1)
    )
    rs = resample_null(controls, sample_size, NUM_SIMS, np.random.default_rng(2))

    assert bq.shape == rs.shape == (NUM_SIMS, 3)
    for j in range(3):
        ks = ks_2samp(bq[:, j], rs[:, j]).statistic
        assert ks < ks_max, f"feature {j}, n={sample_size}: KS={ks:.4f}"
        q_bq = np.quantile(bq[:, j], [0.05, 0.5, 0.95])
        q_rs = np.quantile(rs[:, j], [0.05, 0.5, 0.95])
        assert np.allclose(q_bq, q_rs, rtol=0.03, atol=0.05), (q_bq, q_rs)


@pytest.mark.unit
def test_pvalues_agree_within_monte_carlo_noise():
    rng = np.random.default_rng(10)
    controls = make_controls(rng)
    pools, sizes = prepare_control_pools(controls)
    sample_size = 31

    bq = beta_quantile_null(
        pools, sizes, sample_size, NUM_SIMS, np.random.default_rng(11)
    )
    rs = resample_null(controls, sample_size, NUM_SIMS, np.random.default_rng(12))
    rs_again = resample_null(controls, sample_size, NUM_SIMS, np.random.default_rng(13))

    # 20 fake constructs: medians of random control draws, shifted by growing effect sizes
    feats = controls[:, 1:].astype(np.float64)
    observed = np.array(
        [
            np.median(feats[rng.integers(0, len(feats), size=sample_size)], axis=0) + s
            for s in np.linspace(-1.0, 1.0, 20)
        ]
    )

    def pvals(null):
        return np.array([calculate_pvals(null, obs) for obs in observed]).ravel()

    p_bq, p_rs, p_rs_again = pvals(bq), pvals(rs), pvals(rs_again)
    mc_floor = np.max(np.abs(p_rs - p_rs_again))  # resampler vs itself, different seed

    assert np.max(np.abs(p_bq - p_rs)) <= 1.5 * mc_floor + 0.005
    assert spearmanr(p_bq, p_rs).statistic > 0.99


@pytest.mark.unit
def test_odd_n_is_exact_against_binomial_cdf():
    # For odd n the Beta identity is exact: P(median <= x_(j)) = P(Binom(n, j/n_pool) >= (n+1)/2).
    rng = np.random.default_rng(3)
    pool = np.sort(rng.normal(size=(25, 1)), axis=0)
    n = 7
    bq = beta_quantile_null(
        [pool], np.array([25]), n, 200_000, np.random.default_rng(4)
    )
    for j in (5, 12, 20):
        empirical = np.mean(bq[:, 0] <= pool[j - 1, 0])
        exact = binom.sf((n + 1) / 2 - 1, n, j / 25)
        assert abs(empirical - exact) < 0.005, (j, empirical, exact)


@pytest.mark.unit
def test_pool_choice_is_weighted_by_size():
    rng = np.random.default_rng(5)
    small = np.sort(rng.normal(loc=-10, size=(20, 1)), axis=0)
    large = np.sort(rng.normal(loc=10, size=(180, 1)), axis=0)
    bq = beta_quantile_null(
        [small, large], np.array([20, 180]), 9, 20_000, np.random.default_rng(6)
    )
    frac_small = np.mean(bq[:, 0] < 0)
    assert abs(frac_small - 0.1) < 0.01
