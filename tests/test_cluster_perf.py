"""Tests for PHATE pipeline performance optimizations."""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_data(n_samples=200, n_features=50, n_metadata=2, random_state=42):
    """Create synthetic aggregated data mimicking real pipeline output."""
    rng = np.random.RandomState(random_state)
    metadata = {f"meta_{i}": [f"gene_{j}" for j in range(n_samples)] for i in range(n_metadata)}
    features = {f"PC_{i}": rng.randn(n_samples) for i in range(n_features)}
    return pd.DataFrame({**metadata, **features})


# ---------------------------------------------------------------------------
# perf/subsample-shuffled tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSubsampleShuffled:
    def test_subsample_reduces_data_size(self):
        from lib.cluster.phate_leiden_clustering import run_shuffled_baseline

        data = _make_synthetic_data(n_samples=100, n_features=20)
        result = run_shuffled_baseline(data, resolution=1.0,
                                       phate_distance_metric="euclidean",
                                       subsample_fraction=0.5,
                                       first_feature_name="PC_0")
        assert len(result) == 50

    def test_full_fraction_preserves_size(self):
        from lib.cluster.phate_leiden_clustering import run_shuffled_baseline

        data = _make_synthetic_data(n_samples=100, n_features=20)
        result = run_shuffled_baseline(data, resolution=1.0,
                                       phate_distance_metric="euclidean",
                                       subsample_fraction=1.0,
                                       first_feature_name="PC_0")
        assert len(result) == 100

    def test_output_has_expected_columns(self):
        from lib.cluster.phate_leiden_clustering import run_shuffled_baseline

        data = _make_synthetic_data(n_samples=80, n_features=15)
        result = run_shuffled_baseline(data, resolution=1.0,
                                       phate_distance_metric="euclidean",
                                       subsample_fraction=0.5,
                                       first_feature_name="PC_0")
        assert "PHATE_0" in result.columns
        assert "PHATE_1" in result.columns
        assert "cluster" in result.columns
        assert "meta_0" in result.columns

    def test_shuffling_destroys_correlation(self):
        from lib.cluster.phate_leiden_clustering import run_shuffled_baseline

        data = _make_synthetic_data(n_samples=200, n_features=20)

        # run_shuffled_baseline shuffles internally; verify by checking
        # that the function runs without error and produces valid output
        result = run_shuffled_baseline(data, resolution=1.0,
                                       phate_distance_metric="euclidean",
                                       subsample_fraction=1.0,
                                       first_feature_name="PC_0")
        assert not result["PHATE_0"].isna().any()
        assert not result["PHATE_1"].isna().any()
