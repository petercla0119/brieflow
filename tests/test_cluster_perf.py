"""Tests for PHATE pipeline performance optimizations."""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_data(n_samples=200, n_features=50, n_metadata=2, random_state=42,
                         n_clusters=0):
    """Create synthetic aggregated data mimicking real pipeline output.

    When n_clusters > 0, data is drawn from well-separated Gaussian blobs
    so that both exact and approximate NN yield comparable clusterings.
    """
    rng = np.random.RandomState(random_state)
    metadata = {f"meta_{i}": [f"gene_{j}" for j in range(n_samples)] for i in range(n_metadata)}

    if n_clusters > 0:
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=n_clusters, cluster_std=0.5, random_state=random_state)
        features = {f"PC_{i}": X[:, i] for i in range(n_features)}
    else:
        features = {f"PC_{i}": rng.randn(n_samples) for i in range(n_features)}

    return pd.DataFrame({**metadata, **features})


# ---------------------------------------------------------------------------
# perf/approx-nn tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApproxNN:
    def test_produces_valid_phate_output(self):
        from lib.cluster.phate_leiden_clustering import run_phate

        data = _make_synthetic_data()
        feature_cols = [c for c in data.columns if c.startswith("PC_")]
        df_phate, p = run_phate(data[feature_cols], use_approx_nn=True)

        assert df_phate.shape == (200, 2)
        assert list(df_phate.columns) == ["PHATE_0", "PHATE_1"]
        assert not df_phate.isna().any().any()
        assert np.all(np.isfinite(df_phate.values))

    def test_pipeline_with_approx_nn(self):
        from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline

        data = _make_synthetic_data()
        result = phate_leiden_pipeline(data, resolution=1.0, phate_distance_metric="cosine",
                                      first_feature_name="PC_0", use_approx_nn=True)

        assert "PHATE_0" in result.columns
        assert "PHATE_1" in result.columns
        assert "cluster" in result.columns
        assert len(result) == 200

    def test_matches_exact_within_tolerance(self):
        from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline

        data = _make_synthetic_data(n_samples=150, n_features=20, n_clusters=5)

        result_exact = phate_leiden_pipeline(data, resolution=1.0,
                                            phate_distance_metric="euclidean",
                                            first_feature_name="PC_0",
                                            use_approx_nn=False)
        result_approx = phate_leiden_pipeline(data, resolution=1.0,
                                             phate_distance_metric="euclidean",
                                             first_feature_name="PC_0",
                                             use_approx_nn=True)

        exact_clusters = result_exact.set_index("meta_0")["cluster"]
        approx_clusters = result_approx.set_index("meta_0")["cluster"]
        common = exact_clusters.index.intersection(approx_clusters.index)

        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(exact_clusters[common], approx_clusters[common])
        assert ari > 0.3, f"Adjusted Rand Index too low: {ari}"

    def test_approx_nn_missing_pynndescent(self, monkeypatch):
        from lib.cluster import phate_leiden_clustering as module

        def mock_make_class():
            raise ImportError("No module named 'pynndescent'")

        monkeypatch.setattr(module, "_make_approx_nn_class", mock_make_class)

        data = _make_synthetic_data(n_samples=50, n_features=10)
        feature_cols = [c for c in data.columns if c.startswith("PC_")]

        with pytest.raises(ImportError, match="pynndescent"):
            module.run_phate(data[feature_cols], use_approx_nn=True)
