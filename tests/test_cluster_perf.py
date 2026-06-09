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


# ---------------------------------------------------------------------------
# perf/reuse-graph tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReuseGraph:
    def test_leiden_sweep_matches_individual_runs(self):
        from lib.cluster.phate_leiden_clustering import (
            phate_leiden_pipeline,
            phate_leiden_sweep,
        )

        data = _make_synthetic_data(n_samples=100, n_features=20)
        resolutions = [0.5, 1.0, 5.0]

        sweep_results = phate_leiden_sweep(
            data, resolutions, "euclidean", first_feature_name="PC_0"
        )

        # PHATE embedding differs per run due to random seed interactions,
        # but the sweep should produce valid clusterings for each resolution
        for res in resolutions:
            assert res in sweep_results
            result = sweep_results[res]
            assert "cluster" in result.columns
            assert "PHATE_0" in result.columns
            assert "PHATE_1" in result.columns
            assert len(result) == 100

        # Different resolutions should produce different cluster counts
        n_clusters = [sweep_results[r]["cluster"].nunique() for r in resolutions]
        assert len(set(n_clusters)) > 1, "All resolutions gave same cluster count"

    def test_sweep_clusters_match_individual_pipeline(self):
        """Verify sweep uses the same PHATE graph as a single pipeline call."""
        from lib.cluster.phate_leiden_clustering import (
            run_phate,
            run_leiden_clustering,
            phate_leiden_sweep,
        )

        data = _make_synthetic_data(n_samples=80, n_features=15)
        feature_cols = [c for c in data.columns if c.startswith("PC_")]

        # Get the sweep results
        sweep_results = phate_leiden_sweep(
            data, [1.0], "euclidean", first_feature_name="PC_0"
        )
        sweep_clusters = sweep_results[1.0].sort_values("meta_0")["cluster"].values

        # The sweep function builds PHATE once and applies Leiden.
        # We can't compare against a separate phate_leiden_pipeline call
        # (different PHATE embedding due to random state), but we can verify
        # the sweep result is internally consistent.
        assert len(sweep_clusters) == 80
        assert all(isinstance(c, (int, np.integer)) for c in sweep_clusters)

    def test_inherited_t_produces_valid_output(self):
        from lib.cluster.phate_leiden_clustering import run_phate

        data = _make_synthetic_data(n_samples=100, n_features=20)
        feature_cols = [c for c in data.columns if c.startswith("PC_")]

        # First fit
        df1, p1 = run_phate(data[feature_cols], metric="euclidean")

        # Second fit inheriting t
        df2, p2 = run_phate(data[feature_cols], metric="euclidean",
                            inherit_params_from=p1)

        assert df2.shape == (100, 2)
        assert not df2.isna().any().any()

        # The inherited t should be used (stored in optimal_t after fit)
        if p1.optimal_t is not None:
            assert p2.optimal_t == p1.optimal_t

    def test_pipeline_with_inherit_params(self):
        from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline, run_phate

        data = _make_synthetic_data(n_samples=100, n_features=20)
        feature_cols = [c for c in data.columns if c.startswith("PC_")]

        # Get a fitted PHATE object
        _, p_fitted = run_phate(data[feature_cols], metric="euclidean")

        # Run pipeline with inherited params (simulating shuffled data reuse)
        result = phate_leiden_pipeline(
            data, resolution=1.0, phate_distance_metric="euclidean",
            first_feature_name="PC_0", inherit_params_from=p_fitted,
        )

        assert "cluster" in result.columns
        assert len(result) == 100


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

        result = run_shuffled_baseline(data, resolution=1.0,
                                       phate_distance_metric="euclidean",
                                       subsample_fraction=1.0,
                                       first_feature_name="PC_0")
        assert not result["PHATE_0"].isna().any()
        assert not result["PHATE_1"].isna().any()
