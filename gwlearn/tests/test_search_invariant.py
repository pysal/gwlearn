import numpy as np
import pandas as pd
import pytest
from gwlearn.linear_model import GWLogisticRegression
from gwlearn.search import BandwidthSearch

try:
    from geodatasets import get_path
except ImportError:
    pass


def test_single_class_y_handled_gracefully(sample_data):
    """Test that BandwidthSearch handles single class y without crashing."""
    X, y, geometry = sample_data
    # Force y to be single class
    y_single = pd.Series([1] * len(y))

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=100000,
        max_bandwidth=200000,
        interval=50000,
        verbose=False,
    )

    # This should not crash and return Inf/NaNs
    search.fit(X, y_single, geometry)

    # Check return values
    assert hasattr(search, "scores_")
    # All scores should be Inf
    assert np.isinf(search.scores_).all()
    # All metrics should be NaN
    assert search.metrics_.isna().all().all()


def test_log_loss_with_subset_y(sample_data):
    """Test that log_loss works when y[~mask] has single class but y has mixed classes."""
    X, y_orig, geometry = sample_data

    # Mocking BandwidthSearch._score logic partially
    search = BandwidthSearch(GWLogisticRegression, geometry=geometry)
    search.geometry = geometry
    search.fixed = False

    class MockGWM:
        def __init__(self, y):
            # Proba should have 2 columns if y is mixed
            probs = np.random.rand(len(y), 2)
            probs = probs / probs.sum(axis=1)[:, np.newaxis]
            self.proba_ = pd.DataFrame(probs, columns=[0, 1])
            # Set proba to NaN where y is 1. So valid predictions only for class 0.
            # This simulates models being skipped/invalid for class 1 neighbors
            mask_y = y == 1
            self.proba_.loc[mask_y, :] = np.nan

            self.aicc_ = 100.0
            self.aic_ = 100.0
            self.bic_ = 100.0
            self.prediction_rate_ = 1.0  # Pretend high

        def fit(self, X, y, geometry):
            return self

    class MockModelArg:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, geometry):
            return MockGWM(y)

    search.model = MockModelArg
    search._model_kwargs = {}
    search.criterion = "log_loss"
    search.metrics = ["log_loss"]
    search.n_jobs = 1
    search.verbose = False

    # y must be mixed (0s and 1s)
    y = pd.Series([0, 1] * (len(y_orig) // 2))
    if len(y) != len(X):
        y = pd.Series([0, 1] * ((len(X) + 1) // 2))[: len(X)]

    score, metrics_list = search._score(X, y, bw=100)

    # Should return inf for invariant masked y
    assert np.isinf(metrics_list[3])
