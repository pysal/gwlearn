import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from gwlearn.linear_model import GWLogisticRegression
from gwlearn.search import BandwidthSearch


def test_bandwidth_search_global_invariant_y():
    """
    Verifies that BandwidthSearch handles a target y that is all 0s or all 1s.
    The fix ensures _score returns a correctly sized list of NaNs instead of crashing.
    """
    # Create 10 points with invariant y (all ones)
    n = 10
    X = pd.DataFrame({"feat": np.random.rand(n)})
    y = pd.Series([1] * n)
    geometry = gpd.GeoSeries([Point(i, i) for i in range(n)])

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=1,
        max_bandwidth=5,
        interval=1,
        verbose=False,
    )

    # This should not crash and should return Inf scores
    search.fit(X, y, geometry)

    assert (search.scores_ == np.inf).all()
    # Ensure metrics DataFrame has correct columns (met) and is all NaNs
    assert search.metrics_.isna().all().all()
    assert len(search.metrics_.columns) == 3  # aicc, aic, bic (default)


def test_bandwidth_search_local_invariant_y():
    """
    Constructs data where target is mixed globally, but the subset of locations
    that successfully fit a local model all have the same global label.

    Data Setup:
    - Cluster A (0-19): y=0. Far away. Will fail (invariant neighborhood).
    - Cluster B (20-24): y=1. At x=100.
    - Cluster C (25-29): y=0. At x=101.
    - Cluster D (30-49): y=0. At x=101.5 (closer to C than B is).

    With adaptive bandwidth k=10:
    - Cluster A points see only y=0. Fail.
    - Cluster B points see B (y=1) and C (y=0). Mixed! Success.
    - Cluster C points see C (y=0) and D (y=0). Invariant! Fail.
    - Cluster D points see only y=0. Fail.

    Only Cluster B points (all y=1) succeed. y_masked is all 1s.
    This triggers the fix in log_loss calculation for invariant subsets.
    """
    # y construction
    y = pd.Series([0] * 20 + [1] * 5 + [0] * 5 + [0] * 20)
    n = len(y)
    X = pd.DataFrame({"feat": np.random.rand(n)})

    # Geometry construction
    coords = []
    for i in range(20):
        coords.append(Point(0, i * 0.01))  # A
    for i in range(5):
        coords.append(Point(100, i * 0.01))  # B
    for i in range(5):
        coords.append(Point(101, i * 0.01))  # C
    for i in range(20):
        coords.append(Point(101.5, i * 0.01))  # D
    geometry = gpd.GeoSeries(coords)

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=False,
        search_method="interval",
        min_bandwidth=10,
        max_bandwidth=10,
        interval=1,
        criterion="log_loss",
        metrics=["log_loss"],
        verbose=False,
    )

    # This should not crash and return Inf for the log_loss metric
    search.fit(X, y, geometry)

    # Both score and log_loss metric should be Inf because y_masked is invariant
    assert (search.scores_ == np.inf).all()
    assert (search.metrics_["log_loss"] == np.inf).all()


def test_bandwidth_search_standard_data(sample_data):
    """
    Ensure log_loss calculation still works correctly on mixed data where
    local neighborhoods are also mixed.
    """
    X, y, geometry = sample_data

    search = BandwidthSearch(
        GWLogisticRegression,
        fixed=True,
        search_method="interval",
        min_bandwidth=200000,
        max_bandwidth=200000,
        interval=1,
        criterion="log_loss",
        metrics=["log_loss"],
        verbose=False,
    )

    search.fit(X, y, geometry)

    # Should calculate valid finite log_loss
    assert not np.isinf(search.metrics_["log_loss"]).all()
    assert not search.metrics_["log_loss"].isna().all()
