from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geodatasets import get_path
from sklearn.linear_model import LogisticRegression

from gwlearn.base import BaseClassifier, _kernel_functions


@pytest.fixture
def sample_data():
    """Return sample data from geoda.guerry dataset."""
    gdf = gpd.read_file(get_path("geoda.guerry"))
    # Create point geometries from polygon centroids
    gdf = gdf.set_geometry(gdf.centroid)
    # Create binary target variable
    gdf["binary_target"] = gdf["Donatns"] > gdf["Donatns"].median()

    # Select features
    X = gdf[["Crm_prs", "Litercy", "Wealth"]]
    y = gdf["binary_target"]
    geometry = gdf.geometry

    return X, y, geometry


def test_init_default_parameters():
    """Test BaseClassifier initialization with default parameters."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)

    assert clf.model == LogisticRegression
    assert clf.bandwidth == 100
    assert clf.fixed is False
    assert clf.kernel == "bisquare"
    assert clf.n_jobs == -1
    assert clf.fit_global_model is True
    assert clf.measure_performance is True
    assert clf.strict is False
    assert clf.keep_models is False
    assert clf.temp_folder is None
    assert clf.batch_size is None
    assert clf.min_proportion == 0.2
    assert isinstance(clf.model_kwargs, dict)
    assert len(clf.model_kwargs) == 0


def test_init_custom_parameters():
    """Test BaseClassifier initialization with custom parameters."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=50,
        fixed=True,
        kernel="gaussian",
        n_jobs=2,
        fit_global_model=False,
        measure_performance=False,
        strict=True,
        keep_models=True,
        temp_folder="/tmp",
        batch_size=10,
        min_proportion=0.3,
        max_iter=200,  # A LogisticRegression parameter
    )

    assert clf.model == LogisticRegression
    assert clf.bandwidth == 50
    assert clf.fixed is True
    assert clf.kernel == "gaussian"
    assert clf.n_jobs == 2
    assert clf.fit_global_model is False
    assert clf.measure_performance is False
    assert clf.strict is True
    assert clf.keep_models is True
    assert clf.temp_folder == "/tmp"
    assert clf.batch_size == 10
    assert clf.min_proportion == 0.3
    assert "max_iter" in clf.model_kwargs
    assert clf.model_kwargs["max_iter"] == 200


def test_init_keep_models_path():
    """Test BaseClassifier initialization with keep_models as Path."""
    path_str = "/tmp/models"

    # Test with string
    clf = BaseClassifier(LogisticRegression, bandwidth=100, keep_models=path_str)
    assert isinstance(clf.keep_models, Path)
    assert str(clf.keep_models) == path_str

    # Test with Path object
    path_obj = Path(path_str)
    clf = BaseClassifier(LogisticRegression, bandwidth=100, keep_models=path_obj)
    assert clf.keep_models == path_obj


def test_init_kernel_validation():
    """Test BaseClassifier initialization with various kernel options."""
    # Test with each predefined kernel
    for kernel_name in _kernel_functions:
        clf = BaseClassifier(LogisticRegression, bandwidth=100, kernel=kernel_name)
        assert clf.kernel == kernel_name

    # Test with a custom kernel function
    def custom_kernel(distances, bandwidth):
        return np.exp(-distances / bandwidth) * 2

    clf = BaseClassifier(LogisticRegression, bandwidth=100, kernel=custom_kernel)
    assert clf.kernel == custom_kernel


def test_init_oob_scoring_detection():
    """Test that BaseClassifier correctly detects and configures OOB scoring."""

    # Mock model that supports oob_score
    class ModelWithOOB:
        def __init__(self, oob_score=False, **kwargs):
            self.oob_score = oob_score

    clf = BaseClassifier(ModelWithOOB, bandwidth=100)
    assert clf._measure_oob is True
    assert "oob_score" in clf.model_kwargs

    # Mock model that doesn't support oob_score
    class ModelWithoutOOB:
        def __init__(self, **kwargs):
            pass

    clf = BaseClassifier(ModelWithoutOOB, bandwidth=100)
    assert clf._measure_oob is False
    assert "oob_score" not in clf.model_kwargs


def test_init_with_real_data(sample_data):
    """Test BaseClassifier initialization with real data."""
    X, y, geometry = sample_data

    # Create classifier with default params
    clf = BaseClassifier(LogisticRegression, bandwidth=50000, fixed=True)

    # Just testing that initialization doesn't raise errors
    assert clf.model == LogisticRegression
    assert clf.bandwidth == 50000
    assert clf.fixed is True


@pytest.mark.parametrize("bandwidth", [0, -1, -100])
def test_init_invalid_bandwidth(bandwidth):
    """Test BaseClassifier initialization with invalid bandwidth values."""
    # Bandwidth should be positive, but the class doesn't validate this at init time
    # This test confirms current behavior but might need updating if validation is added
    clf = BaseClassifier(LogisticRegression, bandwidth=bandwidth)
    assert clf.bandwidth == bandwidth


def test_init_multiple_kwargs():
    """Test BaseClassifier initialization with multiple model kwargs."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        C=0.5,
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=1000,
    )

    # Check that all kwargs are passed to model_kwargs
    assert clf.model_kwargs["C"] == 0.5
    assert clf.model_kwargs["penalty"] == "l2"
    assert clf.model_kwargs["solver"] == "liblinear"
    assert clf.model_kwargs["random_state"] == 42
    assert clf.model_kwargs["max_iter"] == 1000


def test_init_preserve_model_class():
    """Test that BaseClassifier preserves the model class without instantiating it."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)

    # The model should be stored as a class, not an instance
    assert clf.model == LogisticRegression
    assert not isinstance(clf.model, LogisticRegression)
