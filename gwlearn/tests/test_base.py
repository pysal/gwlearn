import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geodatasets import get_path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from gwlearn.base import BaseClassifier, BaseRegressor, _kernel_functions


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
    path_str = ["/tmp/models", "\\tmp\\models"]

    # Test with string
    clf = BaseClassifier(LogisticRegression, bandwidth=100, keep_models=path_str[0])
    assert isinstance(clf.keep_models, Path)
    assert str(clf.keep_models) in path_str

    # Test with Path object
    path_obj = Path(path_str[0])
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


def test_init_with_real_data():
    """Test BaseClassifier initialization with real data."""
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
        max_iter=1000,
    )

    # Check that all kwargs are passed to model_kwargs
    assert clf.model_kwargs["C"] == 0.5
    assert clf.model_kwargs["penalty"] == "l2"
    assert clf.model_kwargs["solver"] == "liblinear"
    assert clf.model_kwargs["max_iter"] == 1000


def test_init_preserve_model_class():
    """Test that BaseClassifier preserves the model class without instantiating it."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)

    # The model should be stored as a class, not an instance
    assert clf.model == LogisticRegression
    assert not isinstance(clf.model, LogisticRegression)


def test_fit_basic_functionality(sample_data):
    """Test basic fitting functionality of BaseClassifier."""
    X, y, geometry = sample_data

    # Create classifier with default params
    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    fitted_clf = clf.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_clf is clf

    # Test that the global model was fitted
    assert hasattr(clf, "global_model")
    assert isinstance(clf.global_model, RandomForestClassifier)

    # Test that performance metrics were calculated
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1
    assert hasattr(clf, "f1_macro_")
    assert hasattr(clf, "f1_micro_")
    assert hasattr(clf, "f1_weighted_")


def test_fit_with_keep_models(sample_data):
    """Test fitting with keep_models=True to retain local models."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        random_state=42,
        max_iter=250,
        strict=False,  # To avoid warnings on invariance
        n_jobs=1,
    )

    clf.fit(X, y, geometry)

    # Check that local models were kept
    assert hasattr(clf, "_local_models")
    assert isinstance(clf._local_models, pd.Series)
    assert len(clf._local_models) > 0

    # Check that each local model is a fitted LogisticRegression
    for model in clf._local_models:
        assert isinstance(model, LogisticRegression | None)
        # Check that the model has been fitted by ensuring it has a coef_ attribute
        assert (
            hasattr(model, "coef_") if isinstance(model, LogisticRegression) else True
        )


def test_fit_with_keep_models_path(sample_data):
    """Test fitting with keep_models as a Path to save models to disk."""
    X, y, geometry = sample_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a classifier with keep_models as a path
        clf = BaseClassifier(
            RandomForestClassifier,
            bandwidth=10,
            fixed=False,
            keep_models=temp_dir,
            random_state=42,
            strict=False,  # To avoid warnings on invariance
            n_jobs=1,
        )

        clf.fit(X, y, geometry)

        # Check that models were serialized to disk
        model_files = list(Path(temp_dir).glob("*"))
        assert len(model_files) > 0


@pytest.mark.parametrize("kernel", _kernel_functions)
def test_fit_different_kernels(sample_data, kernel):
    """Test fitting with different kernel functions."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        kernel=kernel,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )

    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1


def test_fit_fixed_bandwidth(sample_data):
    """Test fitting with adaptive bandwidth (fixed=False)."""
    X, y, geometry = sample_data

    # Use a small k for faster testing
    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=100_000,  # Use 10 nearest neighbors
        fixed=True,  # Adaptive bandwidth
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )

    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1


def test_fit_without_global_model(sample_data):
    """Test fitting without computing a global model."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150_000,
        fixed=True,
        fit_global_model=False,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )

    clf.fit(X, y, geometry)

    # Check that global model was not fitted
    assert not hasattr(clf, "global_model")

    # But local results should still be available
    assert hasattr(clf, "proba_")


def test_fit_without_performance_metrics(sample_data):
    """Test fitting without computing performance metrics."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        measure_performance=False,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )

    clf.fit(X, y, geometry)

    # Check that performance metrics were not computed
    assert not hasattr(clf, "score_")
    assert not hasattr(clf, "f1_macro_")

    # But focal probabilities should still be available
    assert hasattr(clf, "proba_")


def test_fit_with_strict_option(sample_data):
    """Test the strict option for invariant y."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=X.shape[0] - 1,  # global bandwidth
        fixed=False,
        strict=True,  # Raise error if invariant
        random_state=42,
    )

    # The fit should complete without error because even with large bandwidth,
    # the target is likely varied enough
    clf.fit(X, y, geometry)

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=5,  # known to produce invariant subsets
        fixed=False,
        strict=True,  # Raise error if invariant
        random_state=42,
    )

    # This should raise a ValueError due to invariant y
    with pytest.raises(ValueError, match="y at locations .* is invariant"):
        clf.fit(X, y, geometry)

    # But with strict=False, it should just warn
    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=5,
        fixed=False,
        strict=None,  # Just warn if invariant
        random_state=42,
    )

    # Should complete with a warning
    with pytest.warns(UserWarning, match="y at locations .* is invariant"):
        clf.fit(X, y, geometry)


def test_non_point_geometry_raises_error(sample_data):
    """Test that non-point geometries raise an error."""
    X, y, _ = sample_data

    # Get the original polygons instead of centroids
    gdf = gpd.read_file(get_path("geoda.guerry"))
    polygon_geometry = gdf.geometry

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=50000,
        fixed=True,
        strict=False,  # To avoid warnings on invariance
    )

    # This should raise a ValueError due to non-point geometries
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        clf.fit(X, y, polygon_geometry)


def test_fit_with_batch_processing(sample_data):
    """Test fitting with batch processing enabled."""
    X, y, geometry = sample_data

    # Create a classifier with a small batch size
    batch_size = 5
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        batch_size=batch_size,  # Process in small batches
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        verbose=True,
    )

    # Capture print output to verify batch processing messages
    f = io.StringIO()
    with redirect_stdout(f):
        clf.fit(X, y, geometry)

    # Get the captured output
    output = f.getvalue()

    # Test that batch processing messages were printed
    expected_batches = ((len(X) + batch_size - 1) // batch_size) + 1  # Ceiling division
    assert f"Processing batch 1 out of {expected_batches}" in output

    # Check that the model was fit successfully
    assert hasattr(clf, "proba_")
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1

    # Compare with a model without batching to ensure results are consistent
    clf_no_batch = BaseClassifier(
        LogisticRegression, bandwidth=150000, fixed=True, random_state=42
    )
    clf_no_batch.fit(X, y, geometry)

    # Results should be similar regardless of batching
    pd.testing.assert_frame_equal(
        clf.proba_, clf_no_batch.proba_, check_exact=False, rtol=1e-5
    )


def test_fit_n_jobs_consistency(sample_data):
    """Test that parallel processing gives the same results as sequential (n_jobs=1)."""
    X, y, geometry = sample_data

    # Create a classifier with n_jobs=1 (sequential)
    clf_sequential = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf_sequential.fit(X, y, geometry)

    # Create a classifier with n_jobs=-1 (parallel)
    clf_parallel = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=-1,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf_parallel.fit(X, y, geometry)

    # Check that the results are the same regardless of parallelization
    pd.testing.assert_frame_equal(
        clf_sequential.proba_,
        clf_parallel.proba_,
        check_exact=False,
        rtol=1e-5,
    )

    # Check that performance metrics are also equal
    assert clf_sequential.score_ == pytest.approx(clf_parallel.score_)
    assert clf_sequential.f1_macro_ == pytest.approx(clf_parallel.f1_macro_)
    assert clf_sequential.f1_weighted_ == pytest.approx(clf_parallel.f1_weighted_)

    # Check that global models have the same coefficients
    np.testing.assert_allclose(
        clf_sequential.global_model.coef_, clf_parallel.global_model.coef_, rtol=1e-5
    )


def test_predict_proba_basic(sample_data):
    """Test basic functionality of predict_proba method."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Predict probabilities for first 5 samples
    proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5])

    # Check output format
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape == (5, 2)  # Binary classification, so 2 columns
    assert all(column in proba.columns for column in [True, False])

    # Check probability values are valid
    assert (proba >= 0).all().all()
    assert (proba <= 1).all().all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_adaptive(sample_data):
    """Test basic functionality of predict_proba method using adaptive kernel."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=7,
        fixed=False,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Predict probabilities for first 5 samples
    proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5])

    # Check output format
    assert isinstance(proba, pd.DataFrame)
    assert proba.shape == (5, 2)  # Binary classification, so 2 columns
    assert all(column in proba.columns for column in [True, False])

    # Check probability values are valid
    assert (proba >= 0).all().all()
    assert (proba <= 1).all().all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_basic(sample_data):
    """Test basic functionality of predict method."""
    X, y, geometry = sample_data

    # Create and fit classifier with keep_models=True (required for prediction)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )
    clf.fit(X, y, geometry)

    # Predict classes for first 5 samples
    pred = clf.predict(X.iloc[:5], geometry.iloc[:5])

    # Check output format
    assert isinstance(pred, pd.Series)
    assert len(pred) == 5

    # Check all predicted values are either True or False
    assert pred.isin([True, False]).all()


def test_predict_with_models_on_disk(sample_data):
    """Test prediction with models stored on disk."""
    X, y, geometry = sample_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and fit classifier with keep_models as a path
        clf = BaseClassifier(
            LogisticRegression,
            bandwidth=150000,
            fixed=True,
            keep_models=temp_dir,
            random_state=42,
            strict=False,  # To avoid warnings on invariance
        )
        clf.fit(X, y, geometry)

        # Predict probabilities
        proba = clf.predict_proba(X.iloc[:5], geometry.iloc[:5])

        # Check output
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape == (5, 2)

        # Also test predict method
        pred = clf.predict(X.iloc[:5], geometry.iloc[:5])
        assert isinstance(pred, pd.Series)
        assert len(pred) == 5


def test_predict_invalid_geometry(sample_data):
    """Test that prediction raises error with non-point geometries."""
    X, y, _ = sample_data

    # Get the original polygons instead of centroids
    gdf = gpd.read_file(get_path("geoda.guerry"))
    polygon_geometry = gdf.geometry

    # Create and fit classifier with point geometries
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
    )
    clf.fit(X, y, sample_data[2])  # Use point geometries for fitting

    # Attempt to predict with polygon geometries
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        clf.predict_proba(X.iloc[:5], polygon_geometry.iloc[:5])


def test_predict_comparison_with_focal_proba(sample_data):
    """Test that prediction for training data matches focal probabilities."""
    X, y, geometry = sample_data

    # Create and fit classifier
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
    )
    clf.fit(X, y, geometry)

    # Get predictions for the same data used for training
    predicted_proba = clf.predict_proba(X, geometry)

    # Compare with proba_ (should be very similar but not identical
    # because proba_ is calculated during training without using the focal point)
    pd.testing.assert_series_equal(
        predicted_proba.loc[2],
        clf.proba_.loc[2],
        check_exact=False,
        atol=0.05,  # Allow some tolerance because they're not identical
    )


def test_binary_target_zero_one(sample_data):
    """Test that 0/1 target values are correctly recognized as binary."""
    X, y, geometry = sample_data

    # Create a 0/1 encoded target
    y_01 = y.astype(int)

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )

    # Should run without errors
    fitted_clf = clf.fit(X, y_01, geometry)
    assert fitted_clf is clf

    # Check that performance metrics were calculated
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1

    # propagation to prediction
    pd.testing.assert_index_equal(clf.proba_.columns, pd.Index([0, 1]))


def test_non_binary_target_raises_error(sample_data):
    """Test that non-binary target variables raise an error."""
    X, _, geometry = sample_data

    # Create a non-binary target with values 1, 2, 3
    y_non_binary = pd.Series(np.random.choice([1, 2, 3], size=len(X)), index=X.index)

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )

    # This should raise a ValueError due to non-binary target
    with pytest.raises(ValueError, match="Only binary dependent variable is supported"):
        clf.fit(X, y_non_binary, geometry)


def test_binary_with_string_values_raises_error(sample_data):
    """Test that binary target with string values raises an error."""
    X, _, geometry = sample_data

    # Create a binary target with string values
    y_str = pd.Series(np.random.choice(["yes", "no"], size=len(X)), index=X.index)

    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )

    # This should raise a ValueError due to string values
    with pytest.raises(ValueError, match="Only binary dependent variable is supported"):
        clf.fit(X, y_str, geometry)


def test_undersample_boolean(sample_data):
    """Test fitting with undersample=True option."""
    X, y, geometry = sample_data

    # Create a classifier with undersample enabled
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # Fit should complete successfully
    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1


def test_undersample_ratio(sample_data):
    """Test fitting with undersample as a float ratio."""
    X, y, geometry = sample_data

    # Create a classifier with undersample ratio
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=0.9,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # Fit should complete successfully
    clf.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(clf, "score_")
    assert 0 <= clf.score_ <= 1


def test_random_state_consistency(sample_data):
    """Test that same random_state produces consistent results."""
    X, y, geometry = sample_data

    # Create two classifiers with same random_state
    clf1 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    clf2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_frame_equal(clf1.proba_, clf2.proba_)
    assert clf1.score_ == clf2.score_


def test_different_random_states(sample_data):
    """Test that different random_states produce different results."""
    X, y, geometry = sample_data

    # Create two classifiers with different random_states
    clf1 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        random_state=42,
        strict=False,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        random_state=99,
        strict=False,
    )
    clf2.fit(X, y, geometry)

    # Results should be different
    assert not clf1.proba_.equals(clf2.proba_)


def test_random_state_with_undersample(sample_data):
    """Test that random_state affects undersample consistently."""
    X, y, geometry = sample_data

    # Create two classifiers with same random_state and undersample
    clf1 = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf1.fit(X, y, geometry)

    clf2 = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        undersample=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )
    clf2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_frame_equal(clf1.proba_, clf2.proba_)
    assert clf1.score_ == clf2.score_


def test_repr_basic():
    """Test basic __repr__ functionality."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)
    repr_str = repr(clf)

    # Check that it contains the class name
    assert "BaseClassifier" in repr_str

    # Check that it contains the model name
    assert "LogisticRegression" in repr_str

    # Check that it contains the bandwidth
    assert "bandwidth=100" in repr_str


def test_repr_single_line_format():
    """Test __repr__ with few parameters uses single line format."""
    clf = BaseClassifier(LogisticRegression, bandwidth=50)
    repr_str = repr(clf)

    # Should be single line (no newlines)
    assert "\n" not in repr_str
    assert repr_str.startswith("BaseClassifier(")
    assert repr_str.endswith(")")


def test_repr_multiline_format():
    """Test __repr__ with many parameters uses multi-line format."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=50,
        fixed=True,
        kernel="gaussian",
        n_jobs=2,
        keep_models=True,
        verbose=True,
        max_iter=200,
    )
    repr_str = repr(clf)

    # Should be multi-line
    assert "\n" in repr_str
    assert repr_str.startswith("BaseClassifier(\n")
    assert repr_str.endswith("\n)")

    # Check key parameters are present
    assert "bandwidth=50" in repr_str
    assert "fixed=True" in repr_str  # non-default value
    assert "kernel='gaussian'" in repr_str  # non-default value
    assert "max_iter=200" in repr_str  # model kwarg


def test_repr_default_values_excluded():
    """Test that default values are excluded from __repr__."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)
    repr_str = repr(clf)

    # Default values should not appear
    assert "fixed=False" not in repr_str  # default value
    assert "kernel='bisquare'" not in repr_str  # default value
    assert "n_jobs=-1" not in repr_str  # default value
    assert "fit_global_model=True" not in repr_str  # default value


def test_repr_non_default_values_included():
    """Test that non-default values are included in __repr__."""
    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=200,
        fixed=True,  # non-default
        kernel="triangular",  # non-default
        n_jobs=4,  # non-default
        random_state=42,  # non-default
        n_estimators=200,  # model kwarg
    )
    repr_str = repr(clf)

    # Non-default values should appear
    assert "fixed=True" in repr_str
    assert "kernel='triangular'" in repr_str
    assert "n_jobs=4" in repr_str
    assert "random_state=42" in repr_str
    assert "n_estimators=200" in repr_str


def test_repr_path_handling():
    """Test __repr__ with Path objects."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        clf = BaseClassifier(
            LogisticRegression,
            bandwidth=100,
            keep_models=temp_dir,
        )
        repr_str = repr(clf)

        # Path should be shown as string
        assert f"keep_models='{temp_dir}'" in repr_str


def test_repr_callable_kernel():
    """Test __repr__ with callable kernel function."""

    def custom_kernel(distances, bandwidth):
        return np.exp(-distances / bandwidth)

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        kernel=custom_kernel,
    )
    repr_str = repr(clf)

    # Function name should appear
    assert "kernel=custom_kernel" in repr_str


def test_repr_model_kwargs_filtering():
    """Test that model kwargs are properly filtered in __repr__."""
    # Test with important kwargs (should be shown)
    clf = BaseClassifier(
        RandomForestClassifier,
        bandwidth=100,
        n_estimators=200,
        max_depth=10,
    )
    repr_str = repr(clf)

    assert "n_estimators=200" in repr_str
    assert "max_depth=10" in repr_str

    # Test with many unimportant kwargs (should be limited)
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        some_param1="value1",
        some_param2="value2",
        some_param3="value3",
        some_param4="value4",
        some_param5="value5",
    )
    repr_str = repr(clf)

    # Should not show all kwargs when there are many unimportant ones
    param_count = repr_str.count("some_param")
    assert param_count == 0  # None of the unimportant params should show


def test_repr_few_model_kwargs():
    """Test __repr__ shows all kwargs when there are only a few."""
    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=100,
        max_iter=500,
        C=0.5,
    )
    repr_str = repr(clf)

    # Should show both kwargs since there are only 2
    assert "max_iter=500" in repr_str
    assert "C=0.5" in repr_str


def test_repr_html_basic():
    """Test basic _repr_html_ functionality."""
    clf = BaseClassifier(LogisticRegression, bandwidth=100)
    html_str = clf._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)

    # Should contain HTML tags
    assert "<" in html_str and ">" in html_str

    # Should contain the class name
    assert "BaseClassifier" in html_str


def test_repr_html_with_fitted_model(sample_data):
    """Test _repr_html_ with a fitted model."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    clf.fit(X, y, geometry)

    html_str = clf._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)
    assert "<" in html_str and ">" in html_str
    assert "BaseClassifier" in html_str


def test_repr_after_fitting(sample_data):
    """Test that __repr__ works correctly after fitting."""
    X, y, geometry = sample_data

    clf = BaseClassifier(
        LogisticRegression,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    # Test repr before fitting
    repr_before = repr(clf)
    assert "BaseClassifier" in repr_before

    # Fit the model
    clf.fit(X, y, geometry)

    # Test repr after fitting (should still work)
    repr_after = repr(clf)
    assert "BaseClassifier" in repr_after

    # Should be the same representation
    assert repr_before == repr_after


def test_fit_focal_inclusion(sample_data):
    """Test basic fitting functionality of BaseClassifier."""
    X, y, geometry = sample_data

    # Create classifier with default params
    no_focal = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        include_focal=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    no_focal = no_focal.fit(X, y, geometry)

    # Create classifier with default params
    focal = BaseClassifier(
        RandomForestClassifier,
        bandwidth=10,
        fixed=False,
        include_focal=True,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    focal = focal.fit(X, y, geometry)

    # RF should 'remember' focal
    assert (no_focal.proba_[True] - no_focal.proba_[False]).abs().mean() < (
        focal.proba_[True] - focal.proba_[False]
    ).abs().mean()


# ------------regression tests----------------


def test_regressor_init_default_parameters():
    """Test BaseRegressor initialization with default parameters."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)

    assert reg.model == LinearRegression
    assert reg.bandwidth == 100
    assert reg.fixed is False
    assert reg.kernel == "bisquare"
    assert reg.n_jobs == -1
    assert reg.fit_global_model is True
    assert reg.measure_performance is True
    assert reg.strict is False
    assert reg.keep_models is False
    assert reg.temp_folder is None
    assert reg.batch_size is None
    assert isinstance(reg.model_kwargs, dict)
    assert len(reg.model_kwargs) == 0


def test_regressor_init_custom_parameters():
    """Test BaseRegressor initialization with custom parameters."""
    reg = BaseRegressor(
        LinearRegression,
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
        fit_intercept=False,  # A LinearRegression parameter
    )

    assert reg.model == LinearRegression
    assert reg.bandwidth == 50
    assert reg.fixed is True
    assert reg.kernel == "gaussian"
    assert reg.n_jobs == 2
    assert reg.fit_global_model is False
    assert reg.measure_performance is False
    assert reg.strict is True
    assert reg.keep_models is True
    assert reg.temp_folder == "/tmp"
    assert reg.batch_size == 10
    assert "fit_intercept" in reg.model_kwargs
    assert reg.model_kwargs["fit_intercept"] is False


def test_regressor_fit_basic_functionality(sample_regression_data):
    """Test basic fitting functionality of BaseRegressor."""
    X, y, geometry = sample_regression_data

    # Create regressor with default params
    reg = BaseRegressor(
        RandomForestRegressor,
        bandwidth=10,
        fixed=False,
        random_state=42,  # For reproducibility
    )

    # Fit the model
    fitted_reg = reg.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_reg is reg

    # Test that the global model was fitted
    assert hasattr(reg, "global_model")
    assert isinstance(reg.global_model, RandomForestRegressor)

    # TODO: Test that performance metrics were calculated
    # assert hasattr(reg, "score_")
    # assert hasattr(reg, "mae_")
    # assert hasattr(reg, "mse_")
    # assert hasattr(reg, "rmse_")


def test_regressor_fit_with_keep_models(sample_regression_data):
    """Test fitting with keep_models=True to retain local models."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=10,
        fixed=False,
        keep_models=True,
        n_jobs=1,
    )

    reg.fit(X, y, geometry)

    # Check that local models were kept
    assert hasattr(reg, "_local_models")
    assert isinstance(reg._local_models, pd.Series)
    assert len(reg._local_models) > 0

    # Check that each local model is a fitted LinearRegression
    for model in reg._local_models:
        assert isinstance(model, LinearRegression | None)
        # Check that the model has been fitted by ensuring it has a coef_ attribute
        assert hasattr(model, "coef_") if isinstance(model, LinearRegression) else True


def test_regressor_fit_with_keep_models_path(sample_regression_data):
    """Test fitting with keep_models as a Path to save models to disk."""
    X, y, geometry = sample_regression_data

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a regressor with keep_models as a path
        reg = BaseRegressor(
            RandomForestRegressor,
            bandwidth=10,
            fixed=False,
            keep_models=temp_dir,
            random_state=42,
            n_jobs=1,
        )

        reg.fit(X, y, geometry)

        # Check that models were serialized to disk
        model_files = list(Path(temp_dir).glob("*"))
        assert len(model_files) > 0


@pytest.mark.parametrize("kernel", _kernel_functions)
def test_regressor_fit_different_kernels(sample_regression_data, kernel):
    """Test fitting with different kernel functions."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        RandomForestRegressor,
        bandwidth=10,
        fixed=False,
        kernel=kernel,
        random_state=42,
    )

    reg.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(reg, "local_r2_")


def test_regressor_fit_fixed_bandwidth(sample_regression_data):
    """Test fitting with fixed bandwidth."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        RandomForestRegressor,
        bandwidth=100_000,
        fixed=True,  # Fixed bandwidth
        random_state=42,
    )

    reg.fit(X, y, geometry)

    # Check that the model was fit successfully
    assert hasattr(reg, "local_r2_")


def test_regressor_fit_without_global_model(sample_regression_data):
    """Test fitting without computing a global model."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150_000,
        fixed=True,
        fit_global_model=False,
    )

    reg.fit(X, y, geometry)

    # Check that global model was not fitted
    assert not hasattr(reg, "global_model")

    # But local results should still be available
    assert hasattr(reg, "pred_")


def test_regressor_fit_without_performance_metrics(sample_regression_data):
    """Test fitting without computing performance metrics."""
    X, y, geometry = sample_regression_data

    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        measure_performance=False,
        strict=False,  # To avoid warnings on invariance
    )

    reg.fit(X, y, geometry)

    # Check that performance metrics were not computed
    # assert not hasattr(reg, "score_")
    assert not hasattr(reg, "mae_")

    # But focal predictions should still be available
    assert hasattr(reg, "pred_")


def test_regressor_fit_with_batch_processing(sample_regression_data):
    """Test fitting with batch processing enabled."""
    X, y, geometry = sample_regression_data

    # Create a regressor with a small batch size
    batch_size = 5
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        batch_size=batch_size,  # Process in small batches
        verbose=True,
    )

    # Capture print output to verify batch processing messages
    f = io.StringIO()
    with redirect_stdout(f):
        reg.fit(X, y, geometry)

    # Get the captured output
    output = f.getvalue()

    # Test that batch processing messages were printed
    expected_batches = ((len(X) + batch_size - 1) // batch_size) + 1  # Ceiling division
    assert f"Processing batch 1 out of {expected_batches}" in output

    # Check that the model was fit successfully
    assert hasattr(reg, "pred_")
    assert hasattr(reg, "local_r2_")


# def test_regressor_predict_basic(sample_regression_data):
#     """Test basic functionality of predict method."""
#     X, y, geometry = sample_regression_data

#     # Create and fit regressor with keep_models=True (required for prediction)
#     reg = BaseRegressor(
#         LinearRegression,
#         bandwidth=150000,
#         fixed=True,
#         keep_models=True,
#     )
#     reg.fit(X, y, geometry)

#     # Predict values for first 5 samples
#     pred = reg.predict(X.iloc[:5], geometry.iloc[:5])

#     # Check output format
#     assert isinstance(pred, pd.Series)
#     assert len(pred) == 5

#     # Check all predicted values are numeric
#     assert pd.api.types.is_numeric_dtype(pred)


# def test_regressor_predict_adaptive(sample_regression_data):
#     """Test basic functionality of predict method using adaptive kernel."""
#     X, y, geometry = sample_regression_data

#     # Create and fit regressor with keep_models=True (required for prediction)
#     reg = BaseRegressor(
#         LinearRegression,
#         bandwidth=7,
#         fixed=False,
#         keep_models=True,
#         strict=False,  # To avoid warnings on invariance
#     )
#     reg.fit(X, y, geometry)

#     # Predict values for first 5 samples
#     pred = reg.predict(X.iloc[:5], geometry.iloc[:5])

#     # Check output format
#     assert isinstance(pred, pd.Series)
#     assert len(pred) == 5

#     # Check all predicted values are numeric
#     assert pd.api.types.is_numeric_dtype(pred)


# def test_regressor_predict_with_models_on_disk(sample_regression_data):
#     """Test prediction with models stored on disk."""
#     X, y, geometry = sample_regression_data

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Create and fit regressor with keep_models as a path
#         reg = BaseRegressor(
#             LinearRegression,
#             bandwidth=150000,
#             fixed=True,
#             keep_models=temp_dir,
#             strict=False,  # To avoid warnings on invariance
#         )
#         reg.fit(X, y, geometry)

#         # Predict values
#         pred = reg.predict(X.iloc[:5], geometry.iloc[:5])

#         # Check output
#         assert isinstance(pred, pd.Series)
#         assert len(pred) == 5


# def test_regressor_predict_comparison_with_focal_pred(sample_regression_data):
#     """Test that prediction for training data matches focal predictions."""
#     X, y, geometry = sample_regression_data

#     # Create and fit regressor
#     reg = BaseRegressor(
#         LinearRegression,
#         bandwidth=150000,
#         fixed=True,
#         keep_models=True,
#         strict=False,  # To avoid warnings on invariance
#     )
#     reg.fit(X, y, geometry)

#     # Get predictions for the same data used for training
#     predicted_values = reg.predict(X, geometry)

#     # Compare with pred_ (should be very similar but not identical
#     # because pred_ is calculated during training without using the focal point)
#     pd.testing.assert_series_equal(
#         predicted_values.loc[[2]],
#         reg.pred_.loc[[2]],
#         check_exact=False,
#         atol=0.1,  # Allow some tolerance because they're not identical
#     )


def test_regressor_random_state_consistency(sample_regression_data):
    """Test that same random_state produces consistent results."""
    X, y, geometry = sample_regression_data

    # Create two regressors with same random_state
    reg1 = BaseRegressor(
        RandomForestRegressor,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    reg1.fit(X, y, geometry)

    reg2 = BaseRegressor(
        RandomForestRegressor,
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
    )
    reg2.fit(X, y, geometry)

    # Results should be identical
    pd.testing.assert_series_equal(reg1.pred_, reg2.pred_)


def test_regressor_n_jobs_consistency(sample_regression_data):
    """Test that parallel processing gives the same results as sequential (n_jobs=1)."""
    X, y, geometry = sample_regression_data

    # Create a regressor with n_jobs=1 (sequential)
    reg_sequential = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
    )
    reg_sequential.fit(X, y, geometry)

    # Create a regressor with n_jobs=-1 (parallel)
    reg_parallel = BaseRegressor(
        LinearRegression,
        bandwidth=150000,
        fixed=True,
        n_jobs=-1,
    )
    reg_parallel.fit(X, y, geometry)

    # Check that the results are the same regardless of parallelization
    pd.testing.assert_series_equal(
        reg_sequential.pred_,
        reg_parallel.pred_,
        check_exact=False,
        rtol=1e-5,
    )
    pd.testing.assert_series_equal(
        reg_sequential.local_r2_,
        reg_parallel.local_r2_,
        check_exact=False,
        rtol=1e-5,
    )

    # TODO: Check that performance metrics are also equal
    # assert reg_sequential.score_ == pytest.approx(reg_parallel.score_)
    # assert reg_sequential.mae_ == pytest.approx(reg_parallel.mae_)
    # assert reg_sequential.mse_ == pytest.approx(reg_parallel.mse_)

    # Check that global models have the same coefficients
    np.testing.assert_allclose(
        reg_sequential.global_model.coef_,
        reg_parallel.global_model.coef_,
        rtol=1e-5,
    )


def test_regressor_repr_basic():
    """Test basic __repr__ functionality."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)
    repr_str = repr(reg)

    # Check that it contains the class name
    assert "BaseRegressor" in repr_str

    # Check that it contains the model name
    assert "LinearRegression" in repr_str

    # Check that it contains the bandwidth
    assert "bandwidth=100" in repr_str


def test_regressor_repr_multiline_format():
    """Test __repr__ with many parameters uses multi-line format."""
    reg = BaseRegressor(
        LinearRegression,
        bandwidth=50,
        fixed=True,
        kernel="gaussian",
        n_jobs=2,
        keep_models=True,
        verbose=True,
        fit_intercept=False,
    )
    repr_str = repr(reg)

    # Should be multi-line
    assert "\n" in repr_str
    assert repr_str.startswith("BaseRegressor(\n")
    assert repr_str.endswith("\n)")

    # Check key parameters are present
    assert "bandwidth=50" in repr_str
    assert "fixed=True" in repr_str  # non-default value
    assert "kernel='gaussian'" in repr_str  # non-default value
    assert "fit_intercept=False" in repr_str  # model kwarg


def test_regressor_repr_html_basic():
    """Test basic _repr_html_ functionality."""
    reg = BaseRegressor(LinearRegression, bandwidth=100)
    html_str = reg._repr_html_()

    # Should return HTML string
    assert isinstance(html_str, str)

    # Should contain HTML tags
    assert "<" in html_str and ">" in html_str

    # Should contain the class name
    assert "BaseRegressor" in html_str


def test_regressor_fit_focal_inclusion(sample_regression_data):
    """Test fitting functionality with focal inclusion parameter."""
    X, y, geometry = sample_regression_data

    # Create regressor with focal exclusion
    no_focal = BaseRegressor(
        RandomForestRegressor,
        bandwidth=10,
        fixed=False,
        include_focal=False,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    no_focal = no_focal.fit(X, y, geometry)

    # Create regressor with focal inclusion
    focal = BaseRegressor(
        RandomForestRegressor,
        bandwidth=10,
        fixed=False,
        include_focal=True,
        random_state=42,  # For reproducibility
        strict=False,  # To avoid warnings on invariance
    )

    # Fit the model
    focal = focal.fit(X, y, geometry)

    # RF should 'remember' focal point when included
    assert (y - no_focal.pred_).mean() > (y - focal.pred_).mean()
