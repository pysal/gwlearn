import pandas as pd
import pytest

from gwlearn.ensemble import GWGradientBoostingClassifier, GWRandomForestClassifier
from gwlearn.tests.test_base import sample_data  # noqa: F401


def test_gwrf_init():
    """Test GWRandomForestClassifier initialization."""
    model = GWRandomForestClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "random_forest"

    # Check custom parameters
    model = GWRandomForestClassifier(
        bandwidth=50, fixed=True, kernel="gaussian", n_estimators=100, max_depth=5
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "gaussian"
    assert model.model_kwargs["n_estimators"] == 100
    assert model.model_kwargs["max_depth"] == 5


def test_gwrf_fit_basic(sample_data):  # noqa: F811
    """Test that GWRandomForestClassifier fit method works and runs as expected."""
    X, y, geometry = sample_data

    model = GWRandomForestClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWRandomForestClassifier
    assert hasattr(model, "feature_importances_")
    assert hasattr(model, "oob_score_")
    assert pytest.approx(0.6033210332) == model.oob_score_

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)
    pd.testing.assert_series_equal(
        model.feature_importances_.mean(),
        pd.Series(
            [0.33556985859855293, 0.36600269841232724, 0.2984274429891198],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
        atol=0.01,
    )


def test_gwrf_local_oob_metrics(sample_data):  # noqa: F811
    """Test the local OOB metrics for GWRandomForestClassifier."""
    X, y, geometry = sample_data

    model = GWRandomForestClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,
        oob_score=True,
    )

    model.fit(X, y, geometry)

    # Check OOB metrics attributes
    assert hasattr(model, "local_oob_score_")
    assert hasattr(model, "local_oob_precision_")
    assert hasattr(model, "local_oob_recall_")
    assert hasattr(model, "local_oob_f1_macro_")
    assert hasattr(model, "local_oob_f1_micro_")
    assert hasattr(model, "local_oob_f1_weighted_")

    # Check structure and values
    assert isinstance(model.local_oob_score_, pd.Series)
    assert len(model.local_oob_score_) == len(X)
    assert (model.local_oob_score_.dropna() >= 0).all()
    assert (model.local_oob_score_.dropna() <= 1).all()

    # Check that values are as expected
    assert pytest.approx(0.595643939) == model.local_oob_score_.mean()
    assert pytest.approx(0.427504960) == model.local_oob_precision_.mean()
    assert pytest.approx(0.434157986) == model.local_oob_recall_.mean()
    assert pytest.approx(0.463288753) == model.local_oob_f1_macro_.mean()
    assert pytest.approx(0.595643939) == model.local_oob_f1_micro_.mean()
    assert pytest.approx(0.564149780) == model.local_oob_f1_weighted_.mean()


def test_gwrf_global_oob_metrics(sample_data):  # noqa: F811
    """Test the global OOB metrics for GWRandomForestClassifier."""
    X, y, geometry = sample_data

    model = GWRandomForestClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,
        oob_score=True,
    )

    model.fit(X, y, geometry)

    # Check global OOB metrics
    assert hasattr(model, "oob_score_")
    assert hasattr(model, "oob_precision_")
    assert hasattr(model, "oob_recall_")
    assert hasattr(model, "oob_f1_macro_")
    assert hasattr(model, "oob_f1_micro_")
    assert hasattr(model, "oob_f1_weighted_")

    # Check that values are as expected
    assert pytest.approx(0.603321033) == model.oob_score_
    assert pytest.approx(0.585470085) == model.oob_precision_
    assert pytest.approx(0.537254901) == model.oob_recall_
    assert pytest.approx(0.599491330) == model.oob_f1_macro_
    assert pytest.approx(0.603321033) == model.oob_f1_micro_
    assert pytest.approx(0.601803603) == model.oob_f1_weighted_


def test_gwgb_init():
    """Test GWGradientBoostingClassifier initialization."""
    model = GWGradientBoostingClassifier(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "gradient_boosting"

    # Check custom parameters
    model = GWGradientBoostingClassifier(
        bandwidth=50,
        fixed=True,
        kernel="gaussian",
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "gaussian"
    assert model.model_kwargs["n_estimators"] == 100
    assert model.model_kwargs["learning_rate"] == 0.1
    assert model.model_kwargs["subsample"] == 0.8


def test_gwgb_fit_basic(sample_data):  # noqa: F811
    """Test that GWGradientBoostingClassifier fit method works as expected."""
    X, y, geometry = sample_data

    model = GWGradientBoostingClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,  # Use fewer trees for faster testing
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWGradientBoostingClassifier
    assert hasattr(model, "feature_importances_")

    # Check structure of feature importances
    assert isinstance(model.feature_importances_, pd.DataFrame)
    assert model.feature_importances_.shape[0] == len(X)
    assert model.feature_importances_.shape[1] == X.shape[1]
    assert all(col in model.feature_importances_.columns for col in X.columns)


def test_gwgb_with_subsample(sample_data):  # noqa: F811
    """Test GWGradientBoostingClassifier with subsample parameter for OOB scoring."""
    X, y, geometry = sample_data

    model = GWGradientBoostingClassifier(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        n_estimators=50,
        subsample=0.8,  # Enable OOB scoring by using subsample < 1.0
    )

    model.fit(X, y, geometry)

    # Check local OOB score (only available with subsample < 1.0)
    assert hasattr(model, "local_oob_score_")
    assert isinstance(model.local_oob_score_, pd.Series)
    assert len(model.local_oob_score_) == len(X)
