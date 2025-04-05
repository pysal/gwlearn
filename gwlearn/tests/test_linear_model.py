import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from gwlearn.linear_model import GWLogisticRegression
from gwlearn.tests.test_base import sample_data  # noqa: F401


def test_gwlogistic_init():
    """Test GWLogisticRegression initialization."""
    model = GWLogisticRegression(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "logistic"
    assert model.model == LogisticRegression

    # Check custom parameters
    model = GWLogisticRegression(
        bandwidth=50, fixed=True, kernel="gaussian", C=0.5, max_iter=200
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "gaussian"
    assert model.model_kwargs["C"] == 0.5
    assert model.model_kwargs["max_iter"] == 200


def test_gwlogistic_fit_basic(sample_data):  # noqa: F811
    """Test that GWLogisticRegression fit method works and as expected."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,  # To avoid warnings on invariance
        max_iter=500,
        n_jobs=1,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWLogisticRegression
    assert hasattr(model, "local_coef_")
    assert hasattr(model, "local_intercept_")

    # Check structure of coefficients
    assert isinstance(model.local_coef_, pd.DataFrame)
    assert model.local_coef_.shape[0] == len(X)
    assert model.local_coef_.shape[1] == X.shape[1]

    pd.testing.assert_series_equal(
        model.local_coef_.mean(),
        pd.Series(
            [-0.000421280677240649, -0.06094205797063275, 0.06581666557051737],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
    )

    # Check structure of intercepts
    assert isinstance(model.local_intercept_, pd.Series)
    assert len(model.local_intercept_) == len(X)
    assert pytest.approx(7.662016468, abs=0.01) == model.local_intercept_.mean()


def test_gwlogistic_coefficients_structure(sample_data):  # noqa: F811
    """Test the structure and consistency of the coefficients."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        keep_models=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    model.fit(X, y, geometry)

    # Check that coefficient names match feature names
    assert all(col in model.local_coef_.columns for col in X.columns)

    # Pick a sample location and check consistency between local_coef_
    # and the stored model
    sample_loc = model.local_models.index[0]
    local_model = model.local_models[sample_loc]

    if local_model is not None:  # Some models might be None due to invariance
        # Compare coefficients from stored model with the ones in local_coef_
        np.testing.assert_allclose(
            local_model.coef_.flatten(),
            model.local_coef_.loc[sample_loc].values,
            rtol=1e-5,
        )

        # Compare intercept
        assert local_model.intercept_[0] == pytest.approx(
            model.local_intercept_[sample_loc]
        )


def test_gwlogistic_prediction_metrics(sample_data):  # noqa: F811
    """Test the prediction-specific metrics created by GWLogisticRegression."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    model.fit(X, y, geometry)

    # Check the prediction metrics attributes
    assert hasattr(model, "pred_score_")
    assert hasattr(model, "pred_precision_")
    assert hasattr(model, "pred_recall_")
    assert hasattr(model, "pred_f1_macro_")
    assert hasattr(model, "pred_f1_micro_")
    assert hasattr(model, "pred_f1_weighted_")

    # Check that values are as expected
    assert pytest.approx(0.859778597) == model.pred_score_
    assert pytest.approx(0.859437751) == model.pred_precision_
    assert pytest.approx(0.839215686) == model.pred_recall_
    assert pytest.approx(0.859085933) == model.pred_f1_macro_
    assert pytest.approx(0.859778597) == model.pred_f1_micro_
    assert pytest.approx(0.859669229) == model.pred_f1_weighted_


def test_gwlogistic_local_prediction_metrics(sample_data):  # noqa: F811
    """Test the local prediction metrics."""
    X, y, geometry = sample_data

    model = GWLogisticRegression(
        bandwidth=150000,
        fixed=True,
        random_state=42,
        strict=False,
        max_iter=500,
    )

    model.fit(X, y, geometry)

    # Check local prediction metrics attributes
    assert hasattr(model, "local_pred_score_")
    assert hasattr(model, "local_pred_precision_")
    assert hasattr(model, "local_pred_recall_")
    assert hasattr(model, "local_pred_f1_macro_")
    assert hasattr(model, "local_pred_f1_micro_")
    assert hasattr(model, "local_pred_f1_weighted_")

    # Check structure and values
    assert isinstance(model.local_pred_score_, pd.Series)
    assert len(model.local_pred_score_) == len(X)
    assert (model.local_pred_score_.dropna() >= 0).all()
    assert (model.local_pred_score_.dropna() <= 1).all()

    # Check that values are as expected
    assert pytest.approx(0.879587166) == model.local_pred_score_.mean()
    assert pytest.approx(0.889862351) == model.local_pred_precision_.mean()
    assert pytest.approx(0.849844990) == model.local_pred_recall_.mean()
    assert pytest.approx(0.859829075) == model.local_pred_f1_macro_.mean()
    assert pytest.approx(0.879587166) == model.local_pred_f1_micro_.mean()
    assert pytest.approx(0.877102172) == model.local_pred_f1_weighted_.mean()
