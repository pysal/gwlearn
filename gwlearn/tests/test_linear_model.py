import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geodatasets import get_path
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import LinearRegression, LogisticRegression

from gwlearn.linear_model import GWLinearRegression, GWLogisticRegression

try:
    from mgwr.gwr import GWR

    HAS_MGWR = True
except ImportError:
    HAS_MGWR = False


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
        include_focal=False,
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
            [-0.0004301675501645129, -0.0620546230731815, 0.06715275989171457],
            index=["Crm_prs", "Litercy", "Wealth"],
        ),
        check_exact=False,
        atol=0.001,
    )

    # Check structure of intercepts
    assert isinstance(model.local_intercept_, pd.Series)
    assert len(model.local_intercept_) == len(X)
    assert pytest.approx(7.873588522, abs=0.01) == model.local_intercept_.mean()


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
        include_focal=False,
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
        include_focal=False,
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


def test_gwlinear_init():
    """Test GWLinearRegression initialization."""
    model = GWLinearRegression(bandwidth=100)

    # Check default parameters
    assert model.bandwidth == 100
    assert model.fixed is False
    assert model.kernel == "bisquare"
    assert model._model_type == "linear"
    assert model.model == LinearRegression

    # Check custom parameters
    model = GWLinearRegression(
        bandwidth=50, fixed=True, kernel="gaussian", fit_intercept=False
    )
    assert model.bandwidth == 50
    assert model.fixed is True
    assert model.kernel == "gaussian"
    assert model.model_kwargs["fit_intercept"] is False


def test_gwlinear_fit_basic(sample_regression_data):
    """Test that GWLinearRegression fit method works as expected."""
    X, y, geometry = sample_regression_data

    model = GWLinearRegression(
        bandwidth=150000,
        fixed=True,
        n_jobs=1,
        include_focal=False,
    )

    fitted_model = model.fit(X, y, geometry)

    # Test that fitting works and returns self
    assert fitted_model is model

    # Test specific attributes of GWLinearRegression
    assert hasattr(model, "local_coef_")
    assert hasattr(model, "local_intercept_")

    # Check structure of coefficients
    assert isinstance(model.local_coef_, pd.DataFrame)
    assert model.local_coef_.shape[0] == len(X)
    assert model.local_coef_.shape[1] == X.shape[1]

    # Check structure of intercepts
    assert isinstance(model.local_intercept_, pd.Series)
    assert len(model.local_intercept_) == len(X)


# def test_gwlinear_coefficients_structure(sample_regression_data):
#     """Test the structure and consistency of the coefficients."""
#     X, y, geometry = sample_regression_data

#     model = GWLinearRegression(
#         bandwidth=150000,
#         fixed=True,
#         keep_models=True,
#     )

#     model.fit(X, y, geometry)

#     # Check that coefficient names match feature names
#     assert all(col in model.local_coef_.columns for col in X.columns)

#     # Pick a sample location and check consistency between local_coef_
#     # and the stored model
#     sample_loc = model.local_models.index[0]
#     local_model = model.local_models[sample_loc]

#     if local_model is not None:  # Some models might be None due to invariance
#         # Compare coefficients from stored model with the ones in local_coef_
#         np.testing.assert_allclose(
#             local_model.coef_.flatten(),
#             model.local_coef_.loc[sample_loc].values,
#             rtol=1e-5,
#         )

#         # Compare intercept
#         assert local_model.intercept_ == pytest.approx(
#             model.local_intercept_[sample_loc]
#         )

# def test_gwlinear_performance_metrics(sample_regression_data):
#     """Test the performance metrics created by GWLinearRegression."""
#     X, y, geometry = sample_regression_data

#     model = GWLinearRegression(
#         bandwidth=150000,
#         fixed=True,
#         include_focal=False,
#     )

#     model.fit(X, y, geometry)

#     # Check the performance metrics attributes
#     assert hasattr(model, "pred_mse_")
#     assert hasattr(model, "pred_mae_")
#     assert hasattr(model, "pred_r2_")

#     # Check that values are reasonable
#     assert isinstance(model.pred_mse_, float)
#     assert isinstance(model.pred_mae_, float)
#     assert isinstance(model.pred_r2_, float)
#     assert model.pred_mse_ >= 0
#     assert model.pred_mae_ >= 0

# def test_gwlinear_local_performance_metrics(sample_regression_data):
#     """Test the local performance metrics."""
#     X, y, geometry = sample_regression_data

#     model = GWLinearRegression(
#         bandwidth=150000,
#         fixed=True,
#         include_focal=False,
#     )

#     model.fit(X, y, geometry)

#     # Check local performance metrics attributes
#     assert hasattr(model, "local_pred_mse_")
#     assert hasattr(model, "local_pred_mae_")
#     assert hasattr(model, "local_pred_r2_")

#     # Check structure and values
#     assert isinstance(model.local_pred_mse_, pd.Series)
#     assert isinstance(model.local_pred_mae_, pd.Series)
#     assert isinstance(model.local_pred_r2_, pd.Series)

#     assert len(model.local_pred_mse_) == len(X)
#     assert len(model.local_pred_mae_) == len(X)
#     assert len(model.local_pred_r2_) == len(X)

#     # Check that values are reasonable (non-negative for MSE and MAE)
#     assert (model.local_pred_mse_.dropna() >= 0).all()
#     assert (model.local_pred_mae_.dropna() >= 0).all()


@pytest.mark.skipif(not HAS_MGWR, reason="needs mgwr")
def test_against_mgwr():
    gdf = gpd.read_file(get_path("geoda.ncovr"))
    gdf = gdf.set_geometry(gdf.representative_point()).to_crs(5070)
    y = gdf["FH90"]

    gwlr = GWLinearRegression(
        bandwidth=250, fixed=False, n_jobs=1, keep_models=False, kernel="bisquare"
    )
    gwlr.fit(
        gdf.iloc[:, 9:15],
        y,
        gdf.geometry,
    )

    gwr = GWR(
        coords=gdf.geometry.get_coordinates(),
        y=y.values.reshape(-1, 1),
        X=gdf.iloc[:, 9:15].values,
        bw=250,
        n_jobs=1,
        fixed=False,
        kernel="bisquare",
    )
    res = gwr.fit()

    assert_array_almost_equal(gwlr.local_r2_, res.localR2.flatten())
    assert_array_almost_equal(gwlr.focal_pred_, res.predy.flatten())
    assert_array_almost_equal(gwlr.TSS_, res.TSS.flatten())
    assert_array_almost_equal(gwlr.RSS_, res.RSS.flatten())
