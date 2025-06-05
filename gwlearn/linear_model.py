from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression

from .base import BaseClassifier, BaseRegressor, _scores


class GWLogisticRegression(BaseClassifier):
    """Geographically weighted logistic regression

    Parameters
    ----------
    bandwidth : int | float
        bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows
        assessment of geographically weighted metrics on unseen data without a need for
        train/test split, hence providing value for all samples. This is needed for
        futher spatial analysis of the model performance (and generalises to models
        that do not support OOB scoring). However, it leaves out the most representative
        sample. By default True
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
    measure_performance : bool, optional
        Calculate performance metrics for the model, by default True
    strict : bool | None, optional
        Do not fit any models if at least one neighborhood has invariant ``y``,
        by default False. None is treated as False but provides a warning if there are
        invariant models.
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that
        for some models, like random forests, the objects can be large. If string or
        Path is provided, the local models are not held in memory but serialized to
        the disk from which they are loaded in prediction.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``, by default
        None
    batch_size : int | None, optional
        Number of models to process in each batch. Specify batch_size fi your models do
        not fit into memory. By default None
    min_proportion : float, optional
        Minimum proportion of minority class for a model to be fitted, by default 0.2
    undersample : bool, optional
        Whether to apply random undersampling to balance classes, by default False
    random_state : int | None, optional
        Random seed for reproducibility, by default None
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation
    """

    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: Literal[
            "triangular",
            "parabolic",
            # "gaussian",
            "bisquare",
            "tricube",
            "cosine",
            "boxcar",
            # "exponential",
        ]
        | Callable = "bisquare",
        include_focal: bool = True,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        strict: bool = False,
        keep_models: bool = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        undersample: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=LogisticRegression,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            undersample=undersample,
            **kwargs,
        )

        self._model_type = "logistic"

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        self._empty_score_data = (
            np.array([]),  # true
            np.array([]),  # pred
            pd.Series(np.nan, index=X.columns),  # local coefficients
            np.array([np.nan]),
        )  # intercept

        super().fit(X=X, y=y, geometry=geometry)

        self.local_coef_ = pd.concat(
            [x[2] for x in self._score_data], axis=1, keys=self._names
        ).T
        self.local_intercept_ = pd.Series(
            np.concatenate([x[3] for x in self._score_data]), index=self._names
        )

        if self.measure_performance:
            true = [x[0] for x in self._score_data]
            pred = [x[1] for x in self._score_data]

            del self._score_data

            all_true = np.concat(true)
            all_pred = np.concat(pred)

            # global pred scores
            self.pred_score_ = metrics.accuracy_score(all_true, all_pred)
            self.pred_precision_ = metrics.precision_score(
                all_true, all_pred, zero_division=0
            )
            self.pred_recall_ = metrics.recall_score(
                all_true, all_pred, zero_division=0
            )
            self.pred_f1_macropred_balanced_accuracy_ = metrics.balanced_accuracy_score(
                all_true, all_pred
            )
            self.pred_f1_macro_ = metrics.f1_score(
                all_true, all_pred, average="macro", zero_division=0
            )
            self.pred_f1_micro_ = metrics.f1_score(
                all_true, all_pred, average="micro", zero_division=0
            )
            self.pred_f1_weighted_ = metrics.f1_score(
                all_true, all_pred, average="weighted", zero_division=0
            )

            # local pred scores
            local_score = pd.DataFrame(
                [
                    _scores(y_true, y_false)
                    for y_true, y_false in zip(true, pred, strict=True)
                ],
                index=self._names,
                columns=[
                    "pred_score",
                    "pred_precision",
                    "pred_recall",
                    "pred_balanced_accuracy",
                    "pred_F1_macro",
                    "pred_F1_micro",
                    "pred_F1_weighted",
                ],
            )
            self.local_pred_score_ = local_score["pred_score"]
            self.local_pred_precision_ = local_score["pred_precision"]
            self.local_pred_recall_ = local_score["pred_recall"]
            self.local_pred_balanced_accuracy_ = local_score["pred_balanced_accuracy"]
            self.local_pred_f1_macro_ = local_score["pred_F1_macro"]
            self.local_pred_f1_micro_ = local_score["pred_F1_micro"]
            self.local_pred_f1_weighted_ = local_score["pred_F1_weighted"]

        return self

    def _get_score_data(self, local_model, X, y):
        local_proba = pd.DataFrame(
            local_model.predict_proba(X), columns=local_model.classes_
        )
        return (
            y,
            local_proba.idxmax(axis=1),
            pd.Series(
                local_model.coef_.flatten(),
                index=local_model.feature_names_in_,
            ),  # coefficients
            local_model.intercept_,  # intercept
        )


class GWLinearRegression(BaseRegressor):
    """Geographically weighted linear regression

    Parameters
    ----------
    bandwidth : int | float
        bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows
        assessment of geographically weighted metrics on unseen data without a need for
        train/test split, hence providing value for all samples. This is needed for
        futher spatial analysis of the model performance (and generalises to models
        that do not support OOB scoring). However, it leaves out the most representative
        sample. By default True
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
    measure_performance : bool, optional
        Calculate performance metrics for the model, by default True
    strict : bool | None, optional
        Do not fit any models if at least one neighborhood has invariant ``y``,
        by default False. None is treated as False but provides a warning if there are
        invariant models.
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that
        for some models, like random forests, the objects can be large. If string or
        Path is provided, the local models are not held in memory but serialized to
        the disk from which they are loaded in prediction.
    temp_folder : str | None, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``, by default
        None
    batch_size : int | None, optional
        Number of models to process in each batch. Specify batch_size fi your models do
        not fit into memory. By default None
    **kwargs
        Additional keyword arguments passed to ``sklearn.linear_model.LinearRegression``
        initialisation
    """

    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: Literal[
            "triangular",
            "parabolic",
            # "gaussian",
            "bisquare",
            "tricube",
            "cosine",
            "boxcar",
            # "exponential",
        ]
        | Callable = "bisquare",
        include_focal: bool = True,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        keep_models: bool = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=LinearRegression,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            **kwargs,
        )

        self._model_type = "linear"

    def _get_score_data(self, local_model, X, y):  # noqa: ARG002
        return (
            pd.Series(
                local_model.coef_.flatten(),
                index=local_model.feature_names_in_,
            ),  # coefficients
            local_model.intercept_,  # intercept
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        self._empty_score_data = (
            pd.Series(np.nan, index=X.columns),  # local coefficients
            np.array([np.nan]),
        )  # intercept

        super().fit(X=X, y=y, geometry=geometry)

        self.local_coef_ = pd.concat(
            [x[0] for x in self._score_data], axis=1, keys=self._names
        ).T
        self.local_intercept_ = pd.Series(
            [x[1] for x in self._score_data], index=self._names
        )

        return self
