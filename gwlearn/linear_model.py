from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression

from .base import BaseClassifier, BaseRegressor, _scores


class GWLogisticRegression(BaseClassifier):
    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
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


class GWLinearRegression(BaseRegressor):
    def __init__(
        self,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
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
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            **kwargs,
        )

        self._model_type = "linear"
