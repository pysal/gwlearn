from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .base import BaseClassifier, _scores


class GWRandomForestClassifier(BaseClassifier):
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
        min_proportion: float = 0.2,
        **kwargs,
    ):
        self._model_type = "random_forest"

        super().__init__(
            model=RandomForestClassifier,
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
            min_proportion=min_proportion,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        super().fit(X=X, y=y, geometry=geometry)

        if self.measure_performance:
            # OOB accuracy for RF can be measured both local and global
            true, pred = zip(*self._score_data, strict=False)
            del self._score_data

            all_true = np.concat(true)
            all_pred = np.concat(pred)

            # global OOB scores
            self.oob_score_ = metrics.accuracy_score(all_true, all_pred)
            self.oob_precision_ = metrics.precision_score(
                all_true, all_pred, zero_division=0
            )
            self.oob_recall_ = metrics.recall_score(all_true, all_pred, zero_division=0)
            self.oob_balanced_accuracy_ = metrics.balanced_accuracy_score(
                all_true, all_pred
            )
            self.oob_f1_macro = metrics.f1_score(
                all_true, all_pred, average="macro", zero_division=0
            )
            self.oob_f1_micro = metrics.f1_score(
                all_true, all_pred, average="micro", zero_division=0
            )
            self.oob_f1_weighted = metrics.f1_score(
                all_true, all_pred, average="weighted", zero_division=0
            )

            # local OOB scores
            local_score = pd.DataFrame(
                [
                    _scores(y_true, y_false)
                    for y_true, y_false in zip(true, pred, strict=True)
                ],
                index=self._names,
                columns=[
                    "oob_score",
                    "oob_precision",
                    "oob_recall",
                    "oob_balanced_accuracy",
                    "oob_F1_macro",
                    "oob_F1_micro",
                    "oob_F1_weighted",
                ],
            )
            self.local_oob_score_ = local_score["oob_score"]
            self.local_oob_precision_ = local_score["oob_precision"]
            self.local_oob_recall_ = local_score["oob_recall"]
            self.local_oob_balanced_accuracy_ = local_score["oob_balanced_accuracy"]
            self.local_oob_f1_macro_ = local_score["oob_F1_macro"]
            self.local_oob_f1_micro_ = local_score["oob_F1_micro"]
            self.local_oob_f1_weighted_ = local_score["oob_F1_weighted"]

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        return self


class GWGradientBoostingClassifier(BaseClassifier):
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
        **kwargs,
    ):
        self._model_type = "gradient_boosting"

        super().__init__(
            model=GradientBoostingClassifier,
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
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        super().fit(X=X, y=y, geometry=geometry)

        if self.measure_performance:
            # OOB accuracy for stochastic GB can be measured as local only. GB is
            # stochastic if subsample < 1.0. Otherwise, oob_score_ is not available
            # as all samples were used in training
            self.local_oob_score_ = pd.Series(self._score_data, index=self._names)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        return self
