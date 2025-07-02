from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .base import BaseClassifier


class GWRandomForestClassifier(BaseClassifier):
    """Geographically weighted random forest classifier

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
        Include focal in the local model training. Excluding it allows assessment of
        geographically weighted metrics on unseen data without a need for train/test
        split, hence providing value for all samples. This is needed for futher spatial
        analysis of the model performance (and generalises to models that do not support
        OOB scoring). However, it leaves out the most representative sample. By default
        False
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations. If given, it is used directly and `bandwidth`, `fixed`, `kernel`,
        and `include_focal` keywords are ignored.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors by
        default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside the
        geographically weighted, by default True
    measure_performance : bool, optional
        Calculate performance metrics for the model. If True, measures accurace score,
        precision, recall, balanced accuracy, and F1 scores (based on focal prediction,
        pooled local out-of-bag predictions and individual local out-of-bag
        predictions). A subset of these can be specified by passing a list of strings.
        By default True
    strict : bool | None, optional
        Do not fit any models if at least one neighborhood has invariant ``y``, by
        default False. None is treated as False but provides a warning if there are
        invariant models.
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that for
        some models, like random forests, the objects can be large. If string or Path is
        provided, the local models are not held in memory but serialized to the disk
        from which they are loaded in prediction.
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

    Attributes
    ----------
    proba_ : pd.DataFrame
        Probability predictions for focal locations based on a local model trained
        around the point itself.
    pred_ : pd.Series
        Binary predictions for focal locations based on a local model trained around the
        location itself.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of hat matrix)
    effective_df_ : float
        Effective degrees of freedom (sum of hat values)
    score_ : float
        Accuracy score of the model based on ``pred_``.
    precision_ : float
        Precision score of the model based on ``pred_``.
    recall_ : float
        Recall score of the model based on ``pred_``.
    balanced_accuracy_ : float
        Balanced accuracy score of the model based on ``pred_``.
    f1_macro_ : float
        F1 score with macro averaging based on ``pred_``.
    f1_micro_ : float
        F1 score with micro averaging based on ``pred_``.
    f1_weighted_ : float
        F1 score with weighted averaging based on ``pred_``.
    log_likelihood_ : float
        Global log likelihood of the model
    aic_ : float
        Akaike inofrmation criterion of the model
    aicc_ : float
        Corrected Akaike information criterion to account to account for model
        complexity (smaller bandwidths)
    bic_ : float
        Bayesian information criterion
    oob_score_ : float
        Out-of-bag accuracy score based on pooled OOB predictions
    oob_precision_ : float
        Out-of-bag precision score based on pooled OOB predictions
    oob_recall_ : float
        Out-of-bag recall score based on pooled OOB predictions
    oob_balanced_accuracy_ : float
        Out-of-bag balanced accuracy score based on pooled OOB predictions
    oob_f1_macro_ : float
        Out-of-bag F1 score with macro averaging based on pooled OOB predictions
    oob_f1_micro_ : float
        Out-of-bag F1 score with micro averaging based on pooled OOB predictions
    oob_f1_weighted_ : float
        Out-of-bag F1 score with weighted averaging based on pooled OOB predictions
    local_oob_score_ : pd.Series
        Out-of-bag accuracy score for each local model
    local_oob_precision_ : pd.Series
        Out-of-bag precision score for each local model
    local_oob_recall_ : pd.Series
        Out-of-bag recall score for each local model
    local_oob_balanced_accuracy_ : pd.Series
        Out-of-bag balanced accuracy score for each local model
    local_oob_f1_macro_ : pd.Series
        Out-of-bag F1 score with macro averaging for each local model
    local_oob_f1_micro_ : pd.Series
        Out-of-bag F1 score with micro averaging for each local model
    local_oob_f1_weighted_ : pd.Series
        Out-of-bag F1 score with weighted averaging for each local model
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
    """

    def __init__(
        self,
        *,
        bandwidth: float,
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
        include_focal: bool = False,
        graph: graph.Graph = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool | list = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        min_proportion: float = 0.2,
        undersample: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=RandomForestClassifier,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            min_proportion=min_proportion,
            undersample=undersample,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "random_forest"
        self._model_kwargs["oob_score"] = self._get_oob_score_data

        self._empty_score_data = (np.array([]).reshape(-1, 1), np.array([]))

    def _get_oob_score_data(self, true, pred):
        return true, pred

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "GWRandomForestClassifier":
        """Fit the geographically weighted model

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        geometry : gpd.GeoSeries
            Geographic location
        """
        self._empty_feature_imp = np.array([np.nan] * (X.shape[1]))
        super().fit(X=X, y=y, geometry=geometry)

        if self.measure_performance:
            if self.measure_performance is True:
                metrics_to_measure = [
                    "accuracy",
                    "precision",
                    "recall",
                    "balanced_accuracy",
                    "f1_macro",
                    "f1_micro",
                    "f1_weighted",
                ]
            else:
                metrics_to_measure = self.measure_performance
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Measuring pooled performance")

            # OOB accuracy for RF can be measured both local and global
            true, pred = zip(*self._score_data, strict=False)
            del self._score_data

            all_true = np.concatenate(true)
            all_pred = np.concatenate(pred)

            # global OOB scores
            if "accuracy" in metrics_to_measure:
                self.oob_score_ = metrics.accuracy_score(all_true, all_pred)

            if "precision" in metrics_to_measure:
                self.oob_precision_ = metrics.precision_score(
                    all_true, all_pred, zero_division=0
                )

            if "recall" in metrics_to_measure:
                self.oob_recall_ = metrics.recall_score(
                    all_true, all_pred, zero_division=0
                )

            if "balanced_accuracy" in metrics_to_measure:
                self.oob_balanced_accuracy_ = metrics.balanced_accuracy_score(
                    all_true, all_pred
                )

            if "f1_macro" in metrics_to_measure:
                self.oob_f1_macro_ = metrics.f1_score(
                    all_true, all_pred, average="macro", zero_division=0
                )

            if "f1_micro" in metrics_to_measure:
                self.oob_f1_micro_ = metrics.f1_score(
                    all_true, all_pred, average="micro", zero_division=0
                )

            if "f1_weighted" in metrics_to_measure:
                self.oob_f1_weighted_ = metrics.f1_score(
                    all_true, all_pred, average="weighted", zero_division=0
                )

            if self.verbose:
                print(
                    f"{(time() - self._start):.2f}s: Measuring local pooled performance"
                )

            # local OOB scores
            local_score = pd.DataFrame(
                [
                    self._scores(y_true, y_false)
                    for y_true, y_false in zip(true, pred, strict=True)
                ],
                index=self._names,
                columns=["oob_" + c for c in metrics_to_measure],
            )
            if "accuracy" in metrics_to_measure:
                self.local_oob_score_ = local_score["oob_accuracy"]
            if "precision" in metrics_to_measure:
                self.local_oob_precision_ = local_score["oob_precision"]
            if "recall" in metrics_to_measure:
                self.local_oob_recall_ = local_score["oob_recall"]
            if "balanced_accuracy" in metrics_to_measure:
                self.local_oob_balanced_accuracy_ = local_score["oob_balanced_accuracy"]
            if "f1_macro" in metrics_to_measure:
                self.local_oob_f1_macro_ = local_score["oob_f1_macro"]
            if "f1_micro" in metrics_to_measure:
                self.local_oob_f1_micro_ = local_score["oob_f1_micro"]
            if "f1_weighted" in metrics_to_measure:
                self.local_oob_f1_weighted_ = local_score["oob_f1_weighted"]

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self

    def _get_score_data(self, local_model, X, y):  # noqa: ARG002
        return local_model.oob_score_


class GWGradientBoostingClassifier(BaseClassifier):
    """Geographically weighted gradient boosting classifier

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
        sample. By default False
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations. If given, it is used directly and `bandwidth`, `fixed`, `kernel`,
        and `include_focal` keywords are ignored.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
    measure_performance : bool, optional
        Calculate performance metrics for the model. If True, measures accurace score,
        precision, recall, balanced accuracy, and F1 scores. A subset of these can
        be specified by passing a list of strings. By default True
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

    Attributes
    ----------
    proba_ : pd.DataFrame
        Probability predictions for focal locations based on a local model trained
        around the point itself.
    pred_ : pd.Series
        Binary predictions for focal locations based on a local model trained around the
        location itself.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of hat matrix)
    effective_df_ : float
        Effective degrees of freedom (sum of hat values)
    score_ : float
        Accuracy score of the model based on ``pred_``.
    precision_ : float
        Precision score of the model based on ``pred_``.
    recall_ : float
        Recall score of the model based on ``pred_``.
    balanced_accuracy_ : float
        Balanced accuracy score of the model based on ``pred_``.
    f1_macro_ : float
        F1 score with macro averaging based on ``pred_``.
    f1_micro_ : float
        F1 score with micro averaging based on ``pred_``.
    f1_weighted_ : float
        F1 score with weighted averaging based on ``pred_``.
    log_likelihood_ : float
        Global log likelihood of the model
    aic_ : float
        Akaike inofrmation criterion of the model
    aicc_ : float
        Corrected Akaike information criterion to account to account for model
        complexity (smaller bandwidths)
    bic_ : float
        Bayesian information criterion
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
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
        include_focal: bool = False,
        graph: graph.Graph = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool | list = True,
        strict: bool = False,
        keep_models: bool = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(
            model=GradientBoostingClassifier,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            **kwargs,
        )

        self._model_type = "gradient_boosting"
        self._empty_score_data = np.nan

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "GWGradientBoostingClassifier":
        """Fit the geographically weighted model

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        geometry : gpd.GeoSeries
            Geographic location
        """
        self._empty_feature_imp = np.array([np.nan] * (X.shape[1]))
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

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self
