from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import graph
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
    geometry : gpd.GeoSeries, optional
        Geographic location of the observations in the sample. Used to determine the
        spatial interaction weight based on specification by ``bandwidth``, ``fixed``,
        ``kernel``, and ``include_focal`` keywords.  Either ``geometry`` or ``graph``
        need to be specified. To allow prediction, it is required to specify
        ``geometry``.
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations in the sample. If given, it is used directly and ``bandwidth``,
        ``fixed``, ``kernel``, and ``include_focal`` keywords are ignored. Either
        ``geometry`` or ``graph`` need to be specified. To allow prediction, it is
        required to specify ``geometry``. Potentially, both can be specified where
        ``graph`` encodes spatial interaction between observations in ``geometry``.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors by
        default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside the
        geographically weighted, by default True
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
        Number of models to process in each batch. Specify batch_size if your models do
        not fit into memory. By default None
    min_proportion : float, optional
        Minimum proportion of minority class for a model to be fitted, by default 0.2
    undersample : bool, optional
        Whether to apply random undersampling to balance classes, by default False
    leave_out : float | int, optional
        Leave out a fraction (when float) or a set number (when int) of random
        observations from each local model to be used to measure out-of-sample log loss
        based on pooled samples from all the models. This is useful for bandwidth
        selection for cases where some local models are not fitted due to local
        invariance and resulting information criteria are not comparable.
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
    log_likelihood_ : float
        Global log likelihood of the model
    aic_ : float
        Akaike information criterion of the model
    aicc_ : float
        Corrected Akaike information criterion to account for model
        complexity (smaller bandwidths)
    bic_ : float
        Bayesian information criterion
    feature_importances_ : pd.DataFrame
        Feature importance values for each local model
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
    left_out_y : np.ndarray
        Array of ``y`` values left out when ``leave_out`` is set.
    left_out_proba_ : np.ndarray
        Array of probabilites on left out observations in local models when
        ``leave_out`` is set.
    left_out_w_ : np.ndarray
        Array of weights on left out observations in local models when
        ``leave_out`` is set.
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
        geometry: gpd.GeoSeries | None = None,
        graph: graph.Graph = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        min_proportion: float = 0.2,
        undersample: bool = False,
        leave_out: float | int | None = None,
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
            geometry=geometry,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            min_proportion=min_proportion,
            undersample=undersample,
            leave_out=leave_out,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

        self._model_type = "random_forest"
        self._model_kwargs["oob_score"] = self._get_oob_score_data

        self._empty_score_data = (np.array([]).reshape(-1, 1), np.array([]))

    def _get_oob_score_data(self, true, pred):
        return true, pred

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GWRandomForestClassifier":
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
        super().fit(X=X, y=y)

        self._y_local = [x[0] for x in self._score_data]
        self._pred_local = [x[1] for x in self._score_data]

        del self._score_data

        self.oob_y_pooled_ = np.concatenate(self._y_local)
        self.oob_pred_pooled_ = np.concatenate(self._pred_local)

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
    geometry : gpd.GeoSeries, optional
        Geographic location of the observations in the sample. Used to determine the
        spatial interaction weight based on specification by ``bandwidth``, ``fixed``,
        ``kernel``, and ``include_focal`` keywords.  Either ``geometry`` or ``graph``
        need to be specified. To allow prediction, it is required to specify
        ``geometry``.
    graph : Graph, optional
        Custom libpysal.graph.Graph object encoding the spatial interaction between
        observations in the sample. If given, it is used directly and ``bandwidth``,
        ``fixed``, ``kernel``, and ``include_focal`` keywords are ignored. Either
        ``geometry`` or ``graph`` need to be specified. To allow prediction, it is
        required to specify ``geometry``. Potentially, both can be specified where
        ``graph`` encodes spatial interaction between observations in ``geometry``.
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
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
        *,
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
        geometry: gpd.GeoSeries | None = None,
        graph: graph.Graph = None,
        n_jobs: int = -1,
        fit_global_model: bool = True,
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
            geometry=geometry,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            **kwargs,
        )

        self._model_type = "gradient_boosting"
        self._empty_score_data = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GWGradientBoostingClassifier":
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
        super().fit(X=X, y=y)

        # feature importances
        self.feature_importances_ = pd.DataFrame(
            self._feature_importances, index=self._names, columns=X.columns
        )

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Finished")

        return self
