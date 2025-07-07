import inspect
import warnings
from collections.abc import Callable, Hashable
from pathlib import Path
from time import time
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from libpysal import graph
from scipy.spatial import KDTree
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

# TODO: summary
# TODO: formal documentation
# TODO: comments in code

__all__ = ["BaseClassifier", "BaseRegressor"]


def _triangular(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u**2


def _gaussian(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-((u / 2) ** 2))


def _bisquare(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (1 - u**2) ** 2


def _cosine(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return np.cos(np.pi / 2 * u)


def _exponential(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    r = (distances < bandwidth).astype(int)
    return r


def _tricube(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (1 - u**3) ** 3


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    # "gaussian": _gaussian,
    "bisquare": _bisquare,
    "tricube": _tricube,
    "cosine": _cosine,
    "boxcar": _boxcar,
    # "exponential": _exponential,
}


class _BaseModel(BaseEstimator):
    """Base class for geographically weighted models"""

    def __init__(
        self,
        model,
        *,
        bandwidth: float | None = None,
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
        measure_performance: bool | list = True,
        strict: bool | None = False,
        keep_models: bool | str | Path = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.include_focal = include_focal
        self.geometry = geometry
        self.graph = graph
        self.fixed = fixed
        self._model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.fit_global_model = fit_global_model
        self.measure_performance = measure_performance
        self.strict = strict
        if isinstance(keep_models, str):
            keep_models = Path(keep_models)
        self.keep_models = keep_models
        self.temp_folder = temp_folder
        self.batch_size = batch_size
        self.verbose = verbose
        self._model_type = None

    def _validate_geometry(self, geometry):
        """Validate that geometry contains only Point geometries"""
        if geometry is not None and not (geometry.geom_type == "Point").all():
            raise ValueError(
                "Unsupported geometry type. Only point geometry is allowed."
            )

    def _build_weights(self) -> graph.Graph:
        """Build spatial weights graph"""
        if self.fixed:  # fixed distance
            weights = graph.Graph.build_kernel(
                self.geometry,
                kernel=_kernel_functions[self.kernel],
                bandwidth=self.bandwidth,
            )
        else:  # adaptive KNN
            weights = graph.Graph.build_kernel(
                self.geometry,
                kernel="identity",
                k=self.bandwidth - 1 if self.include_focal else self.bandwidth,
            )
            # post-process identity weights by the selected kernel
            # and kernel bandwidth derived from each neighborhood
            # the epsilon comes from MGWR to avoid division by zero
            bandwidth = weights._adjacency.groupby(level=0).transform("max") * 1.0000001
            weights = graph.Graph(
                adjacency=_kernel_functions[self.kernel](weights._adjacency, bandwidth),
                is_sorted=True,
            )
        if self.include_focal:
            weights = weights.assign_self_weight(1)
        return weights

    def _setup_model_storage(self):
        """Setup model storage directory if needed"""
        if isinstance(self.keep_models, Path):
            self.keep_models.mkdir(exist_ok=True)

    def _fit_models_batch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: graph.Graph,
    ) -> list:
        """Fit models in batches or all at once"""
        if self.batch_size:
            training_output = []
            num_groups = len(y)
            indices = np.arange(num_groups)
            for i in range(0, num_groups, self.batch_size):
                if self.verbose:
                    print(
                        f"Processing batch {i // self.batch_size + 1} "
                        f"out of {(num_groups // self.batch_size) + 1}."
                    )

                batch_indices = indices[i : i + self.batch_size]
                subset_weights = weights._adjacency.loc[batch_indices, :]

                index = subset_weights.index
                _weight = subset_weights.values
                X_focals = X.values[batch_indices]

                batch_training_output = self._batch_fit(X, y, index, _weight, X_focals)
                training_output.extend(batch_training_output)
        else:
            index = weights._adjacency.index
            _weight = weights._adjacency.values
            X_focals = X.values

            training_output = self._batch_fit(X, y, index, _weight, X_focals)

        return training_output

    def _batch_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        index: pd.MultiIndex,
        _weight: np.ndarray,
        X_focals: np.ndarray,
    ) -> list:
        """Fit a batch of local models"""
        data = X.copy()
        data["_y"] = y
        data = data.loc[index.get_level_values(1)]
        data["_weight"] = _weight
        grouper = data.groupby(index.get_level_values(0))

        invariant = grouper["_y"].nunique() == 1
        if invariant.any():
            if self.strict:
                raise ValueError(
                    f"y at locations {invariant.index[invariant]} is invariant."
                )
            elif self.strict is None:
                warnings.warn(
                    f"y at locations {invariant.index[invariant]} is invariant.",
                    stacklevel=3,
                )

        return Parallel(n_jobs=self.n_jobs, temp_folder=self.temp_folder)(
            delayed(self._fit_local)(
                self.model,
                group,
                name,
                focal_x,
                self._model_kwargs,
            )
            for (name, group), focal_x in zip(grouper, X_focals, strict=False)
        )

    def _fit_global_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit global baseline model"""
        if self._model_type == "random_forest":
            self._model_kwargs["oob_score"] = True
        # fit global model as a baseline
        if "n_jobs" in inspect.signature(self.model).parameters:
            self.global_model = self.model(n_jobs=self.n_jobs, **self._model_kwargs)
        else:
            self.global_model = self.model(**self._model_kwargs)

        self.global_model.fit(X=X, y=y)

    def _store_model(self, local_model, name: Hashable):
        """Store or serialize local model"""
        if self.keep_models is True:  # if True, models are kept in memory
            return local_model
        elif isinstance(self.keep_models, Path):  # if Path, models are saved to disk
            p = f"{self.keep_models.joinpath(f'{name}.joblib')}"
            with open(p, "wb") as f:
                dump(local_model, f, protocol=5)
            del local_model
            return p
        else:
            del local_model
            return None

    def _compute_hat_value(
        self, X: pd.DataFrame, weights: np.ndarray, focal_x: np.ndarray
    ) -> float:
        """
        Compute the hat value (leverage) for the focal point.

        For classification problems, this is an approximation rather than an ideal
        solution but it should be good enough for bandwidth search.

        Parameters:
        -----------
        X : pd.DataFrame
            Design matrix of the local neighborhood
        weights : np.ndarray
            Spatial weights for the neighborhood
        focal_x : np.ndarray
            Feature vector of the focal point

        Returns:
        --------
        float : hat value for the focal point
        """
        try:
            # Add intercept if not present
            if not (X.iloc[:, 0] == 1).all():
                X_with_intercept = np.column_stack([np.ones(len(X)), X.values])
                focal_with_intercept = np.concatenate([[1], focal_x.flatten()])
            else:
                X_with_intercept = X.values
                focal_with_intercept = focal_x.flatten()

            # Compute (X^T W X)^(-1)
            XtWX = X_with_intercept.T @ np.diag(weights) @ X_with_intercept
            XtWX_inv = np.linalg.pinv(
                XtWX
            )  # Use pseudo-inverse for numerical stability

            # Hat value: h_ii = x_i^T (X^T W X)^(-1) x_i * w_i
            hat_value = focal_with_intercept.T @ XtWX_inv @ focal_with_intercept

            return hat_value

        except (np.linalg.LinAlgError, ValueError):
            # Return NaN if computation fails (singular matrix, etc.)
            return np.nan

    def _compute_information_criteria(self):
        """Compute AIC, BIC, and AICc using the global log likelihood"""
        n = (
            self._n_fitted_models
            if hasattr(self, "_n_fitted_models")
            else len(self.pred_)
        )

        # Use effective degrees of freedom as the number of parameters
        k = self.effective_df_

        if not np.isnan(self.log_likelihood_) and not np.isnan(k):
            # Akaike Information Criterion
            self.aic_ = 2 * (k + 1) - 2 * self.log_likelihood_

            # Bayesian Information Criterion
            self.bic_ = np.log(n) * (k + 1) - 2 * self.log_likelihood_

            # Corrected AIC for small samples
            if n - k - 1 > 0:
                self.aicc_ = self.aic_ + (2 * k * (k + 1)) / (n - k - 1)
                # the implementation below matches MGWR but the formula above is
                # typical AICc implementation. The difference is minor.
                # self.aicc_ = -2.0 * self.log_likelihood_ + 2.0 * n * (k + 1.0) / (
                #     n - k - 2.0
                # )
            else:
                self.aicc_ = np.nan
        else:
            self.aic_ = np.nan
            self.bic_ = np.nan
            self.aicc_ = np.nan

    # Abstract methods that subclasses must implement
    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> tuple:
        raise NotImplementedError("Subclasses must implement _fit_local")

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
        raise NotImplementedError("Subclasses must implement fit")

    def _get_score_data(self, local_model, X, y):  # noqa: ARG002
        """Subclasses should implement custom function"""
        return np.nan

    # def __sklearn_tags__(self):
    #     tags = super().__sklearn_tags__(self)
    #     return tags


class BaseClassifier(ClassifierMixin, _BaseModel):
    """Generic geographically weighted classification meta-class

    Parameters
    ----------
    model : model class
        Scikit-learn model class
    bandwidth : int | float
        Bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows assessment of
        geographically weighted metrics on unseen data without a need for train/test
        split, hence providing value for all samples. This is needed for further spatial
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
        Determines if the global baseline model shall be fitted alongside the
        geographically weighted, by default True
    measure_performance : bool | list, optional
        Calculate performance metrics for the model. If True, measures accuracy score,
        precision, recall, balanced accuracy, F1 scores and log loss. A subset of these
        can be specified by passing a list of strings. By default True
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
    log_loss_ : float
        Log loss of the model based on ``pred_``.
    log_likelihood_ : float
        Global log likelihood of the model
    aic_ : float
        Akaike information criterion of the model
    aicc_ : float
        Corrected Akaike information criterion to account for model complexity (smaller
        bandwidths)
    bic_ : float
        Bayesian information criterion
    prediction_rate_ : float
        Proportion of models that are fitted, where the rest are skipped due to not
        fulfilling ``min_proportion``.
    oos_log_loss_ : float
        Out-of-sample log loss of the model. It is based on pooled data of randomly left
        out observations from training of local models. Log loss is measured as weighted
        using the set bandwidth and a kernel. Available only when ``leave_out`` is not
        None.
    """

    def __init__(
        self,
        model,
        *,
        bandwidth: float | None = None,
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
        measure_performance: bool = True,
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
            model=model,
            bandwidth=bandwidth,
            fixed=fixed,
            kernel=kernel,
            include_focal=include_focal,
            geometry=geometry,
            graph=graph,
            n_jobs=n_jobs,
            fit_global_model=fit_global_model,
            measure_performance=measure_performance,
            strict=strict,
            keep_models=keep_models,
            temp_folder=temp_folder,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        self.min_proportion = min_proportion
        self.undersample = undersample
        self.random_state = random_state
        self.leave_out = leave_out
        self._empty_score_data = None
        self._empty_feature_imp = None

        if undersample:
            try:
                from imblearn.under_sampling import RandomUnderSampler  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "imbalance-learn is required for undersampling."
                ) from err

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseClassifier":
        """Fit the geographically weighted model

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        """
        self._start = time()

        def _is_binary(series: pd.Series) -> bool:
            """Check if a pandas Series encodes a binary variable (bool or 0/1)."""
            unique_values = set(np.unique(series))

            # Check for boolean type
            if series.dtype == bool or unique_values.issubset({True, False}):
                return True

            # Check for 0, 1 encoding
            return bool(unique_values.issubset({0, 1}))

        self._validate_geometry(self.geometry)

        if not _is_binary(y):
            raise ValueError("Only binary dependent variable is supported.")

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Building weights")

        weights = self.graph if self.graph is not None else self._build_weights()
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Weights ready")
        self._setup_model_storage()

        self._global_classes = np.unique(y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(X.shape[1])

        # fit the models
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Fitting the models")
        training_output = self._fit_models_batch(X, y, weights)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Models fitted")

        if self.keep_models:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,
                left_out_proba,
                models,
            ) = zip(*training_output, strict=False)
            self._local_models = pd.Series(models, index=self._names)
        else:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,
                left_out_proba,
            ) = zip(*training_output, strict=False)

        self._n_labels = pd.Series(self._n_labels, index=self._names)
        self.proba_ = pd.DataFrame(focal_proba, index=self._names)

        # Store hat values and compute effective degrees of freedom
        self.hat_values_ = pd.Series(hat_values, index=self._names)
        self.effective_df_ = np.nansum(self.hat_values_)

        # support both bool and 0, 1 encoding of binary variable
        col = True if True in self.proba_.columns else 1
        # global GW accuracy
        nan_mask = self.proba_[col].isna()
        self.pred_ = self.proba_[col][~nan_mask] > 0.5

        self._n_fitted_models = (~self.proba_[col].isna()).sum()
        self.prediction_rate_ = self._n_fitted_models / nan_mask.shape[0]

        if self.leave_out:
            if self.prediction_rate_ > 0:
                self.oos_log_loss_ = np.nan

            else:
                y_pred = np.concatenate([arr[0] for arr in left_out_proba])
                y_true = np.concatenate([arr[1] for arr in left_out_proba])
                w = np.concatenate([arr[2] for arr in left_out_proba])

                # TODO: this could potentially follow the logic of measure_performance
                # and measure more than log loss
                self.oos_log_loss_ = metrics.log_loss(y_true, y_pred, sample_weight=w)

        if self.fit_global_model:
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Fitting global model")
            self._fit_global_model(X, y)

        if self.measure_performance:
            if self.measure_performance is True:
                metrics_to_measure = [
                    "score",
                    "precision",
                    "recall",
                    "balanced_accuracy",
                    "f1_macro",
                    "f1_micro",
                    "f1_weighted",
                    "log_loss",
                ]
            else:
                metrics_to_measure = self.measure_performance
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Measuring focal performance")
            masked_y = y[~nan_mask]

            if "score" in metrics_to_measure:
                self.score_ = (
                    metrics.accuracy_score(masked_y, self.pred_)
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "precision" in metrics_to_measure:
                self.precision_ = (
                    metrics.precision_score(masked_y, self.pred_, zero_division=0)
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "recall" in metrics_to_measure:
                self.recall_ = (
                    metrics.recall_score(masked_y, self.pred_, zero_division=0)
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "balanced_accuracy" in metrics_to_measure:
                self.balanced_accuracy_ = (
                    metrics.balanced_accuracy_score(masked_y, self.pred_)
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "f1_macro" in metrics_to_measure:
                self.f1_macro_ = (
                    metrics.f1_score(
                        masked_y, self.pred_, average="macro", zero_division=0
                    )
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "f1_micro" in metrics_to_measure:
                self.f1_micro_ = (
                    metrics.f1_score(
                        masked_y, self.pred_, average="micro", zero_division=0
                    )
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "f1_weighted" in metrics_to_measure:
                self.f1_weighted_ = (
                    metrics.f1_score(
                        masked_y, self.pred_, average="weighted", zero_division=0
                    )
                    if self.prediction_rate_ > 0
                    else np.nan
                )

            if "log_loss" in metrics_to_measure:
                self.log_loss_ = (
                    metrics.log_loss(
                        masked_y,
                        self.proba_[~nan_mask],
                    )
                    if self.prediction_rate_ > 0
                    else np.nan
                )

        # Compute global log likelihood and information criteria
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Computing global likelihood")
        self.log_likelihood_ = self._compute_global_log_likelihood(y)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Computing information criteria")
        self._compute_information_criteria()

        return self

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> tuple:
        """Fit individual local model"""

        if self.undersample:
            from imblearn.under_sampling import RandomUnderSampler

        vc = data["_y"].value_counts()
        n_labels = len(vc)
        skip = n_labels == 1
        if n_labels > 1:
            skip = (vc.iloc[1] / vc.iloc[0]) < self.min_proportion

        # empty data for skipped models
        score_data = self._empty_score_data
        feature_imp = self._empty_feature_imp
        output = [
            name,
            n_labels,
            score_data,
            feature_imp,
            pd.Series(np.nan, index=self._global_classes),
            np.nan,
            (np.zeros(shape=(0, 2)), data["_y"].iloc[:0], data["_weight"].iloc[:0]),
        ]
        if self.keep_models:
            output.append(None)

        if skip:
            return output

        local_model = model(random_state=self.random_state, **model_kwargs)

        if self.undersample:
            if isinstance(self.undersample, float):
                rus = RandomUnderSampler(
                    sampling_strategy=self.undersample, random_state=self.random_state
                )
            else:
                rus = RandomUnderSampler(random_state=self.random_state)
            data, _ = rus.fit_resample(data, data["_y"])

        if self.leave_out:
            try:
                data, left_out_data = train_test_split(
                    data, test_size=self.leave_out, stratify=data["_y"]
                )
            except ValueError:  # only 1 observation of True
                return output
            if len(data["_y"].value_counts()) == 1:
                return output

        X = data.drop(columns=["_y", "_weight"])
        y = data["_y"]

        local_model.fit(
            X=X,
            y=y,
            sample_weight=data["_weight"],
        )
        focal_x = pd.DataFrame(
            focal_x.reshape(1, -1),
            columns=X.columns,
            index=[name],
        )
        focal_proba = pd.Series(
            local_model.predict_proba(focal_x).flatten(), index=local_model.classes_
        )

        hat_value = self._compute_hat_value(X, data["_weight"], focal_x.values)

        if self.leave_out:
            left_out_proba = local_model.predict_proba(
                left_out_data.drop(columns=["_y", "_weight"])
            )
            left_out_proba = (
                left_out_proba,
                left_out_data["_y"],
                left_out_data["_weight"],
            )
        else:
            left_out_proba = None

        output = [
            name,
            n_labels,
            self._get_score_data(local_model, X, y),
            getattr(local_model, "feature_importances_", None),
            focal_proba,
            hat_value,
            left_out_proba,
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

    def _compute_global_log_likelihood(self, y: pd.Series) -> float:
        """
        Compute global log likelihood for classification
        """
        # Get valid predictions (non-NaN)
        mask = ~self.proba_.isna().any(axis=1)

        if not mask.any():
            return np.nan

        y_valid = y[mask]
        proba_valid = self.proba_[mask]

        # Handle both boolean and 0/1 encoding
        if True in proba_valid.columns:
            p = proba_valid[True]
            y_binary = y_valid.astype(int) if y_valid.dtype == bool else y_valid
        else:
            p = proba_valid[1]
            y_binary = y_valid

        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        log_likelihood = np.sum(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p))

        return log_likelihood

    def predict_proba(self, X: pd.DataFrame, geometry: gpd.GeoSeries) -> pd.DataFrame:
        """Predict probabiliies using the ensemble of local models"""
        self._validate_geometry(geometry)

        if self.fixed:
            input_ids, local_ids = self.geometry.sindex.query(
                geometry, predicate="dwithin", distance=self.bandwidth
            )
            distance = _kernel_functions[self.kernel](
                self.geometry.iloc[local_ids].distance(
                    geometry.iloc[input_ids], align=False
                ),
                self.bandwidth,
            )
        else:
            training_coords = self.geometry.get_coordinates()
            tree = KDTree(training_coords)
            query_coords = geometry.get_coordinates()

            distances, indices_array = tree.query(query_coords, k=self.bandwidth)

            # Flatten arrays for consistent format
            input_ids = np.repeat(np.arange(len(geometry)), self.bandwidth)
            local_ids = indices_array.flatten()
            distances = distances.flatten()

            # For adaptive KNN, determine the bandwidth for each neighborhood
            # by finding the max distance in each neighborhood
            kernel_bandwidth = (
                pd.Series(distances).groupby(input_ids).transform("max") + 1e-6
            )  # can't have 0
            distance = _kernel_functions[self.kernel](distances, kernel_bandwidth)

        split_indices = np.where(np.diff(input_ids))[0] + 1
        local_model_ids = np.split(local_ids, split_indices)
        distances = np.split(distance.values, split_indices)
        data = np.split(X.to_numpy(), range(1, len(X)))

        probabilities = []
        for x_, models_, distances_ in zip(
            data, local_model_ids, distances, strict=True
        ):
            probabilities.append(
                self._predict_proba(x_, models_, distances_, X.columns)
            )

        return pd.DataFrame(probabilities, columns=self._global_classes, index=X.index)

    def _predict_proba(
        self,
        x_: np.ndarray,
        models_: np.ndarray,
        distances_: np.ndarray,
        columns: pd.Index,
    ) -> pd.Series:
        x_ = pd.DataFrame(np.array(x_).reshape(1, -1), columns=columns)
        pred = []
        for i in models_:
            local_model = self._local_models[i]
            if isinstance(local_model, str):
                with open(local_model, "rb") as f:
                    local_model = load(f)

            if local_model is not None:
                pred.append(
                    pd.Series(
                        local_model.predict_proba(x_).flatten(),
                        index=local_model.classes_,
                    )
                )
            else:
                pred.append(
                    pd.Series(
                        np.nan,
                        index=self._global_classes,
                    )
                )
        pred = pd.DataFrame(pred)

        mask = pred.isna().any(axis=1)
        if mask.all():
            return pd.Series(np.nan, index=pred.columns)

        weighted = np.average(pred[~mask], axis=0, weights=distances_[~mask])

        # normalize
        weighted = weighted / weighted.sum()
        return pd.Series(weighted, index=pred.columns)

    def predict(self, X: pd.DataFrame, geometry: gpd.GeoSeries) -> pd.Series:
        proba = self.predict_proba(X, geometry)
        return proba.idxmax(axis=1)

    def _scores(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics_to_measure: list | None = None,
    ) -> tuple:
        if metrics_to_measure is None:
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

        if y_true.shape[0] == 0:
            return (np.nan,) * len(metrics_to_measure)

        results = []

        if "accuracy" in metrics_to_measure:
            results.append(metrics.accuracy_score(y_true, y_pred))
        if "precision" in metrics_to_measure:
            results.append(metrics.precision_score(y_true, y_pred, zero_division=0))
        if "recall" in metrics_to_measure:
            results.append(metrics.recall_score(y_true, y_pred, zero_division=0))
        if "balanced_accuracy" in metrics_to_measure:
            results.append(metrics.balanced_accuracy_score(y_true, y_pred))
        if "f1_macro" in metrics_to_measure:
            results.append(
                metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if "f1_micro" in metrics_to_measure:
            results.append(
                metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)
            )
        if "f1_weighted" in metrics_to_measure:
            results.append(
                metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )

        return results


class BaseRegressor(_BaseModel, RegressorMixin):
    """Generic geographically weighted regression meta-class

    TODO:
        - tvalues & adj_alpha & critical_t val
        - predict
        - performance measurements

    Parameters
    ----------
    model : model class
        Scikit-learn model class
    bandwidth : int | float
        Bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default "bisquare"
    include_focal : bool, optional
        Include focal in the local model training. Excluding it allows
        assessment of geographically weighted metrics on unseen data without a need for
        train/test split, hence providing value for all samples. This is needed for
        further spatial analysis of the model performance (and generalises to models
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
        The number of jobs to run in parallel. ``-1`` means using all processors by
        default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alongside
        the geographically weighted, by default True
    measure_performance : bool, optional
        Calculate performance metrics for the model, by default True. If True, measures
        R2 and adjusted R2.
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
        Number of models to process in each batch. Specify batch_size if your models do
        not fit into memory. By default None
    random_state : int | None, optional
        Random seed for reproducibility, by default None
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation

    Attributes
    ----------
    pred_ : pd.Series
        Focal predictions for each location.
    resid_ : pd.Series
        Residuals for each location (y - pred_).
    RSS_ : pd.Series
        Residual sum of squares for each location.
    TSS_ : pd.Series
        Total sum of squares for each location.
    y_bar_ : pd.Series
        Weighted mean of y for each location.
    local_r2_ : pd.Series
        Local R2 for each location.
    focal_r2_ : float
        Global R2 for focal predictions.
    score_ : float
        Alias for focal_r2_ (global R2 for focal predictions).
    focal_adj_r2_ : float
        Adjusted R2 for focal predictions.
    hat_values_ : pd.Series
        Hat values for each location (diagonal elements of hat matrix).
    effective_df_ : float
        Effective degrees of freedom (sum of hat values).
    log_likelihood_ : float
        Global log likelihood of the model.
    aic_ : float
        Akaike information criterion of the model.
    aicc_ : float
        Corrected Akaike information criterion to account for model
        complexity (smaller bandwidths).
    bic_ : float
        Bayesian information criterion.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRegressor":
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
        self._validate_geometry(self.geometry)

        weights = self.graph if self.graph is not None else self._build_weights()
        self._setup_model_storage()

        # fit the models
        training_output = self._fit_models_batch(X, y, weights)

        if self.keep_models:
            (
                self._names,
                focal_pred,
                y_bar,
                tss,
                hat_values,  # Add this
                self._score_data,
                models,
            ) = zip(*training_output, strict=False)
            self._local_models = pd.Series(models, index=self._names)
        else:
            (
                self._names,
                focal_pred,
                y_bar,
                tss,
                hat_values,  # Add this
                self._score_data,
            ) = zip(*training_output, strict=False)

        self.pred_ = pd.Series(focal_pred, index=self._names)
        self.resid_ = y - self.pred_
        resids_ = (
            weights.adjacency.values
            * self.resid_.loc[weights.adjacency.index.get_level_values(1)] ** 2
        )
        self.RSS_ = resids_.groupby(weights.adjacency.index.get_level_values(0)).sum()
        self.TSS_ = pd.Series(tss, index=self._names)
        self.y_bar_ = pd.Series(y_bar, index=self._names)
        self.local_r2_ = (self.TSS_ - self.RSS_) / self.TSS_
        self.focal_r2_ = 1 - (
            np.sum((self.pred_ - y) ** 2) / np.sum((y - y.mean()) ** 2)
        )
        self.score_ = self.focal_r2_

        if self.fit_global_model:
            self._fit_global_model(X, y)

        # Store hat values
        self.hat_values_ = pd.Series(hat_values, index=self._names)
        # Compute effective degrees of freedom (trace of hat matrix)
        self.effective_df_ = np.nansum(self.hat_values_)

        # adjusted R2
        n = len(self.pred_)
        if not np.isnan(self.focal_r2_) and not np.isnan(self.effective_df_):
            if n - self.effective_df_ - 1 > 0:
                self.focal_adj_r2_ = 1 - (
                    (1 - self.focal_r2_) * (n - 1) / (n - self.effective_df_ - 1)
                )
            else:
                self.focal_adj_r2_ = np.nan

        self.log_likelihood_ = self._compute_global_log_likelihood()
        self._compute_information_criteria()

        return self

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x: np.ndarray,
        model_kwargs: dict,
    ) -> tuple:
        local_model = model(**model_kwargs)

        X = data.drop(columns=["_y", "_weight"])
        y = data["_y"]

        local_model.fit(
            X=X,
            y=y,
            sample_weight=data["_weight"],
        )
        focal_x = pd.DataFrame(
            focal_x.reshape(1, -1),
            columns=X.columns,
            index=[name],
        )
        focal_pred = local_model.predict(focal_x).flatten()[0]

        y_bar = self._y_bar(y, data["_weight"])
        tss = self._tss(y, y_bar, data["_weight"])

        # Compute hat value for this location
        hat_value = self._compute_hat_value(X, data["_weight"], focal_x.values)

        output = [
            name,
            focal_pred,
            y_bar,
            tss,
            hat_value,  # Add hat value to output
            self._get_score_data(local_model, X, y),
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

    def _compute_global_log_likelihood(self) -> float:
        """
        Compute the global log likelihood for the entire GWR model

        Parameters:
        -----------
        y : pd.Series
            Original target values

        Returns:
        --------
        float : Global log likelihood value for the entire GWR model
        """
        residuals = self.resid_
        n = len(residuals)

        # Estimate sigma from the residuals
        sigma = np.sqrt(np.sum(residuals**2) / n)

        if sigma <= 0:
            return np.nan

        # Global log likelihood assuming Gaussian errors
        log_likelihood = (
            -n / 2 * np.log(2 * np.pi)
            - n * np.log(sigma)
            - np.sum(residuals**2) / (2 * sigma**2)
        )

        return log_likelihood

    def _y_bar(self, y, w_i):
        """weighted mean of y"""
        sum_yw = np.sum(y * w_i)
        return sum_yw / np.sum(w_i)

    def _tss(self, y, y_bar, w_i):
        """geographically weighted total sum of squares"""
        return np.sum(w_i * (y - y_bar) ** 2)
