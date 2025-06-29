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
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.include_focal = include_focal
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

    def _validate_geometry(self, geometry: gpd.GeoSeries):
        """Validate that geometry contains only Point geometries"""
        if not (geometry.geom_type == "Point").all():
            raise ValueError(
                "Unsupported geometry type. Only point geometry is allowed."
            )

    def _build_weights(self, geometry: gpd.GeoSeries) -> graph.Graph:
        """Build spatial weights graph"""
        if self.fixed:  # fixed distance
            weights = graph.Graph.build_kernel(
                geometry,
                kernel=_kernel_functions[self.kernel],
                bandwidth=self.bandwidth,
            )
        else:  # adaptive KNN
            weights = graph.Graph.build_kernel(
                geometry,
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
        geometry: gpd.GeoSeries,
    ) -> list:
        """Fit models in batches or all at once"""
        if self.batch_size:
            training_output = []
            num_groups = len(geometry)
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
        n = len(self.pred_)

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


class BaseClassifier(_BaseModel, ClassifierMixin):
    """Generic geographically weighted classification meta-class

    Parameters
    ----------
    model :  model class
        Scikit-learn model class
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
    measure_performance : bool | str, optional
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
        Binary predictions for focal locations based on a local model trained around
        the location itself.
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
    """

    def __init__(
        self,
        model,
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
        measure_performance: bool = True,
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
            model=model,
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
            verbose=verbose,
            **kwargs,
        )
        self.min_proportion = min_proportion
        self.undersample = undersample
        self.random_state = random_state
        self._empty_score_data = None
        self._empty_feature_imp = None

        if undersample:
            try:
                from imblearn.under_sampling import RandomUnderSampler  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "imbalance-learn is required for undersampling."
                ) from err

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "BaseClassifier":
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
        self._start = time()

        def _is_binary(series: pd.Series) -> bool:
            """Check if a pandas Series encodes a binary variable (bool or 0/1)."""
            unique_values = set(series.unique())

            # Check for boolean type
            if series.dtype == bool or unique_values.issubset({True, False}):
                return True

            # Check for 0, 1 encoding
            return bool(unique_values.issubset({0, 1}))

        self._validate_geometry(geometry)

        if not _is_binary(y):
            raise ValueError("Only binary dependent variable is supported.")

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Building weights")

        if self.graph is not None:
            weights = self.graph
        else:
            weights = self._build_weights(geometry)
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Weights ready")
        self._setup_model_storage()

        self._global_classes = np.unique(y)

        # fit the models
        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Fitting the models")
        training_output = self._fit_models_batch(X, y, weights, geometry)

        if self.verbose:
            print(f"{(time() - self._start):.2f}s: Models fitted")

        if self.keep_models:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,  # Add hat values
                models,
            ) = zip(*training_output, strict=False)
            self._local_models = pd.Series(models, index=self._names)
            self._geometry = geometry
        else:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                hat_values,  # Add hat values
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

        if self.fit_global_model:
            if self.verbose:
                print(f"{(time() - self._start):.2f}s: Fitting global model")
            self._fit_global_model(X, y)

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
                print(f"{(time() - self._start):.2f}s: Measuring focal performance")
            masked_y = y[~nan_mask]

            if "accuracy" in metrics_to_measure:
                self.score_ = metrics.accuracy_score(masked_y, self.pred_)

            if "precision" in metrics_to_measure:
                self.precision_ = metrics.precision_score(
                    masked_y, self.pred_, zero_division=0
                )

            if "recall" in metrics_to_measure:
                self.recall_ = metrics.recall_score(
                    masked_y, self.pred_, zero_division=0
                )

            if "balanced_accuracy" in metrics_to_measure:
                self.balanced_accuracy_ = metrics.balanced_accuracy_score(
                    masked_y, self.pred_
                )

            if "f1_macro" in metrics_to_measure:
                self.f1_macro_ = metrics.f1_score(
                    masked_y, self.pred_, average="macro", zero_division=0
                )

            if "f1_micro" in metrics_to_measure:
                self.f1_micro_ = metrics.f1_score(
                    masked_y, self.pred_, average="micro", zero_division=0
                )

            if "f1_weighted" in metrics_to_measure:
                self.f1_weighted_ = metrics.f1_score(
                    masked_y, self.pred_, average="weighted", zero_division=0
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
        if skip:
            score_data = self._empty_score_data
            feature_imp = self._empty_feature_imp
            output = [
                name,
                n_labels,
                score_data,
                feature_imp,
                pd.Series(np.nan, index=self._global_classes),
                np.nan,
            ]
            if self.keep_models:
                output.append(None)
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

        output = [
            name,
            n_labels,
            self._get_score_data(local_model, X, y),
            getattr(local_model, "feature_importances_", None),
            focal_proba,
            hat_value,
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
            input_ids, local_ids = self._geometry.sindex.query(
                geometry, predicate="dwithin", distance=self.bandwidth
            )
            distance = _kernel_functions[self.kernel](
                self._geometry.iloc[local_ids].distance(
                    geometry.iloc[input_ids], align=False
                ),
                self.bandwidth,
            )
        else:
            training_coords = self._geometry.get_coordinates()
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


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    if y_true.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return (
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred, zero_division=0),
        metrics.recall_score(y_true, y_pred, zero_division=0),
        metrics.balanced_accuracy_score(y_true, y_pred),
        metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
        metrics.f1_score(y_true, y_pred, average="micro", zero_division=0),
        metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


class BaseRegressor(_BaseModel, RegressorMixin):
    """Generic geographically weighted regression meta-class

    TODO:
        - tvalues & adj_alpha & critical_t val
        - predict
        - performance measurements

    Parameters
    ----------
    model :  model class
        Scikit-learn model class
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
    random_state : int | None, optional
        Random seed for reproducibility, by default None
    verbose : bool, optional
        Whether to print progress information, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation
    """

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "BaseRegressor":
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
        self._validate_geometry(geometry)

        if self.graph is not None:
            weights = self.graph
        else:
            weights = self._build_weights(geometry)
        self._setup_model_storage()

        # fit the models
        training_output = self._fit_models_batch(X, y, weights, geometry)

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
            self._geometry = geometry
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
