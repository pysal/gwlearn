import inspect
import warnings
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from libpysal import graph
from scipy.spatial import KDTree
from sklearn import metrics, utils

# TODO: summary
# TODO: formal documentation
# TODO: comments in code

__all__ = ["BaseClassifier", "BaseRegressor"]


def _triangular(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-((u / 2) ** 2)) / (np.sqrt(2 * np.pi))


def _bisquare(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (1 - u**2) ** 2


def _cosine(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = np.clip(distances / bandwidth, 0, 1)
    return (np.pi / 4) * np.cos(np.pi / 2 * u)


def _exponential(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    r = (distances < bandwidth).astype(int)
    return r


_kernel_functions = {
    "triangular": _triangular,
    "parabolic": _parabolic,
    "gaussian": _gaussian,
    "bisquare": _bisquare,
    "cosine": _cosine,
    "boxcar": _boxcar,
    "exponential": _exponential,
}


class _BaseModel:
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
            "gaussian",
            "bisquare",
            "cosine",
            "boxcar",
            "exponential",
        ]
        | Callable = "bisquare",
        include_focal: bool = False,
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
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
        self.fixed = fixed
        self.model_kwargs = kwargs
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
                self.model_kwargs,
            )
            for (name, group), focal_x in zip(grouper, X_focals, strict=False)
        )

    def _fit_global_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit global baseline model"""
        if self._model_type == "random_forest":
            self.model_kwargs["oob_score"] = True
        # fit global model as a baseline
        if "n_jobs" in inspect.signature(self.model).parameters:
            self.global_model = self.model(n_jobs=self.n_jobs, **self.model_kwargs)
        else:
            self.global_model = self.model(**self.model_kwargs)

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

    def __repr__(self) -> str:
        """Return a string representation of the instance"""
        # Get the class name
        class_name = self.__class__.__name__

        # Core parameters to display
        params = []

        # Add model type if available
        if class_name in ["BaseClassifier", "BaseRegressor"] and hasattr(self, "model"):
            if hasattr(self.model, "__name__"):
                params.append(f"model={self.model.__name__}")
            else:
                params.append(f"model={self.model}")

        # Add key parameters
        params.append(f"bandwidth={self.bandwidth}")

        if self.fixed:
            params.append("fixed=True")

        if self.kernel != "bisquare":
            if callable(self.kernel):
                params.append(f"kernel={self.kernel.__name__}")
            else:
                params.append(f"kernel='{self.kernel}'")

        if self.n_jobs != -1:
            params.append(f"n_jobs={self.n_jobs}")

        if not self.fit_global_model:
            params.append("fit_global_model=False")

        if not self.measure_performance:
            params.append("measure_performance=False")

        if self.keep_models:
            if isinstance(self.keep_models, Path):
                params.append(f"keep_models='{self.keep_models}'")
            else:
                params.append("keep_models=True")

        if self.batch_size is not None:
            params.append(f"batch_size={self.batch_size}")

        if self.verbose:
            params.append("verbose=True")

        # Add any additional model kwargs (limit to avoid overly long repr)
        if self.model_kwargs:
            # Show only a few key kwargs to keep repr readable
            important_kwargs = [
                "max_depth",
                "n_estimators",
                "C",
                "alpha",
                "learning_rate",
            ]
            shown_kwargs = {
                k: v for k, v in self.model_kwargs.items() if k in important_kwargs
            }
            if len(self.model_kwargs) <= 3:
                # Show all if there are only a few
                for k, v in self.model_kwargs.items():
                    if isinstance(v, str):
                        params.append(f"{k}='{v}'")
                    else:
                        params.append(f"{k}={v}")
            elif shown_kwargs:
                for k, v in shown_kwargs.items():
                    if isinstance(v, str):
                        params.append(f"{k}='{v}'")
                    else:
                        params.append(f"{k}={v}")

        # Join parameters with proper formatting
        param_str = ",\n".join(f"    {param}" for param in params)

        if len(params) > 3:  # Multi-line format for many parameters
            return f"{class_name}(\n{param_str}\n)"
        else:  # Single line for few parameters
            return f"{class_name}({', '.join(params)})"

    def _repr_html_(self):
        return utils.estimator_html_repr(self)

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


class BaseClassifier(_BaseModel):
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
        model,
        *,
        bandwidth: float,
        fixed: bool = False,
        kernel: Literal[
            "triangular",
            "parabolic",
            "gaussian",
            "bisquare",
            "cosine",
            "boxcar",
            "exponential",
        ]
        | Callable = "bisquare",
        include_focal: bool = False,
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

        weights = self._build_weights(geometry)
        self._setup_model_storage()

        self._global_classes = np.unique(y)

        # fit the models
        training_output = self._fit_models_batch(X, y, weights, geometry)

        if self.keep_models:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
                models,
            ) = zip(*training_output, strict=False)
            self.local_models = pd.Series(models, index=self._names)
            self._geometry = geometry
        else:
            (
                self._names,
                self._n_labels,
                self._score_data,
                self._feature_importances,
                focal_proba,
            ) = zip(*training_output, strict=False)

        self._n_labels = pd.Series(self._n_labels, index=self._names)
        self.focal_proba_ = pd.DataFrame(focal_proba, index=self._names)

        if self.fit_global_model:
            self._fit_global_model(X, y)

        if self.measure_performance:
            # support both bool and 0, 1 encoding of binary variable
            col = True if True in self.focal_proba_.columns else 1
            # global GW accuracy
            nan_mask = self.focal_proba_[col].isna()
            self.focal_pred_ = self.focal_proba_[col][~nan_mask] > 0.5
            masked_y = y[~nan_mask]
            self.score_ = metrics.accuracy_score(masked_y, self.focal_pred_)
            self.precision_ = metrics.precision_score(
                masked_y, self.focal_pred_, zero_division=0
            )
            self.recall_ = metrics.recall_score(
                masked_y, self.focal_pred_, zero_division=0
            )
            self.balanced_accuracy_ = metrics.balanced_accuracy_score(
                masked_y, self.focal_pred_
            )
            self.f1_macro_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="macro", zero_division=0
            )
            self.f1_micro_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="micro", zero_division=0
            )
            self.f1_weighted_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="weighted", zero_division=0
            )

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

        output = [
            name,
            n_labels,
            self._get_score_data(local_model, X, y),
            getattr(local_model, "feature_importances_", None),
            focal_proba,
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

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
            local_model = self.local_models[i]
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

    def __repr__(self) -> str:
        """Return a string representation of the BaseClassifier instance"""
        # Get the base representation
        base_repr = super().__repr__()

        # If we need to add classifier-specific params, we can modify the string here
        # For now, the classifier-specific params are handled in the constructor
        # and passed to the parent class
        if hasattr(self, "strict") and self.strict is not False:
            # Insert strict parameter if non-default
            base_repr = base_repr.replace(")", f", strict={self.strict})")

        if hasattr(self, "min_proportion") and self.min_proportion != 0.2:
            base_repr = base_repr.replace(
                ")", f", min_proportion={self.min_proportion})"
            )

        if hasattr(self, "undersample") and self.undersample:
            base_repr = base_repr.replace(")", ", undersample=True)")

        if hasattr(self, "random_state") and self.random_state is not None:
            base_repr = base_repr.replace(")", f", random_state={self.random_state})")

        return base_repr


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


class BaseRegressor(_BaseModel):
    """Generic geographically weighted regression meta-class
    TODO:
        - r2, adj_r2
        - degrees of freedom???
        - tvalues & adj_alpha & critical_t val
        - predict
        - performance measurements
    """

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "BaseRegressor":
        self._validate_geometry(geometry)

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
                self._score_data,
                models,
            ) = zip(*training_output, strict=False)
            self.local_models = pd.Series(models, index=self._names)
            self._geometry = geometry
        else:
            (
                self._names,
                focal_pred,
                y_bar,
                tss,
                self._score_data,
            ) = zip(*training_output, strict=False)

        self.focal_pred_ = pd.Series(focal_pred, index=self._names)
        self.resid_ = y - self.focal_pred_
        resids_ = (
            weights.adjacency.values
            * self.resid_.loc[weights.adjacency.index.get_level_values(1)] ** 2
        )
        self.RSS_ = resids_.groupby(weights.adjacency.index.get_level_values(0)).sum()
        self.TSS_ = pd.Series(tss, index=self._names)
        self.y_bar_ = pd.Series(y_bar, index=self._names)
        self.local_r2_ = (self.TSS_ - self.RSS_) / self.TSS_
        self.focal_r2_ = 1 - (
            np.sum((self.focal_pred_ - y) ** 2) / np.sum((y - y.mean()) ** 2)
        )
        self.score_ = self.focal_r2_

        if self.fit_global_model:
            self._fit_global_model(X, y)

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

        output = [
            name,
            focal_pred,
            y_bar,
            tss,
            self._get_score_data(local_model, X, y),
        ]

        if self.keep_models:
            output.append(self._store_model(local_model, name))
        else:
            del local_model

        return output

    def _y_bar(self, y, w_i):
        """weighted mean of y"""
        sum_yw = np.sum(y * w_i)
        return sum_yw / np.sum(w_i)

    def _tss(self, y, y_bar, w_i):
        """geographically weighted total sum of squares"""
        return np.sum(w_i * (y - y_bar) ** 2)
