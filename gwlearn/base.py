import inspect
import os
import warnings
from collections.abc import Callable, Hashable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from libpysal import graph
from sklearn import metrics

# TODO: testing
# TODO: formal documentation
# TODO: comments in code
# TODO: better handling of verbosity
# TODO: summary
# TODO: repr
# TODO: predict_proba for adaptive kernel
# TODO: better type hinting (Literal etc)
# TODO: catch non-binary cases
# TODO: UndefinedMetricWarning

__all__ = ["BaseClassifier"]


def _triangular(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return 1 - u


def _parabolic(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return 0.75 * (1 - u**2)


def _gaussian(distances, bandwidth):
    u = distances / bandwidth
    return np.exp(-((u / 2) ** 2)) / (np.sqrt(2) * np.pi)


def _bisquare(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return (15 / 16) * (1 - u**2) ** 2


def _cosine(distances, bandwidth):
    u = np.clip(distances / bandwidth, 0, 1)
    return (np.pi / 4) * np.cos(np.pi / 2 * u)


def _exponential(distances, bandwidth):
    u = distances / bandwidth
    return np.exp(-u)


def _boxcar(distances, bandwidth):
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


class BaseClassifier:
    """Generic geographically weighted modelling meta-class

    NOTE: local models leave out focal, unlike in traditional approaches. This allows
    assessment of geographically weighted metrics on unseen data without a need for
    train/test split, hence providing value for all samples. This is needed for
    futher spatial analysis of the model performance (and generalises to models
    that do not support OOB scoring).

    Parameters
    ----------
    model :  model class
        Scikit-learn model class
    bandwidth : int | float
        bandwidth value consisting of either a distance or N nearest neighbors
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str, optional
        type of kernel function used to weight observations, by default "bisquare"
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside
        the geographically weighted, by default True
    strict : bool, optional
        Do not fit any models if at least one neighborhood has invariant ``y``,
        by default False
    keep_models : bool | str | Path, optional
        Keep all local models (required for prediction), by default False. Note that
        for some models, like random forests, the objects can be large. If string or
        Path is provided, the local models are not held in memory but serialized to
        the disk from which they are loaded in prediction.
    temp_folder : str, optional
        Folder to be used by the pool for memmapping large arrays for sharing memory
        with worker processes, e.g., ``/tmp``. Passed to ``joblib.Parallel``.
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation
    """

    def __init__(
        self,
        model,
        bandwidth: int | float,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        n_jobs: int = -1,
        fit_global_model: bool = True,
        measure_performance: bool = True,
        strict: bool | str | Path = False,
        keep_models: bool = False,
        temp_folder: str | None = None,
        batch_size: int | None = None,
        min_proportion: float = 0.2,
        **kwargs,
    ):
        self.model = model
        self.bandwidth = bandwidth
        self.kernel = kernel
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
        self.min_proportion = min_proportion
        self._measure_oob = "oob_score" in inspect.signature(model).parameters
        if self._measure_oob:
            self.model_kwargs["oob_score"] = self._get_score_data

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries):
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
        if not (geometry.geom_type == "Point").all():
            raise ValueError(
                "Unsupported geometry type. Only point geometry is allowed."
            )

        # build graph
        if self.fixed:  # fixed distance
            weights = graph.Graph.build_kernel(
                geometry, kernel=self.kernel, bandwidth=self.bandwidth
            )
        else:  # adaptive KNN
            weights = graph.Graph.build_kernel(
                geometry, kernel="identity", k=self.bandwidth
            )
            # post-process identity weights by the selected kernel
            # and kernel bandwidth derived from each neighborhood
            bandwidth = weights._adjacency.groupby(level=0).transform("max")
            weights = graph.Graph(
                adjacency=_kernel_functions[self.kernel](weights._adjacency, bandwidth),
                is_sorted=True,
            )

        if isinstance(self.keep_models, Path):
            self.keep_models.mkdir(exist_ok=True)

        self._global_classes = np.unique(y)
        # fit the models
        data = X.copy()
        data["_y"] = y
        data = data.loc[weights._adjacency.index.get_level_values(1)]
        data["_weight"] = weights._adjacency.values
        grouper = data.groupby(weights._adjacency.index.get_level_values(0))

        invariant = (
            data["_y"]
            .groupby(weights._adjacency.index.get_level_values(0))
            .nunique()
            == 1
        )
        if invariant.any():
            if self.strict:
                raise ValueError(
                    f"y at locations {invariant.index[invariant]} is invariant."
                )
            else:
                warnings.warn(
                    f"y at locations {invariant.index[invariant]} is invariant.",
                    stacklevel=3,
                )

        if self.batch_size:
            training_output = []
            num_groups = len(list(grouper))
            indices = np.arange(num_groups)
            for i in range(0, num_groups, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch_grouper = [
                    item for j, item in enumerate(grouper) if j in batch_indices
                ]
                batch_X = X.values[batch_indices]

                print(
                    f"Processing batch {i // self.batch_size + 1} "
                    f"out of {(num_groups // self.batch_size) + 1}."
                )
                batch_training_output = Parallel(
                    n_jobs=self.n_jobs, temp_folder=self.temp_folder
                )(
                    delayed(self._fit_local)(
                        self.model,
                        group,
                        name,
                        focal_x,
                        self.model_kwargs,
                    )
                    for (name, group), focal_x in zip(
                        batch_grouper, batch_X, strict=False
                    )
                )
                training_output.extend(batch_training_output)
        else:
            training_output = Parallel(
                n_jobs=self.n_jobs, temp_folder=self.temp_folder
            )(
                delayed(self._fit_local)(
                    self.model,
                    group,
                    name,
                    focal_x,
                    self.model_kwargs,
                )
                for (name, group), focal_x in zip(grouper, X.values, strict=False)
            )

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
            if self._measure_oob:
                self.model_kwargs["oob_score"] = True
            # fit global model as a baseline
            if "n_jobs" in inspect.signature(self.model).parameters:
                self.model_kwargs["n_jobs"] = self.n_jobs
            self.global_model = self.model(**self.model_kwargs)
            self.global_model.fit(X=X, y=y)

        if self.measure_performance:
            # global GW accuracy
            nan_mask = self.focal_proba_[True].isna()
            self.focal_pred_ = self.focal_proba_[True][~nan_mask] > 0.5
            masked_y = y[~nan_mask]
            self.score_ = metrics.accuracy_score(masked_y, self.focal_pred_)
            self.precision_ = metrics.precision_score(masked_y, self.focal_pred_)
            self.recall_ = metrics.recall_score(masked_y, self.focal_pred_)
            self.balanced_accuracy_ = metrics.balanced_accuracy_score(
                masked_y, self.focal_pred_
            )
            self.f1_macro_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="macro"
            )
            self.f1_micro_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="micro"
            )
            self.f1_weighted_ = metrics.f1_score(
                masked_y, self.focal_pred_, average="weighted"
            )

        return self

    def _fit_local(
        self,
        model,
        data: pd.DataFrame,
        name: Hashable,
        focal_x,
        model_kwargs: dict,
    ) -> tuple:
        """Fit individual local model

        In case of an invariant y, model is not fitted and empty placeholder output
        is returned.

        Parameters
        ----------
        model : model class
            Scikit-learn model class
        data : pd.DataFrame
            data for training
        name : Hashable
            group name, matching the index of the focal geometry
        model_kwargs : dict
            additional keyword arguments for the model init

        Returns
        -------
        tuple
            name, fitted model
        """
        vc = data["_y"].value_counts()
        n_labels = len(vc)
        skip = n_labels == 1
        if n_labels > 1:
            skip = (vc.iloc[1] / vc.iloc[0]) < self.min_proportion
        if skip:
            if self._model_type in ["random_forest", "gradient_boosting"]:
                score_data = (np.array([]).reshape(-1, 1), np.array([]))
                feature_imp = np.array([np.nan] * (data.shape[1] - 2))
            elif self._model_type == "logistic":
                score_data = (
                    np.array([]),  # true
                    np.array([]),  # pred
                    pd.Series(
                        np.nan, index=data.columns.drop(["_y", "_weight"])
                    ),  # local coefficients
                    np.array([np.nan]),  # intercept
                )
                feature_imp = None
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
        focal_proba = pd.Series(
            local_model.predict_proba(focal_x).flatten(), index=local_model.classes_
        )

        local_proba = pd.DataFrame(
            local_model.predict_proba(X), columns=local_model.classes_
        )

        if self._model_type == "random_forest":
            score_data = local_model.oob_score_
        elif self._model_type == "logistic":
            score_data = (
                y,
                local_proba.idxmax(axis=1),
                pd.Series(
                    local_model.coef_.flatten(),
                    index=local_model.feature_names_in_,
                ),  # coefficients
                local_model.intercept_,  # intercept
            )
        else:
            score_data = np.nan

        output = [
            name,
            n_labels,
            score_data,
            getattr(local_model, "feature_importances_", None),
            focal_proba,
        ]

        if self.keep_models is True:  # if True, models are kept in memory
            output.append(local_model)
        elif isinstance(self.keep_models, Path):  # if Path, models are saved to disk
            p = f"{self.keep_models.joinpath(f'{name}.joblib')}"
            with open(p, "wb") as f:
                dump(local_model, f, protocol=5)
            output.append(p)

            del local_model
        else:
            del local_model

        return output

    def _get_score_data(self, true, pred):
        return true, pred

    def predict_proba(self, X: pd.DataFrame, geometry: gpd.GeoSeries) -> pd.DataFrame:
        """Predict probabiliies using the ensemble of local models

        For any given location, this uses the

        Parameters
        ----------
        X : pd.DataFrame
            _description_
        geometry : gpd.GeoSeries
            _description_

        Returns
        -------
        pd.DataFrame
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        if not (geometry.geom_type == "Point").all():
            raise ValueError(
                "Unsupported geometry type. Only point geometry is allowed."
            )
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
            raise NotImplementedError

        split_indices = np.where(np.diff(input_ids))[0] + 1
        local_model_ids = np.split(local_ids, split_indices)
        distances = np.split(distance.values, split_indices)
        data = np.split(X.to_numpy(), range(1, len(X)))

        probabilities = []
        for x_, models_, distances_ in zip(
            data, local_model_ids, distances, strict=True
        ):
            # there are likely ways of speeding this up using parallel processing
            # but I failed to do so efficiently. We are hitting GIL due to accessing
            # same local models many times so iterative loop is in the end faster
            probabilities.append(
                self._predict_proba(x_, models_, distances_, X.columns)
            )

        return pd.DataFrame(probabilities, columns=self._global_classes, index=X.index)

    def _predict_proba(self, x_, models_, distances_, columns):
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

    def predict(self, X, geometry):
        proba = self.predict_proba(X, geometry)

        return proba.idxmax(axis=1)


def _scores(y_true, y_pred):
    if y_true.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return (
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred),
        metrics.recall_score(y_true, y_pred),
        metrics.balanced_accuracy_score(y_true, y_pred),
        metrics.f1_score(y_true, y_pred, average="macro"),
        metrics.f1_score(y_true, y_pred, average="micro"),
        metrics.f1_score(y_true, y_pred, average="weighted"),
    )
