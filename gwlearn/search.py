from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn import metrics


class BandwidthSearch:
    """Optimal bandwidth search for geographically weighted estimators.

    Reports information criteria and (optionally) other scores from multiple models with
    varying bandwidth. When using golden section search, it minimizes (or maximizes)
    the chosen ``criterion``.

    When using classification models with a defined ``min_proportion``, keep in mind
    that some locations may be excluded from the final model. In such a case, the
    information criteria are typically not comparable across models with different
    bandwidths and shall not be used to determine the optimal one.

    Parameters
    ----------
    model : type
        A geographically weighted estimator class (e.g.
        :class:`gwlearn.linear_model.GWLogisticRegression`) that can be instantiated as
        ``model(bandwidth=..., geometry=..., fixed=..., kernel=..., n_jobs=..., ...)``
        and exposes information criteria attributes like ``aicc_``/``aic_``/``bic_``.
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default ``False``
    kernel : str | Callable, optional
        Type of kernel function used to weight observations, by default ``"bisquare"``
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors,
        by default ``-1``
    search_method : {"golden_section", "interval"}, optional
        Method used to search for optimal bandwidth. When using ``"golden_section"``,
        the Golden section optimization is used to find the optimal bandwidth while
        attempting to minimize or maximise ``criterion``. When using ``"interval"``,
        fits all models within the specified bandwidths at a set interval without any
        attempt to optimize the selection. By default ``"golden_section"``.
    criterion : str, optional
        Criterion used to select the optimal bandwidth.

        Supported values include
        ``{"aicc", "aic", "bic", "prediction_rate", "log_loss"}``.
        If you pass another string, it is interpreted as an attribute name ``m`` and
        retrieved from the fitted model as ``getattr(model, m + "_")``.
        By default ``"aicc"``.
    metrics : list[str] | None, optional
        Additional metrics to report for each bandwidth. Metrics follow the same
        conventions as ``criterion`` (including special cases ``"log_loss"`` and
        ``"prediction_rate"``). By default ``None``.
    minimize : bool, optional
        Minimize or maximize the ``criterion``. When using information criterions,
        like AICc, the optimal solution is the lowest value. When using other metrics,
        the optimal may be the highest value. By default True, assuming lower is
        better.
    min_bandwidth : int | float | None, optional
        Minimum bandwidth to consider, by default ``None``
    max_bandwidth : int | float | None, optional
        Maximum bandwidth to consider, by default ``None``
    interval : int | float | None, optional
        Interval for bandwidth search when using "interval" method, by default ``None``
    max_iterations : int, optional
        Maximum number of iterations for golden section search, by default ``100``
    tolerance : float, optional
        Tolerance for convergence in golden section search, by default ``1e-2``
    verbose : bool | int, optional
        Verbosity level, by default False
    **kwargs
        Additional keyword arguments passed to ``model`` initialization

    Attributes
    ----------
    scores_ : pd.Series
        Series of criterion scores for each bandwidth tested (index is bandwidth).
    metrics_ : pd.DataFrame
        DataFrame of additional metrics for each bandwidth tested.
    optimal_bandwidth_ : int | float
        The optimal bandwidth found by the search method.

    Examples
    --------
    Interval search over a small set of candidate bandwidths:

    >>> import geopandas as gpd
    >>> from geodatasets import get_path
    >>> from gwlearn.linear_model import GWLogisticRegression
    >>> from gwlearn.search import BandwidthSearch

    >>> gdf = gpd.read_file(get_path('geoda.guerry'))
    >>> X = gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']]
    >>> y = gdf["Region"] == 'E'

    >>> search = BandwidthSearch(
    ...     GWLogisticRegression,
    ...     fixed=False,
    ...     search_method="interval",
    ...     criterion="aicc",
    ...     min_bandwidth=20,
    ...     max_bandwidth=80,
    ...     interval=10,
    ...     max_iter=200,
    ... ).fit(X, y, geometry=gdf.representative_point())
    >>> search.optimal_bandwidth_
    np.int64(40)
    """

    def __init__(
        self,
        model,
        *,
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
        n_jobs: int = -1,
        search_method: Literal["golden_section", "interval"] = "golden_section",
        criterion: str = "aicc",
        metrics: list | None = None,
        minimize: bool = True,
        min_bandwidth: int | float | None = None,
        max_bandwidth: int | float | None = None,
        interval: int | float | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-2,
        verbose: bool | int = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.kernel = kernel
        self.fixed = fixed
        self._model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.search_method = search_method
        self.criterion = criterion
        self.minimize = minimize
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.interval = interval
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.metrics = metrics
        self.verbose = verbose

    def fit(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries
    ) -> "BandwidthSearch":
        """
        Fit the searcher by evaluating candidate bandwidths on the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used to evaluate candidate bandwidths (rows are samples).
        y : pd.Series
            Target values corresponding to X.
        geometry : gpd.GeoSeries
            Geographic location of the observations in the sample. Used to determine the
            spatial interaction weight based on specification by ``bandwidth``,
            ``fixed``, ``kernel``, and ``include_focal`` keywords.

        Returns
        -------
        self
            The fitted instance.

        Notes
        -----
        The optimal bandwidth is selected as the index of the minimum score if
        ``minimize=True``, otherwise as the index of the maximum score.
        """
        self.geometry = geometry

        if self.search_method == "interval":
            self._interval(X=X, y=y)
        elif self.search_method == "golden_section":
            self._golden_section(X=X, y=y, tolerance=self.tolerance)

        self.optimal_bandwidth_ = (
            self.scores_.idxmin() if self.minimize else self.scores_.idxmax()
        )

        return self

    def _score(
        self, X: pd.DataFrame, y: pd.Series, bw: int | float
    ) -> tuple[float, list[float]]:
        """Fit the model and report criterion score.

        In case of invariant y in a local model, returns np.inf
        """
        if len(np.unique(y)) == 1:
            return (np.inf, [])

        gwm = self.model(
            bandwidth=bw,
            fixed=self.fixed,
            kernel=self.kernel,
            n_jobs=self.n_jobs,
            fit_global_model=False,
            strict=False,
            verbose=self.verbose == 2,
            **self._model_kwargs,
        ).fit(X=X, y=y, geometry=self.geometry)

        met = ["aicc", "aic", "bic"]
        if self.metrics is not None:
            met += self.metrics

        if hasattr(gwm, "prediction_rate_") and gwm.prediction_rate_ == 0:
            # prediction rate should report 0, everything else is undefined
            score = (
                gwm.prediction_rate_ if self.criterion == "prediction_rate" else np.nan
            )

            all_metrics = [
                gwm.prediction_rate_ if m == "prediction_rate" else np.nan for m in met
            ]

            return score, all_metrics

        all_metrics = []
        for m in met:
            if m == "log_loss":
                mask = gwm.proba_.isna().any(axis=1)
                all_metrics.append(metrics.log_loss(y[~mask], gwm.proba_[~mask]))
            else:
                all_metrics.append(getattr(gwm, m + "_"))

        return all_metrics[met.index(self.criterion)], all_metrics

    def _interval(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit models using the equal interval search.

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        """
        if not (
            isinstance(self.min_bandwidth, float | int)
            and isinstance(self.max_bandwidth, float | int)
            and isinstance(self.interval, float | int)
        ):
            raise ValueError(
                "All 'min_bandwidth', 'max_bandwidth' and 'interval' need "
                "to be set when using interval search method."
            )

        scores = {}
        metrics = {}
        bw = self.min_bandwidth
        while bw <= self.max_bandwidth:
            score, metric = self._score(X=X, y=y, bw=bw)
            scores[bw] = score
            metrics[bw] = metric
            if self.verbose:
                print(f"Bandwidth: {bw:.2f}, {self.criterion}: {scores[bw]:.3f}")
            bw += self.interval
        self.scores_ = pd.Series(scores, name=self.criterion)
        self.metrics_ = pd.DataFrame(
            metrics,
            index=pd.Index(
                ["aicc", "aic", "bic"] + self.metrics
                if self.metrics
                else ["aicc", "aic", "bic"]
            ),
        ).T

    def _golden_section(self, X: pd.DataFrame, y: pd.Series, tolerance: float) -> None:
        delta = 0.38197
        if self.fixed:
            pairwise_distance = pdist(self.geometry.get_coordinates())
            min_dist = np.min(pairwise_distance)
            max_dist = np.max(pairwise_distance)

            a = min_dist / 2.0
            c = max_dist * 2.0
        else:
            a = 40 + 2 * X.shape[1]
            c = len(self.geometry)

        if self.min_bandwidth:
            a = self.min_bandwidth
        if self.max_bandwidth:
            c = self.max_bandwidth

        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)

        diff = 1.0e9
        iters = 0
        scores = {}
        metrics = {}
        while diff > tolerance and iters < self.max_iterations and a != np.inf:
            if not self.fixed:  # ensure we use int as possible bandwidth
                b = int(b)
                d = int(d)

            if b in scores:
                score_b = scores[b]
            else:
                score_b, metric_b = self._score(X=X, y=y, bw=b)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{b:.2f}'.rstrip('0').rstrip('.')}, "
                        f"score: {score_b:.3f}"
                    )
                scores[b] = score_b
                metrics[b] = metric_b

            if d in scores:
                score_d = scores[d]
            else:
                score_d, metric_d = self._score(X=X, y=y, bw=d)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{d:.2f}'.rstrip('0').rstrip('.')}, "
                        f"score: {score_d:.3f}"
                    )
                scores[d] = score_d
                metrics[d] = metric_d

            if self.minimize:
                if score_b <= score_d:
                    c = d
                    d = b
                    b = a + delta * np.abs(c - a)

                else:
                    a = b
                    b = d
                    d = c - delta * np.abs(c - a)

                diff = np.abs(score_b - score_d)

            else:
                if score_b >= score_d:
                    c = d
                    d = b
                    b = a + delta * np.abs(c - a)
                else:
                    a = b
                    b = d
                    d = c - delta * np.abs(c - a)

                diff = np.abs(score_b - score_d)

            iters += 1

        self.scores_ = pd.Series(scores)
        self.metrics_ = pd.DataFrame(
            metrics,
            index=pd.Index(
                ["aicc", "aic", "bic"] + self.metrics
                if self.metrics
                else ["aicc", "aic", "bic"]
            ),
        ).T
