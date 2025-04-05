from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn import metrics


class BandwidthSearch:
    """Optimal bandwidth search for geographically-weighted models

    Minimises one of AIC, AICc, BIC based on prediction probability on focal geometries.

    Parameters
    ----------
    model :  model class
        Scikit-learn model class
    fixed : bool, optional
        True for distance based bandwidth and False for adaptive (nearest neighbor)
        bandwidth, by default False
    kernel : str, optional
        type of kernel function used to weight observations, by default "bisquare"
    n_jobs : int, optional
        The number of jobs to run in parallel. ``-1`` means using all processors
        by default ``-1``
    fit_global_model : bool, optional
        Determines if the global baseline model shall be fitted alognside the g
        eographically weighted.
    **kwargs
        Additional keyword arguments passed to ``model`` initialisation
    """

    def __init__(
        self,
        model,
        fixed: bool = False,
        kernel: str | Callable = "bisquare",
        n_jobs: int = -1,
        search_method: str = "golden_section",
        criterion: str = "aic",
        min_bandwidth: int | float | None = None,
        max_bandwidth: int | float | None = None,
        interval: int | float | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-2,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.kernel = kernel
        self.fixed = fixed
        self.model_kwargs = kwargs
        self.n_jobs = n_jobs
        self.search_method = search_method
        self.criterion = criterion
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.interval = interval
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries) -> None:
        if self.search_method == "interval":
            self._interval(X=X, y=y, geometry=geometry)
        elif self.search_method == "golden_section":
            self._golden_section(X=X, y=y, geometry=geometry, tolerance=self.tolerance)

        self.optimal_bandwidth = self.scores.idxmin()

        return self

    def _score(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries, bw: int | float
    ) -> float:
        """Fit the model ans report criterion score.

        In case of invariant y in a local model, returns np.inf
        """
        if len(np.unique(y)) == 1:
            return np.inf

        gwm = self.model(
            bandwidth=bw,
            fixed=self.fixed,
            kernel=self.kernel,
            n_jobs=self.n_jobs,
            fit_global_model=False,
            measure_performance=False,
            strict=False,
            **self.model_kwargs,
        ).fit(X=X, y=y, geometry=geometry)
        mask = (gwm._n_labels < 2) | np.isnan(gwm.focal_proba_).any(axis=1)
        log_likelihood = -metrics.log_loss(y[~mask], gwm.focal_proba_[~mask])
        n, k = X[~mask].shape
        print(log_likelihood, n, k)

        match self.criterion:
            case "aic":
                return self._aic(k, n, log_likelihood)
            case "bic":
                return self._bic(k, n, log_likelihood)
            case "aicc":
                return self._aicc(k, n, log_likelihood)

    def _interval(self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries) -> None:
        """Fit models using the equal interval search.

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        geometry : gpd.GeoSeries
            Geographic location
        """
        scores = {}
        bw = self.min_bandwidth
        while bw <= self.max_bandwidth:
            scores[bw] = self._score(X=X, y=y, geometry=geometry, bw=bw)
            if self.verbose:
                print(f"Bandwidth: {bw:.2f}, score: {scores[bw]:.3f}")
            bw += self.interval
        self.scores_ = pd.Series(scores, name=self.criterion)

    def _golden_section(
        self, X: pd.DataFrame, y: pd.Series, geometry: gpd.GeoSeries, tolerance: float
    ) -> None:
        delta = 0.38197
        if self.fixed:
            pairwise_distance = pdist(geometry.get_coordinates())
            min_dist = np.min(pairwise_distance)
            max_dist = np.max(pairwise_distance)

            a = min_dist / 2.0
            c = max_dist * 2.0
        else:
            a = 40 + 2 * X.shape[1]
            c = len(geometry)

        if self.min_bandwidth:
            a = self.min_bandwidth
        if self.max_bandwidth:
            c = self.max_bandwidth

        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)

        diff = 1.0e9
        iters = 0
        scores = {}
        while diff > tolerance and iters < self.max_iterations and a != np.inf:
            if not self.fixed:  # ensure we use int as possible bandwidth
                b = int(b)
                d = int(d)

            if b in scores:
                score_b = scores[b]
            else:
                if self.verbose:
                    print(f"Fitting bandwidth: {f'{b:.2f}'.rstrip('0').rstrip('.')}")
                score_b = self._score(X=X, y=y, geometry=geometry, bw=b)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{b:.2f}'.rstrip('0').rstrip('.')}, "
                        f"Score: {score_b:.3f}"
                    )
                scores[b] = score_b

            if d in scores:
                score_d = scores[d]
            else:
                if self.verbose:
                    print(f"Fitting bandwidth: {f'{d:.2f}'.rstrip('0').rstrip('.')}")
                score_d = self._score(X=X, y=y, geometry=geometry, bw=d)
                if self.verbose:
                    print(
                        f"Bandwidth: {f'{d:.2f}'.rstrip('0').rstrip('.')}, "
                        f"score: {score_d:.3f}"
                    )
                scores[d] = score_d

            if score_b <= score_d:
                c = d
                d = b
                b = a + delta * np.abs(c - a)

            else:
                a = b
                b = d
                d = c - delta * np.abs(c - a)

            diff = np.abs(score_b - score_d)

        self.scores = pd.Series(scores, name="oob_score")

    def _aic(self, k, _, log_likelihood):
        return 2 * k - 2 * log_likelihood

    def _bic(self, k, n, log_likelihood):
        return -2 * log_likelihood + k * np.log(n)

    def _aicc(self, k, n, log_likelihood):
        return self._aic(k, n, log_likelihood) + 2 * k * (k + 1) / (n - k - 1)
