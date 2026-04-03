import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin


class GWPCA(BaseEstimator, TransformerMixin):
    """
    Geographically Weighted Principal Component Analysis (GWPCA).
    """

    def __init__(self, bandwidth, kernel="bisquare", fixed=True):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fixed = fixed
        self.loadings_ = None
        self.explained_variance_ = None
        self.coords_fit_ = None
        self.local_means_ = None

    def _get_weights(self, dists, local_bw):
        """Calculates spatial weights based on the kernel function."""
        if self.kernel == "bisquare":
            weights = (1 - (dists / local_bw) ** 2) ** 2
            weights[dists > local_bw] = 0
        else:
            raise ValueError(f"Kernel {self.kernel} not supported yet.")
        return weights

    def fit(self, X, geometry):
        """
        Fits the GWPCA model by computing local loadings for each location.
        """
        X_data = X.values if hasattr(X, "values") else np.array(X)
        n_samples, n_features = X_data.shape
        """--- Global Standardization (important for PCA) ---
        """
        self.global_mean_ = X_data.mean(axis=0)
        self.global_std_ = X_data.std(axis=0)
        """Avoid division by zero"""
        self.global_std_[self.global_std_ == 0] = 1

        X_data = (X_data - self.global_mean_) / self.global_std_
        """Initialize storage"""
        self.loadings_ = np.zeros((n_samples, n_features, n_features))
        self.explained_variance_ = np.zeros((n_samples, n_features))

        """Extract coordinates and compute distance matrix"""
        self.coords_fit_ = np.array([(p.x, p.y) for p in geometry])
        self.local_means_ = np.zeros((n_samples, n_features))
        self.loadings_ = np.zeros((n_samples, n_features, n_features))
        self.explained_variance_ = np.zeros((n_samples, n_features))

        tree = cKDTree(self.coords_fit_)
        for i in range(n_samples):
            dists_all, idx_all = tree.query(self.coords_fit_[i], k=n_samples)
            """---NEW ADPTIVE LOGIC-----"""
            if self.fixed:
                bw_i = self.bandwidth
            else:
                """ Use the distance to the K-th neighbor as the bandwidth for this point
                 We sort distances and pick the one at index 'bandwidth'"""
                k = int(self.bandwidth)
                bw_i = np.sort(dists_all)[k - 1]

            neighbors = tree.query_ball_point(self.coords_fit_[i], bw_i)
            local_coords = self.coords_fit_[neighbors]
            local_X = X_data[neighbors]
            dists = np.linalg.norm(local_coords - self.coords_fit_[i], axis=1)
            """Check for numerical stability: Do we have enough neighbors?"""
            wi = self._get_weights(dists, bw_i)
            if bw_i == 0 or np.sum(wi > 0) < 2:
                self.loadings_[i] = np.nan
                self.explained_variance_[i] = np.nan
                continue

            """Local Weighted Mean centering"""
            local_mean = (wi @ local_X) / np.sum(wi)
            self.local_means_[i] = local_mean
            X_centered = local_X - local_mean

            """Weighted Covariance Matrix: (X.T @ W @ X) / sum(W)
              Optimization: Instead of np.diag (slow), we use element-wise multiplication"""
            W_sqrt = np.sqrt(wi)[:, np.newaxis]
            X_weighted = X_centered * W_sqrt
            local_cov = (X_weighted.T @ X_weighted) / np.sum(wi)

            """Singular Value Decomposition"""
            try:
                # vh contains the components (loadings)
                _, s, vh = np.linalg.svd(local_cov)
                self.loadings_[i] = vh
                self.explained_variance_[i] = s
            except np.linalg.LinAlgError:
                self.loadings_[i] = np.nan
                self.explained_variance_[i] = np.nan

        return self

    def transform(self, X, geometry):
        if self.loadings_ is None:
            raise ValueError("Model must be fitted before transform.")

        X_data = (np.array(X) - self.global_mean_) / self.global_std_
        coords_new = np.array([(p.x, p.y) for p in geometry])

        """We map new points to the nearest local model we fitted"""
        tree = cKDTree(self.coords_fit_)  # Save coords in fit()
        _, idx = tree.query(coords_new)

        X_transformed = []
        for i, model_idx in enumerate(idx):
            """use the local mean and loading of the closest training point"""
            centered = X_data[i] - self.local_means_[model_idx]
            projected = centered @ self.loadings_[model_idx].T
            X_transformed.append(projected)
        return np.array(X_transformed)

    def score_bandwidth(self, bandwidth, X, geometry):
        """
        Calculates the Cross-Validation score(LOO) for a given bandwidth.
        Higher scores(or lower error) help pick the best bandwidth
        """
        X_data = np.array(X)
        n_samples = X_data.shape[0]
        coords = np.array([(p.x, p.y) for p in geometry])
        tree = cKDTree(coords)
        errors = []
        for i in range(n_samples):
            dists, idx = tree.query(coords[i], k=n_samples)
            # Find local bw by excluding point i itself
            if self.fixed:
                bw_i = bandwidth
            else:
                # k-th neighbor (excluding itself, which is dist 0)
                bw_i = np.sort(dists)[int(bandwidth)]

            wi = self._get_weights(dists, bw_i)
            wi[i] = 0  # Exclude the point itself for LOO,by setting own weight to zero

            if np.sum(wi) == 0:
                continue

            # Predict local mean from neighbors
            pred_mean = np.average(X_data, weights=wi, axis=0)
            # Prediction error for this point
            error = np.sum((X_data[i] - pred_mean) ** 2)
            errors.append(error)
        return np.mean(errors) if errors else np.inf
