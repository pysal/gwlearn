import numpy as np
from shapely.geometry import Point

from gwlearn.gwpca import GWPCA


def generate_dummy_data(n=20, features=3):
    np.random.seed(0)
    X = np.random.rand(n, features)
    geometry = [Point(x, y) for x, y in np.random.rand(n, 2)]
    return X, geometry


def test_fit_runs():
    X, geometry = generate_dummy_data()

    model = GWPCA(bandwidth=5, fixed=True)
    model.fit(X, geometry)

    assert model.loadings_ is not None
    assert model.explained_variance_ is not None


def test_output_shapes():
    X, geometry = generate_dummy_data()

    model = GWPCA(bandwidth=5, fixed=True)
    model.fit(X, geometry)

    n_samples, n_features = X.shape

    assert model.loadings_.shape == (n_samples, n_features, n_features)
    assert model.explained_variance_.shape == (n_samples, n_features)


def test_transform_runs():
    X, geometry = generate_dummy_data()

    model = GWPCA(bandwidth=5, fixed=True)
    model.fit(X, geometry)

    X_transformed = model.transform(X, geometry)

    assert X_transformed.shape == X.shape


def test_no_nan_in_outputs():
    X, geometry = generate_dummy_data()

    model = GWPCA(bandwidth=5, fixed=True)
    model.fit(X, geometry)

    assert not np.isnan(model.explained_variance_).all()
