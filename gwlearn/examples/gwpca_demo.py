import numpy as np
from shapely.geometry import Point

from gwlearn.gwpca import GWPCA

# Create simple data
np.random.seed(0)
X = np.random.rand(30, 2)
geometry = [Point(x, y) for x, y in np.random.rand(30, 2)]

# Fit model
model = GWPCA(bandwidth=5, fixed=True)
model.fit(X, geometry)

# Print outputs
print("Loadings shape:", model.loadings_.shape)
print("Explained variance (first point):", model.explained_variance_[0])

# Transform
X_transformed = model.transform(X, geometry)
print("Transformed shape:", X_transformed.shape)

print("\n Basic GWPCA demo completed")
