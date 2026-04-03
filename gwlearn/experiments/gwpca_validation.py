import libpysal
import numpy as np
from esda.moran import Moran
from shapely.geometry import Point
from sklearn.decomposition import PCA

from gwlearn.gwpca import GWPCA

"""1. Create grid"""
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
gx, gy = np.meshgrid(x, y)
coords = np.vstack([gx.ravel(), gy.ravel()]).T
geometry = [Point(p[0], p[1]) for p in coords]

"""2. Generate data"""
n = len(geometry)
X1 = np.random.normal(0, 1, n)
X2 = np.zeros(n)

for i, p in enumerate(geometry):
    if p.x < 5:
        X2[i] = X1[i]
    else:
        X2[i] = -X1[i]

X = np.column_stack([X1, X2])

"""3. Fit models"""
global_pca = PCA(n_components=1).fit(X)
model = GWPCA(bandwidth=30, fixed=False).fit(X, geometry)

"""4. Residuals"""
X_centered = X - np.mean(X, axis=0)

global_recon = (X_centered @ global_pca.components_.T) @ global_pca.components_
global_res = np.linalg.norm(X_centered - global_recon, axis=1)

gwpca_res = []
for i in range(n):
    local_centered = X[i] - model.local_means_[i]
    local_recon = (local_centered @ model.loadings_[i].T) @ model.loadings_[i]
    gwpca_res.append(np.linalg.norm(local_centered - local_recon))

"""5. Moran's I """
w = libpysal.weights.DistanceBand.from_array(coords, threshold=1.5)
w.transform = "R"

mi_global = Moran(global_res, w)
mi_gwpca = Moran(np.array(gwpca_res), w)

print(f"\nMoran's I (Global Residuals): {mi_global.I:.4f} (p={mi_global.p_sim:.4f})")
print(f"Moran's I (GWPCA Residuals):  {mi_gwpca.I:.4f} (p={mi_gwpca.p_sim:.4f})")

"""6. Monte Carlo"""
n_iterations = 50
mc_global = []
mc_gwpca = []

print(f"\nRunning Monte Carlo ({n_iterations} iterations)...")

for _ in range(n_iterations):
    X1_sim = np.random.normal(0, 1, n)
    X2_sim = np.zeros(n)

    for i, p in enumerate(geometry):
        X2_sim[i] = (X1_sim[i] if p.x < 5 else -X1_sim[i]) + np.random.normal(0, 0.2)

    X_sim = np.column_stack([X1_sim, X2_sim])

    g_pca = PCA(n_components=1).fit(X_sim)
    gw_pca = GWPCA(bandwidth=30, fixed=False).fit(X_sim, geometry)

    mc_global.append(g_pca.explained_variance_ratio_[0])
    mc_gwpca.append(
        np.mean(
            gw_pca.explained_variance_[:, 0]
            / np.sum(gw_pca.explained_variance_, axis=1)
        )
    )

print("\nMonte Carlo Results:")
print(f"Average Global EVR: {np.mean(mc_global):.2f}")
print(f"Average GWPCA EVR:  {np.mean(mc_gwpca):.2f}")
