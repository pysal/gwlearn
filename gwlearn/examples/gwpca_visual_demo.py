import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.decomposition import PCA
from gwlearn.gwpca import GWPCA

# 1. Create grid
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
gx, gy = np.meshgrid(x, y)
coords = np.vstack([gx.ravel(), gy.ravel()]).T
geometry = [Point(p[0], p[1]) for p in coords]

# 2. Generate data (West vs East behavior)
n = len(geometry)
X1 = np.random.normal(0, 1, n)
X2 = np.zeros(n)

for i, p in enumerate(geometry):
    if p.x < 5:
        X2[i] = X1[i] + np.random.normal(0, 0.1)
    else:
        X2[i] = -X1[i] + np.random.normal(0, 0.1)

X = np.column_stack([X1, X2])

# 3. Global PCA
global_pca = PCA(n_components=1).fit(X)
global_evr = global_pca.explained_variance_ratio_[0]

# 4. GWPCA
model = GWPCA(bandwidth=30, fixed=False).fit(X, geometry)
local_evr = model.explained_variance_[:, 0] / np.sum(model.explained_variance_, axis=1)

# 5. Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(coords[:, 0], coords[:, 1], c=[global_evr]*n, cmap='viridis', vmin=0.5, vmax=1.0)
ax1.set_title(f"Standard Global PCA\nExplained Variance: {global_evr:.2f}")

im = ax2.scatter(coords[:, 0], coords[:, 1], c=local_evr, cmap='viridis', vmin=0.5, vmax=1.0)
plt.colorbar(im, ax=ax2, label='Local Variance Explained')
ax2.set_title(f"GWPCA\nMean Local Explained Variance: {np.mean(local_evr):.2f}")

plt.show()

print(f"Improvement: {((np.mean(local_evr) - global_evr) / global_evr)*100:.1f}%")

# 6. Loadings visualization
pc1_loadings_var2 = model.loadings_[:, 0, 1]

plt.figure(figsize=(10, 8))
im = plt.scatter(coords[:, 0], coords[:, 1],
                 c=pc1_loadings_var2,
                 cmap='coolwarm',
                 s=70,
                 edgecolor='white',
                 linewidth=0.1)

plt.colorbar(im, label='PC1 Loading for Variable 2')
plt.title("GWPCA Proof: Spatial Non-Stationarity")
plt.xlabel("West <---> East")
plt.ylabel("North <---> South")

plt.axvline(x=5, color='black', linestyle='--', alpha=0.5)

plt.text(2, 9, "Variable 2 follows Var 1", fontweight='bold')
plt.text(6, 9, "Variable 2 opposes Var 1", fontweight='bold')

plt.show()