import sys
import os
import libpysal
from esda.moran import Moran
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.decomposition import PCA
from gwlearn.decomposition import GWPCA
'''Add the parent directory to the path so it can find  gwlearn'''
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

'''1. Create a 20x20 Grid of Points'''
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
gx, gy = np.meshgrid(x, y)
coords = np.vstack([gx.ravel(), gy.ravel()]).T
geometry = [Point(p[0], p[1]) for p in coords]

'''2. Generate Data: West has Positive correlation, East has Negative Correlation'''
n = len(geometry)
X1 = np.random.normal(0, 1, n)
X2 = np.zeros(n)
for i, p in enumerate(geometry):
    if p.x < 5: # West
        X2[i] = X1[i] + np.random.normal(0, 0.1)
    else:       # East
        X2[i] = -X1[i] + np.random.normal(0, 0.1)
X = np.column_stack([X1, X2])

'''3. Fit Global PCA (The Control)'''
global_pca = PCA(n_components=1).fit(X)
global_evr = global_pca.explained_variance_ratio_[0]

'''4. Fit Your GWPCA (The Experiment) - Using 30 neighbors'''
model = GWPCA(bandwidth=30, fixed=False).fit(X, geometry)
local_evr = model.explained_variance_[:, 0] / np.sum(model.explained_variance_, axis=1)

'''5. Visualize the Proof'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

'''Plot Global PCA (The Fail)'''
ax1.scatter(coords[:, 0], coords[:, 1], c=[global_evr]*n, cmap='viridis', vmin=0.5, vmax=1.0)
ax1.set_title(f"Standard Global PCA\nExplained Variance: {global_evr:.2f}")

'''Plot GWPCA (The Success)'''
im = ax2.scatter(coords[:, 0], coords[:, 1], c=local_evr, cmap='viridis', vmin=0.5, vmax=1.0)
plt.colorbar(im, ax=ax2, label='Local Variance Explained')
ax2.set_title(f"Your GWPCA\nMean Local Explained Variance: {np.mean(local_evr):.2f}")

plt.show()

print(f"Improvement: {((np.mean(local_evr) - global_evr) / global_evr)*100:.1f}%")
'''1. Extract the Loadings for the first Principal Component (PC1)
 model.loadings_ shape is (n_samples, n_components, n_features)
 We want: All samples [ : ], PC1 [ 0 ], and the 2nd Variable [ 1 ]'''
pc1_loadings_var2 = model.loadings_[:, 0, 1]

''' 2. Create the Plot'''
plt.figure(figsize=(10, 8))
'''We use 'coolwarm' cmap because it clearly separates Positive (Red) from Negative (Blue)'''
im = plt.scatter(coords[:, 0], coords[:, 1], c=pc1_loadings_var2, 
                 cmap='coolwarm', s=70, edgecolor='white', linewidth=0.1)

plt.colorbar(im, label='PC1 Loading for Variable 2')
plt.title("GWPCA Proof: Spatial Non-Stationarity\n(Red: Positive Correlation | Blue: Negative Correlation)")
plt.xlabel("West <---> East")
plt.ylabel("North <---> South")

''' Draw a dashed line where we programmed the data to flip (x=5)'''
plt.axvline(x=5, color='black', linestyle='--', alpha=0.5)
plt.text(2, 9, "Variable 2 follows Var 1", fontweight='bold')
plt.text(6, 9, "Variable 2 opposes Var 1", fontweight='bold')

plt.show()

''' --- MORAN'S I TEST --- 
1. Calculate Residuals for both models'''
X_centered = X - np.mean(X, axis=0)
''' Global Reconstruction error'''
global_recon = (X_centered @ global_pca.components_.T) @ global_pca.components_
global_res = np.linalg.norm(X_centered - global_recon, axis=1)

'''GWPCA Reconstruction error'''
gwpca_res = []
for i in range(n):
    local_centered = X[i] - model.local_means_[i]
    local_recon = (local_centered @ model.loadings_[i].T) @ model.loadings_[i]
    gwpca_res.append(np.linalg.norm(local_centered - local_recon))

'''2. Create Spatial Weights (using the grid coordinates)'''
w = libpysal.weights.DistanceBand.from_array(coords, threshold=1.5)
w.transform = 'R'

'''3. Compute Moran's I'''
mi_global = Moran(global_res, w)
mi_gwpca = Moran(np.array(gwpca_res), w)

print(f"\nMoran's I (Global Residuals): {mi_global.I:.4f} (p={mi_global.p_sim:.4f})")
print(f"Moran's I (GWPCA Residuals):  {mi_gwpca.I:.4f} (p={mi_gwpca.p_sim:.4f})")

''' --- MONTE CARLO SIMULATION ---'''
n_iterations = 50
mc_global = []
mc_gwpca = []

print(f"\nRunning Monte Carlo ({n_iterations} iterations)...")
for _ in range(n_iterations):
    ''' Re-generate noise but keep the West/East structure'''
    X1_sim = np.random.normal(0, 1, n)
    X2_sim = np.zeros(n)
    for i, p in enumerate(geometry):
        X2_sim[i] = (X1_sim[i] if p.x < 5 else -X1_sim[i]) + np.random.normal(0, 0.2)
    X_sim = np.column_stack([X1_sim, X2_sim])
    
    # Fit both
    g_pca = PCA(n_components=1).fit(X_sim)
    gw_pca = GWPCA(bandwidth=30, fixed=False).fit(X_sim, geometry)
    
    mc_global.append(g_pca.explained_variance_ratio_[0])
    mc_gwpca.append(np.mean(gw_pca.explained_variance_[:, 0] / np.sum(gw_pca.explained_variance_, axis=1)))

print(f"Monte Carlo Results:")
print(f"Average Global EVR: {np.mean(mc_global):.2f}")
print(f"Average GWPCA EVR:  {np.mean(mc_gwpca):.2f}")