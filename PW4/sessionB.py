"""PW4 Session B -- Unsupervised learning: Wine data (PCA, K-Means)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================================================
# B1: Load and hide the labels
# ============================================================
print("=" * 60)
print("B1: Load Wine dataset")
print("=" * 60)

wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target  # Hidden until B6

# Print a short summary of DESCR
print(wine.DESCR[:1000], "\n... (see full DESCR in sklearn docs)")

print("\nX.describe():")
print(X.describe())

# ============================================================
# B2: PCA on the raw data
# ============================================================
print("\n" + "=" * 60)
print("B2: PCA on raw data")
print("=" * 60)

pca_raw = PCA()
pca_raw.fit(X)

print(f"Explained variance ratio (first 3 PCs): {pca_raw.explained_variance_ratio_[:3]}")

# Loadings of PC1
loadings_pc1 = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)
print("\nPC1 Loadings (sorted):")
print(loadings_pc1)

# ============================================================
# B3: Scale, then PCA again
# ============================================================
print("\n" + "=" * 60)
print("B3: Scale then PCA")
print("=" * 60)

# Scale
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# Verify scaling
print(f"Mean of scaled data: {Z.mean(axis=0).round(5)}")
print(f"Std of scaled data:  {Z.std(axis=0).round(5)}")

# PCA on scaled data
pca_scaled = PCA()
pca_scaled.fit(Z)

print("\nExplained variance ratio (first 5 PCs):")
for i, ratio in enumerate(pca_scaled.explained_variance_ratio_[:5], 1):
    print(f"  PC{i}: {ratio:.4f}")

print("\nCumulative variance:")
cumsum = np.cumsum(pca_scaled.explained_variance_ratio_)
for i, c in enumerate(cumsum, 1):
    print(f"  PC1-{i}: {c:.4f}")

# How many components for 90%?
n_90 = np.argmax(cumsum >= 0.90) + 1
print(f"\nComponents needed for 90% variance: {n_90}")

# Scree plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(pca_scaled.explained_variance_ratio_) + 1), 
       pca_scaled.explained_variance_ratio_, alpha=0.6, label="Individual")
ax.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', label="Cumulative")
ax.axhline(0.90, color="green", linestyle="--", label="90% threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("Scree Plot (scaled data)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scree_plot.png", dpi=150)
print("Saved scree_plot.png")

# Loadings PC1 and PC2
loadings = pd.DataFrame(
    pca_scaled.components_[:2].T,
    columns=["PC1", "PC2"],
    index=X.columns
)
print("\nLoadings PC1 & PC2 (sorted by PC1):")
print(loadings.sort_values("PC1", ascending=False))

# Scatter on PC1-PC2
scores = pca_scaled.transform(Z)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(scores[:, 0], scores[:, 1], s=20, alpha=0.6)
ax.set_xlabel(f"PC1 ({pca_scaled.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca_scaled.explained_variance_ratio_[1]:.1%})")
ax.set_title("Wines in PC1-PC2 plane (single colour)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("pca_scatter.png", dpi=150)
print("Saved pca_scatter.png")
# ============================================================
# B4: K-Means
# ============================================================
print("\n" + "=" * 60)
print("B4: K-Means clustering")
print("=" * 60)

# (a) Effect of n_init and random_state
print("\n(a) Inertia for 8 different random_states (k=3, n_init=1):")
inertias = []
for seed in range(8):
    km = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=seed)
    km.fit(Z)
    inertias.append(km.inertia_)
    print(f"  seed={seed}: inertia = {km.inertia_:.2f}")
print(f"  Spread: min={min(inertias):.2f}, max={max(inertias):.2f}")

# (b) Sweep k, compute inertia
print("\n(b) Inertia for k = 1..10 (n_init=10, random_state=0):")
k_values = range(1, 11)
inertias_k = []
for k in k_values:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(Z)
    inertias_k.append(km.inertia_)
    print(f"  k={k}: inertia = {km.inertia_:.2f}")

# Elbow plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_values), inertias_k, 'bo-')
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Inertia (within-cluster sum of squares)")
ax.set_title("Elbow method")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("elbow_plot.png", dpi=150)
print("Saved elbow_plot.png")
# ============================================================
# B5: What are the clusters?
# ============================================================
print("\n" + "=" * 60)
print("B5: Cluster interpretation")
print("=" * 60)

# Fit KMeans with k=3 (our chosen k) and n_init=10
km_final = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_final.fit(Z)
labels = km_final.labels_

# Cluster sizes
sizes = np.bincount(labels)
print(f"Cluster sizes: {sizes}")

# Centers in real units
centers_real = scaler.inverse_transform(km_final.cluster_centers_)
centers_df = pd.DataFrame(centers_real, columns=X.columns)
print("\nCluster centers (real units):")
print(centers_df.round(2))

# Which features differ most between clusters?
# Normalize by range to compare
centers_range = centers_df.max() - centers_df.min()
print("\nFeatures that differ most between clusters (range):")
print(centers_range.sort_values(ascending=False).head(5))

# Plot PC1-PC2 colored by cluster
scores = pca_scaled.transform(Z)
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(3):
    mask = labels == c
    ax.scatter(scores[mask, 0], scores[mask, 1], s=30, alpha=0.7, label=f"Cluster {c}")
ax.set_xlabel(f"PC1 ({pca_scaled.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca_scaled.explained_variance_ratio_[1]:.1%})")
ax.set_title("K-Means clusters in PC1-PC2 plane")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("kmeans_clusters.png", dpi=150)
print("Saved kmeans_clusters.png")
# ============================================================
# B6: The reveal
# ============================================================
print("\n" + "=" * 60)
print("B6: The reveal -- compare with true cultivars")
print("=" * 60)

# Crosstab: true cultivars vs our cluster labels
crosstab = pd.crosstab(cultivar, labels, rownames=["True Cultivar"], colnames=["Cluster"])
print("\nCrosstab (True vs Cluster):")
print(crosstab)

# Agreement score
# For each cluster, find the most common true cultivar, sum those counts, divide by 178
max_counts = crosstab.max(axis=0)
agreement = max_counts.sum() / len(cultivar)
print(f"\nAgreement score (scaled K-Means): {agreement:.4f} ({agreement:.2%})")

# Let's also test K-Means on RAW unscaled data (X)
km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_raw.fit(X)  # X = raw, unscaled
labels_raw = km_raw.labels_
crosstab_raw = pd.crosstab(cultivar, labels_raw)
agreement_raw = crosstab_raw.max(axis=0).sum() / len(cultivar)
print(f"\nAgreement score (raw, unscaled K-Means): {agreement_raw:.4f} ({agreement_raw:.2%})")

# And K-Means on PCA (first 2 PCs of scaled data)
scores_2d = pca_scaled.transform(Z)[:, :2]
km_pca = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_pca.fit(scores_2d)
labels_pca = km_pca.labels_
crosstab_pca = pd.crosstab(cultivar, labels_pca)
agreement_pca = crosstab_pca.max(axis=0).sum() / len(cultivar)
print(f"\nAgreement score (PCA 2D, scaled): {agreement_pca:.4f} ({agreement_pca:.2%})")

# Save agreement table for Snakemake
os.makedirs("results", exist_ok=True)
agreement_df = pd.DataFrame([
    {"method": "KMeans_scaled", "agreement": agreement},
    {"method": "KMeans_raw", "agreement": agreement_raw},
    {"method": "KMeans_PCA2D", "agreement": agreement_pca},
])
agreement_df.to_csv("results/B_agreement.csv", index=False)
print("\nSaved results/B_agreement.csv")