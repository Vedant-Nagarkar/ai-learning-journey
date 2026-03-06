import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════
# K-MEANS FROM SCRATCH
# ═══════════════════════════════════════════════

class KMeansScratch:

    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k            = k
        self.max_iters    = max_iters
        self.random_state = random_state
        self.centroids    = None
        self.labels       = None
        self.inertia_     = None

    def fit(self, X):
        np.random.seed(self.random_state)

        # ── Step 1: Random init centroids ─────
        random_idx     = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idx].copy()

        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()

            # ── Step 2: Assign each point ─────
            self.labels = self._assign(X)

            # ── Step 3: Move centroids ────────
            for k in range(self.k):
                points = X[self.labels == k]
                if len(points) > 0:
                    self.centroids[k] = points.mean(axis=0)

            # ── Step 4: Check convergence ─────
            if np.allclose(old_centroids, self.centroids):
                print(f"  Converged at iteration {iteration+1}")
                break

        # ── Compute inertia ───────────────────
        self.inertia_ = sum(
            np.sum((X[self.labels == k] - self.centroids[k])**2)
            for k in range(self.k)
        )

    def _assign(self, X):
        # Distance from each point to each centroid
        distances = np.array([
            np.linalg.norm(X - c, axis=1)
            for c in self.centroids
        ])
        return np.argmin(distances, axis=0)

    def predict(self, X):
        return self._assign(X)


# ─── 1. LOAD DATA ───────────────────────────────
print("="*55)
print("  K-MEANS FROM SCRATCH")
print("="*55)

iris    = load_iris()
X       = iris.data
y_true  = iris.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 2. ELBOW METHOD ────────────────────────────
print("\nRunning Elbow Method...")
inertias = []
K_range  = range(1, 11)

for k in K_range:
    km = KMeansScratch(k=k)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  K={k}: inertia={km.inertia_:.2f}")

# ─── 3. TRAIN WITH K=3 ──────────────────────────
print("\nTraining K-Means with K=3...")
km = KMeansScratch(k=3, random_state=42)
km.fit(X_scaled)
labels = km.labels

print(f"Final Inertia : {km.inertia_:.4f}")

# ─── 4. COMPARE WITH SKLEARN ────────────────────
print("\n" + "="*55)
print("  COMPARISON WITH SKLEARN K-MEANS")
print("="*55)

sk_km = KMeans(n_clusters=3, random_state=42, n_init=10)
sk_km.fit(X_scaled)
print(f"Scratch Inertia : {km.inertia_:.4f}")
print(f"Sklearn Inertia : {sk_km.inertia_:.4f}")

# ═══════════════════════════════════════════════
# PCA FROM SCRATCH
# ═══════════════════════════════════════════════

print("\n" + "="*55)
print("  PCA FROM SCRATCH")
print("="*55)

class PCAScratch:

    def __init__(self, n_components=2):
        self.n_components   = n_components
        self.components     = None
        self.explained_var  = None
        self.mean           = None

    def fit_transform(self, X):
        # Step 1: Center data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Step 2: Covariance matrix
        cov = np.cov(X_centered.T)

        # Step 3: Eigenvectors & eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Step 4: Sort by eigenvalue descending
        idx             = np.argsort(eigenvalues)[::-1]
        eigenvalues     = eigenvalues[idx]
        eigenvectors    = eigenvectors[:, idx]

        # Step 5: Keep top n_components
        self.components    = eigenvectors[:, :self.n_components]
        total_var          = eigenvalues.sum()
        self.explained_var = eigenvalues[:self.n_components] / total_var

        # Step 6: Project data
        return X_centered @ self.components

# ─── 5. APPLY PCA ───────────────────────────────
pca_scratch = PCAScratch(n_components=2)
X_pca = pca_scratch.fit_transform(X_scaled)

print(f"Original shape  : {X_scaled.shape}")
print(f"Reduced shape   : {X_pca.shape}")
print(f"\nExplained Variance per Component:")
for i, var in enumerate(pca_scratch.explained_var):
    bar = "█" * int(var * 40)
    print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)  {bar}")
print(f"  Total: {pca_scratch.explained_var.sum()*100:.1f}% variance kept")

# ─── 6. COMPARE WITH SKLEARN PCA ────────────────
print("\n" + "="*55)
print("  COMPARISON WITH SKLEARN PCA")
print("="*55)

sk_pca   = PCA(n_components=2)
X_sk_pca = sk_pca.fit_transform(X_scaled)

print(f"Scratch PC1 variance: {pca_scratch.explained_var[0]:.4f}")
print(f"Sklearn PC1 variance: {sk_pca.explained_variance_ratio_[0]:.4f}")
print(f"Scratch PC2 variance: {pca_scratch.explained_var[1]:.4f}")
print(f"Sklearn PC2 variance: {sk_pca.explained_variance_ratio_[1]:.4f}")

# ─── 7. PLOTS ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='K=3 (elbow)')
axes[0].set_title("Elbow Method — Finding Best K")
axes[0].set_xlabel("Number of Clusters K")
axes[0].set_ylabel("Inertia")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: K-Means clusters
colors = ['red', 'blue', 'green']
for k in range(3):
    mask = labels == k
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[k], label=f'Cluster {k}', alpha=0.6)
axes[1].set_title("K-Means Clusters (PCA view)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: True labels
class_colors = ['red', 'blue', 'green']
for i, name in enumerate(iris.target_names):
    mask = y_true == i
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=class_colors[i], label=name, alpha=0.6)
axes[2].set_title("True Labels (PCA view)")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_pca.png")
plt.show()
print("\nkmeans_pca.png saved!")
