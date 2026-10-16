# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.4 B (SOLUTION): two rings, and why the silhouette can vote for the wrong answer.

Lesson 8.4 used two half-moons. Here the shape is a ring inside a ring, which no method that cuts
space with straight lines can separate. You choose eps from a k-distance check, count clusters
and noise correctly, and then see which clustering the silhouette score prefers.
"""
# %% Step 1: two rings, scaled
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

X, y_true = make_circles(n_samples=600, factor=0.4, noise=0.06, random_state=42)
Z = StandardScaler().fit_transform(X)
print(Z.shape, "points per ring:", np.bincount(y_true))

# %% Step 2: K-Means with the right k
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Z)
print(f"K-Means k=2  ARI {adjusted_rand_score(y_true, km):.3f}")

# %% Step 3: a starting eps from the k-distances (min_samples=5)
from sklearn.neighbors import NearestNeighbors

dist, _ = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)
k_dist = np.sort(dist[:, -1])
print("k-distance percentiles (50, 90, 99):", np.percentile(k_dist, [50, 90, 99]).round(3))

# %% Step 4: a short sweep; count clusters WITHOUT the noise label
for eps in [0.1, 0.2, 0.3, 0.5, 0.8]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Z)
    n_clusters = len(set(labels) - {-1})
    print(f"eps={eps:<4} clusters={n_clusters:<3} noise={(labels == -1).sum():<4} "
          f"ARI={adjusted_rand_score(y_true, labels):.3f}")

# %% Step 5: which clustering does the silhouette prefer?
db = DBSCAN(eps=0.3, min_samples=5).fit_predict(Z)
clustered = db != -1
print(f"\nsilhouette, K-Means (wrong answer)  : {silhouette_score(Z, km):.3f}")
print(f"silhouette, DBSCAN (right answer)   : {silhouette_score(Z[clustered], db[clustered]):.3f}")
print("The silhouette rewards round, separated blobs; two rings around one center are neither.")
