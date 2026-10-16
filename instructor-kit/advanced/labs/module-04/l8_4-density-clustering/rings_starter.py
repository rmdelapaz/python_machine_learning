# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.4 B: two rings, and why the silhouette can vote for the wrong answer.

Lesson 8.4 used two half-moons. Here the shape is a ring inside a ring, which no method that cuts
space with straight lines can separate. Choose eps from a k-distance check, count clusters and noise
correctly, then see which clustering the silhouette score prefers. Work through TODO 1 to 4.
"""
# %% Step 1: two rings, scaled (done for you)
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
# TODO 1: KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Z); print its ARI against y_true.
km = None
print("TODO 1: K-Means and its ARI. Predict first: can two centers separate a ring from the ring inside it?")

# %% Step 3: k-distances
from sklearn.neighbors import NearestNeighbors

# TODO 2: distances to the 5th neighbor (self included); print the 50th, 90th and 99th percentiles.
print("TODO 2: k-distance percentiles")

# %% Step 4: the sweep
# TODO 3: for eps in [0.1, 0.2, 0.3, 0.5, 0.8] print clusters (without -1), noise count and ARI.
print("TODO 3: the eps sweep")

# %% Step 5: the silhouette's vote
# TODO 4: silhouette of K-Means on all points; silhouette of DBSCAN(eps=0.3, min_samples=5) on the
#         clustered points only (mask = labels != -1). Which one does the silhouette prefer, and is it right?
print("TODO 4: two silhouettes")

print("\nExpected when done: K-Means ARI -0.002; k-distance percentiles [0.118 0.186 0.251]; eps 0.1 gives 18")
print("clusters and 307 noise, 0.3 and 0.5 give ARI 1.000, 0.8 one cluster; silhouette 0.328 for the wrong")
print("K-Means answer against 0.158 for the right DBSCAN one.")
