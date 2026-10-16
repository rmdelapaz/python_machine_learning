# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.4 A: dense payment patterns, and the payments that fit none of them.

Lesson 8.4 clustered customers. Here the payments team asks a density question about card payments:
what do the crowded, ordinary kinds of payment look like, and which payments sit in no crowd at all?
The fraud label is NEVER a feature; you use it only at the end, to profile what density found.
Work through TODO 1 to 6; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: a sample of 5,000 payments, five features, logged where skewed, then scaled (done for you)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
tx = pd.read_csv(DATA / "transactions.csv")
pay = tx.sample(5000, random_state=42).reset_index(drop=True)
features = pd.DataFrame({
    "log_amount": np.log(pay["amount"]),
    "log_account_age": np.log1p(pay["account_age_days"]),
    "hour": pay["hour"],
    "velocity_24h": pay["velocity_24h"],
    "items": pay["items"],
})
Z = StandardScaler().fit_transform(features)
print(Z.shape, f"frauds in the sample: {pay['is_fraud'].sum()} ({pay['is_fraud'].mean():.2%})")

# %% Step 2: DBSCAN first
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

min_samples = 10
# TODO 1: the k-distances: NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z) and print the
#         50th, 80th and 95th percentiles of the LAST column (np.percentile(..., [50, 80, 95]).round(2)).
# TODO 2: for eps in [0.3, 0.5, 0.7, 1.0]: fit DBSCAN(eps=eps, min_samples=min_samples), then print the
#         number of clusters (leave out -1!), the size of the largest one and the noise share.
#         Is there one eps you would trust?
print("TODO 1-2: k-distance percentiles and the eps sweep")

# %% Step 3: HDBSCAN
from sklearn.cluster import HDBSCAN

# TODO 3: fit HDBSCAN(min_cluster_size=50, min_samples=10, copy=True) on Z, store the labels in
#         pay["cluster"], and print the number of clusters (without -1) and the noise share.
hdb = None
if hdb is None:
    print("TODO 3: HDBSCAN, number of clusters and noise share")

# %% Step 4: profile each cluster in original units
# TODO 4: pay.groupby("cluster").agg(...) with the count of payments, the MEDIAN amount and account age,
#         the mean hour, velocity_24h, items and is_fraud. Read the items and velocity columns: what do
#         most clusters have in common? (Lesson 8.4, Common Mistake 5.)
print("TODO 4: the profile table")

# %% Step 5: noise against everyone else
# TODO 5: the fraud rate in the noise and in the clustered payments, and how many of the sample's
#         frauds are in the noise.
print("TODO 5: fraud rate in the noise vs the clusters")

# %% Step 6: stability
# TODO 6: repeat Step 3 with min_cluster_size 25, 50, 100 and 200 (min_samples=10). Print clusters, noise
#         share and the fraud rate in the noise each time. Does the finding survive?
print("TODO 6: other min_cluster_size values")

print("\nExpected when done: k-distance percentiles [0.45 0.77 1.24]; no eps works (69.7% noise at 0.3,")
print("one cluster of 1,434 and 5.6% noise at 1.0); HDBSCAN finds 8 clusters and 8.8% noise; most clusters")
print("sit on one exact items count and velocity value; noise holds 24 of the 38 frauds (5.4% vs 0.31%")
print("in the clusters); 50 and 100 give the same answer, 25 and 200 still put 3.9% to 4.5% fraud in the noise.")
