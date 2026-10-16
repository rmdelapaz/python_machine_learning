# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.4 A (SOLUTION): dense payment patterns, and the payments that fit none of them.

Lesson 8.4 clustered customers. Here the payments team asks a density question about card payments:
what do the crowded, ordinary kinds of payment look like, and which payments sit in no crowd at all?
The fraud label is NEVER a feature; it is used only at the end, to profile what density found.
"""
# %% Step 1: a sample of 5,000 payments, five features, logged where skewed, then scaled
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

# %% Step 2: DBSCAN first: k-distances and a short eps sweep
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

min_samples = 10                                   # about twice the number of features
distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
print("k-distance percentiles (50, 80, 95):", np.percentile(distances[:, -1], [50, 80, 95]).round(2))
for eps in [0.3, 0.5, 0.7, 1.0]:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    print(f"eps={eps}: {len(sizes):2d} clusters, largest {sizes.iloc[0]:5d}, noise {(labels == -1).mean():.1%}")

# %% Step 3: HDBSCAN, smallest pattern worth naming = 50 payments
from sklearn.cluster import HDBSCAN

hdb = HDBSCAN(min_cluster_size=50, min_samples=10, copy=True).fit(Z)
pay["cluster"] = hdb.labels_
n_clusters = len(set(hdb.labels_) - {-1})
print(f"\nHDBSCAN: {n_clusters} clusters, noise share {(hdb.labels_ == -1).mean():.1%}")

# %% Step 4: profile every cluster in original units; the fraud rate was never used for fitting
profile = pay.groupby("cluster").agg(
    payments=("transaction_id", "size"),
    amount=("amount", "median"),
    account_age=("account_age_days", "median"),
    hour=("hour", "mean"),
    velocity=("velocity_24h", "mean"),
    items=("items", "mean"),
    fraud=("is_fraud", "mean"),
).round({"amount": 2, "account_age": 0, "hour": 1, "velocity": 2, "items": 2, "fraud": 3})
print(profile.to_string())
print("Most clusters sit on whole-number shelves: one exact items count (1 to 4) and velocity 0, 1 or 2.")

# %% Step 5: noise against everyone else
noise = pay["cluster"] == -1
print(f"\nfraud rate: noise {pay.loc[noise, 'is_fraud'].mean():.1%}, "
      f"clustered {pay.loc[~noise, 'is_fraud'].mean():.2%}")
print(f"noise holds {noise.mean():.1%} of the payments and "
      f"{pay.loc[noise, 'is_fraud'].sum()} of the {pay['is_fraud'].sum()} frauds")

# %% Step 6: does the answer survive other reasonable settings?
for size in [25, 50, 100, 200]:
    labels = HDBSCAN(min_cluster_size=size, min_samples=10, copy=True).fit_predict(Z)
    in_noise = labels == -1
    print(f"min_cluster_size={size:<4} clusters={len(set(labels) - {-1}):2d}  noise={in_noise.mean():.1%}  "
          f"fraud in noise={pay.loc[in_noise, 'is_fraud'].mean():.1%}")
