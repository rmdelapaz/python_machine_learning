# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.3 B: do frauds form an island?

A colleague shows a t-SNE map of card payments with the frauds in red and says "look, fraud is its
own cluster". Check that claim with numbers, at two perplexities, against the original space. The
sample over-represents fraud (all 438 frauds and 1,062 honest payments). Work through TODO 1 to 3.
"""
# %% Step 1: a fraud-rich sample of 1,500 payments, scaled (done for you)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
sample = pd.concat([tx[tx["is_fraud"] == 1], tx[tx["is_fraud"] == 0].sample(1062, random_state=42)])
Z = StandardScaler().fit_transform(sample[numeric])
fraud = sample["is_fraud"].to_numpy()
print(Z.shape, f"fraud share in the sample {fraud.mean():.3f}")


def fraud_neighbor_share(points, k=10):
    """Among frauds: the share of their k nearest neighbors that are frauds too."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points, return_distance=False)[:, 1:]
    return fraud[idx][fraud == 1].mean()

# %% Step 2: the original space first
# TODO 1: print fraud_neighbor_share(Z). This is the number a map must reproduce, not beat.
print("TODO 1: fraud neighbors in the original space")

# %% Step 3: two t-SNE maps
from sklearn.manifold import TSNE
# TODO 2: for perplexity 5 and 30: TSNE(perplexity=..., random_state=42).fit_transform(Z); print neighbors kept
#         (10 nearest, original vs map, as in Lesson 9.3) and fraud_neighbor_share(emb).
print("TODO 2: two maps, two checks each")

# %% Step 4: outside the picture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# TODO 3: KMeans(n_clusters=2, n_init=10, random_state=42) on Z; silhouette; pd.crosstab(cluster, fraud).
#         Is the colleague right, partly right, or wrong? Write one sentence.
print("TODO 3: a clustering in the original space")

print("\nExpected when done: 0.614 in the original space; perplexity 5: neighbors kept 0.633, fraud share 0.630;")
print("perplexity 30: 0.696 and 0.631; K-Means silhouette 0.406, one cluster holds 239 frauds and 20 honest")
print("payments, the other 199 frauds sit among 1,042 honest ones.")
