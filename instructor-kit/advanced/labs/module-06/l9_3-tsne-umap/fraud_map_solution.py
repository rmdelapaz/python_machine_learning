# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.3 B (SOLUTION): do frauds form an island?

A colleague shows a t-SNE map of card payments with the frauds in red and says "look, fraud is its
own cluster". Check that claim with numbers, at two perplexities, against the original space. The
sample over-represents fraud (all 438 frauds and 1,062 honest payments) so that frauds are visible.
"""
# %% Step 1: a fraud-rich sample of 1,500 payments, scaled
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
sample = pd.concat([tx[tx["is_fraud"] == 1], tx[tx["is_fraud"] == 0].sample(1062, random_state=42)])
Z = StandardScaler().fit_transform(sample[numeric])
fraud = sample["is_fraud"].to_numpy()
print(Z.shape, f"fraud share in the sample {fraud.mean():.3f} (in all payments 0.009)")

# %% Step 2: how often is a fraud's nearest neighbor a fraud, in the ORIGINAL space?
from sklearn.neighbors import NearestNeighbors


def fraud_neighbor_share(points, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points, return_distance=False)[:, 1:]
    return fraud[idx][fraud == 1].mean()


print(f"original space: {fraud_neighbor_share(Z):.3f} of a fraud's 10 nearest neighbors are frauds")

# %% Step 3: two t-SNE maps
from sklearn.manifold import TSNE

for perp in [5, 30]:
    emb = TSNE(perplexity=perp, random_state=42).fit_transform(Z)
    nn_o = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    nn_e = NearestNeighbors(n_neighbors=11).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    kept = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_o, nn_e)])
    print(f"perplexity {perp:2d}: neighbors kept {kept:.3f}   "
          f"fraud share among a fraud's map neighbors {fraud_neighbor_share(emb):.3f}")

# %% Step 4: does "an island" survive outside the picture?
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Z)
print(f"\nK-Means k=2 in the original space: silhouette {silhouette_score(Z, km):.3f}")
print(pd.crosstab(km, fraud, rownames=["cluster"], colnames=["is_fraud"]))
