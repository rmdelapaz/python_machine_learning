# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.2 A (SOLUTION): how many kinds of shoppers, by what they buy?

Lesson 8.2 chose k for HOW customers shop (orders, basket size, discounts, returns). Merchandising asks a
different question: what do they buy? Describe each of the 3,000 customers in interactions.csv by the
share of their purchases in each activity (camping, climbing, hiking, paddling, running, general), then
collect the four kinds of evidence for k: elbow, silhouette, stability and usefulness.
"""
# %% Step 1: one row per customer, one column per activity (done for you)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

pd.set_option("display.width", 120)
interactions = pd.read_csv(DATA / "interactions.csv")
products = pd.read_csv(DATA / "products_ml.csv")
bought = interactions.merge(products[["product_id", "primary_activity"]], on="product_id")
share = pd.crosstab(bought["customer_id"], bought["primary_activity"], values=bought["purchases"],
                    aggfunc="sum").fillna(0)
share = share.div(share.sum(axis=1), axis=0)          # each row adds up to 1: shares, already on one scale
print(share.shape, "average shares:", share.mean().round(2).to_dict())
X = share.to_numpy()

# %% Step 2: elbow and silhouette for k = 2 to 9
fits, rows = {}, []
for k in range(2, 10):
    fits[k] = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"k": k, "inertia": fits[k].inertia_, "silhouette": silhouette_score(X, fits[k].labels_)})
scores = pd.DataFrame(rows).set_index("k")
scores["drop"] = -scores["inertia"].diff()
print(scores.round({"inertia": 1, "silhouette": 3, "drop": 1}))

# %% Step 3: stability under bootstrap resampling
rng = np.random.default_rng(42)
for k in [4, 5, 6, 7]:
    aris = []
    for b in range(8):
        rows_b = rng.choice(len(X), size=len(X), replace=True)
        refit = KMeans(n_clusters=k, n_init=5, random_state=b).fit(X[rows_b])
        aris.append(adjusted_rand_score(fits[k].labels_, refit.predict(X)))
    print(f"k={k}: mean ARI {np.mean(aris):.3f}   worst {np.min(aris):.3f}")

# %% Step 4: usefulness: profile the chosen k
best = fits[5]
profile = share.groupby(best.labels_).mean().round(2)
profile["customers"] = np.bincount(best.labels_)
profile["main_taste"] = share.drop(columns="general").groupby(best.labels_).mean().idxmax(axis=1)
print(profile.sort_values("customers", ascending=False))

# %% Step 5: the runner-up, k = 6
six = share.groupby(fits[6].labels_).mean().round(2)
six["customers"] = np.bincount(fits[6].labels_)
print(six.sort_values("customers", ascending=False))
