# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.2 A: how many kinds of shoppers, by what they buy?

Lesson 8.2 chose k for HOW customers shop (orders, basket size, discounts, returns). Merchandising asks a
different question: what do they buy? Describe each of the 3,000 customers in interactions.csv by the
share of their purchases in each activity (camping, climbing, hiking, paddling, running, general), then
collect the four kinds of evidence for k: elbow, silhouette, stability and usefulness. Work through TODO 1 to 4; the file runs as it is.
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
# TODO 1: for k in 2..9 fit KMeans(n_clusters=k, n_init=10, random_state=42) on X, keep each fit in a dict
#         `fits`, and collect inertia and silhouette_score(X, labels). Print a table with the drop in inertia.
#         Why is no scaling needed here? (Hint: what does each row add up to?)
fits = None
if fits is None:
    print("TODO 1: inertia, drop and silhouette for k = 2 to 9")

# %% Step 3: stability under bootstrap resampling
# TODO 2: for k in 4..7: eight bootstrap samples (rng = np.random.default_rng(42), one rng for the whole loop),
#         refit with n_init=5, random_state=b, and compare refit.predict(X) with fits[k].labels_ using
#         adjusted_rand_score. Print the mean and the worst ARI per k.
print("TODO 2: bootstrap stability for k = 4 to 7")

# %% Step 4: usefulness: profile the chosen k
# TODO 3: choose k from the evidence. Profile it: the mean share per cluster, the cluster sizes, and each
#         cluster's main taste (largest share, leaving out "general").
# TODO 4 (think): repeat the profile for k = 6. Which group appears? Would merchandising rather have 5 or 6?
print("TODO 3: profile of the chosen clustering")

print("\nExpected when done: (3000, 6); the silhouette peaks at k = 5 (0.453; k = 6 0.429) and the inertia drops")
print("115 from 4 to 5 but only 51 from 5 to 6; stability: k = 4 worst ARI 0.641, k = 5 0.925, k = 6 0.950,")
print("k = 7 0.658; five tastes: hiking 1,009, camping 738, climbing 424, paddling 423, running 406; at k = 6 a")
print("'general gear' group of 484 (general share 0.56) splits off, mostly from the hiking group.")
