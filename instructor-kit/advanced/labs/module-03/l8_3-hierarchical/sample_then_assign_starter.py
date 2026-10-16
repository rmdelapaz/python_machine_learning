# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.3 B: hierarchical clustering when the data is too big for it.

Lesson 8.3 warned that agglomerative clustering needs every pairwise distance: 18 million for 6,000
customers, 5 billion for 100,000. Its practical recipe: build the tree on a random SAMPLE, then assign
everyone to the nearest cluster mean. Try it on the customers (Lesson 8.2's log + scale features) and
check it against clustering everyone: does a sample of 500 or 1,500 find the same five groups? Work through TODO 1 to 4; the file runs as it is.
"""
# %% Step 1: Lesson 8.2's features and the two full-data answers (done for you)
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
log_some = ColumnTransformer(
    [("log", FunctionTransformer(np.log, feature_names_out="one-to-one"), ["orders_12m", "avg_order_value"])],
    remainder="passthrough", verbose_feature_names_out=False,
)
Z = make_pipeline(log_some, StandardScaler()).set_output(transform="pandas").fit_transform(customers[behavior])
km5 = KMeans(n_clusters=5, n_init=10, random_state=42).fit(Z)
ward_all = AgglomerativeClustering(n_clusters=5, linkage="ward").fit(Z)
print(f"pairwise distances for all {len(Z):,} customers: {len(Z) * (len(Z) - 1) // 2:,}")
print(f"full Ward vs K-Means (ARI): {adjusted_rand_score(km5.labels_, ward_all.labels_):.3f}")

# %% Step 2: build the tree on a sample, assign everyone to the nearest mean
# TODO 1: for n in 500 and 1500: sample = Z.sample(n, random_state=42); Ward with 5 clusters on the sample;
#         means = sample.groupby(labels).mean(); labels for ALL customers with pairwise_distances_argmin(Z, means).
#         Print the number of pairwise distances the sample needed, the sorted sizes, and the ARI against
#         km5 and against ward_all. Keep the 500-sample means for Step 3.
print("TODO 1: Ward on a sample, then everyone to the nearest mean")

# %% Step 3: polish with K-Means started from the sample's means
# TODO 2: KMeans(n_clusters=5, init=<the 500-sample means as an array>, n_init=1, random_state=42).fit(Z).
#         Compare its inertia and labels (ARI) with km5.
print("TODO 2: K-Means started from the sample's means")

# %% Step 4: do the five groups still mean the same thing?
# TODO 3: profile the polished groups: the mean of each behavior column and the churn rate, sorted by churn.
#         Name them with Lesson 8.2's segment names.
# TODO 4 (think): for 100,000 customers, which steps of this recipe would still be fast, and which not?
print("TODO 3: profile of the polished groups")

print("\nExpected when done: 17,997,000 distances for everyone, full Ward vs K-Means 0.907; a 500-customer sample")
print("(124,750 distances) gives ARI 0.971 with K-Means and 0.910 with full Ward; 1,500 gives 0.957 and 0.927;")
print("K-Means started from the 500-sample means reaches inertia 5,525 and ARI 1.000; churn 0.04 to 0.31.")
