# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.3 B (SOLUTION): hierarchical clustering when the data is too big for it.

Lesson 8.3 warned that agglomerative clustering needs every pairwise distance: 18 million for 6,000
customers, 5 billion for 100,000. Its practical recipe: build the tree on a random SAMPLE, then assign
everyone to the nearest cluster mean. Try it on the customers (Lesson 8.2's log + scale features) and
check it against clustering everyone: does a sample of 500 or 1,500 find the same five groups?
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
results = {}
for n in [500, 1500]:
    sample = Z.sample(n, random_state=42)
    ward = AgglomerativeClustering(n_clusters=5, linkage="ward").fit(sample)
    means = sample.groupby(ward.labels_).mean()               # one row per cluster, in label order
    labels = pairwise_distances_argmin(Z, means)              # every customer to the nearest mean
    results[n] = means
    print(f"sample {n:5,d} ({n * (n - 1) // 2:>9,} distances): sizes {sorted(np.bincount(labels).tolist())}, "
          f"ARI vs K-Means {adjusted_rand_score(km5.labels_, labels):.3f}, vs full Ward {adjusted_rand_score(ward_all.labels_, labels):.3f}")

# %% Step 3: polish with K-Means started from the sample's means
polished = KMeans(n_clusters=5, init=results[500].to_numpy(), n_init=1, random_state=42).fit(Z)
print(f"K-Means from the 500-sample means: inertia {polished.inertia_:,.0f} (ten random starts: {km5.inertia_:,.0f}), "
      f"ARI {adjusted_rand_score(km5.labels_, polished.labels_):.3f}")

# %% Step 4: do the five groups still mean the same thing?
customers["group"] = polished.labels_
print(customers.groupby("group")[behavior + ["churned"]].mean().round(2).sort_values("churned").to_string())
