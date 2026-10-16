# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 8.1: a segmentation script with four bugs (deliberately broken handout).

Two bugs stop the script with an error. Two do not: the script prints centers and a "big-ticket" segment
that are wrong. Fix them one at a time. When all four are fixed it uses the four behavior columns from
Lesson 8.1 and prints the five centers in real units (orders, dollars, shares), then
  big-ticket segment: 1073 customers, average order $336.92, churn 0.159
"""
# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate", "home_store", "email_opens_90d"]
X = customers[behavior]

scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled)

# %% The centers, to show the marketing team
centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print(centers.round(2))

# %% The big-ticket segment (it was cluster 1 when this script was written)
customers["cluster"] = kmeans.labels_
big = customers[customers["cluster"] == 1]
print(f"big-ticket segment: {len(big)} customers, average order ${big['avg_order_value'].mean():.2f}, "
      f"churn {big['churned'].mean():.3f}")
