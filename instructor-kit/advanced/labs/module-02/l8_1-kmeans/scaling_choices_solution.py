# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.1 B (SOLUTION): scaling is a choice about what "similar" means.

The retention team wants four groups of customers built from three columns they care about: days since
the last order, 12-month spend and support tickets. Cluster them three ways (raw, standardized, and with
recency counted twice), read each result in plain units, and compare the churn rate of the groups,
which none of the clusterings saw.
"""
# %% Step 1: three columns on very different scales (done for you)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
customers = pd.read_csv(DATA / "customers_churn.csv")
cols = ["days_since_last_order", "spend_12m", "support_tickets_12m"]
X = customers[cols]
print("standard deviations:", X.std().round(1).to_dict())


def profile(labels):
    """Size, means in ORIGINAL units and churn rate per cluster, safest first (given)."""
    g = customers.assign(cluster=labels).groupby("cluster")
    out = g.agg(customers=("churned", "size"), days=("days_since_last_order", "mean"),
                spend=("spend_12m", "mean"), tickets=("support_tickets_12m", "mean"), churn=("churned", "mean"))
    return out.sort_values("churn").round(2)


# %% Step 2: raw columns
raw = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
print(profile(raw.labels_))

# %% Step 3: standardized columns, and the centers back in real units
scaler = StandardScaler().set_output(transform="pandas")
Z = scaler.fit_transform(X)
std = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Z)
print(profile(std.labels_))
print("centers in real units:")
print(pd.DataFrame(scaler.inverse_transform(std.cluster_centers_), columns=cols).round(1))

# %% Step 4: a business choice: recency counts twice
W = Z.copy()
W["days_since_last_order"] *= 2
weighted = KMeans(n_clusters=4, n_init=10, random_state=42).fit(W)
print(profile(weighted.labels_))

# %% Step 5: how different are the raw and standardized groupings?
print(pd.crosstab(raw.labels_, std.labels_, rownames=["raw"], colnames=["standardized"]))
