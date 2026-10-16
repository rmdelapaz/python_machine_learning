# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.1 B: scaling is a choice about what "similar" means.

The retention team wants four groups of customers built from three columns they care about: days since
the last order, 12-month spend and support tickets. Cluster them three ways (raw, standardized, and with
recency counted twice), read each result in plain units, and compare the churn rate of the groups,
which none of the clusterings saw. Work through TODO 1 to 4; the file runs as it is.
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
# TODO 1: KMeans(n_clusters=4, n_init=10, random_state=42) on X as it is; print profile(labels_).
#         Which column do the four groups differ on most? Why that one?
print("TODO 1: K-Means on the raw columns")

# %% Step 3: standardized columns, and the centers back in real units
# TODO 2: scaler = StandardScaler().set_output(transform="pandas"); Z = scaler.fit_transform(X); K-Means
#         on Z; print the profile, then the centers in real units with scaler.inverse_transform(...).
#         Which new group appears that the raw version could not find?
print("TODO 2: K-Means on standardized columns, centers in real units")

# %% Step 4: a business choice: recency counts twice
# TODO 3: W = Z.copy(); W["days_since_last_order"] *= 2; K-Means on W; print the profile.
print("TODO 3: K-Means with recency weighted twice")

# %% Step 5: how different are the raw and standardized groupings?
# TODO 4: pd.crosstab(raw labels, standardized labels). Are they the same groups with other numbers
#         (one big count per row), or different groups?
print("TODO 4: crosstab of the raw and standardized clusterings")

print("\nExpected when done: raw groups are spend bands ($236 to $1,692) with churn 4% to 25%; standardized")
print("finds 780 silent customers (271 days, churn 51%) and 1,201 who complain (2.5 tickets, churn 22%);")
print("recency x2 gives a 539-customer group at 307 days with churn 53%; the crosstab spreads raw groups")
print("over several standardized ones: different groupings, not renamed ones.")
