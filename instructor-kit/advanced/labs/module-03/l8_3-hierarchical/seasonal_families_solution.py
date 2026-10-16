# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.3 A (SOLUTION): which products sell in the same season?

Lesson 8.3 grouped products by WHO buys them. The planning team asks WHEN: which of the 24 products in
weekly_demand.csv rise and fall together through the year, so that their stock can be planned together?
Describe each product by its average weekly sales in each calendar month, build the tree with average
linkage and correlation distance, read the linkage matrix and the cuts, and see what Euclidean distance
would have done instead.
"""
# %% Step 1: one row per product, one column per month (done for you)
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist

pd.set_option("display.width", 140)
weekly = pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["week_start"])
weekly["month"] = weekly["week_start"].dt.month
months = weekly.pivot_table(index=["product_name", "category"], columns="month", values="units_sold", aggfunc="mean")
category = months.index.get_level_values("category")
print(months.shape, "units per week, product averages from", round(months.mean(axis=1).min(), 1),
      "to", round(months.mean(axis=1).max(), 1))
X = months.to_numpy()

# %% Step 2: the tree, and its highest merges
Z_corr = linkage(X, method="average", metric="correlation")
print("merges:", len(Z_corr), "  five highest merge heights:", Z_corr[-5:, 2][::-1].round(3).tolist())

# %% Step 3: cut by height, then by number of families
for h in [0.15, 0.25, 0.5, 1.0]:
    cut = fcluster(Z_corr, t=h, criterion="distance")
    print(f"height {h:.2f}: {len(np.unique(cut))} families, sizes {np.bincount(cut)[1:].tolist()}")
families = fcluster(Z_corr, t=3, criterion="maxclust")
print(pd.crosstab(families, category).to_string())

# %% Step 4: what each family's year looks like
season = months.groupby(families).mean().round(1)
season.index = [f"family {f}" for f in season.index]
print(season.to_string())
print("family 2:", months.index.get_level_values("product_name")[families == 2].tolist())

# %% Step 5: the same tree with Euclidean distance
Z_euc = linkage(X, method="average", metric="euclidean")
euc = fcluster(Z_euc, t=3, criterion="maxclust")
print("Euclidean sizes:", np.bincount(euc)[1:].tolist())
print("Euclidean small families:", months.index.get_level_values("product_name")[euc != np.bincount(euc).argmax()].tolist())
d = pdist(X, metric="correlation")
print(f"cophenetic correlation (average linkage, correlation distance): {cophenet(Z_corr, d)[0]:.3f}")
