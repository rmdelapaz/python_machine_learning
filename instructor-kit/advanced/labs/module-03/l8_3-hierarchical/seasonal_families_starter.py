# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.3 A: which products sell in the same season?

Lesson 8.3 grouped products by WHO buys them. The planning team asks WHEN: which of the 24 products in
weekly_demand.csv rise and fall together through the year, so that their stock can be planned together?
Describe each product by its average weekly sales in each calendar month, build the tree with average
linkage and correlation distance, read the linkage matrix and the cuts, and see what Euclidean distance
would have done instead. Work through TODO 1 to 5; the file runs as it is.
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
# TODO 1: Z_corr = linkage(X, method="average", metric="correlation"). Print the number of merges and the five
#         highest merge heights (the last rows of column 2, highest first). How many merges does 24 products give?
Z_corr = None
if Z_corr is None:
    print("TODO 1: build the tree and read its highest merges")

# %% Step 3: cut by height, then by number of families
# TODO 2: cut at heights 0.15, 0.25, 0.5 and 1.0 (fcluster(..., criterion="distance")) and print how many
#         families each gives and their sizes. Where is the big gap?
# TODO 3: cut into 3 families (criterion="maxclust") and crosstab them against `category`.
print("TODO 2-3: cuts by height and by number, and the families against the categories")

# %% Step 4: what each family's year looks like
# TODO 4: the average of `months` per family (one row per family, 12 months). Name each family's season.
#         Which two products make up the small family?
print("TODO 4: each family's average year")

# %% Step 5: the same tree with Euclidean distance
# TODO 5: linkage with metric="euclidean", cut into 3 families. Which products end up on their own, and
#         why? Finally print the cophenetic correlation of the correlation tree (cophenet(Z_corr, pdist(...))).
print("TODO 5: the Euclidean tree, and how faithful the correlation tree is")

print("\nExpected when done: 23 merges; highest heights [1.54, 0.292, 0.281, 0.215, 0.179]; a huge gap between")
print("0.29 and 1.54: two families (6 winter products: 4 Apparel + 2 Snow, and 18 others); in 3 families the")
print("two footwear products (Foothill Trail Runners, Talus Hiking Boots) peak in May; Euclidean: [22, 1, 1],")
print("the socks and the mug alone (the two best sellers); cophenetic correlation 0.979.")
