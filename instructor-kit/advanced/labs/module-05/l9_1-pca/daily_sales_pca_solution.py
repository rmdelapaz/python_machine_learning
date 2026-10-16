# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.1 B (SOLUTION): four stores, one day at a time.

Lesson 9.1's project compressed product sales histories. Here the rows are days and the columns are
the four stores' sales: one unit, strongly correlated, PCA's home ground. How much of the story is
"the whole chain had a big day", and what is the second component?
"""
# %% Step 1: one row per day, one column per store, log sales; closed days left out
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pd.set_option("display.width", 120)
sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
wide = sales.pivot(index="date", columns="store", values="sales")
closed = (wide == 0).any(axis=1)
logged = np.log(wide[~closed])
print(f"{len(wide)} days, {closed.sum()} closed days left out:", logged.shape)

# %% Step 2: same unit, so no scaling; PCA centers the columns itself
pca = PCA().fit(logged)
print("share of variance:", pca.explained_variance_ratio_.round(3))
print(pd.DataFrame(pca.components_[:2].T, index=logged.columns, columns=["PC1", "PC2"]).round(2))

# %% Step 3: what does a high PC1 score mean?
scores = pd.DataFrame(pca.transform(logged)[:, :2], index=logged.index, columns=["PC1", "PC2"])
events = sales.groupby("date")["event"].first().reindex(scores.index).fillna("none")
print(scores.groupby(events)["PC1"].agg(["size", "mean"]).round(2).sort_values("mean"))

# %% Step 4: what does PC2 follow?
print(scores.groupby(scores.index.year)["PC2"].mean().round(3))
growth = wide[~closed].groupby(wide[~closed].index.year).mean().round(0)
print(growth)

# %% Step 5: rebuild from PC1 alone; how big is what is left?
one = PCA(n_components=1).fit(logged)
rebuilt = one.inverse_transform(one.transform(logged))
gap = np.abs(logged.to_numpy() - rebuilt).mean()
print(f"\nmean absolute gap from PC1 alone: {gap:.3f} log units (about {np.expm1(gap):.1%} of a day's sales)")
