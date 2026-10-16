# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.1 B: four stores, one day at a time.

Lesson 9.1's project compressed product sales histories. Here the rows are days and the columns are
the four stores' sales: one unit, strongly correlated, PCA's home ground. How much of the story is
"the whole chain had a big day", and what is the second component? Work through TODO 1 to 4.
"""
# %% Step 1: one row per day, one column per store, log sales; closed days left out (done for you)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

pd.set_option("display.width", 120)
sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
wide = sales.pivot(index="date", columns="store", values="sales")
closed = (wide == 0).any(axis=1)
logged = np.log(wide[~closed])
print(f"{len(wide)} days, {closed.sum()} closed days left out:", logged.shape)

# %% Step 2: PCA without scaling (one unit)
# TODO 1: pca = PCA().fit(logged). Print the shares of variance and the loadings of PC1 and PC2 per store.
pca = None
if pca is None:
    print("TODO 1: shares of variance and loadings")

# %% Step 3: PC1 against the calendar
# TODO 2: scores = the first two columns of pca.transform(logged) (index = logged.index); attach each day's
#         event (sales.groupby("date")["event"].first(), missing = "none") and print the mean PC1 per event.
print("TODO 2: mean PC1 score per event")

# %% Step 4: PC2 against time
# TODO 3: the mean PC2 score per year, next to each store's mean daily sales per year. What is PC2 following?
print("TODO 3: PC2 by year, and the stores' sales by year")

# %% Step 5: rebuild from one component
# TODO 4: PCA(n_components=1), transform then inverse_transform; print the mean absolute gap in log units.
print("TODO 4: the gap left by PC1 alone")

print("\nExpected when done: 1,090 days; shares 0.841 0.067 0.051 0.042; PC1 weights all about 0.5 (the whole")
print("chain's day), PC2 Asheville 0.70 against Flagstaff -0.66; mean PC1 -0.05 on ordinary days and 1.16 to")
print("1.23 on the three sale events; PC2 falls from 0.117 (2023) to -0.130 (2025) while Flagstaff grows")
print("from $4,997 to $6,597 a day and Asheville stays flat; PC1 alone leaves a gap of 0.105 log units.")
