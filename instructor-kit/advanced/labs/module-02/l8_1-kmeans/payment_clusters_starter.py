# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.1 A: what kinds of payments does Pinecrest take?

Lesson 8.1 grouped customers. The payments team asks the same question about 50,000 card payments,
using four columns that describe how a payment looks: amount, account age, items and card velocity.
Cluster them (raw first, then the right way), profile the groups in plain units, and check one thing
the clustering never saw: the fraud rate of each group. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1: four columns that describe a payment (done for you)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
tx = pd.read_csv(DATA / "transactions.csv")
cols = ["amount", "account_age_days", "items", "velocity_24h"]
print(tx[cols].describe().loc[["mean", "std", "min", "max"]].round(1))



def profile(labels):
    """Size, column means and fraud rate per cluster, sorted by fraud rate (given)."""
    t = tx.assign(cluster=labels)
    prof = t.groupby("cluster")[cols + ["is_fraud"]].mean()
    prof.insert(0, "payments", t["cluster"].value_counts().sort_index())
    return prof.sort_values("is_fraud")


# %% Step 2: K-Means on the raw columns
# TODO 1: KMeans(n_clusters=4, n_init=10, random_state=42) on tx[cols] as they are; print profile(labels_).
#         Which column decides the groups? (Compare the four columns across the clusters.)
print("TODO 1: K-Means on the raw columns, and its profile")

# %% Step 3: log the skewed money and age columns, scale everything, cluster again
# TODO 2: X = a float copy of tx[cols]; replace amount by np.log(amount) and account_age_days by
#         np.log1p(account_age_days) (why log1p?); StandardScaler().fit_transform(X); KMeans as above;
#         print the profile.
print("TODO 2: log + scale, then K-Means, and its profile")

# %% Step 4: is it a fluke of the start?
# TODO 3: refit with random_state 0, 1, 2 and print each inertia next to the seed-42 one.
print("TODO 3: inertia for three more seeds")

# %% Step 5: name the groups and read the fraud column
# TODO 4: name each cluster from its profile (brand-new accounts, busy cards, big baskets, small everyday
#         orders), then print payments, fraud rate and frauds per named group.
# TODO 5 (think): the clustering never saw is_fraud. Is "Busy cards" a fraud detector? What would you
#         tell the payments team about the 438 frauds and these four groups?
print("TODO 4: named groups with their fraud rate")

print("\nExpected when done: raw clusters are bands of account age (about 197, 797, 1,633 and 2,982 days, every")
print("other column the same); after log + scale: Small everyday orders 20,287 payments, fraud 0.27%; Big baskets")
print("14,316, 0.76%; Brand-new accounts 3,996, 0.78%; Busy cards 11,401, 2.13% (243 of the 438 frauds);")
print("inertia 97,067 for every seed.")
