# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 8.1 A (SOLUTION): what kinds of payments does Pinecrest take?

Lesson 8.1 grouped customers. The payments team asks the same question about 50,000 card payments,
using four columns that describe how a payment looks: amount, account age, items and card velocity.
Cluster them (raw first, then the right way), profile the groups in plain units, and check one thing
the clustering never saw: the fraud rate of each group.
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

# %% Step 2: K-Means on the raw columns


def profile(labels):
    t = tx.assign(cluster=labels)
    prof = t.groupby("cluster")[cols + ["is_fraud"]].mean()
    prof.insert(0, "payments", t["cluster"].value_counts().sort_index())
    return prof.sort_values("is_fraud")


raw = KMeans(n_clusters=4, n_init=10, random_state=42).fit(tx[cols])
print(profile(raw.labels_).round(3))
# Account age (std about 700 days) dwarfs everything else: the four groups are bands of account age.

# %% Step 3: log the skewed money and age columns, scale everything, cluster again
X = tx[cols].astype(float).copy()
X["amount"] = np.log(X["amount"])
X["account_age_days"] = np.log1p(X["account_age_days"])    # log1p: brand-new accounts have age 0
Z = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Z)
prof = profile(km.labels_)
print(prof.round(3))

# %% Step 4: is it a fluke of the start?
for seed in [0, 1, 2]:
    print(f"seed {seed}: inertia {KMeans(n_clusters=4, n_init=10, random_state=seed).fit(Z).inertia_:,.0f}")
print(f"seed 42: inertia {km.inertia_:,.0f}")

# %% Step 5: name the groups and read the fraud column
names = {}
for c, row in prof.iterrows():
    if row["account_age_days"] < 30:
        names[c] = "Brand-new accounts"
    elif row["velocity_24h"] > 1:
        names[c] = "Busy cards"
    elif row["items"] > 2.5:
        names[c] = "Big baskets"
    else:
        names[c] = "Small everyday orders"
summary = tx.assign(group=pd.Series(km.labels_).map(names)).groupby("group").agg(
    payments=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"), frauds=("is_fraud", "sum"))
print(summary.sort_values("fraud_rate").round(4))
