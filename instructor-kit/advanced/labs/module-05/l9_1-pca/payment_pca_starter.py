# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.1 A: principal components of card payments.

Lesson 9.1 ran PCA on the churn columns. Here the same moves meet the nine numeric payment features:
what does PCA find without scaling, how many components does it need with scaling, what do the first
two mean, and do they separate fraud (which PCA never sees)? Work through TODO 1 to 5.
"""
# %% Step 1: nine numeric features, the January to May payments (done for you)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)
tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
numeric = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
train = tx[tx["timestamp"] < "2025-06-01"]
X = train[numeric]
print(X.shape)

# %% Step 2: unscaled
# TODO 1: PCA().fit(X): print the first three explained_variance_ratio_ values and PC1's weights as a dict
#         (pd.Series(raw.components_[0], index=numeric).round(3).to_dict()). Which column IS PC1, and why?
print("TODO 1: unscaled PCA")

# %% Step 3: scaled
# TODO 2: StandardScaler().set_output(transform="pandas").fit_transform(X), then PCA().fit(...). Print the
#         shares, the cumulative shares, and PCA(n_components=0.90).fit(Xs).n_components_.
Xs = pca = None
if pca is None:
    print("TODO 2: scaled PCA and the 90% count")

# %% Step 4: loadings
# TODO 3: a DataFrame of pca.components_[:2].T with the feature names. Give PC1 and PC2 a name each.
print("TODO 3: loadings of PC1 and PC2")

# %% Step 5: fraud on the components
# TODO 4: the mean PC1 and PC2 score of frauds and of honest payments.
# TODO 5: the fraud rate among the 2% highest PC1 scores and among the 2% highest PC2 scores
#         (n_top = len(scores) // 50). Which direction of spread carries the fraud signal?
print("TODO 4-5: frauds on PC1 and PC2")

print("\nExpected when done: unscaled PC1 explains 0.95 and is account_age_days alone (weight 1.0); scaled")
print("shares 0.159 0.123 ... 0.064, eight components for 90%; PC1 is basket size (amount 0.70, items 0.70),")
print("PC2 is risk signals (card_attempts and velocity_24h 0.52, country_mismatch 0.39, device_new 0.35);")
print("frauds average PC2 3.10 against -0.03; fraud rate 0.084 in the top 2% of PC1 and 0.206 in the top 2%")
print("of PC2 (base rate 0.009).")
