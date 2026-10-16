# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 9.1 A (SOLUTION): principal components of card payments.

Lesson 9.1 ran PCA on the churn columns. Here the same moves meet the nine numeric payment features:
what does PCA find without scaling, how many components does it need with scaling, what do the first
two mean, and do they separate fraud (which PCA never sees)?
"""
# %% Step 1: nine numeric features, the January to May payments
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

# %% Step 2: unscaled PCA, then read PC1's weights
raw = PCA().fit(X)
print("unscaled, share of variance (first 3):", raw.explained_variance_ratio_[:3].round(3))
print(pd.Series(raw.components_[0], index=numeric).round(3).to_dict())

# %% Step 3: scaled PCA and the scree numbers
scaler = StandardScaler().set_output(transform="pandas")
Xs = scaler.fit_transform(X)
pca = PCA().fit(Xs)
print("\nscaled, share of variance:", pca.explained_variance_ratio_.round(3))
print("cumulative:               ", pca.explained_variance_ratio_.cumsum().round(3))
print("components kept for 90%:", PCA(n_components=0.90).fit(Xs).n_components_)

# %% Step 4: read the first two components
loadings = pd.DataFrame(pca.components_[:2].T, index=numeric, columns=["PC1", "PC2"])
print(loadings.round(2))

# %% Step 5: do frauds sit somewhere special on PC1 and PC2?
scores = pd.DataFrame(pca.transform(Xs)[:, :2], columns=["PC1", "PC2"], index=Xs.index)
scores["is_fraud"] = train["is_fraud"]
print(scores.groupby("is_fraud")[["PC1", "PC2"]].mean().round(2))
n_top = len(scores) // 50                                           # the top 2% of the training payments
for pc in ["PC1", "PC2"]:
    top = scores[pc].nlargest(n_top).index
    print(f"fraud rate among the {n_top} highest {pc} scores: {scores.loc[top, 'is_fraud'].mean():.3f} "
          f"(base rate {scores['is_fraud'].mean():.3f})")
