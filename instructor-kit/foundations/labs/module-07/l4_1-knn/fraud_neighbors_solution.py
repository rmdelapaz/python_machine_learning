# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.1 A (SOLUTION): k-nearest neighbors for fraud.

Lesson 4.1 asked churn customers' lookalikes. Here a payment is judged by the payments most like it. Fraud
columns mix dollars (up to $3,161), days (up to 7,030) and 0/1 flags, so scaling matters even more than for
churn. Fraud is rare, so the score is average precision (AP), with ROC AUC beside it.
"""
# %% Step 1: the stratified split from Lesson 2.5, numeric columns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
cols = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
        "country_mismatch", "card_attempts", "velocity_24h"]
X_train, X_test, y_train, y_test = train_test_split(tx[cols], tx["is_fraud"], test_size=0.3,
                                                    stratify=tx["is_fraud"], random_state=42)
print(X_train.shape, f"random-ranking AP {y_test.mean():.4f}")

# %% Step 2: who decides the unscaled distance?
a, b = X_train.iloc[0], X_train.iloc[1]
share = (a - b) ** 2 / ((a - b) ** 2).sum()
print("\nshare of the squared distance, first two payments:")
print(share.sort_values(ascending=False).head(3).round(3).to_string())

# %% Step 3: unscaled against scaled, with logistic regression as the reference
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print()
for name, model in [("kNN unscaled", KNeighborsClassifier(n_neighbors=51)),
                    ("kNN scaled", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=51))),
                    ("logistic regression", make_pipeline(StandardScaler(), LogisticRegression()))]:
    p = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"{name:20s} ROC AUC {roc_auc_score(y_test, p):.3f}   AP {average_precision_score(y_test, p):.3f}")

# %% Step 4: choose k by cross-validated AP on the training rows
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k in [5, 51, 201]:
    s = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
                        X_train, y_train, cv=cv, scoring="average_precision")
    print(f"k={k:3d}   CV AP {s.mean():.3f} +/- {s.std():.3f}")

# %% Step 5: why a small k ranks badly (ties), and the chosen k on the test set, once
for k in [5, 201]:
    p = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"k={k:3d}: {np.unique(p).size:3d} distinct probabilities; {int((p == 0).sum()):,} test payments at exactly 0;"
          f" test AP {average_precision_score(y_test, p):.3f}")
