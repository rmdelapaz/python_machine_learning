# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 4.1 A: k-nearest neighbors for fraud.

Lesson 4.1 asked churn customers' lookalikes. Here a payment is judged by the payments most like it. Fraud
columns mix dollars (up to $3,161), days (up to 7,030) and 0/1 flags, so scaling matters even more than for
churn. Fraud is rare, so the score is average precision (AP), with ROC AUC beside it. Work through TODO 1 to 5; the file runs as it is.
"""
# %% Step 1 (done for you): the stratified split from Lesson 2.5, numeric columns
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
# TODO 1: take the first two training payments (X_train.iloc[0] and .iloc[1]); compute each column's share of
#         their squared Euclidean distance, (a - b) ** 2 / ((a - b) ** 2).sum(), and print the top 3.
print("TODO 1: shares of the unscaled distance")

# %% Step 3: unscaled against scaled
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# TODO 2: fit KNeighborsClassifier(n_neighbors=51) unscaled, the same inside make_pipeline(StandardScaler(), ...),
#         and a scaled LogisticRegression. Print each one's test ROC AUC and AP.
print("TODO 2: unscaled kNN, scaled kNN, logistic regression")

# %% Step 4: choose k
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO 3: cross-validated AP (scaled pipeline) for k = 5, 51 and 201 on the training rows. (About 10 seconds.)
print("TODO 3: CV AP by k")

# %% Step 5: ties, then one test
# TODO 4: for k = 5 and 201, count the distinct predicted probabilities on the test set and the payments at
#         exactly 0. Why does a small k rank badly?
# TODO 5: print the test AP for both. Does tuned kNN catch up with logistic regression?
print("TODO 4-5: ties and test AP")

print("\nExpected when done: account_age_days makes 99.7% of the unscaled distance; unscaled kNN AUC 0.659,")
print("AP 0.056; scaled AUC 0.812, AP 0.397; logistic regression AUC 0.848, AP 0.444; CV AP 0.362, 0.448, 0.468")
print("for k = 5, 51, 201; k=5: 6 distinct probabilities, 14,588 payments at 0, test AP 0.329; k=201: 43,")
print("8,240 and test AP 0.419, still below logistic regression.")
