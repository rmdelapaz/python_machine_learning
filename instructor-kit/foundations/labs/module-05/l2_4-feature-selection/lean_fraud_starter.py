# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.4 A: a lean fraud model.

Lesson 2.4 trimmed the churn model from 16 columns to 6. Here the payments team asks the same question
about fraud: nine raw columns plus five engineered ones. Which of the fourteen earn their place? Fraud
is rare, so the score is average precision (AP), not ROC AUC. Work through TODO 1 to 5 and run the file
after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: features, engineered features and a stratified split (done for you)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

tx = pd.read_csv(DATA / "transactions.csv")
tx["night"] = (tx["hour"] < 6).astype(int)
tx["night_new_device"] = tx["night"] * tx["device_new"]
tx["gift_card"] = (tx["category"] == "Gift Cards").astype(int)
tx["log_amount"] = np.log(tx["amount"])
tx["new_account"] = (tx["account_age_days"] < 30).astype(int)
raw = ["account_age_days", "amount", "hour", "items", "device_new", "billing_shipping_mismatch",
       "country_mismatch", "card_attempts", "velocity_24h"]
engineered = ["night", "night_new_device", "gift_card", "log_amount", "new_account"]
candidates = raw + engineered

train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
y_train, y_test = train["is_fraud"], test["is_fraud"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(len(candidates), "candidates;", len(train), "training payments,", y_train.sum(), "frauds")

# %% Step 2: free wins first
# TODO 1: compute train[candidates].corr(), keep the upper triangle (as in the lesson) and print the
#         pairs whose absolute correlation is above 0.5. Which pair is a near-duplicate, and why?
print("TODO 1: strongly correlated pairs")

# %% Step 3: a filter ranks the columns one at a time
from sklearn.feature_selection import f_classif

# TODO 2: f_classif(train[candidates], y_train) returns (scores, p-values). Put the scores in a Series
#         indexed by candidates and print the top 6. Which column wins, raw or engineered?
print("TODO 2: F-test top 6")

# %% Step 4: forward selection with the real model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def cv_ap(columns):
    """Five cross-validated AP scores of a scaled logistic regression on these columns."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_score(model, train[columns], y_train, cv=cv, scoring="average_precision")


# TODO 3: print the mean and std of cv_ap(candidates), and the mean of cv_ap(raw).
# TODO 4: write the forward-selection loop from the lesson project (Step 4), but with cv_ap(...).mean().
#         Stop after 10 columns. Keep the chosen order in `chosen` and the scores in `history`.
#         (It fits about 500 small models: allow 15 seconds.)
chosen, history = [], []
if not chosen:
    print("TODO 3-4: CV AP of all 14 and of the raw 9; forward selection, ten steps")

# %% Step 5: the tolerance rule, then one look at the test set
from sklearn.metrics import average_precision_score

# TODO 5: the folds disagree by about 0.06, so use a tolerance of 0.01: keep the smallest number of
#         columns whose CV AP is within 0.01 of the best step. Then fit raw 9, all 14 and your lean set on
#         the training rows and print each one's test AP with average_precision_score.
print("TODO 5: the lean set and three test APs")

print("\nExpected when done: amount and log_amount correlate 0.88; the F-test ranks night_new_device first;")
print("CV AP 0.5512 +/- 0.0618 for all 14, 0.5202 for the raw 9; forward selection starts with velocity_24h,")
print("then night_new_device and log_amount; the lean set has 8 columns; test AP raw 9 0.444, all 14 0.505,")
print("lean 8 0.505.")
