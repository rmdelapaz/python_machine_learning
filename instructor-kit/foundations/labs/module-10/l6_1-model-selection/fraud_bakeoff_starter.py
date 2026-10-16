# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 6.1 A: a fair fraud-model bake-off, decided before June is opened.

Lesson 6.1 ran a churn bake-off. The payments team now wants to know which fraud model to put in
front of the review queue. The rules from the lesson: one metric chosen before looking (average
precision: the queue is the top of a ranked list), the same folds for every candidate, paired
differences for close calls, and the test month (June) opened once, after the choice.
Work through TODO 1 to 4; the file runs as it is (about 30 seconds when finished).
"""
# %% Step 1: the time split and four candidates (done for you)
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
num = ["account_age_days", "amount", "hour", "items", "device_new",
       "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[num + ["category"]].copy()
X["category"] = X["category"].astype("category")
y = tx["is_fraud"]
is_june = tx["timestamp"] >= "2025-06-01"
X_train, X_test, y_train, y_test = X[~is_june], X[is_june], y[~is_june], y[is_june]

candidates = {
    "baseline": DummyClassifier(),
    "logistic": make_pipeline(make_column_transformer((StandardScaler(), num),
                                                      (OneHotEncoder(handle_unknown="ignore"), ["category"])),
                              LogisticRegression(max_iter=1000)),
    "boost_default": HistGradientBoostingClassifier(random_state=42),
    "boost_gentle": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                   min_samples_leaf=40, l2_regularization=1.0,
                                                   early_stopping=True, random_state=42),
}

# %% Step 2: every candidate on the same 10 folds of January to May
# TODO 1: ONE splitter, RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42). For each
#         candidate run cross_validate(..., scoring=["average_precision", "roc_auc"]); keep the per-fold AP
#         in a dict fold_ap and print a leaderboard (mean AP, std AP, mean AUC) sorted by mean AP.
#         Before you run it, predict: which model will the AUC column favor, and which the AP column?
fold_ap = None
if fold_ap is None:
    print("TODO 1: a 10-fold leaderboard on Jan-May (AP decides, AUC for context)")

# %% Step 3: paired differences for the close calls
# TODO 2: write paired(a, b) as in the lesson (per-fold difference: mean, std, wins) and run it for
#         boost_gentle vs boost_default and boost_gentle vs logistic. Which gap is real, and which is close?
print("TODO 2: paired differences for two pairs")

# %% Step 4: the choice, made before June
# TODO 3: keep the top of the leaderboard in winner and print it with its mean and std.
winner = None
if winner is None:
    print("TODO 3: name the winner on Jan-May")

# %% Step 5: open June once
# TODO 4: fit the winner (and logistic, for reference only) on all of Jan-May; print each one's June AP
#         and the frauds among its 100 highest scores. What does a random ordering score in AP?
print("TODO 4: June AP and frauds in the top 100, once")

print("\nExpected when done: CV AP boost_gentle 0.521 +/- 0.057, logistic 0.504, boost_default 0.446, baseline 0.009;")
print("AUC puts logistic first (0.855 vs 0.847); gentle - default +0.075 (wins 10 of 10), gentle - logistic +0.017")
print("(std 0.023, wins 7 of 10); June AP 0.556 for boost_gentle and 0.539 for logistic, both 36 frauds in the top 100;")
print("random order AP 0.008.")
