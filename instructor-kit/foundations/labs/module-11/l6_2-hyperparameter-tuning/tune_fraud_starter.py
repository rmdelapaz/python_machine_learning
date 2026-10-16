# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 6.2 A: tune the fraud booster on January to May, and report it honestly.

Lesson 6.2 tuned the churn booster. Fraud is harder: 374 training frauds, and a test month (June) with
only 64. Tune on the training months only, look at how many settings are as good as the winner, then
open June once for the default and the tuned model, and save the settings so the model can be rebuilt.
Work through TODO 1 to 5; the finished file takes about 40 seconds.
"""
# %% Step 1: the time split, a cross-validation plan and the score to beat (done for you)
import json
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

tx = pd.read_csv(DATA / "transactions.csv")
num = ["account_age_days", "amount", "hour", "items", "device_new",
       "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[num + ["category"]].copy()
X["category"] = X["category"].astype("category")
y = tx["is_fraud"]
is_june = tx["timestamp"] >= "2025-06-01"
X_train, X_test, y_train, y_test = X[~is_june], X[is_june], y[~is_june], y[is_june]

booster = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
default_ap = cross_val_score(booster, X_train, y_train, cv=cv, scoring="average_precision")
print(f"default booster (early stopping), 3-fold AP on Jan-May: {default_ap.mean():.3f} +/- {default_ap.std():.3f}")

# %% Step 2: the search space, written down before anything runs
# TODO 1: a dict space with loguniform(0.01, 0.3) for "learning_rate", randint(4, 64) for
#         "max_leaf_nodes", randint(10, 200) for "min_samples_leaf" and loguniform(1e-3, 10) for
#         "l2_regularization" (no "model__" prefix: the booster is not inside a pipeline here).
#         Set budget = 8 and print how many fits that means.
space = None
budget = 8
if space is None:
    print("TODO 1: write the search space (log scales for the rate and the L2 strength)")

# %% Step 3: the search
# TODO 2: RandomizedSearchCV(booster, space, n_iter=budget, cv=cv, scoring="average_precision",
#         random_state=42, n_jobs=-1), fitted on Jan-May. Print best_score_ and the best settings.
search = None
if search is None:
    print("TODO 2: run the search on Jan-May")

# %% Step 4: how many settings are about as good as the winner?
# TODO 3: cv_results_ as a DataFrame sorted by rank_test_score; print the settings and scores, then
#         count the settings whose mean is at least best_score_ minus the winner's std. What do the good
#         settings have in common? What happened to the one with the highest learning rate?
print("TODO 3: the leaderboard and the settings within one std of the best")

# %% Step 5: June, once
# TODO 4: fit the default booster on Jan-May, then for the default and the search print June AP and the
#         frauds among the 100 highest scores.
print("TODO 4: default vs tuned on June, once")

# %% Step 6: save the settings
# TODO 5: write fraud_best_params.json to OUTPUT: the model, the fixed settings, the training months, the
#         metric, the cv and search descriptions, best_params, the CV score and the default's CV score.
print("TODO 5: save fraud_best_params.json")

print("\nExpected when done: default 3-fold AP 0.443 +/- 0.032; 24 fits; best CV AP 0.515 with learning_rate")
print("0.0186, max_leaf_nodes 24, min_samples_leaf 170; 5 of 8 settings within one std (all have small rates and")
print("large leaves; the 0.254 rate collapses to 0.262 +/- 0.169); June AP 0.477 default vs 0.544 tuned,")
print("32 vs 38 frauds in the top 100.")
