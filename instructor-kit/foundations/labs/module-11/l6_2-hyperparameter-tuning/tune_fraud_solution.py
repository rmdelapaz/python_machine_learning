# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 6.2 A (solution): tune the fraud booster on January to May, and report it honestly.

Lesson 6.2 tuned the churn booster. Fraud is harder: 374 training frauds, and a test month (June) with
only 64. Tune on the training months only, look at how many settings are as good as the winner, then
open June once for the default and the tuned model, and save the settings so the model can be rebuilt.
"""
# %% Step 1: the time split, a cross-validation plan and the score to beat
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

# %% Step 2 (TODO 1): the search space, written down before anything runs
space = {"learning_rate": loguniform(0.01, 0.3),
         "max_leaf_nodes": randint(4, 64),
         "min_samples_leaf": randint(10, 200),
         "l2_regularization": loguniform(1e-3, 10)}
budget = 8
print(f"{budget} settings x {cv.get_n_splits()} folds = {budget * cv.get_n_splits()} fits, plus 1 refit")

# %% Step 3 (TODO 2): the search
search = RandomizedSearchCV(booster, space, n_iter=budget, cv=cv, scoring="average_precision",
                            random_state=42, n_jobs=-1).fit(X_train, y_train)
best = {k: round(float(v), 4) if isinstance(v, float) else int(v) for k, v in search.best_params_.items()}
print(f"best CV AP: {search.best_score_:.3f}")
print(best)

# %% Step 4 (TODO 3): how many settings are about as good as the winner?
board = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
print(board[["param_learning_rate", "param_max_leaf_nodes", "param_min_samples_leaf",
             "mean_test_score", "std_test_score"]].round(3).to_string(index=False))
close = board["mean_test_score"] >= search.best_score_ - board["std_test_score"].iloc[0]
print(f"settings within one std of the best: {close.sum()} of {len(board)}")

# %% Step 5 (TODO 4): June, once, for both models
default_model = booster.fit(X_train, y_train)
for name, model in [("default", default_model), ("tuned", search)]:
    p = model.predict_proba(X_test)[:, 1]
    top100 = int(y_test.loc[pd.Series(p, index=y_test.index).nlargest(100).index].sum())
    print(f"June {name:8s} AP {average_precision_score(y_test, p):.3f}   frauds in top 100: {top100} of {y_test.sum()}")

# %% Step 6 (TODO 5): save the settings and the evidence
record = {"model": "HistGradientBoostingClassifier", "fixed": {"max_iter": 300, "early_stopping": True,
          "random_state": 42}, "train": "2025-01-01 to 2025-05-31", "metric": "average_precision",
          "cv": "StratifiedKFold(3, shuffle=True, random_state=42)",
          "search": f"RandomizedSearchCV, n_iter={budget}, random_state=42", "best_params": best,
          "cv_score": round(float(search.best_score_), 4), "default_cv_score": round(float(default_ap.mean()), 4)}
with open(OUTPUT / "fraud_best_params.json", "w") as f:
    json.dump(record, f, indent=2)
print("saved fraud_best_params.json with", sorted(record))
