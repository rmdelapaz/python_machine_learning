# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 6.1 A (solution): a fair fraud-model bake-off, decided before June is opened.

Lesson 6.1 ran a churn bake-off. The payments team now wants to know which fraud model to put in
front of the review queue. The rules from the lesson: one metric chosen before looking (average
precision: the queue is the top of a ranked list), the same folds for every candidate, paired
differences for close calls, and the test month (June) opened once, after the choice.
"""
# %% Step 1: the time split
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

# %% Step 2 (TODO 1): every candidate on the same 10 folds of January to May
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
fold_ap, rows = {}, []
for name, model in candidates.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=["average_precision", "roc_auc"])
    fold_ap[name] = res["test_average_precision"]
    rows.append({"model": name, "ap_mean": res["test_average_precision"].mean(),
                 "ap_std": res["test_average_precision"].std(), "auc_mean": res["test_roc_auc"].mean()})
board = pd.DataFrame(rows).set_index("model").sort_values("ap_mean", ascending=False)
print(board.round(3))

# %% Step 3 (TODO 2): paired differences for the two close calls
def paired(a, b):
    diff = fold_ap[a] - fold_ap[b]
    print(f"{a} - {b}: mean {diff.mean():+.3f}, std {diff.std():.3f}, {a} wins {(diff > 0).sum()} of {len(diff)} folds")

paired("boost_gentle", "boost_default")
paired("boost_gentle", "logistic")

# %% Step 4 (TODO 3): the choice, made before June
winner = board.index[0]
print(f"\nchosen on Jan-May: {winner} (CV AP {board.loc[winner, 'ap_mean']:.3f} +/- {board.loc[winner, 'ap_std']:.3f})")

# %% Step 5 (TODO 4): open June once, for the winner (and the runner-up for reference, not for choosing)
for name in [winner, "logistic"]:
    p = candidates[name].fit(X_train, y_train).predict_proba(X_test)[:, 1]
    caught = int(y_test.loc[pd.Series(p, index=y_test.index).nlargest(100).index].sum())
    print(f"June {name:13s} AP {average_precision_score(y_test, p):.3f}   frauds in top 100: {caught} of {y_test.sum()}")
print(f"June random order  AP {y_test.mean():.3f}")
