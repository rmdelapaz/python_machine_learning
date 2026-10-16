# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Foundations refresher: the six habits Advanced Machine Learning assumes, one small function each.

Kit B builds on Machine Learning Foundations (or equivalent): the canonical churn split, a Pipeline
named "prep" and "model", cross-validation on training rows only, recall and precision, and choosing a
threshold from Pinecrest's campaign economics ($15 per offer, $60 per real churner reached). Each
function below matches one refresher question (D1 to D6 on the workbook's Start Here page). Replace
each `return None` with a real answer, run the file, and watch "not yet" turn into PASS. The last cell
runs your functions on the real churn data once all six pass.
Review if stuck: D1 Lessons 1.1-1.2, D2 Lesson 2.2, D3 Lesson 1.3, D4-D6 Lesson 1.4.
"""
# %% The six functions
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


def split_churn(customers):
    """D1 the canonical split: a quarter for testing, stratified on churned, random_state=42."""
    return None  # TODO 1: return train, test from train_test_split on the WHOLE table


def churn_pipeline(preprocess, model):
    """D2 a Pipeline whose two steps are named "prep" and "model"."""
    return None  # TODO 2: Pipeline([("prep", ...), ("model", ...)])


def cv_auc(pipe, X, y):
    """D3 mean and standard deviation of 5-fold stratified CV ROC AUC (shuffle, random_state=42)."""
    return None  # TODO 3: StratifiedKFold(...), cross_val_score(..., scoring="roc_auc"); return (mean, std)


def recall_precision(y_true, flags):
    """D4 recall and precision of 0/1 flags, as a pair of floats rounded to 3 places."""
    return None  # TODO 4: count true positives; recall = tp / real positives, precision = tp / flags


def campaign_profit(y_true, flags):
    """D5 Pinecrest's campaign economics: $60 per real churner reached, $15 per offer sent."""
    return None  # TODO 5: 60 * true positives - 15 * number flagged, as an int


def best_threshold(y_true, proba, grid):
    """D6 the threshold in `grid` with the highest campaign profit (flag = proba >= threshold)."""
    return None  # TODO 6: try every threshold, keep the most profitable one


# %% Check each one on a tiny example (do not edit below this line)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

tiny_y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
tiny_flags = np.array([1, 1, 0, 0, 0, 1, 0, 0])
tiny_p = np.array([0.9, 0.3, 0.3, 0.6, 0.1, 0.4, 0.05, 0.15])
tiny_customers = pd.DataFrame({"customer_id": [f"T{i}" for i in range(20)],
                               "churned": [1, 0, 0, 0, 0] * 4})
toy_X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) % 7})
toy_y = (toy_X["a"] + 3 * toy_X["b"] > 30).astype(int)


def _d1():
    r = split_churn(tiny_customers)
    return None if r is None else (len(r[0]), len(r[1]), int(r[1]["churned"].sum()))


def _d2():
    r = churn_pipeline(StandardScaler(), LogisticRegression())
    return None if r is None else list(r.named_steps)


def _d3():
    r = cv_auc(churn_pipeline(StandardScaler(), LogisticRegression()), toy_X, toy_y)
    return None if r is None else (round(r[0], 3), round(r[1], 3))


checks = [
    ("D1 canonical split", _d1, (15, 5, 1)),
    ("D2 named pipeline", _d2, ["prep", "model"]),
    ("D3 cross-validation", _d3, (1.0, 0.0)),
    ("D4 recall, precision", lambda: recall_precision(tiny_y, tiny_flags), (0.667, 0.667)),
    ("D5 campaign profit", lambda: campaign_profit(tiny_y, tiny_flags), 75),
    ("D6 best threshold", lambda: best_threshold(tiny_y, tiny_p, [0.1, 0.25, 0.5]), 0.25),
]
passed = 0
for name, fn, want in checks:
    try:
        got = fn()
    except Exception as err:          # a half-written function: show the error, keep going
        got = f"{type(err).__name__}: {err}"
    ok = got == want
    passed += ok
    print(f"{'PASS' if ok else 'not yet':8s} {name:22s} got {got!r}")
print(f"{passed} of {len(checks)} checks pass.")

# %% The same functions on the real churn data (runs once all six pass)
if passed == len(checks):
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline

    customers = pd.read_csv(DATA / "customers_churn.csv")
    features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
                "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user",
                "email_opens_90d"]
    train, test = split_churn(customers)
    print(f"\nsplit: {len(train)} / {len(test)}, first test customer {test['customer_id'].iloc[0]}")
    pipe = churn_pipeline(make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LogisticRegression())
    mean, std = cv_auc(pipe, train[features], train["churned"])
    print(f"logistic regression: CV ROC AUC {mean:.3f} +/- {std:.3f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, train[features], train["churned"], cv=cv, method="predict_proba")[:, 1]
    t = best_threshold(train["churned"].to_numpy(), oof, np.round(np.arange(0.05, 0.96, 0.01), 2))
    print(f"threshold chosen on out-of-fold training probabilities: {t:.2f} (break-even 15/60 = 0.25)")
    flags = (pipe.fit(train[features], train["churned"]).predict_proba(test[features])[:, 1] >= t).astype(int)
    r, pr = recall_precision(test["churned"].to_numpy(), flags)
    print(f"test set, once: {flags.sum()} offers, recall {r:.3f}, precision {pr:.3f}, "
          f"profit ${campaign_profit(test['churned'].to_numpy(), flags):,}")
else:
    print("\nExpected when done: six PASS lines, then split 4500 / 1500 (first test customer C102112);")
    print("CV ROC AUC 0.833 +/- 0.026; threshold 0.21 chosen on out-of-fold probabilities; on the test set")
    print("385 offers, recall 0.641, precision 0.413, profit $3,765.")
