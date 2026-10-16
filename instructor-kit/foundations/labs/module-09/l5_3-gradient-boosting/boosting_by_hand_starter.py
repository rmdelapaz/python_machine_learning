# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.3 A: gradient boosting by hand, for churn.

Lesson 5.3 boosted a wavy regression curve by hand and said that for churn "each tree is fitted to
y - p, the gap between the 0/1 label and the current predicted probability, on the log-odds scale".
Here you write exactly that loop on the churn table, watch a validation set (never the test set) to
choose the number of trees, test once, and compare with HistGradientBoostingClassifier.
Work through TODO 1 to 4; run the file after each one. It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the churn split from Lesson 1.1, plus a validation set carved from the training rows (done)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss, roc_auc_score

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
print(len(X_fit), "rows to fit,", len(X_val), "to validate,", len(X_test), "locked away for the test")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# %% Step 2: start from the average, on the log-odds scale
y = y_fit.to_numpy()
# TODO 1: start = the log-odds of the churn rate, np.log(rate / (1 - rate)). Print the rate, start and
#         sigmoid(start) (it must give the rate back), and the validation log loss of predicting the rate
#         for everyone: log_loss(y_val, np.full(len(y_val), rate)).
start = None
if start is None:
    print("TODO 1: the starting guess")

# %% Step 3: the loop
# TODO 2: F_fit and F_val start at `start` (np.full). For m in 1..300:
#             residual = y - sigmoid(F_fit)                       # what is still wrong
#             tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_fit, residual)
#             F_fit += 0.3 * tree.predict(X_fit);  F_val += 0.3 * tree.predict(X_val)
#         append log_loss(y_val, sigmoid(F_val)) to val_loss, and print the fit loss, validation loss and
#         validation AUC at m = 1, 10, 30, 100, 300. Then print the best m (lowest validation loss).
val_loss = []
best = None
if best is None:
    print("TODO 2: the boosting loop and the best number of trees")


# %% Step 4: refit on all training rows with that many trees, then test once
def boost(X, y, X_new, n_trees, learning_rate=0.3):
    """Hand-made boosting: returns the log-odds for X_new after n_trees rounds."""
    y = np.asarray(y)
    F = np.full(len(X), np.log(y.mean() / (1 - y.mean())))
    F_new = np.full(len(X_new), F[0])
    for _ in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y - sigmoid(F))
        F += learning_rate * tree.predict(X)
        F_new += learning_rate * tree.predict(X_new)
    return F_new


# TODO 3: hand = boost(X_train, y_train, X_test, best); print the test AUC (log-odds rank like
#         probabilities) and the test log loss of sigmoid(hand).
print("TODO 3: test the hand-made booster once")

# %% Step 5: the library version, with its own early stopping
from sklearn.ensemble import HistGradientBoostingClassifier

# TODO 4: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=2000, max_depth=3, early_stopping=True,
#         validation_fraction=0.2, n_iter_no_change=20, random_state=42): fit on X_train, print n_iter_,
#         the test AUC and the test log loss. Same idea, smarter steps.
print("TODO 4: the library version")

# %% Stretch: run the loop again with learning_rate=1.0. How many trees does it want, and how bad is
#    its validation loss after 300 trees?

print("\nExpected when done: rate 0.165, start -1.619, constant-guess validation loss 0.4488; validation")
print("loss 0.4375 (1 tree), 0.3699 (10), 0.3217 (30), 0.2954 (100), 0.2898 (300); best 275 trees")
print("(0.2892); hand-made test AUC 0.870, log loss 0.3082; HistGradientBoosting 165 trees, test AUC")
print("0.866, log loss 0.3091. Stretch: rate 1.0 is best at 72 trees (0.2901) and reaches 0.3072 at 300.")
