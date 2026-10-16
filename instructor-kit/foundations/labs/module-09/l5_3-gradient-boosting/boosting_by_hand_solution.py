# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.3 A (SOLUTION): gradient boosting by hand, for churn.

Lesson 5.3 boosted a wavy regression curve by hand and said that for churn "each tree is fitted to
y - p, the gap between the 0/1 label and the current predicted probability, on the log-odds scale".
Here you write exactly that loop on the churn table, watch a validation set (never the test set) to
choose the number of trees, test once, and compare with HistGradientBoostingClassifier.
"""
# %% Step 1: the churn split from Lesson 1.1, plus a validation set carved from the training rows
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
start = np.log(y.mean() / (1 - y.mean()))
print(f"churn rate {y.mean():.3f} -> starting log-odds {start:.3f} -> starting probability {sigmoid(start):.3f}")
print(f"log loss of predicting {y.mean():.3f} for everyone: {log_loss(y_val, np.full(len(y_val), y.mean())):.4f}")

# %% Step 3: the loop: fit a small regression tree to y - p, take a small step on the log-odds
F_fit = np.full(len(X_fit), start)
F_val = np.full(len(X_val), start)
learning_rate = 0.3
val_loss = []
for m in range(1, 301):
    residual = y - sigmoid(F_fit)                                        # what is still wrong
    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_fit, residual)
    F_fit += learning_rate * tree.predict(X_fit)                         # a small step
    F_val += learning_rate * tree.predict(X_val)
    val_loss.append(log_loss(y_val, sigmoid(F_val)))
    if m in (1, 10, 30, 100, 300):
        print(f"{m:3d} trees  fit loss {log_loss(y, sigmoid(F_fit)):.4f}  validation loss {val_loss[-1]:.4f}"
              f"  validation AUC {roc_auc_score(y_val, F_val):.3f}")
best = int(np.argmin(val_loss)) + 1
print(f"best number of trees on validation: {best} (loss {min(val_loss):.4f})")

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


hand = boost(X_train, y_train, X_test, best)
print(f"\nhand-made boosting, {best} trees: test AUC {roc_auc_score(y_test, hand):.3f}, "
      f"test log loss {log_loss(y_test, sigmoid(hand)):.4f}")

# %% Step 5: the library version, with its own early stopping
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=2000, max_depth=3, early_stopping=True,
                                     validation_fraction=0.2, n_iter_no_change=20, random_state=42)
hgb.fit(X_train, y_train)
p = hgb.predict_proba(X_test)[:, 1]
print(f"HistGradientBoosting, {hgb.n_iter_} trees: test AUC {roc_auc_score(y_test, p):.3f}, test log loss {log_loss(y_test, p):.4f}")

# %% Stretch: the hand loop with a full-strength step (learning_rate=1.0)
F_val = np.full(len(X_val), start)
F_fit = np.full(len(X_fit), start)
fast = []
for m in range(1, 301):
    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_fit, y - sigmoid(F_fit))
    F_fit += 1.0 * tree.predict(X_fit)
    F_val += 1.0 * tree.predict(X_val)
    fast.append(log_loss(y_val, sigmoid(F_val)))
print(f"\nlearning rate 1.0: best {int(np.argmin(fast)) + 1} trees, validation loss {min(fast):.4f} "
      f"(0.3: {best} trees, {min(val_loss):.4f}); after 300 trees {fast[-1]:.4f} vs {val_loss[-1]:.4f}")
