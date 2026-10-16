# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 5.3 B (SOLUTION): three boosting libraries on the payments team's review queue.

The fraud labs so far: logistic regression put 53 frauds in its top 100 (Session 1), the forest 57
(Lab 5.2 A). Now boosting, in all three libraries the lesson met, with the category column handled
natively and early stopping on a validation set carved from the training rows. Lesson 5.3 used
churn; fraud is much rarer, which changes what "early stopping" should watch.
"""
# %% Step 1: nine numeric columns plus category, as the category dtype; the usual stratified split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["amount", "account_age_days", "hour", "items", "device_new",
           "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
X = tx[numeric + ["category"]].copy()
X["category"] = X["category"].astype("category")
y = tx["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
print(f"fit {len(X_fit)} rows ({y_fit.sum()} frauds), validation {len(X_val)} ({y_val.sum()}), test {len(X_test)} ({y_test.sum()})")


def report(name, p, note=""):
    top = int(y_test.to_numpy()[np.argsort(-p, kind="stable")[:100]].sum())
    print(f"{name:22s} test AUC {roc_auc_score(y_test, p):.3f}   AP {average_precision_score(y_test, p):.3f}"
          f"   frauds in top 100: {top:3d}   {note}")


# %% Step 2: scikit-learn's booster, default and gentle
from sklearn.ensemble import HistGradientBoostingClassifier

default = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
report("HGB, defaults", default.predict_proba(X_test)[:, 1], f"trees {default.n_iter_}")
print("treated as categorical:", list(X_train.columns[default.is_categorical_]))

gentle = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=2000, max_leaf_nodes=15, min_samples_leaf=40,
                                        early_stopping=True, validation_fraction=0.2, n_iter_no_change=30,
                                        random_state=42).fit(X_train, y_train)
report("HGB, gentle", gentle.predict_proba(X_test)[:, 1], f"trees {gentle.n_iter_}")

# %% Step 3: XGBoost: early stopping in the constructor, the validation set in fit
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, min_child_weight=5,
                    enable_categorical=True, eval_metric="aucpr", early_stopping_rounds=100,
                    random_state=42, verbosity=0)
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
report("XGBoost", xgb.predict_proba(X_test)[:, 1], f"best iteration {xgb.best_iteration}")

# %% Step 4: LightGBM: eval_X / eval_y in fit, early stopping as a callback
import lightgbm as lgb
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                      random_state=42, verbose=-1)
lgbm.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="average_precision",
         callbacks=[lgb.early_stopping(100, verbose=False)])
report("LightGBM", lgbm.predict_proba(X_test)[:, 1], f"best iteration {lgbm.best_iteration_}")

# %% Step 5: the same XGBoost, early-stopped on the TEST set (the mistake to recognize)
peek = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, min_child_weight=5,
                     enable_categorical=True, eval_metric="aucpr", early_stopping_rounds=100,
                     random_state=42, verbosity=0)
peek.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
report("XGBoost, peeking", peek.predict_proba(X_test)[:, 1], f"best iteration {peek.best_iteration} (chosen ON the test set)")
