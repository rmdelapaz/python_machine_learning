# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 5.3). This file is DELIBERATELY BROKEN.

It should compare HistGradientBoosting and an early-stopped XGBoost on churn, with the text columns
handled as categories and the test set touched once. Two bugs stop it with an error, one at a time:
run it, read the LAST line of the traceback, fix that one bug, run again. A third bug raises no error
at all and makes XGBoost look a little better than it is. Find it too.

When everything is fixed it prints (XGBoost's last digit may vary on your machine):
    HistGradientBoosting  test ROC AUC 0.862
    XGBoost               test ROC AUC 0.864   best iteration 218
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

customers = pd.read_csv(DATA / "customers_churn.csv")
numeric = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
categorical = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]
X = customers[numeric + categorical].copy()
y = customers["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

hgb = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
print(f"HistGradientBoosting  test ROC AUC {roc_auc_score(y_test, hgb.predict_proba(X_test)[:, 1]):.3f}")

xgb = XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=4, enable_categorical=True,
                    eval_metric="logloss", random_state=42, verbosity=0)
xgb.fit(X_fit, y_fit, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
print(f"XGBoost               test ROC AUC {roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]):.3f}"
      f"   best iteration {xgb.best_iteration}")
