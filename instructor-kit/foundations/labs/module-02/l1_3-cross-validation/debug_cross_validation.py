# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 1.3). This file is DELIBERATELY BROKEN.

A colleague's cross-validation of a churn model. Two bugs stop it with an error, one at a time: run it,
read the LAST line of the traceback, fix that one bug, run again. Three more bugs raise no error: the
script runs and reports a number that is not honest. Find those too.

When everything is fixed (with the course's splitter for classification,
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)) it prints:
    cross-validated on 4500 training customers, 5 folds
    ROC AUC 0.829 +/- 0.025
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X, y = customers[features], customers["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

X_scaled = StandardScaler().fit_transform(X)
cv = KFold(n_splits=5, random_state=42)
scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=cv, scoring="roc-auc")
print(f"cross-validated on {len(X_scaled)} training customers, {len(scores)} folds")
print(f"ROC AUC {scores.max():.3f} +/- {scores.std():.3f}")
