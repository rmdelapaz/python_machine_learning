# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 5.2). This file is DELIBERATELY BROKEN.

It should compare 100 bagged trees with a random forest on churn, judge the forest out-of-bag, and
print numbers a colleague can reproduce. Two bugs stop it with an error, one at a time: run it, read the
LAST line of the traceback, fix that one bug, run again. Two more bugs raise no error at all: one makes
the forest look perfect, one makes every run print slightly different numbers. Find them too.

When everything is fixed it prints, every time:
    bagged trees   test ROC AUC 0.849
    forest         OOB ROC AUC 0.866   test ROC AUC 0.865
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)

bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
print(f"bagged trees   test ROC AUC {roc_auc_score(y_test, bagging.predict_proba(X_test)[:, 1]):.3f}")

forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, bootstrap=False, oob_score=True, n_jobs=-1)
forest.fit(X_train, y_train)
honest = roc_auc_score(y_train, forest.predict_proba(X_train)[:, 1])
print(f"forest         OOB ROC AUC {honest:.3f}   test ROC AUC {roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1]):.3f}")
