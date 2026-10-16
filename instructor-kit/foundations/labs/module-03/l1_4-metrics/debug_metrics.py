# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 1.4). This file is DELIBERATELY BROKEN.

It should report four numbers for the retention team. Two bugs stop it with an error, one at a time:
run it, read the LAST line of the traceback, fix that one bug, run again. Two more bugs raise no error
at all and print believable numbers. Those are the dangerous kind: find them too.

When everything is fixed it prints:
    logistic regression ROC AUC: 0.823
    forest precision: 0.705
    forest recall:    0.395
    forest ROC AUC:   0.865
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)
logreg = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_train, y_train)

print(f"logistic regression ROC AUC: {roc_auc_score(y_test, logreg.predict_proba(X_test)):.3f}")

# The care team's report uses words, not 0 and 1
labels = {0: "stayed", 1: "churned"}
text_true = y_test.map(labels)
text_pred = pd.Series(forest.predict(X_test), index=y_test.index).map(labels)
print(f"forest precision: {precision_score(text_true, text_pred):.3f}")
print(f"forest recall:    {recall_score(forest.predict(X_test), y_test):.3f}")
print(f"forest ROC AUC:   {roc_auc_score(y_test, forest.predict(X_test)):.3f}")
