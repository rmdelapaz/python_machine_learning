# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 1.2). This file is DELIBERATELY BROKEN.

A colleague's churn evaluation. Two bugs stop it with an error, one at a time: run it, read the LAST
line of the traceback, fix that one bug, run again. Two more bugs raise no error at all and make the
score look better (or shakier) than it is. Find them too: they are the dangerous kind.

When everything is fixed it prints:
    test customers: 1500 (churn rate 0.165)
    baseline accuracy 0.835
    logistic regression accuracy 0.848
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
customers["has_reason"] = customers["cancellation_reason"].notna().astype(int)
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "has_reason"]
X = customers[features]
y = customers["churned"]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=25, stratify=customers["spend_12m"],
                                                    random_state=42)
print(f"test customers: {len(X_test)} (churn rate {y_test.mean():.3f})")

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"baseline accuracy {baseline.score(X_test, y_test):.3f}")

scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)
print(f"logistic regression accuracy {model.score(scaler.transform(X_test), y_test):.3f}")
