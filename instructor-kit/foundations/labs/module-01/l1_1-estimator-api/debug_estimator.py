# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 1.1). This file is DELIBERATELY BROKEN.

It should print a one-feature model's accuracy and the full model's accuracy. Three bugs stop it with an
error, one at a time: run it, read the LAST line of the traceback, fix that one bug, run again.
A fourth bug raises no error at all. Find it too: it is the dangerous kind.

When everything is fixed it prints:
    one feature (days_since_last_order): accuracy 0.841
    ten features: accuracy 0.848
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X = customers[features]
y = customers["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# A model on one feature only
one = LogisticRegression().fit(X_train["days_since_last_order"], y_train)
print(f"one feature (days_since_last_order): accuracy {one.score(X_test[['days_since_last_order']], y_test):.3f}")

# The full model
scaler = StandardScaler().set_output(transform="pandas")
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)
model = LogisticRegression()
model.fit(X_train_s, y_train)
model = LogisticRegression()
print(f"ten features: accuracy {model.score(X_test_s, y_test):.3f}")
