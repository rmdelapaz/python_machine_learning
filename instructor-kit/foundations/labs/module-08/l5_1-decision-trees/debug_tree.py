# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 5.1). This file is DELIBERATELY BROKEN.

It should print a two-question churn tree with readable class names, then the tree's accuracy on
customers it has never seen next to the do-nothing baseline. Two bugs stop it with an error, one at a
time: run it, read the LAST line of the traceback, fix that one bug, run again. Two more bugs raise no
error at all: one mislabels every leaf, one reports a flattering number that says nothing about new
customers. Find them too.

When everything is fixed (a depth-2 tree) it prints the rules (first leaf "class: stayed", third leaf
"class: churned"), then:
    accuracy on new customers 0.851 (baseline 0.835)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "days_since_last_order", "home_store",
            "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)

tree = DecisionTreeClassifier(max_depth=0, random_state=42).fit(X_train, y_train)
print(export_text(tree, feature_names=features, class_names=["churned", "stayed"]))

print(f"accuracy on new customers {tree.score(X_train, y_train):.3f} (baseline {1 - y_test.mean():.3f})")
