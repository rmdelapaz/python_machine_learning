# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 4.3). This file is DELIBERATELY BROKEN.

It should tune an RBF SVM's C by cross-validation, score the test set once, and count the real
churners among the 50 customers it ranks riskiest. Two bugs stop it with an error, one at a time: run
it, read the LAST line of the traceback, fix that one bug, run again. Two more bugs raise no error at
all and quietly make the model and the list worse. Find them too: they are the dangerous kind.

When everything is fixed it prints:
    best C: 1   CV ROC AUC 0.848
    test ROC AUC 0.850
    real churners among the 50 riskiest: 32 (a random 50 holds 8.3)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)

pipe = make_pipeline(SVC(class_weight="balanced"))
grid = GridSearchCV(pipe, {"C": [0.1, 1, 10]}, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print(f"best C: {grid.best_params_['svc__C']}   CV ROC AUC {grid.best_score_:.3f}")

scores = grid.predict_proba(X_test)[:, 1]
print(f"test ROC AUC {roc_auc_score(y_test, scores):.3f}")

riskiest = np.argsort(scores)[:50]
print(f"real churners among the 50 riskiest: {y_test.iloc[riskiest].sum()} (a random 50 holds {50 * y_test.mean():.1f})")
