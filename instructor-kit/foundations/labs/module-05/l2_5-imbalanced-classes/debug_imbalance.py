# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 2.5). This file is DELIBERATELY BROKEN.

It should compare a plain and a SMOTE fraud model by cross-validated average precision, then price the
plain model's decisions at the formula threshold (review $10, missed fraud $375). Two bugs stop it with
an error, one at a time: run it, read the LAST line of the traceback, fix that bug, run again. Two more
bugs raise no error at all and print wrong numbers. Find them too: they are the dangerous kind.

When everything is fixed it prints:
    plain  CV AP 0.535
    SMOTE  CV AP 0.495
    plain model at the formula threshold: 611 flagged, cost $25,985
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
num_cols = ["account_age_days", "amount", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
cat_cols = ["category"]
X_train, X_test, y_train, y_test = train_test_split(tx[num_cols + cat_cols], tx["is_fraud"], test_size=0.3,
                                                    stratify=tx["is_fraud"], random_state=42)
preprocess = ColumnTransformer([("num", StandardScaler(), num_cols),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plain = make_pipeline(preprocess, LogisticRegression())
with_smote = make_pipeline(SMOTE(random_state=42), preprocess, LogisticRegression())
for name, pipe in [("plain", plain), ("SMOTE", with_smote)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=cv)
    print(f"{name:6s} CV AP {scores.mean():.3f}")


def fraud_cost(y_true, y_pred):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(10 * (tp + fp) + 375 * fn)


plain.fit(X_train, y_train)
pred = (plain.predict_proba(X_test)[:, 1] >= 10 / 375).astype(int)
print(f"plain model at the formula threshold: {pred.sum()} flagged, cost ${fraud_cost(y_test, pred):,}")
