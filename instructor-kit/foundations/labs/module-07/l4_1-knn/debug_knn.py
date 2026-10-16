# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (fast finishers, Lesson 4.1). This file is DELIBERATELY BROKEN.

It should cross-validate a kNN churn model on seven columns and show what the first test customer's 51
lookalikes did. Two bugs stop it with an error, one at a time: run it, read the LAST line of the
traceback, fix that bug, run again. Two more bugs raise no error and print wrong numbers. Find them too.

When everything is fixed it prints:
    CV ROC AUC 0.854
    first test customer: 3 of 51 lookalikes churned; their median days since last order 46 (the customer's: 53)
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
cols = ["tenure_months", "orders_12m", "days_since_last_order", "support_tickets_12m", "discount_share",
        "app_user", "email_opens_90d"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)

# a quick smoke test on a few customers before the real run
smoke = KNeighborsClassifier(n_neighbors=51).fit(train[cols].head(40).fillna(0), train["churned"].head(40))
print("smoke test:", smoke.predict(test[cols].head(3).fillna(0)))

knn = make_pipeline(KNeighborsClassifier(n_neighbors=51))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"CV ROC AUC {cross_val_score(knn, train[cols], train['churned'], cv=cv, scoring='roc_auc').mean():.3f}")

knn.fit(train[cols], train["churned"])
customer = test[cols].head(1)
_, positions = knn[-1].kneighbors(customer)
lookalikes = train.iloc[positions[0]]
print(f"first test customer: {lookalikes['churned'].sum()} of 51 lookalikes churned;"
      f" their median days since last order {lookalikes['days_since_last_order'].median():.0f}"
      f" (the customer's: {customer['days_since_last_order'].iloc[0]})")
