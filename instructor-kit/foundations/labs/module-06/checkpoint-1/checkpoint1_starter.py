# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Checkpoint 1: audit and fix a churn model (30 minutes, open notebook, not graded).

Pinecrest's marketing team will email a win-back voucher to customers the churn model flags. An email with
a voucher costs $5; a churner nobody reaches costs about $50 of margin. A colleague's notebook (Part A
below, it runs) reports about 95% accuracy. Your job:

  1. PLAN (3 min): above the code, write in comments what you will check and how you will judge the model.
  2. AUDIT (5 min): list every problem you find in Part A, in comments, one line each.
  3. REBUILD (15 min): an honest model in a pipeline, cross-validated against a baseline with a metric
     that suits a rare class; choose who gets the email from the $5 / $50 prices, on the training rows only;
     look at the test set once.
  4. REPORT (5 min): print three sentences for the marketing lead: the answer, the numbers, two caveats.

Use your own notebooks from Lessons 1.1 to 2.5. The canonical split is in Part B.
"""
# %% Plan (write it here, as comments)
#

# %% Part A: the colleague's notebook (do not fix it here; audit it)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
X = customers.drop(columns=["customer_id", "signup_date", "churned"])
X = X.fillna({"cancellation_reason": "none", "age_group": "unknown"}).fillna(X.median(numeric_only=True))
X = pd.get_dummies(X, dtype=float)
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
y = customers["churned"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
colleague = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"colleague's accuracy: {colleague.score(X_te, y_te):.3f}")

# %% Audit (one comment line per problem)
# 1.
# 2.
# 3.

# %% Part B: the canonical split and the prices (done for you)
num_cols = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
cat_cols = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, y_train = train[num_cols + cat_cols], train["churned"]
X_test, y_test = test[num_cols + cat_cols], test["churned"]
EMAIL_COST, CHURN_LOSS = 5, 50

# %% Rebuild
# TODO 1: a pipeline that imputes, scales and one-hot encodes inside itself, then a logistic regression.
# TODO 2: 5-fold stratified CV on the training rows: ROC AUC and average precision, for your model and a
#         DummyClassifier.
# TODO 3: a cost function (EMAIL_COST per email, CHURN_LOSS per churner not reached) and a threshold chosen
#         on the training rows (the formula or TunedThresholdClassifierCV).
# TODO 4: the test set, once: ROC AUC, AP, emails sent, churners reached, cost against emailing nobody,
#         everyone, and the default 0.5 cut-off.
print("TODO 1-4: rebuild, cross-validate, choose the threshold, test once")

# %% Report
# TODO 5: print three sentences for the marketing lead.
print("TODO 5: three sentences")

print("\nWhat a solid answer reaches (your numbers may differ slightly if your pipeline does): CV ROC AUC about")
print("0.83 against 0.50 for the baseline, CV AP about 0.53 against 0.165; a threshold near 0.10; on the test")
print("set roughly 650 emails reaching about 214 of 248 churners for about $5,000, against $7,500 for")
print("emailing everyone and $12,400 for emailing nobody.")
