# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Checkpoint 1 (MODEL ANSWER): audit and fix a churn model.

The brief: a colleague's notebook reports 95% accuracy for a churn model that will choose who gets a
win-back email with a voucher (cost $5 per customer emailed; a churner nobody reaches costs $50). Find
what is wrong, rebuild it honestly, choose who to email, and write three sentences for the marketing lead.
Written in about 25 minutes at a learner's pace.
"""
# %% Plan (written before any code)
# Audit: (1) cancellation_reason is filled only after a customer leaves: a leak; (2) gaps filled and the
# scaler fitted on all 6,000 rows before the split; (3) accuracy with no baseline (doing nothing scores
# 0.835); (4) one split, no cross-validation; (5) predict() cuts at 0.5, which ignores the $5 / $50 prices.
# Rebuild: canonical split; a pipeline (impute, scale, one-hot) + logistic regression; 5-fold CV ROC AUC
# and AP against a dummy; threshold tuned on the training rows for cost; test once; three sentences.

# %% 1. The colleague's number, reproduced (kept short)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
filled = customers["cancellation_reason"].notna()
print(f"cancellation_reason filled for {filled.sum()} customers; churn among them {customers.loc[filled, 'churned'].mean():.0%}")

# %% 2. Honest columns and the canonical split
num_cols = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
cat_cols = ["home_store", "region", "age_group", "loyalty_tier", "preferred_category"]   # no cancellation_reason
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, y_train = train[num_cols + cat_cols], train["churned"]
X_test, y_test = test[num_cols + cat_cols], test["churned"]

# %% 3. Everything that learns lives in the pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_model():
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=1000))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, est in [("baseline", DummyClassifier(strategy="prior")), ("logistic regression", make_model())]:
    s = cross_validate(est, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
    print(f"{name:20s} CV ROC AUC {s['test_roc_auc'].mean():.3f}   CV AP {s['test_average_precision'].mean():.3f}")

# %% 4. Who gets the email: a threshold from the prices, chosen on the training rows
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

EMAIL_COST, CHURN_LOSS = 5, 50


def email_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(EMAIL_COST * (tp + fp) + CHURN_LOSS * fn)


tuned = TunedThresholdClassifierCV(make_model(), scoring=make_scorer(email_cost, greater_is_better=False), cv=cv)
tuned.fit(X_train, y_train)
print(f"\nthreshold {tuned.best_threshold_:.3f} (formula {EMAIL_COST / CHURN_LOSS:.2f})")

# %% 5. The test set, once
from sklearn.metrics import average_precision_score, roc_auc_score

p = tuned.predict_proba(X_test)[:, 1]
pred = tuned.predict(X_test)
reached = int(((pred == 1) & (y_test.to_numpy() == 1)).sum())
print(f"test ROC AUC {roc_auc_score(y_test, p):.3f}   AP {average_precision_score(y_test, p):.3f}")
print(f"emails {pred.sum()} of {len(pred)}, churners reached {reached} of {y_test.sum()}")
print(f"cost ${email_cost(y_test, pred):,}   email nobody ${email_cost(y_test, np.zeros(len(y_test), dtype=int)):,}"
      f"   email everyone ${email_cost(y_test, np.ones(len(y_test), dtype=int)):,}"
      f"   at 0.5 ${email_cost(y_test, tuned.estimator_.predict(X_test)):,}")

# %% 6. Three sentences for the marketing lead
print("""
The 95% figure came from a column that is filled in only after a customer has left; without it the model
ranks customers well (ROC AUC about 0.83, against 0.50 for guessing) but is not "95% accurate".
Emailing the 654 test customers above the cost-based cut-off reaches 214 of the 248 churners and costs
$4,970, against $7,500 for emailing everyone, $9,630 at the default cut-off and $12,400 for emailing nobody.
Caveats: the $5 and $50 are estimates (the cut-off moves if they change), and the voucher will not save
every churner it reaches, so the real saving is smaller.""")
