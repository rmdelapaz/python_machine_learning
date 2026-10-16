# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.5 A (SOLUTION): who gets a retention offer?

Lesson 2.5 priced fraud reviews. Here the retention team prices a different decision. Each flagged
customer gets a call and a voucher, costing $15. Each churner nobody contacts costs Pinecrest about $60
of margin. (To keep the arithmetic simple, assume the offer keeps every churner it reaches.) Churn is
16.5%, rarer than "stays" but far less rare than fraud: does the 0.5 cut-off still go wrong?
"""
# %% Step 1: the canonical split, 14 columns, a preprocessing function
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
num_cols = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user", "email_opens_90d"]
cat_cols = ["home_store", "loyalty_tier", "preferred_category"]
train, test = train_test_split(customers, test_size=0.25, stratify=customers["churned"], random_state=42)
X_train, y_train = train[num_cols + cat_cols], train["churned"]
X_test, y_test = test[num_cols + cat_cols], test["churned"]


def make_preprocess():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


print(f"train {len(train)}, test {len(test)}, test churners {y_test.sum()}")

# %% Step 2: the cost of a set of decisions, and the two do-nothing policies
from sklearn.metrics import confusion_matrix

OFFER_COST, CHURN_LOSS = 15, 60


def retention_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(OFFER_COST * (tp + fp) + CHURN_LOSS * fn)


print(f"offer nobody : ${retention_cost(y_test, np.zeros(len(y_test), dtype=int)):,}")
print(f"offer everyone: ${retention_cost(y_test, np.ones(len(y_test), dtype=int)):,}")

# %% Step 3: a plain model: how well does it rank, and what does 0.5 cost?
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

plain = make_pipeline(make_preprocess(), LogisticRegression()).fit(X_train, y_train)
p_plain = plain.predict_proba(X_test)[:, 1]
print(f"\nROC AUC {roc_auc_score(y_test, p_plain):.3f}   AP {average_precision_score(y_test, p_plain):.3f}"
      f"   (random ranking AP {y_test.mean():.3f})")
print(f"at 0.5: {plain.predict(X_test).sum()} offers, cost ${retention_cost(y_test, plain.predict(X_test)):,}")

# %% Step 4: sweep the threshold, then the formula
for t in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
    pred = (p_plain >= t).astype(int)
    print(f"threshold {t:<4}  offers {pred.sum():4d}  cost ${retention_cost(y_test, pred):,}")
print(f"formula threshold: {OFFER_COST} / {CHURN_LOSS} = {OFFER_COST / CHURN_LOSS}")

# %% Step 5: class weights instead
weighted = make_pipeline(make_preprocess(), LogisticRegression(class_weight="balanced")).fit(X_train, y_train)
p_weighted = weighted.predict_proba(X_test)[:, 1]
print(f"\nbalanced @ 0.5      : {weighted.predict(X_test).sum()} offers, cost ${retention_cost(y_test, weighted.predict(X_test)):,}")
print(f"mean probability    : real {y_test.mean():.3f}, plain {p_plain.mean():.3f}, balanced {p_weighted.mean():.3f}")
wrong = (p_weighted >= OFFER_COST / CHURN_LOSS).astype(int)
print(f"balanced @ formula  : {wrong.sum()} offers, cost ${retention_cost(y_test, wrong):,}")

# %% Step 6: let cross-validation choose the threshold on the training rows only
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuned = TunedThresholdClassifierCV(make_pipeline(make_preprocess(), LogisticRegression()),
                                   scoring=make_scorer(retention_cost, greater_is_better=False), cv=cv)
tuned.fit(X_train, y_train)
pred = tuned.predict(X_test)
caught = int(((pred == 1) & (y_test.to_numpy() == 1)).sum())
print(f"\ntuned threshold {tuned.best_threshold_:.3f}: {pred.sum()} offers, {caught} of {y_test.sum()} churners reached,"
      f" cost ${retention_cost(y_test, pred):,}")
